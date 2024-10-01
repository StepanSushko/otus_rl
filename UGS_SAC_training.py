# https://github.com/XinJingHao/SAC-Continuous-Pytorch/tree/main
# 
# %%

%load_ext autoreload
%autoreload 2

import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ! poetry remove nbformat => poetry add nbformat=5.10.4

# from utils import str2bool, evaluate_policy, Action_adapter, Action_adapter_reverse, Reward_adapter
from datetime import datetime
#from SAC import SAC_countinuous
import gymnasium as gym
import os, shutil
#import argparse
import torch

import UGS_environment

# %% Plot 3D surface of reward function for UGS 1 and 2 fixing first and second element of state vector on a field of state space of environment
    
NUMBER_OF_UGS = 2
    
env = UGS_environment.UGSEnv(number_of_ugs = NUMBER_OF_UGS, balance="soft")
    
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.animation as animation



x_1 = np.linspace(env.action_space.low[0], env.action_space.high[0], 20)
x_2 = np.linspace(env.action_space.low[1], env.action_space.high[1], 20)
x_1, x_2 = np.meshgrid(x_1, x_2)
rewards = np.zeros(x_1.shape)
for i in range(x_1.shape[0]):
        for j in range(x_1.shape[1]):
            act = np.array([x_1[i,j], x_2[i,j]])
            env.reset()
            rewards[i,j] = env.step(act)[1]
            

env.reset()
env.step( np.array([-30, -30]),  print_out = True)[1]
env.reset()
env.step( np.array([30, 30]),  print_out = True)[1]


env.reset()
env.step( np.array([-5, -10]),  print_out = True)[1]

env.reset()
env.step( np.array([5, -10]),  print_out = True)[1]

env.reset()
env.step( np.array([-6, -9]),  print_out = True)[1]

env.reset()
env.step( np.array([-7, -8]),  print_out = True)[1]

#productivity = 5.0 + 0.0000001*self.state[i]**3 + 0.00004*self.state[i]**2 - 0.02*self.state[i]
            # UGS2 productivity y = 10 - 0.00006*x^2 + 0.05*x

#productivity = 10.0 - 0.00006*self.state[i]**2 + 0.05*self.state[i]

import plotly.graph_objects as go

fig = go.Figure(data=[go.Surface(x=x_1, y=x_2, z=rewards, colorscale='RdBu_r')])
fig.update_layout(title='Rewards surface',
                  scene = dict(
                    xaxis_title='UGS 1',
                    yaxis_title='UGS 2',
                    zaxis_title='Reward'))
fig.add_trace(go.Scatter3d(
    x=x_1.ravel(),
    y=x_2.ravel(),
    z=rewards.ravel(),
    mode='markers',
    marker=dict(
        size=1,
        color='black',                # set color to an array/list of desired values
        opacity=0.8
    )))
fig.show()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.animation as animation


if False:




	x_1 = np.linspace(env.action_space.low[0], env.action_space.high[0], 100)
	x_2 = np.linspace(env.action_space.low[1], env.action_space.high[1], 100)
	x_1, x_2 = np.meshgrid(x_1, x_2)
	rewards = np.zeros(x_1.shape)
	for i in range(x_1.shape[0]):
			for j in range(x_1.shape[1]):
				act = np.array([x_1[i,j], x_2[i,j]])
				env.reset()
				rewards[i,j] = env.step(act)[1]
				
	fig = plt.figure(figsize=(8,6))
	ax = fig.add_subplot(111, projection='3d')
	surf = ax.plot_surface(x_1, x_2, rewards, cmap='plasma', edgecolor='none', animated=True)
	ax.view_init(elev=5, azim=20)
	ax.set_xlabel('UGS 1')
	ax.set_ylabel('UGS 2')
	ax.set_zlabel('Reward')

	plt.show()




# %%



def build_net(layer_shape, hidden_activation, output_activation):
	'''Build net with for loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = hidden_activation if j < len(layer_shape)-2 else output_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
	return nn.Sequential(*layers)


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape, hidden_activation=nn.ReLU, output_activation=nn.ReLU):
		super(Actor, self).__init__()
		layers = [state_dim] + list(hid_shape)

		self.a_net = build_net(layers, hidden_activation, output_activation)
		self.mu_layer = nn.Linear(layers[-1], action_dim)
		self.log_std_layer = nn.Linear(layers[-1], action_dim)

		self.LOG_STD_MAX = 2
		self.LOG_STD_MIN = -20

	def forward(self, state, deterministic, with_logprob):
		'''Network with Enforcing Action Bounds'''
		net_out = self.a_net(state)
		mu = self.mu_layer(net_out)
		log_std = self.log_std_layer(net_out)
		log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  #总感觉这里clamp不利于学习
		# we learn log_std rather than std, so that exp(log_std) is always > 0
		std = torch.exp(log_std)
		dist = Normal(mu, std)
		if deterministic: u = mu
		else: u = dist.rsample()

		'''↓↓↓ Enforcing Action Bounds, see Page 16 of https://arxiv.org/pdf/1812.05905.pdf ↓↓↓'''
		a = torch.tanh(u)
		if with_logprob:
			# Get probability density of logp_pi_a from probability density of u:
			# logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
			# Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
			logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)
		else:
			logp_pi_a = None

		return a, logp_pi_a

class Double_Q_Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Double_Q_Critic, self).__init__()
		layers = [state_dim + action_dim] + list(hid_shape) + [1]

		self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
		self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)
		q1 = self.Q_1(sa)
		q2 = self.Q_2(sa)
		return q1, q2

#reward engineering for better training
def Reward_adapter(r, EnvIdex):
	# For Pendulum-v0
	if EnvIdex == 0:
		r = (r + 8) / 8

	# For LunarLander
	elif EnvIdex == 1:
		if r <= -100: r = -10

	# For BipedalWalker
	elif EnvIdex == 4 or EnvIdex == 5:
		if r <= -100: r = -1
	return r


def Action_adapter(a,max_action):
	#from [-1,1] to [-max,max]
	return  a*max_action

def Action_adapter_reverse(act,max_action):
	#from [-max,max] to [-1,1]
	return  act/max_action


def evaluate_policy(env, agent, opt, turns = 1):
	total_scores = 0
	states_records = []
	actions_records = []
	rewards_records = []
	productivity_records = []
 
	total_scores = 0
	for j in range(turns):
		s, info = env.reset()
		done = False
		while not done:
			# Take deterministic actions at test time
			a = agent.select_action(s, deterministic=True)
			act = Action_adapter(a, opt.max_action)  # act∈[-max,max]
			s_next, r, dw, tr, info = env.step(act)
			done = (dw or tr)

			states_records.append(s)
   
			if env.balance == "hard":
				# ! NO DEFICIT
				act = np.append(act, -(s[1] + act.sum()))
			actions_records.append(act)
			rewards_records.append(r)
   
			productivity_records.append(
       			[round( 5.0 + 0.0000001*s[2]**3 + 0.00004*s[2]**2 - 0.02*s[2], 3 ),
                 round(10.0 - 0.00006*s[3]**2 + 0.05*s[3], 3)]
                )

			total_scores += r

			s = s_next
   

	for i in range(0,env.horizon,1):
		print( 
			"state = " +    "".join(
			"     ".join(str([ round(  s, 3) for s in state]) + "" for state in np.array(states_records)[i:i+1])
			),
			"action = " +    "".join( 
				"     ".join(str([ round(  a, 3) for a in action]) + "" for action in np.array(actions_records)[i:i+1]) 
			),
			"reward = " +    "".join( 
				"     ".join(str([ round(  a, 3) for a in action]) + "" for action in (np.array([rewards_records]).T)[i:i+1]) 
			),
			"productivity = " +    "".join( 
				"     ".join(str([ round(  a, 3) for a in action]) + "" for action in (np.array(productivity_records))[i:i+1])
			),
     		"(" +    "".join( 
			"     ".join(str( sum([ round(  a, 3) for a in action])) + ")" for action in (np.array(productivity_records))[i:i+1]) 
			)  )
  
	print( 
		"  Reward  =", round( total_scores, 2), 
		" Productivity =", str( round( sum(np.array(productivity_records)[-1]), 3))
	)
 
	return int(total_scores/turns), sum(np.array(productivity_records)[-1]), (np.array(states_records)[-1])[2], (np.array(states_records)[-1])[3]


def str2bool(v):
	'''transfer str to bool for argparse'''
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')




# %%

#from utils import Actor, Double_Q_Critic
import torch.nn.functional as F
import numpy as np
import torch
import copy


class SAC_countinuous():
	def __init__(self, **kwargs):
		# Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)
		self.tau = 0.005

		self.actor = Actor(self.state_dim, self.action_dim, (self.net_width,self.net_width)).to(self.dvc)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

		self.q_critic = Double_Q_Critic(self.state_dim, self.action_dim, (self.net_width,self.net_width)).to(self.dvc)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)
		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.q_critic_target.parameters():
			p.requires_grad = False

		self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(1e6), dvc=self.dvc)

		if self.adaptive_alpha:
			# Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
			self.target_entropy = torch.tensor(-self.action_dim, dtype=float, requires_grad=True, device=self.dvc)
			# We learn log_alpha instead of alpha to ensure alpha>0
			self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.dvc)
			self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.c_lr)

	def select_action(self, state, deterministic):
		# only used when interact with the env
		with torch.no_grad():
			state = torch.FloatTensor(state[np.newaxis,:]).to(self.dvc)
			a, _ = self.actor(state, deterministic, with_logprob=False)
		return a.cpu().numpy()[0]

	def train(self,):
		s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

		#----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
		with torch.no_grad():
			a_next, log_pi_a_next = self.actor(s_next, deterministic=False, with_logprob=True)
			target_Q1, target_Q2 = self.q_critic_target(s_next, a_next)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = r + (~dw) * self.gamma * (target_Q - self.alpha * log_pi_a_next) #Dead or Done is tackled by Randombuffer

		# Get current Q estimates
		current_Q1, current_Q2 = self.q_critic(s, a)

		q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		#----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
		# Freeze critic so you don't waste computational effort computing gradients for them when update actor
		for params in self.q_critic.parameters(): params.requires_grad = False

		a, log_pi_a = self.actor(s, deterministic=False, with_logprob=True)
		current_Q1, current_Q2 = self.q_critic(s, a)
		Q = torch.min(current_Q1, current_Q2)

		a_loss = (self.alpha * log_pi_a - Q).mean()
		self.actor_optimizer.zero_grad()
		a_loss.backward()
		self.actor_optimizer.step()

		for params in self.q_critic.parameters(): params.requires_grad = True

		#----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
		if self.adaptive_alpha:
			# We learn log_alpha instead of alpha to ensure alpha>0
			alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
			self.alpha_optim.zero_grad()
			alpha_loss.backward()
			self.alpha_optim.step()
			self.alpha = self.log_alpha.exp()

		#----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
		for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self,EnvName, timestep):
		torch.save(self.actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName,timestep))
		torch.save(self.q_critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName,timestep))

	def load(self,EnvName, timestep):
		self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, timestep), map_location=self.dvc))
		self.q_critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep), map_location=self.dvc))


class ReplayBuffer():
	def __init__(self, state_dim, action_dim, max_size, dvc):
		self.max_size = max_size
		self.dvc = dvc
		self.ptr = 0
		self.size = 0

		self.s = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
		self.a = torch.zeros((max_size, action_dim) ,dtype=torch.float,device=self.dvc)
		self.r = torch.zeros((max_size, 1) ,dtype=torch.float,device=self.dvc)
		self.s_next = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
		self.dw = torch.zeros((max_size, 1) ,dtype=torch.bool,device=self.dvc)

	def add(self, s, a, r, s_next, dw):
		#每次只放入一个时刻的数据
		self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
		self.a[self.ptr] = torch.from_numpy(a).to(self.dvc) # Note that a is numpy.array
		self.r[self.ptr] = torch.tensor(r, dtype=torch.float, device=self.dvc)
		self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
		self.dw[self.ptr] = dw

		self.ptr = (self.ptr + 1) % self.max_size #存满了又重头开始存
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]



# %%




'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda:1', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=6, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(5e6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(100e3), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(10e3), help='Model evaluating interval, in steps.')
parser.add_argument('--update_every', type=int, default=50, help='Training Fraquency, in steps')

parser.add_argument('--gamma', type=float, default=1.0, help='Discounted Factor') # !
parser.add_argument('--net_width', type=int, default=512, help='Hidden net width, s_dim-400-300-a_dim') # !
parser.add_argument('--a_lr', type=float, default=5e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256*32, help='batch_size of training') # !
parser.add_argument('--alpha', type=float, default=0.12, help='Entropy coefficient')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive_alpha or Not')

opt, unknown = parser.parse_known_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device


# TODO: 1. add noise to test task demand
# TODO: 2. plot objective function step by step w.r.t. a[1] and a[2]


# %% Plot 3D surface of reward function for UGS 1 and 2 fixing first and second element of state vector on a field of state space of environment
    
NUMBER_OF_UGS = 2
    
env = UGS_environment.UGSEnv(number_of_ugs = NUMBER_OF_UGS, demand="sinusoidal_with_noise", horizon = 180, random_start = True)
    



# %%


def main():

    EnvName = ['Pendulum-v1','LunarLanderContinuous-v2','Humanoid-v4','HalfCheetah-v4','BipedalWalker-v3','BipedalWalkerHardcore-v3', 'UGS-v0']
    BrifEnvName = ['PV1', 'LLdV2', 'Humanv4', 'HCv4','BWv3', 'BWHv3', 'UGS']

    # Build Env
    #env = gym.make(EnvName[opt.EnvIdex], render_mode = "human" if opt.render else None)
    NUMBER_OF_UGS = 3
    
    env = UGS_environment.UGSEnv(number_of_ugs = NUMBER_OF_UGS, balance="hard", demand="sinusoidal_with_noise", horizon = 180, random_start = True)
    eval_env = UGS_environment.UGSEnv(number_of_ugs = NUMBER_OF_UGS, balance="hard", demand="sinusoidal_with_noise", horizon = 180, random_start = True)
	#env.reset(seed=42) # 

    #eval_env = gym.make(EnvName[opt.EnvIdex])
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.shape[0]
    opt.max_action = float(env.action_space.high[0])   #remark: action space【-max,max】
    opt.max_e_steps = 2000 #env.horizon
    print(f'Env:{EnvName[opt.EnvIdex]}  state_dim:{opt.state_dim}  action_dim:{opt.action_dim}  '
          f'max_a:{opt.max_action}  min_a:{env.action_space.low[0]}  max_e_steps:{opt.max_e_steps}')

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))


    # Build SummaryWriter to record training curves
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format( 'UGSenv_SAC') + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)


    # Build DRL model
    if not os.path.exists('model'): os.mkdir('model')
    agent = SAC_countinuous(**vars(opt)) # var: transfer argparse to dictionary
    if opt.Loadmodel: agent.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex)

    if opt.render:
        while True:
            score = evaluate_policy(env, agent, opt, turns=1)
            print('EnvName:', BrifEnvName[opt.EnvIdex], 'score:', score)
    else:
        total_steps = 0
        while total_steps < opt.Max_train_steps:
            s, info = env.reset(seed=env_seed)  # Do not use opt.seed directly, or it can overfit to opt.seed
            env_seed += 1
            done = False

            '''Interact & trian'''
            while not done:
                if total_steps < (5*opt.max_e_steps):
                    act = env.action_space.sample()  # act∈[-max,max]
                    a = Action_adapter_reverse(act, opt.max_action)  # a∈[-1,1]
                else:
                    a = agent.select_action(s, deterministic=False)  # a∈[-1,1]
                    act = Action_adapter(a, opt.max_action)  # act∈[-max,max]
                s_next, r, dw, tr, info = env.step(act)  # dw: dead&win; tr: truncated
                r = Reward_adapter(r, opt.EnvIdex)
                done = (dw or tr)

                agent.replay_buffer.add(s, a, r, s_next, dw)
                s = s_next
                total_steps += 1

                '''train if it's time'''
                # train 50 times every 50 steps rather than 1 training per step. Better!
                if (total_steps >= 2*opt.max_e_steps) and (total_steps % opt.update_every == 0):
                    for j in range(opt.update_every):
                        agent.train()

                '''record & log'''
                if total_steps % opt.eval_interval == 0:
                    ep_r, prod, v1, v2 = evaluate_policy(eval_env, agent, opt, turns=1)
                    if opt.write: writer.add_scalar('ep_r', ep_r, global_step=total_steps)
                    if opt.write: writer.add_scalar('productivity', prod, global_step=total_steps)
                    for i in range(env.number_of_ugs):
                        if opt.write: writer.add_scalar(f'Volume{i+1}', eval(f'v{i+1}'), global_step=total_steps)
                    print(f'EnvName:{BrifEnvName[opt.EnvIdex]}, Steps: {int(total_steps/1000)}k, Episode Reward:{ep_r}', "\n")

                '''save model'''
                if total_steps % opt.save_interval == 0:
                    agent.save(BrifEnvName[opt.EnvIdex], int(total_steps/1000))
        env.close()
        eval_env.close()
        
        
        #evaluate_policy(env, agent, opt, turns = 3)


if __name__ == '__main__':
    main()

# %%
