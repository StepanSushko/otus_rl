import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

        self.maxaction = maxaction

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.maxaction
        return a


class Double_Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Double_Q_Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, net_width)  #没有先提取特征
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, net_width)
        self.l5 = nn.Linear(net_width, net_width)
        self.l6 = nn.Linear(net_width, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

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
			#a = agent.select_action(s, deterministic=True)
			#! act = Action_adapter(a, opt.max_action)  # act∈[-max,max]
            #s_next, r, dw, tr, info = env.step(act)
            # Take deterministic actions at test time   
            
			a = agent.select_action(s, deterministic=True)
			s_next, r, dw, tr, info = env.step(a)
			done = (dw or tr)
			total_scores += r
   
			states_records.append(s)
			if env.balance == "hard":
				# ! NO DEFICIT
				a = np.append(a, -(s[1] + a.sum()))
			actions_records.append(a)
			rewards_records.append(r)

			productivity_records.append(
       			[round( 5.0 + 0.0000001*s[2]**3 + 0.00004*s[2]**2 - 0.02*s[2], 3 ),
                 round(10.0 - 0.00006*s[3]**2 + 0.05*s[3], 3)]
                )
            
			s = s_next

   
	for i in range(0,env.horizon,1):
		print( 
					
					#" V_loss =", round( np.mean(loss_q_records[-200:].cpu()), 2),
					#" Pi_loss =", round( np.mean(loss_mu_records[-200:]), 2), 
					#" Episode_len =", episode_len,
					#" Computation time:", round(computation_time/60, 2), " minutes",
					#"  \na =", actions[0:4]
					"state = " +    "".join(
						"     ".join(str([ round(  s, 3) for s in state]) + "" for state in np.array(states_records)[i:i+1]) # actions[0]
						
					),
					"action = " +    "".join( 
						"     ".join(str([ round(  a, 3) for a in action]) + "" for action in np.array(actions_records)[i:i+1]) # actions[0]
						
					),
					"reward = " +    "".join( 
						"     ".join(str([ round(  a, 3) for a in action]) + "" for action in (np.array([rewards_records]).T)[i:i+1]) # actions[0]
						
					),
					"productivity = " +    "".join( 
						"     ".join(str([ round(  a, 3) for a in action]) + "" for action in (np.array(productivity_records))[i:i+1]) # actions[0]
						
					),
     					"(" +    "".join( 
						"     ".join(str( sum([ round(  a, 3) for a in action])) + ")" for action in (np.array(productivity_records))[i:i+1]) # actions[0]
						
					)
					#"             a = " + " ".join(   str([ round(  a, 2) for a in action]) for action in actions[0:10]) # type(actions[0])
					#"             a = " + " ".join(str(a) for a in actions[0:10])
					)   
	print( 
					"  Reward  =", round( total_scores, 2), 
					" Productivity =", str( round( sum(np.array(productivity_records)[-1]), 3))
	)
 
	return int(total_scores/turns), sum(np.array(productivity_records)[-1]), (np.array(states_records)[-1])[2], (np.array(states_records)[-1])[3]




#Just ignore this function~
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