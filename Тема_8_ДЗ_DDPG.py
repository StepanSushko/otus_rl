# %%
import random
import numpy as np
import gymnasium as gym
import torch # !!! pip install torch==2.3.0 torchvision==0.18 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm 

from IPython.display import HTML
import os
import datetime
import matplotlib.animation as animation

import time
import cv2


# %%

torch.cuda.is_available()

# %%

def preprocess(img):
    img = img[:84, 6:90] # CarRacing-v2-specific cropping
    # img = cv2.resize(img, dsize=(84, 84)) # or you can simply use rescaling
    
    # Remove all green squares
    green_mask = ( (np.abs( img[:,:,1] )>= 208) & (np.abs(img[:,:,1] ) <= 248) )
    img[green_mask] = np.array([100, 202, 100])

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img


class ImageEnv_GrayScaled(gym.Wrapper):
    def __init__(
        self,
        env,
        skip_frames=5,
        stack_frames=10,
        initial_no_op=50,
        **kwargs
    ):
        super(ImageEnv_GrayScaled, self).__init__(env, **kwargs)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
    
    def reset(self):
        # Reset the original environment.
        s, info = self.env.reset()

        # Do nothing for the next `self.initial_no_op` steps
        for i in range(self.initial_no_op):
            s, r, terminated, truncated, info = self.env.step([ 0.0, 0.0, 0.0])
        
        # Convert a frame to 84 X 84 gray scale one
        s = preprocess(s)
        s = s[..., np.newaxis]

        # The initial observation is simply a copy of the frame `s`
        self.stacked_state = np.tile(s, (1, 1, self.stack_frames))  # [96, 96, 12]  # [4, 84, 84]
        #self.stacked_state = np.tile(s, (self.stack_frames, 1, 1))  # [4, 84, 84]
        return self.stacked_state, info
    
    def step(self, action):
        # We take an action for self.skip_frames steps
        reward = 0
        for _ in range(self.skip_frames):
            s, r, terminated, truncated, info = self.env.step(action)
            reward += r
            if terminated or truncated:
                break

        # Convert a frame to 84 X 84 gray scale one
        s = preprocess(s)
        s = s[..., np.newaxis]

        # Push the current frame `s` at the end of self.stacked_state
        #self.stacked_state = np.concatenate((self.stacked_state[1:], s[np.newaxis]), axis=0)
        self.stacked_state = np.concatenate((self.stacked_state[:,:,1:], s), axis=2)

        return self.stacked_state, reward, terminated, truncated, info






# %% [markdown]
# ## Original

# %%


def prepare_state(s):
        s_batch = s
        #s_batch = np.expand_dims(s, axis=0)
        
        # Assuming obs is your observation in the form Box(0, 255, (96, 96, 3), uint8)
        #obs = np.random.randint(0, 256, size=(96, 96, 3), dtype=np.uint8)  # Example random observation

        # Normalize the observation values
        s_batch_normalized = s_batch.astype(np.float32) # !!!/ 255.0  # Normalizing to [0, 1] range

        # Convert the observation to a PyTorch tensor
        # NCHW stands for: batch N, channels C, depth D, height H, width W
        # from 96x96x3 to 3x96x96 (3 - channels, 96 - height, 96 - width)  three channel picture
        s_batch_tensor = torch.tensor(s_batch_normalized).permute(2, 0, 1)#.unsqueeze(0)  # Assuming NHWC to NCHW format
        # s_batch_tensor.size()

        #s_batch = s_batch_tensor.unsqueeze(0).clone().detach().to(device) # s_batch.size()
        
        return s_batch_tensor

class QNet(nn.Module):
    def __init__(self, hidden_dim=16, frames_number = 10, channels = 1):
        super().__init__()
        
        #self.hidden = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3) 

        #self.hidden = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=3) #nn.Linear( in_features = 96*96*3, out_features = hidden_dim)
        
        # Формула расчета свёрточной сети:
        # N=(W-F+2P)/S+1 где N: размер вывода W: размер ввода F: размер ядра свертки P: размер значения заполнения S: размер шага
        
        # Variant 2.2 Larger and better than Variant 2.1  "Racetrack Navigation on OpenAIGym with Deep Reinforcement Learning"" (DDQN)
        self.hidden = nn.Conv2d(in_channels=channels*frames_number, out_channels=64,  kernel_size=8, stride=4) 
        self.hidden2 = nn.Conv2d(in_channels=64,                    out_channels=128, kernel_size=4, stride=2)  # (8, 47, 47) 
        self.hidden3 = nn.Conv2d(in_channels=128,                   out_channels=128, kernel_size=3, stride=1)  # (8, 47, 47) 
        
        self.hidden_linear1 = nn.Linear(  in_features = 6272, out_features = 128)
        self.hidden_linear2 = nn.Linear( in_features = 128, out_features = 4)
        
        self.hidden_linear3 = nn.Linear( in_features = 4 + 3, out_features = 7)
        self.output_linear = nn.Linear( in_features = 7, out_features = 1)

        #self.hidden = nn.Linear(96*96*3, hidden_dim)
        #self.output = nn.Linear(hidden_dim, 1)

    def forward(self, s, a):
        
        outs = self.hidden(s)
        outs = F.relu(outs)
        
        outs = self.hidden2(outs)
        outs = F.relu(outs)
        
        outs = self.hidden3(outs)
        outs = F.relu(outs)
        
        outs = torch.flatten( outs, 1 )
        
        outs = self.hidden_linear1( outs )
        #outs = F.relu(outs)
        
        outs = self.hidden_linear2( outs )
        #outs = F.relu(outs)
        
        outs = torch.concat((outs, a), dim=-1) # !!!!
        
        outs = self.hidden_linear3( outs )
        outs = F.relu(outs)
        
        value = self.output_linear(outs)
        
        return value



# %%
class PolicyNet(nn.Module):
    def __init__(self, hidden_dim=16, frames_number = 10, channels = 1):
        super().__init__()

        self.hidden = nn.Conv2d(in_channels=channels*frames_number, out_channels=64,  kernel_size=8, stride=4) 
        self.hidden2 = nn.Conv2d(in_channels=64,                    out_channels=128, kernel_size=4, stride=2)  # (8, 47, 47) 
        self.hidden3 = nn.Conv2d(in_channels=128,                   out_channels=128, kernel_size=3, stride=1)  # (8, 47, 47) 
        
        self.hidden_linear1 = nn.Linear(  in_features = 6272, out_features = 64)
        self.hidden_linear2 = nn.Linear( in_features = 64, out_features = 32)
        
        self.steering = nn.Linear( in_features = 32, out_features = 1)
        self.acceleration = nn.Linear( in_features = 32, out_features = 1)
        self.brake = nn.Linear( in_features = 32, out_features = 1)
        
    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        
        outs = self.hidden2(outs)
        outs = F.relu(outs)
        
        outs = self.hidden3(outs)
        outs = F.relu(outs)

        outs = torch.flatten( outs, 1 )
        
        outs = self.hidden_linear1( outs )
        outs = self.hidden_linear2( outs )
        
        steering     = F.tanh(    self.steering(outs) )/4.0 #- F.tanh(    self.steering2(outs) )
        acceleration = F.sigmoid( self.acceleration(outs) )/10.0
        brake        = F.sigmoid( self.brake(outs) )/1000.0
        
        return steering, acceleration, brake


# %%
def optimize(states, actions, rewards, next_states, dones):
    # Convert to tensor
    states = torch.tensor(np.array(states), dtype=torch.float).to(device) # np.array(states).shape states.size()
    actions = torch.tensor(np.array(actions), dtype=torch.float).to(device)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float).to(device)
    rewards = rewards.unsqueeze(dim=1)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(device) # next_states.shape
    dones = torch.tensor(np.array(dones), dtype=torch.float).to(device)
    dones = dones.unsqueeze(dim=1)

    
    # Optimize critic loss
    opt_q.zero_grad()
    q_org = q_origin_model( states, actions ) # states.shape, actions.shape  outs = torch.concat((states, actions), dim=-1) states[0].size()
    #q_org = q_origin_model( states, prepare_state( actions ) ) # states.shape, actions.shape  outs = torch.concat((states, actions), dim=-1)
    mu_tgt_next = mu_target_model(next_states)  # PolicyNet
    mu_tgt_next = torch.stack( (mu_tgt_next[0][:,0], mu_tgt_next[1][:,0], mu_tgt_next[2][:,0]), dim=1)
    
    q_tgt_next = q_target_model(next_states, mu_tgt_next)
    q_tgt = rewards + gamma * (1.0 - dones) * q_tgt_next
    loss_q = F.mse_loss(
        q_org,
        q_tgt,
        reduction="none")
    loss_q.sum().backward()
    opt_q.step()

    # Optimize actor loss
    opt_mu.zero_grad()
    mu_org = mu_origin_model(states)
    mu_org = torch.stack( (mu_org[0][:,0], mu_org[1][:,0], mu_org[2][:,0]), dim=1)
    
    for p in q_origin_model.parameters():
        p.requires_grad = False # disable grad in q_origin_model before computation
    q_tgt_max = q_origin_model(states, mu_org)
    (-q_tgt_max).sum().backward()
    opt_mu.step()
    
    for p in q_origin_model.parameters():
        p.requires_grad = True # enable grad again
        
    return loss_q.sum(), q_tgt.sum()

# %%
tau = 0.002

def update_target():
    for var, var_target in zip(q_origin_model.parameters(), q_target_model.parameters()):
        var_target.data = tau * var.data + (1.0 - tau) * var_target.data
    
    for var, var_target in zip(mu_origin_model.parameters(), mu_target_model.parameters()):
        var_target.data = tau * var.data + (1.0 - tau) * var_target.data

# %%
class replayBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = []
        self._next_idx = 0

    def add(self, item):
        if len(self.buffer) > self._next_idx:
            self.buffer[self._next_idx] = item
        else:
            self.buffer.append(item)
        if self._next_idx == self.buffer_size - 1:
            self._next_idx = 0
        else:
            self._next_idx = self._next_idx + 1

    def sample(self, batch_size):
        indices = [random.randint(0, len(self.buffer) - 1) for _ in range(batch_size)]
        states   = [self.buffer[i][0] for i in indices]
        actions  = [self.buffer[i][1] for i in indices]
        rewards  = [self.buffer[i][2] for i in indices]
        n_states = [self.buffer[i][3] for i in indices]
        dones    = [self.buffer[i][4] for i in indices]
        return states, actions, rewards, n_states, dones

    def length(self):
        return len(self.buffer)


# %%
"""
Ornstein-Uhlenbeck noise implemented by OpenAI
Copied from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
"""
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


# %%



# pick up action with Ornstein-Uhlenbeck noise
def pick_sample(s):
    with torch.no_grad():
        # s = np.array(s)
        #s_batch = np.expand_dims(s, axis=0)
        #s_batch = torch.tensor(s_batch, dtype=torch.float).to(device) # s_batch.size()
        
        s_batch = s
        #s_batch = np.expand_dims(s, axis=0)
        
        # Assuming obs is your observation in the form Box(0, 255, (96, 96, 3), uint8)
        #obs = np.random.randint(0, 256, size=(96, 96, 3), dtype=np.uint8)  # Example random observation

        # Normalize the observation values
        s_batch_normalized = s_batch.astype(np.float32) #/ 255.0  # Normalizing to [0, 1] range

        # Convert the observation to a PyTorch tensor
        # NCHW stands for: batch N, channels C, depth D, height H, width W
        # from 96x96x3 to 3x96x96 (3 - channels, 96 - height, 96 - width)  three channel picture
        s_batch_tensor = torch.tensor(s_batch_normalized).permute(2, 0, 1)#.unsqueeze(0)  # Assuming NHWC to NCHW format
        # s_batch_tensor.size()

        s_batch = s_batch_tensor.unsqueeze(0).clone().detach().to(device) # s_batch.size()
        
        steering, acceleration, brake  = mu_origin_model(s_batch)
        #steering = steering.squeeze(dim=1)
        #acceleration = acceleration.squeeze(dim=1)
        #brake = brake.squeeze(dim=1)
        
        noise_steering = ou_steering_noise()
        noise_acceleration = ou_acceleration_noise()
        #noise_brake = ou_action_noise()
        
        steering = steering.cpu().numpy() + noise_steering
        acceleration = acceleration.cpu().numpy() + noise_acceleration
        brake = brake.cpu().numpy() #+ noise_brake # !!!!!
        
        action = np.array([steering, acceleration, brake])
        
        action = np.clip(action, -1.0, 1.0)
        
        #return np.array([float(action.item())])
        return [steering.tolist()[0][0], acceleration.tolist()[0][0], brake.tolist()[0][0]], s_batch_tensor #a.tolist()[0]
    
    
# %%

def weights_init_uniform_rule(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # get the number of the inputs
            n = m.in_features
            y = 1.0/np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)

    
# %%
stacked_frames = 10
channels = 1

env = gym.make(
    "CarRacing-v2", 
    options={"randomize": False}
    )  # среда

env = ImageEnv_GrayScaled(
    env,
    skip_frames = 3,
    stack_frames = stacked_frames,
    initial_no_op = 50)



# %% Initialization

# Create objects

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

q_origin_model = QNet( frames_number = stacked_frames, channels=channels ).to(device)  # Q_phi
q_target_model = QNet( frames_number = stacked_frames, channels=channels ).to(device)  # Q_phi'
_ = q_target_model.requires_grad_(False)  # target model doen't need grad

mu_origin_model = PolicyNet( frames_number = stacked_frames, channels=channels ).to(device)  # mu_theta
mu_target_model = PolicyNet( frames_number = stacked_frames, channels=channels ).to(device)  # mu_theta'
_ = mu_target_model.requires_grad_(False)  # target model doen't need grad

q_origin_model.apply(weights_init_uniform_rule) # !!!!
q_target_model.apply(weights_init_uniform_rule) # !!!!
mu_origin_model.apply(weights_init_uniform_rule) # !!!!
mu_target_model.apply(weights_init_uniform_rule) # !!!!

gamma = 0.99
opt_q  = torch.optim.AdamW(q_origin_model.parameters(), lr=0.0005)
opt_mu = torch.optim.AdamW(mu_origin_model.parameters(), lr=0.0005)

buffer = replayBuffer(buffer_size=20000)

ou_steering_noise     = OrnsteinUhlenbeckActionNoise(mu=np.zeros(1), sigma=np.ones(1) * 0.01)
ou_acceleration_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(1), sigma=np.ones(1) * 0.0025)

# Create directory once
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = './gif_animations/DDPG/' + timestamp
os.makedirs(folder_name, exist_ok=True)     

reward_records = []

episodes_number = 5000
episode = 0

loss_q_records = []
loss_mu_records = []

if False:
    for p in q_origin_model.parameters():
        p.requires_grad = True # enable grad again

if False:
    # Load model checkpoint
    checkpoint = torch.load('./gif_animations/DDPG/DDPG_agent_q_origin.pt')
    q_origin_model.load_state_dict(checkpoint['model_state_dict'])
    
    reward_records = checkpoint['reward_records']
    loss_q_records = checkpoint['loss_q_records']
    loss_mu_records = checkpoint['loss_mu_records']
    
    checkpoint = torch.load('./gif_animations/DDPG/DDPG_agent_q_target.pt')
    q_target_model.load_state_dict(checkpoint['model_state_dict'])
    checkpoint = torch.load('./gif_animations/DDPG/DDPG_agent_mu_origin.pt')
    mu_origin_model.load_state_dict(checkpoint['model_state_dict'])
    checkpoint = torch.load('./gif_animations/DDPG/DDPG_agent_mu_target.pt')
    mu_target_model.load_state_dict(checkpoint['model_state_dict'])

    episode = checkpoint['epoch']


# %%

batch_size = 1000

while episode < episodes_number:
    # Run episode till done
    episode = episode + 1
    
    # Episode compuation time
    start_time = time.time()
    
    s = env.reset()[0] # s.shape
    done = False
    cum_reward = 0
    episode_len = 0
    rewards_buffer = []
    actions_buffer = []
    frames = []
    while not done:
        a, s_batch_tensor = pick_sample(s)
        s_next, r, done, truncate, _ = env.step(a) # a.size() s.shape s_batch_tensor.size()
        frames.append(s_next) # !
        done = done or truncate
        buffer.add([ np.array( s_batch_tensor ), a, r, np.array(prepare_state(s_next)), float(done)]) # s_next.shape prepare_state(s_next).size()  type(s_batch_tensor) s_batch_tensor.size()
        cum_reward += r
        
        actions_buffer.append(a)
    
        # Train (optimize parameters)
        if buffer.length() >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)  #  next_states.shape  buffer.length() buffer[0]
            loss_q, loss_mu = optimize(states, actions, rewards, next_states, dones)
            loss_q_records.append( loss_q )
            loss_mu_records.append( loss_mu )
            update_target()
        s = s_next
        
        episode_len = episode_len + 1
        
        # Terminate failed episodes
        if (len([reward for reward in rewards_buffer[-25:] if reward > 0]) == 0) and (len(rewards_buffer) > 25):
                #rewards.append( - 200.0 )
                rewards_buffer.append(r)
                done = True
        else:
                rewards_buffer.append(r)
    
    # Output total rewards in episode (max 500)
    # print("Run episode{} with rewards {}".format(i, cum_reward), end="\r")
    reward_records.append(cum_reward)
    
    end_time = time.time()
    computation_time = end_time - start_time

    print ( "  Reward  =", round( reward_records[-1:][0], 2),
            #" V_loss =", round(  v_loss_records[-1:][0], 2),
            #" Pi_loss =", round( pi_loss_records[-1:][0], 2), 
            " Episode_len =", episode_len,
            " Computation time:", round(computation_time/60, 2), " minutes",
            #"  \na =", actions[0:4]
            "\n" +    "".join( 
                "     ".join(str([ round(  a, 2) for a in action]) + "\n" for action in np.array(actions_buffer)[i:i+1]) # actions[0]
                for i in range(0,100,10)
            )
            #"             a = " + " ".join(   str([ round(  a, 2) for a in action]) for action in actions[0:10]) # type(actions[0])
            #"             a = " + " ".join(str(a) for a in actions[0:10])
            )
    
    if episode % 5 == 0:
            print("\nRun episode {} with average Reward {} ".format(
                episode, 
                round(np.mean(reward_records[-50:]), 2), 
                #round(np.mean(v_loss_records[-20:]), 2), 
                #round(np.mean(pi_loss_records[-20:]))
                2), 
                  end="\n\n\n")
            
    if episode % 10 == 0:
            if True:
                # Create animation
                if channels == 1:
                    cmap='gray'
                else:
                    cmap=None
                
                fig = plt.figure(figsize=(5, 5))
                plt.axis('off')
                im = plt.imshow(frames[0][:,:, (stacked_frames-1):(stacked_frames-1 + channels)], cmap=cmap)
                def animate(i):
                    im.set_array(frames[i][:,:,(stacked_frames-1):(stacked_frames-1 + channels)])
                    return im,
                anim = animation.FuncAnimation(fig, animate, frames=len(frames))

                # Save animation to file
                anim.save(folder_name + '/animation_' + str(episode) + '.gif') #, writer='imagemagick')
                #HTML(anim.to_jshtml())
                
                
                
                # Plot v_loss_records  k = 0
                plt.figure(figsize=(5, 5))
                plt.plot( [loss_q_records[k].cpu().tolist() for k in range(len(loss_q_records))] )
                plt.xlabel('i')
                plt.ylabel('q_loss')
                plt.title('q_loss vs i')
                #plt.show()
                
                plt.savefig(folder_name + '/q_loss_' + str(episode) + '.png')
                
                
                # Plot v_loss_records
                plt.figure(figsize=(5, 5))
                plt.plot( [loss_mu_records[k].cpu().tolist() for k in range(len(loss_mu_records))] )
                plt.xlabel('i')
                plt.ylabel('mu_loss')
                plt.title('mu_loss vs i')
                #plt.show()
                
                plt.savefig(folder_name + '/mu_loss_' + str(episode) + '.png')
                
                
                
                
                plt.figure(figsize=(5, 5))
                average_reward = []
                for idx in range(len(reward_records)):
                    avg_list = np.empty(shape=(1,), dtype=int)
                    if idx < 50:
                        avg_list = reward_records[:idx+1]
                    else:
                        avg_list = reward_records[idx-49:idx+1]
                    average_reward.append(np.average(avg_list))

                # Plot
                plt.plot(reward_records, label='reward')
                plt.plot(average_reward, label='average reward')
                plt.xlabel('N episode')
                plt.ylabel('Reward')
                plt.legend()
                #plt.show();
                
                plt.savefig(folder_name + '/reward_' + str(episode) + '.png')
                
                plt.close("all")
                
                #EPOCH = episode
                #PATH = folder_name + "/DDPG_agent.pt"
                #LOSS = loss#.item()

                torch.save({
                            'epoch': episode,
                            'model_state_dict': q_origin_model.state_dict(),
                            'optimizer_state_dict': opt_q.state_dict(),
                            'reward_records': reward_records,
                            'loss_q_records': loss_q_records,
                            'loss_mu_records': loss_mu_records,
                            #'buffer': buffer
                            #'loss': LOSS,
                            }, "./gif_animations/DDPG//DDPG_agent_q_origin.pt")
                
                torch.save({
                            'epoch': episode,
                            'model_state_dict': q_target_model.state_dict(),
                            'optimizer_state_dict': opt_q.state_dict(),
                            #'loss': LOSS,
                            }, "./gif_animations/DDPG//DDPG_agent_q_target.pt")
                
                torch.save({
                            'epoch': episode,
                            'model_state_dict': mu_origin_model.state_dict(),
                            'optimizer_state_dict': opt_mu.state_dict(),
                            #'loss': LOSS,
                            }, "./gif_animations/DDPG//DDPG_agent_mu_origin.pt")
                
                torch.save({
                            'epoch': episode,
                            'model_state_dict': mu_target_model.state_dict(),
                            'optimizer_state_dict': opt_mu.state_dict(),
                            #'loss': LOSS,
                            }, "./gif_animations/DDPG//DDPG_agent_mu_target.pt")



    # stop if reward mean > 475.0
    if np.average(reward_records[-50:]) > 475.0:
        break

print("\nDone")

# %%
# Generate recent 50 interval average
average_reward = []
for idx in range(len(reward_records)):
    avg_list = np.empty(shape=(1,), dtype=int)
    if idx < 50:
        avg_list = reward_records[:idx+1]
    else:
        avg_list = reward_records[idx-49:idx+1]
    average_reward.append(np.average(avg_list))
plt.plot(reward_records, label='reward')
plt.plot(average_reward, label='average reward')
plt.xlabel('N episodes')
plt.ylabel('Reward')
plt.legend()
plt.show()

# %%



