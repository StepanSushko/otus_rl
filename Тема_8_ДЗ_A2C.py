# %% [markdown]
# # Advantage Actor-Critic A2C
# 
# Здесь рассмотрим основные идеи алгоритма, приведем псевдокод и его реализацию на Python.
# 
# Actor-Critic это смешанный подход, включающий обновление V-функции и обновление политики.
# 
# В этом алгоритме агент одновременно обучает две сети. Одну для политики, она используется для предсказания действия на каждом шаге и вторую для аппроксимации V-функции на следующем шаге.

# %% [markdown]
# ## Реализация:
# 1. Инициализируем случайным образом сети политики (actor) $\pi^{\mu}(a|s)|_{\theta^{\mu}}$ и V-функции (critic) $V^{\theta}(s)|_{\theta^{V}}$ с весами $\theta^V$ и $\theta^{\mu}$ и целевые сети $V'$ и $\pi'$: $\theta^{V'} \gets \theta^V$ и $\theta^{\mu'} \gets \theta^{\mu}$
# 2. Устанавливаем число эпизодов обучения $M$ и для каждого эпизода выполняем:
# 3. Проходим траекторию, пока не достигнем конечного состояния.
#     - Находясь в состоянии $s_t$ действуем в силу текущей политики и выбираем действие $a_t = \pi^{\mu}(s_t)|_{\theta^{\mu}}$
#     - Выполняем действие $a_t$ и переходим в состояние $s_{t+1}$ и получаем награду $r_t$
#     - В состоянии $s_{t+1}$ действуя в силу текущей политики выбираем действие $a_{t+1} = \pi^{\mu}(s_{t+1})|_{\theta^{\mu}}$
#     - Вычисляем $Loss(\theta^V)=\big( r_t + \gamma V^{\theta}(s_{t+1}) - V^{\theta}(s_t) \big)^2$
#     - Вычисляем $Loss(\theta^{\mu}) = \ln{\pi^{\mu}(a_t|s_t)}(r_t + \gamma V^{\theta}(s_{t+1}) - V^{\theta}(s_t))$
#     - Обновляем веса: </br>
#     __Внимание!__ У V-функции мы ___минимизируем___ веса, а в политике ___максимизируем_!__ </br>
#       $\quad \quad \theta^V \gets \theta^V - \alpha \nabla_{\theta^V}Loss(\theta^V)$, </br>
#       $\quad \quad \theta^{\mu} \gets \theta^{\mu} + \beta \nabla_{\theta^{\mu}}Loss(\theta^{\mu})$ 
#     - Обновляем целевые сети: </br>
#     $\quad \quad \theta^{V'} \gets \tau \theta^V + (1 - \tau) \theta^{V'}$, </br>
#     $\quad \quad \theta^{\mu'} \gets \tau \theta^{\mu} + (1 - \tau) \theta^{\mu'}$


# %%
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

#import matplotlib
from IPython.display import HTML
import os
import datetime

from tqdm import tqdm, trange
gym.__version__


# %% [markdown]
# # Imports

torch.cuda.is_available()

# %% Simple CNN


# Assuming obs is your observation in the form Box(0, 255, (96, 96, 3), uint8)
obs = np.random.randint(0, 256, size=(96, 96, 3), dtype=np.uint8)  # Example random observation

# Normalize the observation values
obs_normalized = obs.astype(np.float32) / 255.0  # Normalizing to [0, 1] range

# Convert the observation to a PyTorch tensor
# NCHW stands for: batch N, channels C, depth D, height H, width W
# from 96x96x3 to 3x96x96 (3 - channels, 96 - height, 96 - width)  three channel picture
obs_tensor = torch.tensor(obs_normalized).permute(2, 0, 1).unsqueeze(0)  # Assuming NHWC to NCHW format
# obs_tensor[2][0].size()

# Define a simple CNN model with Conv2d layer
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Output size = (Input size - Kernel size + 2 * Padding) / Stride + 1
        # Assuming a default padding of 0 and a stride of 1, the output size would be:
        # Output size = (3 - 3 + 2 * 0) / 1 + 1 Output size = 1
        # So, the output size of the convolution would be 1.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3) # 3 x 96 x 96 to 1 ??

    def forward(self, x):
        x = self.conv1(x)
        return x

# Create an instance of the SimpleCNN model
model = SimpleCNN()

# Pass the observation tensor through the model
output = model(obs_tensor)

# output.size()
# 3*96*96
# 16*94*94 ??
# :
# The number of output channels is 16 as specified in the Conv2d layer.
# The formula to calculate the output size of a convolution operation is: Output size = ((Input size - Kernel size + 2 * Padding) / Stride) + 1 Here, assuming default padding of 0 and a stride of 1:
# For the height dimension: ((96 - 3 + 2 * 0) / 1) + 1 = 94
# For the width dimension: ((96 - 3 + 2 * 0) / 1) + 1 = 94
# The output size will have a single batch dimension, 16 channels, and a spatial resolution of 94x94 pixels after the convolution operation.

#!!! SimpleCNN is not used in the A2C algorithm.

# %%

if False:
    s.shape
    stacked_state = np.tile(s, (1, 1, 4)).shape

    np.concatenate((stacked_state[:,:,3:], s), axis=2).shape

import cv2

def preprocess(img):
    img = img[:84, 6:90] # CarRacing-v2-specific cropping
    # img = cv2.resize(img, dsize=(84, 84)) # or you can simply use rescaling
    
    # Remove all green squares
    green_mask = ( (np.abs( img[:,:,1] )>= 208) & (np.abs(img[:,:,1] ) <= 248) )
    img[green_mask] = np.array([100, 202, 100])

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img


if False:
    s_new = s.copy()
    
    green_mask = ( (np.abs(s[:,:,1] )>= 208) & (np.abs(s[:,:,1] ) <= 248) )
    s_new[green_mask] = np.array([100, 202, 100])
    plt.imshow( s_new )
    
    plt.savefig('image.png')


    # Find all unique colors
    unique_colors = np.unique(s.reshape(-1, s.shape[-1]), axis=0)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)

    # Print the number of unique colors and the shape of the grayscale image
    print(f"Number of unique colors: {len(unique_colors)}")
    print(f"Shape of the grayscale image: {gray_image.shape}")


    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')
    plt.show()
    gray_image

    s


class ImageEnv(gym.Wrapper):
    def __init__(
        self,
        env,
        skip_frames=5,
        stack_frames=10,
        initial_no_op=50,
        **kwargs
    ):
        super(ImageEnv, self).__init__(env, **kwargs)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
    
    def reset(self):
        # Reset the original environment.
        s, info = self.env.reset()

        # Do nothing for the next `self.initial_no_op` steps
        for i in range(self.initial_no_op):
            s, r, terminated, truncated, info = self.env.step([0.0, 0.0, 0.0])
        
        # Convert a frame to 84 X 84 gray scale one
        #s = preprocess(s)

        # The initial observation is simply a copy of the frame `s`
        #self.stacked_state = np.tile(s, (self.stack_frames, 1, 1))  # [4, 84, 84]
        self.stacked_state = np.tile(s, (1, 1, self.stack_frames))  # [96, 96, 12]
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
        #s = preprocess(s)

        # Push the current frame `s` at the end of self.stacked_state
        #self.stacked_state = np.concatenate((self.stacked_state[1:], s[np.newaxis]), axis=0)
        self.stacked_state = np.concatenate((self.stacked_state[:,:,(3):], s), axis=2)

        return self.stacked_state, reward, terminated, truncated, info

#s = preprocess(s) # s.shape
if False:
    stacked_state = np.tile(s, (4, 1, 1))  # [96, 96, 12]  # [4, 84, 84]
    stacked_state.shape

    action = [0, 0, 0]

    reward = 0
    for _ in range(4):
                s, r, terminated, truncated, info =env.step(action)
                reward += r
                if terminated or truncated:
                    break

            # Convert a frame to 84 X 84 gray scale one
    s_p = preprocess(s) # s.shape
    s_p = s_p[..., np.newaxis].shape
    s_p

            # Push the current frame `s` at the end of self.stacked_state
            #self.stacked_state = np.concatenate((self.stacked_state[1:], s[np.newaxis]), axis=0)
    stacked_state = np.concatenate((stacked_state[1:, :, :].shape, s.shape), axis=0)
    np.concatenate((stacked_state[1:], s[np.newaxis]), axis=0).shape
            

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


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 
class ActorCriticNet(nn.Module):
    def __init__(self, hidden_dim=16, frames_number = 10, channels = 1):
        super().__init__()

        # Variant 1  (DQN)
        if False:
            self.hidden = nn.Conv2d(in_channels=channels*frames_number, out_channels=16, kernel_size=8, stride=4) 
            self.hidden2 = nn.Conv2d(in_channels=16,                    out_channels=32, kernel_size=4, stride=2)  # (8, 47, 47) 
            self.in_features = 32 * 9 * 9
            self.fc1 = nn.Linear(self.in_features, 256)
            #self.fc2 = nn.Linear(256, action_dim)
        
        # Variant 2.1   "Racetrack Navigation on OpenAIGym with Deep Reinforcement Learning"" (DDQN)
        if False:
            self.hidden = nn.Conv2d(in_channels=channels*frames_number, out_channels=32, kernel_size=8, stride=4) 
            self.hidden2 = nn.Conv2d(in_channels=32,                    out_channels=64, kernel_size=4, stride=2)  # (8, 47, 47) 
            self.hidden3 = nn.Conv2d(in_channels=64,                    out_channels=64, kernel_size=3, stride=1)  # (8, 47, 47) 
            self.hidden_linear = nn.Linear( in_features = 512, out_features = 64)
            #Dense Size 1024 (no activation!)
        
        # Variant 2.2 Larger and better than Variant 2.1  "Racetrack Navigation on OpenAIGym with Deep Reinforcement Learning"" (DDQN)
        if True:
            self.hidden = nn.Conv2d(in_channels=channels*frames_number, out_channels=64,  kernel_size=8, stride=4) 
            self.hidden2 = nn.Conv2d(in_channels=64,                    out_channels=128, kernel_size=4, stride=2)  # (8, 47, 47) 
            self.hidden3 = nn.Conv2d(in_channels=128,                   out_channels=128, kernel_size=3, stride=1)  # (8, 47, 47) 
            self.hidden_linear = nn.Linear( in_features = 6272, out_features = 128)
        
        # Variant 2.3 Even Larger and better than Variant 2.1  "Racetrack Navigation on OpenAIGym with Deep Reinforcement Learning"" (DDQN)
        if False:
            self.hidden = nn.Conv2d(in_channels=channels*frames_number, out_channels=128,  kernel_size=8, stride=4) 
            self.hidden2 = nn.Conv2d(in_channels=128,                    out_channels=256, kernel_size=4, stride=2)  # (8, 47, 47) 
            self.hidden3 = nn.Conv2d(in_channels=256,                    out_channels=256, kernel_size=3, stride=1)  # (8, 47, 47) 
            self.hidden_linear = nn.Linear( in_features = 1024*2, out_features = 64)
        
        #Flatten
        #Dense Size 1024 (no activation!)

        #self.hidden3 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride = 2) # (16*23*23)      
        #self.hidden4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride = 2) # (32, 11, 11)
        
        self.mu    = nn.Linear(128, 3) 
        self.sigma = nn.Linear(128, 3)
        self.distribution = torch.distributions.Normal
        
        self.value = nn.Linear(128, 1)
        
        
        
    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        outs = self.hidden2(outs)
        outs = F.relu(outs)
        outs = self.hidden3(outs)
        outs = F.relu(outs)

        
        #outs = self.hidden5(outs)
        #outs = F.relu(outs)
        #outs = self.hidden6(outs)
        #outs = F.relu(outs)

        outs = torch.flatten( outs, 1 )
        
        outs = self.hidden_linear( outs )
        #outs = F.relu(outs)
        
        #outs = torch.clamp(outs[:, 0], -1, 1)
        #outs = torch.clamp(outs[:, 1:], 0, 0.1)
        
        mu      = self.mu(outs)
        mu_s    = F.tanh(mu[:, 0])/2 # * 2
        mu_a    = F.sigmoid(mu[:, 1])/10.0 # * 2
        mu_b    = F.sigmoid(mu[:, 2])/1000.0 # * 2
        mu      = torch.stack([mu_s, mu_a, mu_b], dim=1)
        
        sigma = (F.softplus(self.sigma(outs)) + 1e-5)/10.0
        
        state_value = self.value(outs)
        
        dist = self.distribution(mu.view(1, 3).data, sigma.view(1, 3).data)
        #dist = self.distribution(mu.view(3, ).data, sigma.view(3, ).data)
        
        #steering     = F.tanh(    self.steering(outs) )/4.0 #- F.tanh(    self.steering2(outs) )
        #acceleration = F.sigmoid( self.acceleration(outs) )/10.0
        #brake        = F.sigmoid( self.brake(outs) )/1000.0
        
        return dist, state_value  #steering, acceleration, brake

        

        

        

        
        return value

#actor_func = ActorNet().to(device)
#value_func = ValueNet().to(device)

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
device
# %%
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

ou_action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(1), sigma=np.ones(1) * 0.25)

# %%

import random

rng = np.random.default_rng()
#rng.normal(3, 2.5, size=(2, 4))

# Получить действие на основе текущей политики
def pick_sample(s, episode_, stochastic_policy = False):
    with torch.no_grad():
        #   --> size : (1, 4)
        #s_batch = np.expand_dims(s, axis=0)
        
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
        #s_batch = torch.tensor(s_batch_tensor.unsqueeze(0), dtype=torch.float).to(device) # s_batch.size()

        if True:
            
            dist, state_value = actor_critic_func(s_batch)
            #dist.scale
            action = dist.sample()#.numpy()
            log_prob = dist.log_prob(action[0])
            #action
            
            return action.cpu().numpy()[0, :], s_batch_tensor, (log_prob[0,0] + log_prob[0,1] + log_prob[0,2]).unsqueeze(0), state_value[0]

        if False:
            a_steering, a_acceleration, a_brake = actor_func(s_batch) # s_batch.shape
            
            std_ = 0.5 * .995**(episode_)  # 0.05 * .995**(500)
            
            a_steering = a_steering.cpu().numpy() + ou_action_noise() # random.randint(-100, 100)/100 #ou_action_noise()
            a_acceleration = a_acceleration.cpu().numpy() + rng.normal(0, std_) # random.randint(-100, 100)/100 ou_action_noise()
            a_brake = a_brake.cpu().numpy() #+ rng.normal(0, std_) # random.randint(-100, 100)/100  #ou_action_noise()
            

            #print( "   a_steering = ", a_steering, "  a_acceleration = ", a_acceleration, "  a_brake = ", a_brake)
            
            #a_steering = np.clip(a_steering, -0.3, 0.3)
            a_acceleration = np.clip(a_acceleration, 0.0, 0.1)
            #if a_acceleration.tolist()[0][0] < 0.1: 
            #     a_acceleration = 0.1
            #else:
            #    a_acceleration = 0.0

            #a_brake = np.clip(a_brake, 0.0, 0.0)
            
            return [a_steering.tolist()[0][0], a_acceleration.tolist()[0][0], a_brake.tolist()[0][0]], s_batch_tensor #a.tolist()[0]
        
        if False:    
            steering, acceleration, brake = actor_func(s_batch)
            
            # STOCHASTIC POLICY 1472*92
            # From logits to probabilities
            probs_steering = F.softmax(steering, dim=-1)
            probs_acceleration = F.softmax(acceleration, dim=-1)
            probs_brake = F.softmax(brake, dim=-1)
            
            # Pick up action's sample
            # ! VALUES ARE TOO LARGE
            a_steering = torch.multinomial(probs_steering, num_samples=1)
            a_acceleration = torch.multinomial(probs_acceleration, num_samples=1)
            a_brake = torch.multinomial(probs_brake, num_samples=1)
        
        # Return
        

def update(log_probs, state_values, returns ):

    """ Обновляет веса сети исполнитель–критик на основе переданных ... обучающих примеров
... @param returns: доход (накопительное вознаграждение) на ... каждом шаге эпизода
... @param log_probs: логарифм вероятности на каждом шаге ... @param state_values: ценности состояний на каждом шаге
"""
    loss = 0
    for log_prob, value, Gt in zip(log_probs, state_values, returns):
        
        advantage = Gt - value.item()
        policy_loss = -log_prob * advantage
        value_loss = F.smooth_l1_loss(value, Gt)
        loss += policy_loss + value_loss
        
    return loss
    #self.optimizer.zero_grad()
    #loss.backward()
    #self.optimizer.step()


# %%

stacked_frames = 10
channels = 1

env = gym.make("CarRacing-v2")  # среда
env = ImageEnv_GrayScaled(
    env,
    skip_frames = 3,
    stack_frames = stacked_frames,
    initial_no_op = 50)
env.observation_space



# %%

# create a new model with these weights
actor_critic_func = ActorCriticNet( frames_number = stacked_frames, channels=channels )
actor_critic_func.apply(weights_init_uniform_rule) # !!!!
actor_critic_func        = actor_critic_func.to(device)
#actor_target_func = ActorNet( frames_number = stacked_frames, channels=channels ).to(device)

# Подгрузить в целевую сеть коэффициенты из сети политики
#actor_target_func.load_state_dict(actor_func.state_dict())



# Подгрузить в целевую сеть коэффициенты из сети политики
#value_target_func.load_state_dict(value_func.state_dict())

gamma = 0.95  # !!! дисконтирование
#env = gym.make("CartPole-v1")  # среда
reward_records = []  # массив наград
v_loss_records = []
pi_loss_records = []

# Оптимизаторы
opt1 = torch.optim.AdamW(actor_critic_func.parameters(), lr=0.0005) # !!! greater lr ?
#opt2 = torch.optim.AdamW(actor_func.parameters(), lr=0.0005)

# количество циклов обучения
num_episodes = 3600
i = 0

cum_reward_ = 0
first_action = 0



# Create directory once
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = './gif_animations/A2C/' + timestamp
os.makedirs(folder_name, exist_ok=True)                
                
import time


#for i in tqdm(range(num_episodes), desc="Episodes", postfix={"Last reward": cum_reward_, "First action": first_action}):
#for i in trange(num_episodes, desc="Episodes", postfix={"Last reward": cum_reward_}):

#with tqdm(total=num_episodes, postfix=[ {"Reward": 0, "V_loss": 0, "Pi_loss": 0}]) as t:
#    for i in range(num_episodes): i = 0
for i in range(num_episodes):
        
        #for i in range(num_episodes):
            
        # в начале эпизода обнуляем массивы и сбрасываем среду
        done = False
        states = [] 
        actions = []
        rewards = []
        state_values = []
        # Episode compuation time
        start_time = time.time()
    
        s, info = env.reset() # s.shape
        
        #frames = []  zooming cut
        #for k in range(50):
        #    s, r, terminated, truncated, info = env.step([0, 0, 0])  # 0-th action is no_op action
        #    #frames.append(s)

        # пока не достигнем конечного состояния продолжаем выполнять действия
        episode_len = 0
        frames = []
        log_probs = []
        while not done:
            # добавить состояние в список состояний
            #states.append(s.tolist())
            
            # по текущей политике получить действие
            a, s_prepared, log_prob, state_value = pick_sample(s, i)
            a = a.clip( env.action_space.low, env.action_space.high)
            
            states.append(s_prepared.tolist()) # states.size()
            log_probs.append(log_prob)
            
            # выполнить шаг, получить награду (r), следующее состояние (s) и флаги конечного состояния (term, trunc)
            s, r, term, trunc, _ = env.step(a) # s.shape
            frames.append(s) # !
            
            # если конечное состояние - устанавливаем флаг окончания в True
            done = term or trunc
            
            # добавляем действие и награду в соответствующие массивы
            actions.append(a)
            state_values.append(state_value)
            
            episode_len = episode_len + 1
            
            # Terminate failed episodes
            if (len([reward for reward in rewards[-100:] if reward > 0]) == 0) and (len(rewards) > 100):
                #rewards.append( - 200.0 )
                rewards.append(r)
                done = True
            else:
                rewards.append(r)


        
        
        #print( "   reward = ", rewards[-1], "\n" )

        #
        # Если траектория закончилась (достигли финального состояния)
        #
        # формируем массив полной награды для каждого состояния
        cum_rewards = np.zeros_like(rewards) # Q(s,a)
        reward_len = len(rewards)
        for j in reversed(range(reward_len)):
            cum_rewards[j] = rewards[j] + (cum_rewards[j+1]*gamma if j+1 < reward_len else 0) # Q(s,a) = r_t+1 + gamme * V(s_t+1)

        #
        # Оптимизируем параметры сетей
        #

        # ! Оптимизируем value loss (Critic)
        # Обнуляем градиенты в оптимизаторе
        opt1.zero_grad()

        #loss = update(log_probs, state_values, cum_rewards )
        
        
        
        log_probs = torch.cat(log_probs, dim=0)
        log_probs.requires_grad_(True)
        
        state_values = torch.cat(state_values, dim=0)
        state_values.requires_grad_(True)
        
        cum_rewards = torch.tensor( cum_rewards, dtype=torch.float).to(device) # cum_rewards.size()
        
        loss = 0 #  log_prob, value, Gt = log_probs[0], state_values[0], cum_rewards[0]
        for log_prob, value, Gt in zip(log_probs, state_values, cum_rewards):
            
            advantage = Gt - value.item()
            policy_loss = -log_prob * advantage
            value_loss = F.smooth_l1_loss(value, Gt)
            loss += policy_loss + value_loss
        
        loss.backward()
        
        
        
        # делаем шаг оптимизатора
        opt1.step()



        # Выводим итоговую награду в эпизоде  
        reward_records.append(sum(rewards))
        cum_reward_ = sum(rewards)

        #print( "  i =", i, "  episode_len =", episode_len, "  reward =", cum_reward_, "\n" )
        
        #t.set_description('Episode %i' % i)
        #t.set_postfix(loss=reward_records[-20:], str='h',  lst=[1, 2]) # , gen=randint(1,999)
        #tqdm.write("n == 6 completed.")
        
        #tqdm.write( "  a = " + " ".join(str(a) for a in actions[0:4]))

        #t.set_postfix(
        #    Reward  = round( reward_records[-20:][0], 2),
        #    V_loss  = round( vf_loss.sum().tolist(), 2),
        #    Pi_loss = round( pi_loss.sum().tolist(), 2),
        #)
            
        #t.update()
        
        end_time = time.time()
        computation_time = end_time - start_time
             
        print ( "  Reward  =", round( reward_records[-1:][0], 2),
            #" V_loss =", round(  v_loss_records[-1:][0], 2),
            #" Pi_loss =", round( pi_loss_records[-1:][0], 2), 
            " loss =", round( loss.cpu().tolist(), 2), 
            " Episode_len =", episode_len,
            " Computation time:", round(computation_time/60, 2), " minutes",
            #"  \na =", actions[0:4]
                "\n" +    "".join( 
                "     ".join(str([ round(  a, 2) for a in np.array(action)]) + "\n" for action in actions[i:i+1]) # actions[0]
                for i in range(0,200,20)
            )
            )
        
        
        
        if i % 5 == 0:
            print("\nRun episode {} with average Reward {} ".format(
                i, 
                round(np.mean(reward_records[-20:]), 2), 
                #round(np.mean(v_loss_records[-20:]), 2), 
                #round(np.mean(pi_loss_records[-20:])), 2), 
            ),
                  end="\n\n\n")
            
        if i % 10 == 0:
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
                anim.save(folder_name + '/animation_' + str(i) + '.gif') #, writer='imagemagick')
                #HTML(anim.to_jshtml())
        
            

        # stop if mean reward for 100 episodes > 475.0
        if np.average(reward_records[-100:]) > 475.0:
            break
        

print("\nDone")
env.close()




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

# Plot
plt.plot(reward_records, label='reward')
plt.plot(average_reward, label='average reward')
plt.xlabel('N episode')
plt.ylabel('Reward')
plt.legend()
plt.show();

# %%



