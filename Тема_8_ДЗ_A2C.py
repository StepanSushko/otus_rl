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

# %% [markdown]
# # Imports

# %%
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


from tqdm import tqdm, trange
gym.__version__


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
    #img = img[:84, 6:90] # CarRacing-v2-specific cropping
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
class ActorNet(nn.Module):
    def __init__(self, hidden_dim=16, frames_number = 10, channels = 1):
        super().__init__()

        self.hidden = nn.Conv2d(in_channels=channels*frames_number, out_channels=16, kernel_size=16, stride=2) #nn.Linear( in_features = 96*96*3*frames_number, out_features = hidden_dim)
        self.hidden2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=8, stride=2)  # (8, 47, 47) 
        #self.hidden = nn.Conv2d(in_channels=channels*frames_number, out_channels=4, kernel_size=3, stride = 2) #nn.Linear( in_features = 96*96*3*frames_number, out_features = hidden_dim)
        #self.hidden2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride = 2)  # (8, 47, 47) 
        self.hidden3 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride = 2) # (16*23*23)      
        self.hidden4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride = 2) # (32, 11, 11)
        #self.hidden5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride = 1) # (64* 5*5)
        #self.hidden6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride = 1) # (128, 3, 3)
        
        #self.hidden4 = nn.Linear( in_features = 32*11*11, out_features = 11*11)
        #self.hidden_linear = nn.Linear( in_features = 256, out_features = 64)
        self.hidden_linear = nn.Linear( in_features = 576, out_features = 64)
        
        #self.output = nn.Linear( in_features = hidden_dim*94*94, out_features = 3) 
        # #nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3)
        
        #self.output = nn.Linear( in_features = hidden_dim*94*94, out_features = 3)
        
        self.steering = nn.Linear( in_features = 64, out_features = 1)
        self.steering2 = nn.Linear( in_features = 64, out_features = 1)
          
        self.acceleration = nn.Linear( in_features = 64, out_features = 1)
        self.brake = nn.Linear( in_features = 64, out_features = 1)
        
    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        outs = self.hidden2(outs)
        outs = F.relu(outs)
        outs = self.hidden3(outs)
        outs = F.relu(outs)
        outs = self.hidden4(outs)
        outs = F.relu(outs)
        #outs = self.hidden5(outs)
        #outs = F.relu(outs)
        #outs = self.hidden6(outs)
        #outs = F.relu(outs)

        outs = torch.flatten( outs, 1 )
        
        outs = self.hidden_linear( outs )
        
        steering     = F.tanh(    self.steering(outs) )/4.0 #- F.tanh(    self.steering2(outs) )
        
        acceleration = F.sigmoid( self.acceleration(outs) )/10.0
        brake        = F.sigmoid( self.brake(outs) )/1000.0
        
        return steering, acceleration, brake

class ValueNet(nn.Module):
    def __init__(self, hidden_dim=16, frames_number = 10, channels = 1):
        super().__init__()
        
        #self.hidden = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3) 

        #self.hidden = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=3) #nn.Linear( in_features = 96*96*3, out_features = hidden_dim)
        
        self.hidden = nn.Conv2d(in_channels=channels*frames_number, out_channels=16, kernel_size=16, stride=2) #nn.Linear( in_features = 96*96*3*frames_number, out_features = hidden_dim)
        self.hidden2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=8, stride=2)  # (8, 47, 47) 
        #self.hidden = nn.Conv2d(in_channels=channels*frames_number, out_channels=4, kernel_size=3, stride = 2) #nn.Linear( in_features = 96*96*3*frames_number, out_features = hidden_dim)
        #self.hidden2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride = 2)  # (8, 47, 47) 
        self.hidden3 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride = 2) # (16*23*23)      
        self.hidden4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride = 2) # (32, 11, 11)
        
        #self.hidden_linear = nn.Linear( in_features = 64*5*5, out_features = 4*5*5)
        self.hidden_linear = nn.Linear( in_features = 576, out_features = 64)
        
        
        self.output_linear = nn.Linear( in_features = 64, out_features = 1)

        #self.hidden = nn.Linear(96*96*3, hidden_dim)
        #self.output = nn.Linear(hidden_dim, 1)

    def forward(self, s):
        
        outs = self.hidden(s)
        outs = F.relu(outs)
        outs = self.hidden2(outs)
        outs = F.relu(outs)
        outs = self.hidden3(outs)
        outs = F.relu(outs)
        outs = self.hidden4(outs)
        outs = F.relu(outs)
        #outs = self.hidden5(outs)
        #outs = F.relu(outs)
        #outs = self.hidden6(outs)
        #outs = F.relu(outs)

        
        outs = torch.flatten( outs, 1 )
        
        outs = self.hidden_linear( outs )
        
        value = self.output_linear(outs)
        
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

ou_action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(1), sigma=np.ones(1) * 0.05)

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

        if False:
            
            a_steering, a_acceleration, a_brake = actor_func(s_batch) # s_batch.shape
            
            sigma = tf.squeeze(self.sigma)
            sigma = tf.nn.softplus(self.sigma) + 1e-5
            
            normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
            action = self.normal_dist._sample_n(1)

        if True:
            a_steering, a_acceleration, a_brake = actor_func(s_batch) # s_batch.shape
            
            std_ = 0.0 # !!! * .995**(episode_)
            
            a_steering = a_steering.cpu().numpy() + rng.normal(0, std_) # random.randint(-100, 100)/100 #ou_action_noise()
            a_acceleration = a_acceleration.cpu().numpy() + rng.normal(0, std_) # random.randint(-100, 100)/100 ou_action_noise()
            a_brake = a_brake.cpu().numpy() + rng.normal(0, std_) # random.randint(-100, 100)/100  #ou_action_noise()
            

            #print( "   a_steering = ", a_steering, "  a_acceleration = ", a_acceleration, "  a_brake = ", a_brake)
            
            #a_steering = np.clip(a_steering, -0.3, 0.3)
            #a_acceleration = np.clip(a_acceleration, 0.0, 0.1)
            #if a_acceleration.tolist()[0][0] < 0.1: 
            #     a_acceleration = 0.1
            #else:
            #    a_acceleration = 0.0

            #a_brake = np.clip(a_brake, 0.0, 0.0)
            
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
        return [a_steering.tolist()[0][0], a_acceleration.tolist()[0][0], a_brake.tolist()[0][0]], s_batch_tensor #a.tolist()[0]


# %%
env = gym.make("CarRacing-v2")  # среда
env = ImageEnv_GrayScaled(env)
env.observation_space

# %%
env.action_space


# %%

stacked_frames = 10
channels = 1

# create a new model with these weights
actor_func = ActorNet( frames_number = stacked_frames, channels=channels )
actor_func.apply(weights_init_uniform_rule)
actor_func        = actor_func.to(device)
actor_target_func = actor_func.to(device)

# Подгрузить в целевую сеть коэффициенты из сети политики
actor_target_func.load_state_dict(actor_func.state_dict())


value_func = ValueNet()
value_func.apply(weights_init_uniform_rule)
value_func = ValueNet().to(device)
value_target_func = ValueNet().to(device)

# Подгрузить в целевую сеть коэффициенты из сети политики
value_target_func.load_state_dict(value_func.state_dict())

gamma = 0.99  # дисконтирование
#env = gym.make("CartPole-v1")  # среда
reward_records = []  # массив наград
v_loss_records = []
pi_loss_records = []

# Оптимизаторы
opt1 = torch.optim.AdamW(value_func.parameters(), lr=0.0005) # !!! greater lr ?
opt2 = torch.optim.AdamW(actor_func.parameters(), lr=0.0005)

# количество циклов обучения
num_episodes = 3600
i = 0

cum_reward_ = 0
first_action = 0

#import matplotlib
from IPython.display import HTML
import os
import datetime

# Create directory once
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = './gif_animations/' + timestamp
os.makedirs(folder_name, exist_ok=True)                
                



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
        s, info = env.reset() # s.shape
        
        #frames = []  zooming cut
        #for k in range(50):
        #    s, r, terminated, truncated, info = env.step([0, 0, 0])  # 0-th action is no_op action
        #    #frames.append(s)

        # пока не достигнем конечного состояния продолжаем выполнять действия
        episode_len = 0
        frames = []
        while not done:
            # добавить состояние в список состояний
            #states.append(s.tolist())
            
            # по текущей политике получить действие
            a, s_prepared = pick_sample(s, i)
            states.append(s_prepared.tolist()) # states.size()
            
            # выполнить шаг, получить награду (r), следующее состояние (s) и флаги конечного состояния (term, trunc)
            s, r, term, trunc, _ = env.step(a) # s.shape
            frames.append(s) # !
            
            # если конечное состояние - устанавливаем флаг окончания в True
            done = term or trunc
            
            # добавляем действие и награду в соответствующие массивы
            actions.append(a)
            
            episode_len = episode_len + 1
            
            # Terminate failed episodes
            if (len([reward for reward in rewards[-15:] if reward > 0]) == 0) and (len(rewards) > 15):
                #rewards.append( - 200.0 )
                rewards.append(r)
                done = True
            else:
                rewards.append(r)


        first_action = actions[0]
        
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
        # преобразуем состояния и суммарные награды для каждого состояния в тензор
        states      = torch.tensor( states, dtype=torch.float).to(device) # states.size()
        cum_rewards = torch.tensor( cum_rewards, dtype=torch.float).to(device) # cum_rewards.size()

        # Вычисляем лосс
        values = value_func(states) # values.size()
        values = values.squeeze(dim=1)
        vf_loss = F.mse_loss(
            values,
            cum_rewards,
            reduction="none")
        # считаем градиенты This function accumulates gradients in the leaves - you might need to zero .grad attributes or set them to None before calling it.
        v_loss_records.append(vf_loss.sum().tolist())
        vf_loss.sum().backward()
        
        # делаем шаг (одну итерацию) оптимизатора 
        opt1.step()

        # обновляем параметры сети
        with torch.no_grad():
            values = value_func(states)



        # ! Оптимизируем ACTOR loss (обновляем параметры сети)
        # Обнуляем градиенты
        opt2.zero_grad()
        # преобразуем к тензорам
        actions = torch.tensor(actions, dtype=torch.float64).to(device) # actions.size()
        # считаем advantage функцию
        advantages = cum_rewards - values # A(s,a) = Q(s,a) - V(s)

        
        
        #logits = actor_func(states)
        # states_normalized.size()
        #states_normalized = states / 255.0  # Normalizing to [0, 1] range

        # Convert the observation to a PyTorch tensor
        # NCHW stands for: batch N, channels C, depth D, height H, width W
        # from 96x96x3 to 3x96x96 (3 - channels, 96 - height, 96 - width)  three channel picture
        #states_tensor = states_normalized.permute(2, 0, 1).unsqueeze(0)  # Assuming NHWC to NCHW format
        
        #states_tensor = torch.tensor(states_tensor, dtype=torch.float).to(device)

        
        logits_steering, logits_acceleration, logits_brake = actor_func(states) # logits
        
        #logits_steering = torch.clamp(logits_steering, -0.3, 0.3)

        #logits_acceleration = torch.clamp(logits_acceleration, 0.0, 0.1)
        #for k in range(len(logits_acceleration)):
        #    if logits_acceleration[k][0] < 0.1: 
        #       logits_acceleration[k][0] = 0.1
        #    else:
        #        logits_acceleration[k][0] = 0.0
        #logits_acceleration.requires_grad_(True)
                
        #logits_brake = torch.clamp(logits_brake, 0.0, 0.0)
            
        logits = torch.cat((logits_steering, logits_acceleration, logits_brake), dim=1)
        #logits.requires_grad_(True)
        
            
        #logits = torch.tensor(logits, dtype=torch.int64).to(device)
        
        # [steering,acceleration, brake].shape() actions.size()
        log_probs = F.cross_entropy( 
            logits, 
            actions, 
            reduction="none")
        pi_loss = log_probs * advantages  # log_probs.size()  advantages.size()
        
        # считаем градиент
        pi_loss_records.append(pi_loss.sum().tolist())
        pi_loss.sum().backward()
        
        # делаем шаг оптимизатора
        opt2.step()



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
        
             
        print ( "  Reward  =", round( reward_records[-1:][0], 2),
            " V_loss =", round(  v_loss_records[-1:][0], 2),
            " Pi_loss =", round( pi_loss_records[-1:][0], 2), 
            " Episode_len =", episode_len,
            #"  \na =", actions[0:4]
            "             a = " + " ".join(str(a) for a in actions[0:10])
            )
        
        
        
        if i % 5 == 0:
            print("\nRun episode {} with average Reward {} V_loss = {} Pi_loss = {}".format(
                i, 
                round(np.mean(reward_records[-20:]), 2), 
                round(np.mean(v_loss_records[-20:]), 2), 
                round(np.mean(pi_loss_records[-20:])), 2), 
                  end="\n\n\n")
            
        if i % 10 == 0:
            if False:
                #s, info = env.reset()
                print(s.shape)

                plt.figure(figsize=(5, 5))
                plt.imshow(s)
                plt.axis('off')
                plt.show()
                
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



