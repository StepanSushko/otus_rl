"""
@author: Stepan S. Sushko
"""

import math
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import gymnasium as gym
#from gym import spaces
#from gym.envs.classic_control import utils
#from gym.error import DependencyNotInstalled

from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

class UGSEnv(gym.Env):
    """
    ### Description

    The UGS MDP is a deterministic MDP that consists of a gas volume set stochastically/deterministically
    at the top of a sinusoidal multidimensional hill, with the only possible actions being the withdrawal/штоусешщт 
    that can be applied to the set of UGSs in either direction. The goal of the MDP is to strategically
    maximize the UGS's productivity and to reach the goal state on bottom of the valley.
    This version assumes continuous actions in the limits set for each UGS.


    ### Observation Space

    The observation is a `ndarray` with shape `(30 x 1,)` where the elements correspond to the following:

    | Num | Observation                          | Min   | Max     | Unit             |
    |-----|--------------------------------------|-------|---------|------------------|
    | 0   | total gas demand (stachistic)        | 0     | ??      | mlns of q.meters |
    | 1   | UGS #1 stored gaz volume             | Min_1 | Max_1   | mlns of q.meters |
    ...
    | 28   | UGS #29 stored gaz volume           | Min_29 | Max_29 | mlns of q.meters |

    ### Action Space

    The action is a `ndarray` with shape `(30,)`, representing the withdrawal|injection in/out the UGSs.
    Action vector is clipped by min/max vectors, where min is maximal withdrawal, max is maximal injection to UGS vector.

    ### Transition Dynamics:

    Given an action, the mountain car follows the following transition dynamics:

    The collisions at either end are inelastic with the velocity set to 0 upon collision with the wall.
    The position is clipped to the range [-1.2, 0.6] and velocity is clipped to the range [-0.07, 0.07].

    ### Reward

    A negative reward of *-0.1 * action<sup>2</sup>* is received at each timestep to penalise for
    taking actions of large magnitude? 
    
    If the UGSs reaches the goal then a positive reward of +100 is added to the negative reward for that timestep.

    ### Starting State

    The initial state is assigned a uniform random value in normal/avarage volume of UGS +- 10% ?


    ### Episode End

    The episode ends if either of the following happens:
    1. Termination: The productivity of UGSs is greater than or equal to X (the goal position on top of the winter valley)
    2. Truncation: The length of the episode is 180 days??
    3. Out of capacity?

    ### Arguments

    ```
    gym.make('UGS-v0')
    ```

    ### Version History

    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    
    def __init__(self, render_mode: Optional[str] = None, number_of_ugs=2, balance = "hard", demand = "test_task", horizon = 30, random_start = False):
        """
        S_0 = 450, 450
        а_1 + а_2 = D

        0 <= a_1 <= f_1 (S^1)  -- производительность
        0 <= a_2 <= f_2 (S^2)  -- производительность

        S_1 = S_0 - (а_1, а_2)

        f_1 (S^1) + f_2 (S^2) -> max

        Terminal state -> out of boundaries?

        """
        
        self.random_start = random_start
        self.demand = demand
        self.number_of_ugs = number_of_ugs
        
        self.balance = balance

        self.min_volume = [1.0, 2.0]
        self.max_volume = [500.0, 500.0]
        
        
        # UGS1 productivity y = 5 + 0.0000001*x^3 + 0.00004*x^2 - 0.02*x
        # UGS2 productivity y = 10 - 0.00006*x^2 + 0.05*x
        
        self.max_withdrwal = [-30.1, -30.2]
        self.max_injection = [0.1, 0.1]
        self.max_injection = [30.1, 30.1] # ! 
        
        # goals definition?
        # ??
        self.low_state = np.array(
            [0] + self.min_volume, dtype=np.float32
        )
        
        self.high_state = np.array(
           [0] + self.max_volume, dtype=np.float32
        )

        #self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        #self.screen = None
        #self.clock = None
        #self.isopen = True
        
        self.start_date = 0 # pd.Timestamp('2024-01-01')
        self.timedelta = 1 # pd.Timedelta(days=1)
        self.horizon = horizon # pd.Timedelta(days=180)
        
        if demand == "test_task":
            self.demand_with_noise = np.array([15.46, 15.93, 16.41, 16.9, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0])
        
        if demand == "sinusoidal":
            self.demand_with_noise = self.get_sinusoidal_signal_with_noise( add_noise = False)
            
        if demand == "sinusoidal_with_noise":
            self.demand_with_noise = self.get_sinusoidal_signal_with_noise( add_noise = True)

        if balance == "hard":
            self.action_space = spaces.Box(
                low=np.array(self.max_withdrwal[:-1]), 
                high=np.array(self.max_injection[:-1]), 
                shape=(number_of_ugs - 1,), 
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=np.array(self.max_withdrwal), 
                high=np.array(self.max_injection), 
                shape=(number_of_ugs,), 
                dtype=np.float32
            )
        
        self.observation_space = spaces.Box(
            low  = np.concatenate([ 
                np.array([0]),            
                np.array([self.demand_with_noise.min() * 0.8]), 
                self.min_volume], axis=0, dtype=np.float32),
            high = np.concatenate([ 
                np.array([self.horizon]), 
                np.array([self.demand_with_noise.max() * 1.2]), 
                self.max_volume], axis=0, dtype=np.float32),
            dtype=np.float32
        )
        
        


    #state = 10

    #10 - 0.00006*state**2 + 0.05*state

    def get_sinusoidal_signal_with_noise(self, baseline_level = 2.0, noise_level = 0.1, season_level = 2.0, period = 365, plot_figures = False, add_noise = False, randomize = True):
        
        t = np.arange(0, period, 1)
        signal = season_level * np.sin(2 * np.pi * t / 365)
        
        if randomize:
            noise = np.random.normal(0, noise_level, len(t))
        else:
            noise = np.random.seed(0)
            noise = np.random.normal(0, noise_level, len(t))
        
        signal_with_noise = 0.0
        if add_noise:
            signal_with_noise = signal + noise + baseline_level
        else:
            signal_with_noise = signal + baseline_level

        if plot_figures:
            plt.figure(figsize=(10, 6))
            plt.plot(t, signal, label='Signal')
            plt.plot(t, noise, label='Noise')
            plt.plot(t, signal_with_noise, label='Signal with noise')
            plt.legend()
            plt.show()

        return signal_with_noise

    



    def step(self, act: np.ndarray, print_out = False):

        self.next_state = self.state.copy()
        
        # Increment date time.
        self.date_time     += self.timedelta
        self.next_state[0] += self.timedelta
        
        demand             = self.demand_with_noise[self.date_time]
        self.next_state[1] = self.demand_with_noise[self.date_time]
        
        reward = 0.0 
        
        action = act.copy()
        
        if self.balance == "hard":
            # ! NO DEFICIT
            action = np.append(action, -(self.state[1] + action.sum()))
        
        for i in range(2,self.number_of_ugs + 2):
            
            j = i - 2
            
            productivity = 0
            # UGS1 productivity y = 5 + 0.0000001*x^3 + 0.00004*x^2 - 0.02*x
            if i == 2:
                productivity = 5.0 + 0.0000001*self.state[i]**3 + 0.00004*self.state[i]**2 - 0.02*self.state[i]
            # UGS2 productivity y = 10 - 0.00006*x^2 + 0.05*x
            if i == 3:
                productivity = 10.0 - 0.00006*self.state[i]**2 + 0.05*self.state[i]
            
            
            #if action[i] > self.max_injection[i]:
            #    print(f"Action {i} is out of bounds, clipping to max injection")
            #    action[i] = self.max_injection[i]
            #if action[i] < self.max_withdrwal[i]:
            #    print(f"Action {i} is out of bounds, clipping to min withdrawal")
            #    action[i] = self.max_withdrwal[i]
            if action[j] < -productivity:
                #print(f"Action {i} is out of bounds, clipping to max withdrawal")
                
                penalty = - 111.0  - 1.0* (- productivity - action[j]) **2
                reward += penalty
                
                action[j] = -productivity
                
                if print_out:
                    print(f"Action {j} is greater than productivity, clipping to max withdrawal. Penalty = {penalty}, total reward = {reward}")
                
        
            if action[j] > 0:
                
                penalty = -666.0 - 10.0 * action[j]**2
                reward += penalty
                if print_out:
                    print(f"Action {j} is positive. Penalty = {penalty}, total reward = {reward}")
                    
                action[j] = 0


            if self.state[i] + action[j] > self.max_volume[j]:
                #print(f"State {i} is out of bounds, clipping to max volume")
                penalty = -333.0 - 1.0 *( self.max_volume[j] - (self.state[i] + action[j])  ) **2
                reward += penalty
                
                action[j] = self.max_volume[j] - self.state[i]
                
                if print_out:
                    print(f"Action {i} is out of bounds, clipping to max volume. Penalty = {penalty}, total reward = {reward}")
                
                
            if self.state[i] + action[j] < self.min_volume[j]:
                #print(f"State {i} is out of bounds, clipping to min volume")

                penalty = -444.0 - 1.0 * ( self.min_volume[j] - (self.state[i] + action[j])  ) **2
                reward += penalty

                action[j] = self.min_volume[j] - self.state[i]
                
                if print_out:
                    print(f"Action {i} is out of bounds, clipping to min volume. Penalty = {penalty}, total reward = {reward}")

                
            self.next_state[i] = self.state[i] + action[j]



        if self.balance == "soft":
            # ! Deficit level
            if np.abs(self.state[1] + action.sum()) >= 0.5:
                penalty = - 100.0 - 100.0* (np.abs(self.state[1] + action.sum()) - 0.5)**2 # !!! next state ???
                reward += penalty
                
                if print_out:
                    print(f"Deficit. Penalty = {penalty}, total reward = {reward}")
                
                # !!! Break if DEFICIT ????
                #print("Demand ",-demand," does not match action ",action.sum(), " reward ",reward)
            
        

        
        # Convert a possible numpy bool to a Python bool.
        truncated  = False # bool(
            #spaces.Box(
            #    low  = np.concatenate( np.array([self.date_time]), np.array(self.min_volume)), 
            #    high = np.concatenate( np.array([self.horizon  ]), np.array(self.max_volume))).contains(self.next_state)
        #    self.observation_space.contains(self.next_state)
        #    
        #)
        #print(f"Terminated: {terminated}",  " min_volume: ", self.min_volume," State: ", self.state, " max_volume: ", self.max_volume)
        
        if self.date_time >= self.horizon :
            terminated = True
        else:
            terminated = False



        if truncated:
            reward += -1000.0
        #reward = sum( action )
        

        

        
        if self.date_time == self.horizon:
            cum_productivity = 0.0
            for i in range(2,self.number_of_ugs + 2):
                    if i == 2:
                        cum_productivity += 1*(5.0 + 0.0000001*self.next_state[i]**3 + 0.00004*self.next_state[i]**2 - 0.02*self.next_state[i])
                    # UGS2 productivity y = 10 - 0.00006*x^2 + 0.05*x
                    if i == 3:
                        cum_productivity += 1*(10.0 - 0.00006*self.next_state[i]**2 + 0.05*self.next_state[i])
            
            premium = 10.0*cum_productivity**2
            
            reward += premium
            
            if print_out:
                print(f"Premium. Premium = {premium}, total reward = {reward}")
            
            

        #self.state = np.array([position, velocity], dtype=np.float32)

        #if self.render_mode == "human":
        #    self.render()
        
        #self.next_state = np.concatenate([ np.array([self.date_time]), self.next_state], axis=0, dtype=np.float32)

        self.state = self.next_state
            
        return  self.next_state, reward, terminated, truncated, {}  # s_next, r, done, truncate, _





    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        
        self.date_time = self.start_date
        #print("Date: ", self.date_time)
        #low, high = utils.maybe_parse_reset_bounds(options, 480, 520)
        
        if self.random_start:
            self.state = self.np_random.uniform(low= np.array(self.max_volume) * 0.8, high=np.array( self.max_volume) * 1.0)
        else:
            self.state = 0.9* np.array(self.max_volume)
            
        #if self.render_mode == "human":
        #    self.render()
        
        if not self.random_start:
            for i in range(self.number_of_ugs):
                # UGS1 productivity y = 5 + 0.0000001*x^3 + 0.00004*x^2 - 0.02*x
                # UGS2 productivity y = 10 - 0.00006*x^2 + 0.05*x

                if self.state[i]  > self.max_volume[i]:
                    print(f"State {i} is out of bounds, clipping to max volume")
                    self.state[i] = self.max_volume[i] 
                if self.state[i] < self.min_volume[i]:
                    print(f"State {i} is out of bounds, clipping to min volume")
                    self.state[i] = self.min_volume[i]
                
        if self.random_start:
            
            if self.demand == "sinusoidal":
                self.demand_with_noise = self.get_sinusoidal_signal_with_noise( add_noise = False, randomize = True)
                
            if self.demand == "sinusoidal_with_noise":
                self.demand_with_noise = self.get_sinusoidal_signal_with_noise( add_noise = True, randomize = True)

        #if self.render_mode == "human":
        #    self.render()
        
        self.state = np.concatenate([ 
            np.array([self.date_time]), 
            np.array([ self.demand_with_noise[self.date_time]]), 
            self.state], axis=0, dtype=np.float32)

        return np.array(self.state, dtype=np.float32), {}

    #def _height(self, xs):
    #    return np.sin(3 * xs) * 0.45 + 0.55

    def render(self, delay=0.01):
        
        """
        Renders the environment.

        This function renders the current state of the environment into a plot using matplotlib.

        Parameters
        ----------
        delay : float, optional
            The delay between frames in seconds, by default 0.01

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axs : list of matplotlib.axes.Axes
            The list of axes objects.
        """
        
        if True:
            self.fig, self.axs = plt.subplots( 
                self.number_of_ugs, 
                1, 
                figsize=(10, 5*self.number_of_ugs), 
                sharex=True)
            
            self.lines = []

            for i in range(self.number_of_ugs):
                self.axs[i].set_xlim(self.start_date, self.start_date + self.horizon )
                self.axs[i].set_ylim(0, self.max_volume[i])
                self.axs[i].xaxis.set_major_locator(plt.MaxNLocator(6))
        
        for ax, s, ns in zip(self.axs, self.state[1:], self.next_state[1:]):
            
            ax.plot( [self.date_time, self.date_time + self.timedelta], [s, ns], marker='o', linestyle='-')
                #ax.arrow( self.date_time, s, self.date_time + pd.Timedelta(days=1), ns-s, head_width=0.1, head_length=10, fc='k', ec='k')
                
            self.lines.append(ax)
                
        plt.draw()
        plt.pause(delay)
        plt.show()
        
        return self.fig, self.axs


    def close(self):
        super().close()
