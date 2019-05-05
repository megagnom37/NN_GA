##########################################################
#                   CARTPOLE API                         #
##########################################################
# Author:                                Ivan Skripachev #
# Create date:                                21.04.2019 #
# Organization:                                     ITMO #
# Version:                                         1.0.1 #
##########################################################
#                      Example:                          #
##########################################################
# env = CartPole()                                       #
# for i in range(num_agents):                            #
#     env.prepare_env()                                  #
#     while not env.is_done():                          #
#         obs = env.get_obs()                            #
#         action = agent[i].predict(obs)                 #
#         env.step(action)                               #
#     results.append(env.get_reward())                   #
#                                                        #
##########################################################

import gym
import numpy as np
import utils
import math

# Working area of the window
ENV_HEIGHT_SLICE = slice(150, 350)
ENV_WIDTH_SLICE  = slice(200, 400)
# Max bright of image pixel 
MAX_IMAGE_BRIGHT = 255.0
# Limits for discret observation
MAX_SHIFT = 2.4
MAX_ANGLE = math.radians(12)

class CartPole:
    def __init__(self, img_mode=False, img_size=(50, 50), num_prev_states=3):
        # Image size for tensor
        self.img_size = img_size
        # Number of previous states used
        self.num_prev_states = num_prev_states
        # Mode of observation
        self.img_mode = img_mode
        
        if self.img_mode:
            # The last argument is +1 because previous states + current
            self.tensor_shape = (img_size[0], img_size[1], num_prev_states + 1)
        else:
            self.tensor_shape = (4,)
        # Create target environment
        self.env = gym.make('CartPole-v0')

    def prepare_env(self):
        # Reset environment before first step
        self.last_obs = self.env.reset()
        # Reset internal parameters
        self.reward = 0
        self.done = False
        if self.img_mode:
            # Init previous states with zeros matrices
            self.prev_states = np.zeros((self.tensor_shape[0], 
                                        self.tensor_shape[1],
                                        self.tensor_shape[2] - 1))
    
    def get_obs(self):
        # Show window with environment
        self.env.render()

        if self.img_mode:
            # Get matrix of pixels of environment
            env_pxls = self.env.render(mode='rgb_array')
            # Convert to grayscale 
            env_pxls = utils.rgb2gray(env_pxls)
            # Cut target area
            env_pxls = env_pxls[ENV_HEIGHT_SLICE, ENV_WIDTH_SLICE]
            # Reduce image size
            env_pxls = utils.resize(env_pxls, (self.img_size))
            # Normalize values
            env_pxls = env_pxls.astype('float32') / MAX_IMAGE_BRIGHT

            # Expand tensor (height, width, 1)
            tensor = np.expand_dims(env_pxls, axis=2)
            # Add previous states (height, width, num_prev_states + 1)
            for i in range(self.num_prev_states):
                tensor = np.append(tensor, self.prev_states[:,:,i:i+1], axis=2)
            # Expand tensor (1, height, width, num_prev_states + 1)
            tensor = np.expand_dims(tensor, axis=0)
            
            # Save last observation
            self.last_obs = tensor
        else:
            # Normalize parameters
            self.last_obs[0] = (self.last_obs[0] + MAX_SHIFT) / (2 * MAX_SHIFT)
            self.last_obs[1] = (self.last_obs[1] + 2) / 4
            self.last_obs[2] = (self.last_obs[2] + MAX_ANGLE) / (2 * MAX_ANGLE)
            self.last_obs[3] = (self.last_obs[3] + 2) / 4
            
        return self.last_obs
    
    def step(self, action):
        # Execute step with got action [0; 1]
        obs, _, self.done, _ = self.env.step(action)
        # Add point to reward for step
        self.reward += 1
        
        if self.img_mode:
            # Reset excess dims
            self.last_obs = self.last_obs.reshape(self.last_obs.shape[1],
                                                self.last_obs.shape[2],
                                                self.last_obs.shape[3])
            # Update previous states
            self.prev_states = self.last_obs[:,:,:-1]
        else:
            self.last_obs = obs

        return self.done

    def get_reward(self):
        # Return current reward
        return self.reward

    def is_done(self):
        # Return True if environment has finished
        return self.done
