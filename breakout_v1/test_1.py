
import gc
import os
import gym
import time
import copy
import random
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras.models import load_model
from keras import backend as K
from keras.applications import NASNetMobile
from list_converter_v2 import Weights
from keras.utils import plot_model

import subprocess

from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#################### DEFINES ####################
NUM_GENERATION = 1000
NUM_PARENT_NETWORKS = 10
CHILDREN_PER_PARENT = 5
NUM_MUTATION_WEIGHTS = 600
MUTATION_FACTOR = np.float32(0.1)
MAX_REWARD = 1000
MAX_SAMPLES = 1000
RANDOM_SELECTED_NETWORKS = 1
NUM_STARTS_FOR_AVRG = 1
NUM_PREVIOUS_USING_STATES = 3
#################################################

def timer(function):
    def wrap(*args):
        a = time.time()
        res = function(*args)
        print(function.__name__ + ' : {0:.3f}'.format(time.time() - a))
        return res
    return wrap


def current_window_img(offset):
    window = win32gui.GetActiveWindow()
    window_rect = win32gui.GetWindowRect(window)
    target_rect = (window_rect[0] + offset[0], 
                   window_rect[1] + offset[1], 
                   window_rect[2] + offset[2], 
                   window_rect[3] + offset[3])
    window_img = ImageGrab.grab(bbox=target_rect).convert('L')
    # window_img = window_img.resize((int(window_img.size[0] / 4), 
    #                                 int(window_img.size[1] / 4)), 
    #                                 Image.BICUBIC)              
    return window_img


def generate_model(input_size):
    model = models.Sequential()
    model.add(layers.Conv2D(8, (8, 8), activation='relu', input_shape=input_size))#, strides=(4, 4), padding='valid'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (4, 4), activation='relu'))#, strides=(2, 2), padding='valid'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))#, strides=(2, 2), padding='valid'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))#, strides=(2, 2), padding='valid'))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(32, (3, 3), activation='relu'))#, strides=(2, 2), padding='valid'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))#, strides=(2, 2), padding='valid'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))#, strides=(2, 2), padding='valid'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(32, (3, 3), activation='relu'))#, strides=(1, 1), padding='valid'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(32, (3, 3), activation='relu'))#, strides=(1, 1), padding='valid'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Reshape(target_shape=(32, 10*16)))
    # model.add(layers.LSTM(64))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    model.trainable = False

    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    # plot_model(model, to_file='model1.png', show_shapes=True)

    # model.summary()
    # exit()
    
    return model


def get_partner_idx(partner_1_idx, num_networks):
    allowed_idx = [i for i in range(num_networks)]
    allowed_idx.remove(partner_1_idx)
    partner_2_idx = random.randint(0, len(allowed_idx))
    return partner_2_idx


def crossingover1(model1, model2):
    result = []

    for layer_idx in range(len(model1.layers)):
        m1_weights = Weights(model1.layers[layer_idx])
        m2_weights = Weights(model2.layers[layer_idx])
        m1_weights_list = m1_weights.get_weights_list()
        m2_weights_list = m2_weights.get_weights_list()

        model_idx = random.randint(0, 1)
        if model_idx == 0:
            result.extend(m1_weights_list)
        else:
            result.extend(m2_weights_list)
    
    return result

def crossingover2(model1, model2):
    result = []

    for layer_idx in range(len(model1.layers)):
        m1_weights = Weights(model1.layers[layer_idx])
        m2_weights = Weights(model2.layers[layer_idx])
        m1_weights_list = m1_weights.get_weights_list()
        m2_weights_list = m2_weights.get_weights_list()

        for weight_idx in range(len(m1_weights_list)):
            model_idx = random.randint(0, 1)
            if model_idx == 0:
                result.append(m1_weights_list[weight_idx])
            else:
                result.append(m2_weights_list[weight_idx])
    
    return result

def crossingover3(model1, model2):
    result = []

    for layer_idx in range(len(model1.layers)):
        m1_weights = Weights(model1.layers[layer_idx])
        m2_weights = Weights(model2.layers[layer_idx])
        m1_weights_list = m1_weights.get_weights_list()
        m2_weights_list = m2_weights.get_weights_list()
        
        if len(m1_weights_list) == 0:
            continue

        separate_idx = random.randint(0, len(m1_weights_list) - 1)
        for weight_idx in range(0, separate_idx):
            result.append(m1_weights_list[weight_idx])
        for weight_idx in range(separate_idx, len(m1_weights_list)):
            result.append(m2_weights_list[weight_idx])
    
    return result

def mutation(weights, num_weights, mutation_value):
    for i in range(num_weights):
        idx = random.randint(0, len(weights) - 1)
        direction = random.randint(0, 1)
        if direction == 0:
            weights[idx] -= mutation_value
        else:
            weights[idx] += mutation_value

def mutation2(weights, num_weights, mutation_value):
    idx = random.sample(range(len(weights) - 1), num_weights)
    for i in idx:
        direction = random.randint(0, 1)
        if direction == 0:
            weights[i] -= mutation_value
            if weights[i] < -1.0:
                weights[i] = -1.0
        else:
            weights[i] += mutation_value
            if weights[i] > 1.0:
                weights[i] = 1.0

@timer
def generate_child(model1, model2, tesnsor_size, layers_info):
    child_weights = crossingover3(model1, model2)
    mutation2(child_weights, NUM_MUTATION_WEIGHTS, MUTATION_FACTOR)
    child_model = generate_model(tesnsor_size)
    
    for layer_idx in range(len(child_model.layers)):
        weights_mtrx = layers_info[layer_idx].get_weights_mtrx(
                        child_weights[:layers_info[layer_idx].size()])
        child_model.layers[layer_idx].set_weights(weights_mtrx)
        child_weights = child_weights[layers_info[layer_idx].size():]
    
    return child_model


def selection(num_networks, rewards, num_selected, num_random):
    result = []
    for i in range(num_selected - num_random):
        best_network_idx = rewards.index(max(rewards))
        nn = models.load_model('nn' + str(best_network_idx) + '.h5')
        nn.save('nn' + str(i) + '.h5')
        rewards[best_network_idx] = 0
        # rewards.pop(best_network_idx)

    for i in range(num_random):
        random_idx = random.randint(0, num_networks - 1)
        nn = models.load_model('nn' + str(random_idx) + '.h5')
        nn.save('nn' + str(num_selected + i) + '.h5')

    return result


def update_prev_states(prev_states, curr_states):
    for i in range(1, NUM_PREVIOUS_USING_STATES):
        prev_states[:,:,i-1:i] = prev_states[:,:,i:i+1]
    prev_states[:,:,-1:] = np.reshape(curr_states, (curr_states.shape[1],
                                                    curr_states.shape[2],
                                                    curr_states.shape[3])) 

def save_states(states):
    for i in range(states.shape[2]):
        print(states[:,:,i:i+1].shape)
        img = Image.fromarray(np.uint8(states[:,:,i:i+1].reshape((states.shape[0],states.shape[1]))))
        img.save('state[%s].png' % i)

def rgb2gray(img_mtrx):
    return np.dot(img_mtrx[:,:,:3], [0.2989, 0.5870, 0.1140])

def resize(img_mtrx, size):
    height, width = size
    img = Image.fromarray(np.uint8(img_mtrx))
    # img.save('test1.png')
    img = img.resize((width, height), Image.BICUBIC)
    # img.save('test2.png')
    return np.array(img)

def main():
    global NUM_PARENT_NETWORKS
    global CHILDREN_PER_PARENT
    global NUM_MUTATION_WEIGHTS
    global MUTATION_FACTOR

    env = gym.make('Breakout-v0')
    env.frameskip = 1
    env.reset()

    # print(env.get_action_meanings())
    # exit()
    # time.sleep(0.5)
    
    gym_img = env.render(mode='rgb_array')
    gym_img = rgb2gray(gym_img)
    gym_img = gym_img[20:, :]
    gym_img = resize(gym_img, (90, 80))

    gym_img = gym_img.astype('float32') / 255.0

    img_tensor = np.array(gym_img, dtype='float')
    img_tensor = img_tensor.reshape((img_tensor.shape[0],
                                     img_tensor.shape[1],
                                     1))

    for _ in range(NUM_PREVIOUS_USING_STATES):
        img_tensor = np.append(img_tensor, img_tensor[:,:,0:1], axis=2)

    for i in range(NUM_PARENT_NETWORKS):
        nn = generate_model(img_tensor.shape)
        nn.save('nn' + str(i) + '.h5')
        K.clear_session()
        gc.collect()
    
    K.clear_session()
    gc.collect()

    # nnetworks = [generate_model(img_tensor.shape) 
    #              for i in range(NUM_PARENT_NETWORKS)]

    nn = models.load_model('nn0.h5')

    layers_info = []
    for i in range(len(nn.layers)):
        layers_info.append(Weights(nn.layers[i]))

    max_reward = 0
    for gen_idx in range(NUM_GENERATION):
        print('Generation {}'.format(gen_idx))
        with open('GAConfig.txt') as cfg:
            NUM_PARENT_NETWORKS = int(cfg.readline())
            CHILDREN_PER_PARENT = int(cfg.readline())
            NUM_MUTATION_WEIGHTS = int(cfg.readline())
            MUTATION_FACTOR = np.float32(float(cfg.readline()))
            print(NUM_PARENT_NETWORKS, CHILDREN_PER_PARENT, NUM_MUTATION_WEIGHTS, MUTATION_FACTOR)
        
        for net_idx in range(NUM_PARENT_NETWORKS):
            for child_idx in range(CHILDREN_PER_PARENT):
                partner_idx = get_partner_idx(net_idx, NUM_PARENT_NETWORKS)
                nn_parent1 = models.load_model('nn' + str(net_idx) + '.h5')
                nn_parent2 = models.load_model('nn' + str(partner_idx) + '.h5')
                child_model = generate_child(nn_parent1,
                                             nn_parent2,
                                             img_tensor.shape,
                                             layers_info)
                safe_idx = NUM_PARENT_NETWORKS + net_idx * CHILDREN_PER_PARENT + child_idx
                child_model.save('nn' + str(safe_idx) + '.h5')
                K.clear_session()
                gc.collect()
            K.clear_session()
            gc.collect()
                # nnetworks.append(child_model)

        num_networks = NUM_PARENT_NETWORKS + CHILDREN_PER_PARENT * NUM_PARENT_NETWORKS
        
        rewards = [0 for i in range(num_networks)]
        for network_idx in range(num_networks):
            current_nn = models.load_model('nn' + str(network_idx) + '.h5')
            run_results = np.array([])
            for start_id in range(NUM_STARTS_FOR_AVRG):
                reward = 0
                sample = 0
                env.reset()
                # env.seed()
                env.step(1)
                
                prev_states = np.zeros((img_tensor.shape[0],
                                        img_tensor.shape[1],
                                        img_tensor.shape[2] - 1))
                # for i in range(img_tensor.shape[2] - 1):
                #     prev_states[:,:,i:i+1] = img_tensor[:,:,0:1]

                prev_lives = 5
                while sample < MAX_SAMPLES:
                    # time.sleep(0.01)
                    env.render()
                    gym_img = env.render(mode='rgb_array') #current_window_img(WINDOW_OFFSET)
                    gym_img = rgb2gray(gym_img)
                    # gym_img = gym_img[150:350, 200:400]
                    gym_img = gym_img[20:, :]
                    gym_img = resize(gym_img, (90, 80))
                    gym_img = gym_img.astype('float32') / 255.0

                    gym_tensor = np.array(gym_img, dtype='float')
                    gym_tensor = gym_tensor.reshape((gym_tensor.shape[0],
                                                    gym_tensor.shape[1],
                                                    1))
                    for i in range(NUM_PREVIOUS_USING_STATES):
                        gym_tensor = np.append(gym_tensor, prev_states[:,:,i:i+1], axis=2)

                    gym_tensor = np.expand_dims(gym_tensor, axis=0)
                    
                    predict = current_nn.predict(gym_tensor)
                    predict = np.argmax(predict)
                    if predict == 0:
                        action = 0
                    elif predict == 1:
                        action = 2
                    elif predict == 2:
                        action = 3
                    # else:
                    #     action = 3
                    
                    obs, rew, done, lives = env.step(action)
                    reward += rew
                    sample += 1

                    if lives['ale.lives'] < prev_lives:
                        env.step(1)

                    # print(rew, done, lives)
                    if done:
                        # run_results = np.append(run_results, rew)
                        break
                    else:
                        # if reward % 2 == 0:
                        update_prev_states(prev_states, gym_tensor[:,:,:,0:1])
                run_results = np.append(run_results, reward)

            rewards[network_idx] = int(np.mean(run_results))
            if max_reward < max(rewards):
                max_reward = max(rewards)
                with open("max_reward.txt", "w") as f:
                    f.writelines(['MAX REWARD COMMON: {}'.format(max_reward)])
                current_nn.save('best_network.h5')
            print('Network {}: {}'.format(network_idx, rewards[network_idx]))
            
            K.clear_session()
            gc.collect()

        print('-'*40)
        print('MAX REWARD CURRENT: {}'.format(max(rewards)))
        print('MAX REWARD COMMON: {}'.format(max_reward))
        print('-'*40)

        nnetworks = selection(num_networks,
                              rewards,
                              NUM_PARENT_NETWORKS,
                              RANDOM_SELECTED_NETWORKS)

        # for i in range(len(nnetworks)):
        #     nnetworks[i].save('tmp'+str(i) + '.h5')
        
        # nnetworks.clear()
        
        K.clear_session()
        gc.collect()

        # nnetworks = []
        # for i in range(NUM_PARENT_NETWORKS):
        #     nnetworks.append(load_model('tmp' + str(i) + '.h5'))
            
if __name__ == '__main__':
    main()