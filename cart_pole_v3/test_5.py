import gc
import gym
import time
import random
import win32gui
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras.models import load_model
from keras import backend as K
from keras.applications import VGG16
from list_converter import *
from list_converter_v2 import Weights

from PIL import ImageGrab, Image, ImageDraw


#################### DEFINES ####################
WINDOW_OFFSET = (8, 189, -8, -80)
NUM_GENERATION = 300
NUM_PARENT_NETWORKS = 2
CHILDREN_PER_PARENT = 1
NUM_MUTATION_WEIGHTS = 10000
MUTATION_FACTOR = np.float32(0.1)
MAX_REWARD = 500
RANDOM_SELECTED_NETWORKS = 1
#################################################

def timer(function):
    def wrap(*args):
        a = time.time()
        res = function(*args)
        print(function.__name__ + ':' + str(time.time() - a))
        return res
    return wrap


def current_window_img(offset):
    window = win32gui.GetActiveWindow()
    window_rect = win32gui.GetWindowRect(window)
    target_rect = (window_rect[0] + offset[0], 
                   window_rect[1] + offset[1], 
                   window_rect[2] + offset[2], 
                   window_rect[3] + offset[3])
    window_img = ImageGrab.grab(bbox=target_rect)
    return window_img


def generate_model(input_size):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=input_size))
    # model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.trainable = False

    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    # model.summary()
    # exit()
    
    return model


def get_partner_idx(partner_1_idx, networks):
    allowed_idx = [i for i in range(len(networks))]
    allowed_idx.remove(partner_1_idx)
    partner_2_idx = random.randint(0, len(allowed_idx))
    return partner_2_idx


def crossingover(model1, model2):
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


def mutation(weights, num_weights, mutation_value):
    for i in range(num_weights):
        idx = random.randint(0, len(weights) - 1)
        direction = random.randint(0, 1)
        if direction == 0:
            weights[idx] -= mutation_value
        else:
            weights[idx] += mutation_value

@timer
def generate_child(model1, model2, tesnsor_size, layers_info):
    child_weights = crossingover(model1, model2)
    mutation(child_weights, NUM_MUTATION_WEIGHTS, MUTATION_FACTOR)
    child_model = generate_model(tesnsor_size)
    
    for layer_idx in range(len(child_model.layers)):
        weights_mtrx = layers_info[layer_idx].get_weights_mtrx(
                        child_weights[:layers_info[layer_idx].size()])
        child_model.layers[layer_idx].set_weights(weights_mtrx)
        child_weights = child_weights[layers_info[layer_idx].size():]
    
    return child_model


def selection(networks, rewards, num_selected, num_random):
    result = []
    for i in range(num_selected - num_random):
        best_network_idx = rewards.index(max(rewards))
        result.append(networks[best_network_idx])
        networks.pop(best_network_idx)
        rewards.pop(best_network_idx)

    for i in range(num_random):
        random_idx = random.randint(0, len(networks) - 1)
        result.append(networks[random_idx])
        networks.pop(random_idx)
    
    gc.collect()

    return result


def main():
    env = gym.make('CartPole-v1')
    env.reset()
    env.render()
    time.sleep(0.5)
    gym_img = current_window_img(WINDOW_OFFSET)
    
    img_tensor = np.array(gym_img, dtype='float')

    # vgg16 = VGG16(weights='imagenet', 
    #               include_top=False,
    #               input_shape=img_tensor.shape)
    
    # conv_base = models.Sequential()
    # conv_base.add(vgg16.layers[1])
    # conv_base.add(vgg16.layers[3])
    # conv_base.add(vgg16.layers[4])
    # conv_base.add(vgg16.layers[6])
    # conv_base.add(vgg16.layers[7])
    # conv_base.add(vgg16.layers[10])

    # conv_base.trainable = False

    # conv_base.compile(optimizer=optimizers.RMSprop(lr=1e-4),
    #                   loss='binary_crossentropy',
    #                   metrics=['acc'])

    # gym_img = current_window_img(WINDOW_OFFSET)
    # gym_tensor = np.array(gym_img, dtype='float')
    # gym_tensor = np.expand_dims(gym_tensor, axis=0)                  

    # a = conv_base.predict(gym_tensor)
    # conv_base.summary()
    # exit()
    
    nnetworks = [generate_model((1575,)) 
                 for i in range(NUM_PARENT_NETWORKS)]

    layers_info = []
    for i in range(len(nnetworks[0].layers)):
        layers_info.append(Weights(nnetworks[0].layers[i]))

    for gen_idx in range(NUM_GENERATION):
        print('Generation {}'.format(gen_idx))

        vgg16 = VGG16(weights='imagenet', 
                  include_top=False,
                  input_shape=img_tensor.shape)
        conv_base = models.Sequential()
        conv_base.add(vgg16.layers[1])
        conv_base.add(vgg16.layers[3])
        conv_base.add(vgg16.layers[4])
        conv_base.add(vgg16.layers[6])
        conv_base.add(vgg16.layers[7])
        conv_base.add(vgg16.layers[10])

        conv_base.trainable = False

        conv_base.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                        loss='binary_crossentropy',
                        metrics=['acc'])


        for net_idx in range(NUM_PARENT_NETWORKS):
            for child_idx in range(CHILDREN_PER_PARENT):
                partner_idx = get_partner_idx(net_idx, nnetworks)
                child_model = generate_child(nnetworks[net_idx],
                                             nnetworks[partner_idx],
                                             (1575,),
                                             layers_info)
                nnetworks.append(child_model)

        rewards = [0 for i in range(len(nnetworks))]
        for network_idx in range(len(nnetworks)):
            reward = 0
            env.reset()

            last_tensor = None
            while reward < MAX_REWARD:
                env.render()
                gym_img = current_window_img(WINDOW_OFFSET)
                gym_tensor = np.array(gym_img, dtype='float')
                gym_tensor = np.expand_dims(gym_tensor, axis=0)

                # a = time.time()
                conv_predict = conv_base.predict(gym_tensor)
                # print('conv_base :' + str(time.time() - a))
                conv_predict = conv_predict[:,:,:,0:1]
                conv_predict = conv_predict.reshape((1, 1575))

                # a = time.time()
                predict = nnetworks[network_idx].predict(conv_predict)
                # print('nnetworks :' + str(time.time() - a))
                
                action = 0 if predict[0][0] < 0.5 else 1
                _, _, done, _ = env.step(action)
                reward += 1

                last_tensor = gym_tensor

                if done:
                    rewards[network_idx] = reward
                    break
            
            print('Network {}: {}'.format(network_idx, reward))
        print('MAX REWARD: {}'.format(max(rewards)))
        print('-'*40)

        nnetworks = selection(nnetworks, 
                              rewards, 
                              NUM_PARENT_NETWORKS, 
                              RANDOM_SELECTED_NETWORKS)

        for i in range(len(nnetworks)):
            nnetworks[i].save('tmp'+str(i) + '.h5')
        
        nnetworks.clear()
        
        K.clear_session()
        gc.collect()

        nnetworks = []
        for i in range(NUM_PARENT_NETWORKS):
            nnetworks.append(load_model('tmp' + str(i) + '.h5'))
            
if __name__ == '__main__':
    main()