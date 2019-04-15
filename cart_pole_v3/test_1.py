import gym
import time
import random
import win32gui
import numpy as np
from keras import models
from keras import layers
from list_converter import *
from list_converter_v2 import Weights
from keras import optimizers
from PIL import ImageGrab, Image, ImageDraw


#################### DEFINES ####################
WINDOW_OFFSET = (8, 189, -8, -80)
NUM_GENERATION = 300
NUM_PARENT_NETWORKS = 5
CHILDREN_PER_PARENT = 3
NUM_MUTATION_WEIGHTS = 6000
MUTATION_FACTOR = np.float32(0.1)
MAX_REWARD = 500
RANDOM_SELECTED_NETWORKS = 1
#################################################


def current_window_img(offset):
    window = win32gui.GetActiveWindow()
    window_rect = win32gui.GetWindowRect(window)
    target_rect = (window_rect[0] + offset[0], 
                   window_rect[1] + offset[1], 
                   window_rect[2] + offset[2], 
                   window_rect[3] + offset[3])
    window_img = ImageGrab.grab(bbox=target_rect).convert('L')
    return window_img


def generate_model(input_size):
    model = models.Sequential()
    model.add(layers.Conv2D(8, (5, 5), activation='relu',
                            input_shape=input_size))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (4, 4), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    
    model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
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
        m1_weights = model1.layers[layer_idx].get_weights()
        m2_weights = model2.layers[layer_idx].get_weights()
        m1_weights = to_list(m1_weights, np.float32)
        m2_weights = to_list(m2_weights, np.float32)

        child_weights = []
        for weight_idx in range(len(m1_weights)):
            model_idx = random.randint(0, 1)
            if model_idx == 0:
                child_weights.append(m1_weights[weight_idx])
            else:
                child_weights.append(m2_weights[weight_idx])

        result.append(child_weights)
    
    return result


def mutation(model, num_weights, mutation_value):
    weights = to_list(model, np.float32)
    for i in range(num_weights):
        idx = random.randint(0, len(weights) - 1)
        direction = random.randint(0, 1)
        if direction == 0:
            weights[idx] -= mutation_value
        else:
            weights[idx] += mutation_value
    recover_list(model, weights, np.float32)


def generate_child(model1, model2, tesnsor_size):
    child_weights = crossingover(model1, model2)
    mutation(child_weights, NUM_MUTATION_WEIGHTS, MUTATION_FACTOR)
    child_model = generate_model(tesnsor_size)
    for layer_idx in range(len(child_model.layers)):
        p = child_model.layers[layer_idx].get_weights()
        # print(child_model.layers[layer_idx].get_weights())
        print((child_model.layers[layer_idx].get_weights()[0].shape))
        print(child_model.layers[layer_idx].get_weights())
        recover_list(p, 
                     child_weights[layer_idx],
                     np.float32)
        child_model.layers[layer_idx].set_weights(p)
        # print(child_model.layers[layer_idx].get_weights())
        exit()
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

    return result


def main():
    env = gym.make('CartPole-v1')
    env.reset()
    env.render()
    time.sleep(0.5)
    gym_img = current_window_img(WINDOW_OFFSET)
    
    img_tensor = np.array(gym_img, dtype='float')
    img_tensor = img_tensor.reshape((img_tensor.shape[0],
                                     img_tensor.shape[1],
                                     1))

    nnetworks = [generate_model(img_tensor.shape) 
                 for i in range(NUM_PARENT_NETWORKS)]
    
    for gen_idx in range(NUM_GENERATION):
        print('Generation {}'.format(gen_idx))
        for net_idx in range(NUM_PARENT_NETWORKS):
            for child_idx in range(CHILDREN_PER_PARENT):
                partner_idx = get_partner_idx(net_idx, nnetworks)
                child_model = generate_child(nnetworks[net_idx],
                                             nnetworks[partner_idx],
                                             img_tensor.shape)
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
                gym_tensor = gym_tensor.reshape((gym_tensor.shape[0],
                                                 gym_tensor.shape[1],
                                                 1))
                gym_tensor = np.expand_dims(gym_tensor, axis=0)

                predict = nnetworks[network_idx].predict(gym_tensor)
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
        
if __name__ == '__main__':
    main()