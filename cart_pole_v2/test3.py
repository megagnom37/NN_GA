import gym
from network import *
import math
import random       
import copy
import time
from PIL import ImageGrab, Image, ImageDraw
import time
import win32gui
import numpy as np
import itertools
from scipy import signal
from conv_network import *

coord_max = 2.4
angle_max = math.radians(12)

num_generation = 1000
num_best_networks = 5
num_children = 3
num_weights_mutation = 550
mutation_value = 0.1

def crossingover(weights1, weights2):
    # print(len(weights1))
    result = []
    for i in range(len(weights1)):
        side = random.randint(0, 1)
        if side == 0:
            result.append(weights1[i])
        else:
            result.append(weights2[i])
    return result

def mutation(weights1):
    for i in range(num_weights_mutation):
        idx = random.randint(0, len(weights1)-1)
        sign = random.randint(0, 1)
        if sign == 0:
            weights1[idx] -= mutation_value
        else:
            weights1[idx] += mutation_value

def get_target_img():
    window = win32gui.GetActiveWindow()
    rect = win32gui.GetWindowRect(window)
    rect = (rect[0]+8, rect[1]+189, rect[2]-8, rect[3]-80)
    img = ImageGrab.grab(bbox=rect).convert('L')
    Image.fromarray(np.uint8(img)).save('test.png')
    exit()
    # print("TARGET_IMG: {}".format(img.size))
    return img

def convolution(img, core):
    img_h, img_w = img.shape
    core_h, core_w = core.shape

    res_mtrx = np.zeros((img_h - core_h, img_w - core_w))

    for y in range(img_h - core_h):
        for x in range(img_w - core_w):
            res_mtrx[y][x] = np.sum(img[y:y+core_h, x:x+core_h] * core)

    return res_mtrx

def pulling(img, pull_size):
    img_h, img_w = img.shape
    res_mtrx = np.zeros((math.ceil(img_h / pull_size), math.ceil(img_w / pull_size)))

    for y in range(0, img_h, pull_size):
        for x in range(0, img_w, pull_size):
            res_mtrx[int(y/pull_size)][int(x/pull_size)] = np.min(img[y:y+pull_size,x:x+pull_size])
    
    return res_mtrx

def relu(img):
    return np.clip(img, 0, np.max(img))
    

nnetworks = []
for _ in range(num_best_networks):
    conv_nn = ConvolutionNetwork()
    conv_nn.nn.set_input_layer(95)
    conv_nn.nn.set_hide_layer(50)
    conv_nn.nn.set_hide_layer(10)
    conv_nn.nn.set_output_layer(1)
    conv_nn.add_filter((6, 6))
    conv_nn.add_filter((3, 3))
    nnetworks.append(conv_nn)

env = gym.make('CartPole-v1')

for i_gen in range(num_generation):
    print('Generation : {}'.format(i_gen))
    for i_nn in range(len(nnetworks)):
        for i_chld in range(num_children):
            idx_partner = random.randint(0, num_best_networks-1)
            child_weights = crossingover(nnetworks[i_nn].get_weights_list(),
                                         nnetworks[idx_partner].get_weights_list())
            mutation(child_weights)
            conv_nn = ConvolutionNetwork()
            conv_nn.nn.set_input_layer(95)
            conv_nn.nn.set_hide_layer(50)
            conv_nn.nn.set_hide_layer(10)
            conv_nn.nn.set_output_layer(1)
            conv_nn.add_filter((6, 6))
            conv_nn.add_filter((3, 3))
            conv_nn.set_weights_from_list(child_weights)
            nnetworks.append(conv_nn)

    nn_rewards = []
    for num_nn, conv_nn in enumerate(nnetworks):
        # print('Number of NN : {}'.format(num_nn))
        reward = 0
        observation = env.reset()
        env.render()
        time.sleep(0.5)
        while reward < 1000:
            env.render()
            
            img = get_target_img()
            img = np.array(img, dtype='float')

            img = signal.convolve2d(img, conv_nn.filters[0], boundary='symm', mode='valid')
            # print(img)
            # img = relu(img)
            # print(img)
            img = pulling(img, 4)
            # img = Image.fromarray(np.uint8(img))
            # img = img.resize((img.size[0] // 4, img.size[1] // 4), Image.BILINEAR)
            # img = np.array(img, dtype='float')
            img = signal.convolve2d(img, conv_nn.filters[1], boundary='symm', mode='valid')
            # img = relu(img)
            img = pulling(img, 4)
            # img = Image.fromarray(np.uint8(img))
            # img = img.resize((img.size[0] // 4, img.size[1] // 4), Image.BILINEAR)
            # img = np.array(img, dtype='float')
            # img = signal.convolve2d(img, conv_nn.filters[2], boundary='symm', mode='valid')
            # img = relu(img)
            img = pulling(img, 2)
            
            # print(img.shape)
            # t1 = time.time()
            # img = get_target_img()
            # img = np.array(img, dtype='float')
            # t1 = time.time()
            # img.thumbnail((img.size[0] // 16, img.size[1] // 16), Image.BILINEAR)
            # img = np.array(img, dtype='float')
            # t1 = time.time()
            # grad = signal.convolve2d(img, conv1, boundary='symm', mode='valid')
            # t2 = time.time()
            # print("Time 120 = {}".format(t2 - t1))
            # print(grad.shape)
            # Image.fromarray(np.uint8(grad)).save('test.png')
            # exit()
            # t2 = time.time()
            # print("Time 120 = {}".format(t2 - t1))
            # print(img.size)
            # Image.fromarray(np.uint8(img)).save('test.png')
            # exit()
            
            # t2 = time.time()
            # print("Time 117 = {}".format(t2 - t1))

            # t1 = time.time()
            # conv1 = convolution(img, sobel_horizontal)
            # t2 = time.time()
            # print("Time 119 = {}".format(t2 - t1))
            # # Image.fromarray(np.uint8(conv1)).save('test1.png')
            # t1 = time.time()
            # conv2 = convolution(img, sobel_vertical)
            # t2 = time.time()
            # print("Time 127 = {}".format(t2 - t1))
            # # Image.fromarray(np.uint8(conv2)).save('test2.png')
            # t1 = time.time()
            # conv3 = conv1 + conv2
            # t2 = time.time()
            # print("Time 132 = {}".format(t2 - t1))
            # print("CONV_IMG: {}".format(conv3.shape))
            # Image.fromarray(np.uint8(conv3)).save('test3.png')


            # t1 = time.time()
            # img = pulling(img)
            # print(img.shape)
            img /= np.max(np.absolute(img))
            # t2 = time.time()
            # print("Time 139 = {}".format(t2 - t1))
            # Image.fromarray(np.uint8(img)).save('test.png')
            # exit()
            # print("PUL1_IMG: {}".format(img.shape))
            # t1 = time.time()
            # img = pulling(img)
            # t2 = time.time()
            # print("Time 144 = {}".format(t2 - t1))
            # print("PUL2_IMG: {}".format(img.shape))
            # t1 = time.time()
            # img = pulling(img)
            # t2 = time.time()
            # print("Time 149 = {}".format(t2 - t1))
            # print("PUL3_IMG: {}".format(img.shape))
            # img = pulling(img)
            # print("PUL4_IMG: {}".format(img.shape))
            # Image.fromarray(np.uint8(img)).save('test.png')
            # exit()
            
            reward += 1
            # coord = (observation[0] + coord_max) / (2 * coord_max)
            # x_speed = (observation[1] + 2) / (2 * 2)
            # angle = (observation[2] + angle_max) / (2 * angle_max)
            # angle_speed = (observation[3] + 2) / (2 * 2)
            # t1 = time.time()
            input_data = list(itertools.chain(*img.tolist()))
            # t2 = time.time()
            # print("Time 164 = {}".format(t2 - t1))
            

            # t1 = time.time()
            solution = conv_nn.nn.get_solution(input_data)[0]
            # solution = 1
            # t2 = time.time()
            # print("Time 170 = {}".format(t2 - t1))
            # exit()

            action = 0 if solution < 0.5 else 1
            observation, _, done, _ = env.step(action)
            # print(observation)
            if done:
                nn_rewards.append(reward)
                break
    print(max(nn_rewards))

    new_nnetworks = []
    for i in range(num_best_networks - 2):
        idx = nn_rewards.index(max(nn_rewards))
        nn_rewards.pop(idx)
        new_nnetworks.append(nnetworks[idx])
        nnetworks.pop(idx)
    new_nnetworks.append(nnetworks[random.randint(0, len(nnetworks)-1)])
    new_nnetworks.append(nnetworks[random.randint(0, len(nnetworks)-1)])
    nnetworks = copy.deepcopy(new_nnetworks)
    # print(new_nnetworks)
    # print('_'*50)

# observation = env.reset()
# nn = nnetworks[0]
# while reward < 1000:
#     env.render()
#     coord = (observation[0] + coord_max) / (2 * coord_max)
#     x_speed = (observation[1] + 2) / (2 * 2)
#     angle = (observation[2] + angle_max) / (2 * angle_max)
#     angle_speed = (observation[3] + 2) / (2 * 2)
#     solution = nn.get_solution([coord, x_speed, angle, angle_speed])[0]
#     action = 0 if solution < 0.5 else 1
#     observation, _, done, _ = env.step(action)
#     time.sleep(0.01)
# nn.activate_neurons([coord, angle])

# env = gym.make('CartPole-v1')

# observation = env.reset()
# print(env.__dict__)
# for t in range(100):
#     env.render()
#     print(observation)
#     action = 1
#     observation, reward, done, info = env.step(action)
#     if done:
#         print("Episode finished after {} timesteps".format(t+1))
#         # break