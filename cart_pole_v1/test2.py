import gym
from network import *
import math
import random       
import copy
import time

coord_max = 2.4
angle_max = math.radians(12)

num_generation = 3
num_best_networks = 30
num_children = 25
num_weights_mutation = 3
mutation_value = 0.1

def crossingover(weights1, weights2):
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

nnetworks = []
for _ in range(num_best_networks):
    nn = NNetwork()
    nn.set_input_layer(4)
    nn.set_hide_layer(5)
    nn.set_output_layer(1)
    nnetworks.append(nn)

env = gym.make('FrozenLake8x8-v0')

for i_gen in range(num_generation):
    # print('Generation : {}'.format(i_gen))
    for i_nn in range(len(nnetworks)):
        for i_chld in range(num_children):
            idx_partner = random.randint(0, num_best_networks-1)
            child_weights = crossingover(nnetworks[i_nn].get_weights_as_list(), 
                                         nnetworks[idx_partner].get_weights_as_list())
            mutation(child_weights)
            nn = NNetwork()
            nn.set_input_layer(4)
            nn.set_hide_layer(5)
            nn.set_output_layer(1)
            nn.set_weights_from_list(child_weights)
            nnetworks.append(nn)
    # print(nnetworks)
    nn_rewards = []
    for nn in nnetworks:
        reward = 0
        observation = env.reset()
        while reward < 1000:
            reward += 1
            coord = (observation[0] + coord_max) / (2 * coord_max)
            x_speed = (observation[1] + 2) / (2 * 2)
            angle = (observation[2] + angle_max) / (2 * angle_max)
            angle_speed = (observation[3] + 2) / (2 * 2)
            solution = nn.get_solution([coord, x_speed, angle, angle_speed])[0]
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

observation = env.reset()
nn = nnetworks[0]
while reward < 1000:
    env.render()
    coord = (observation[0] + coord_max) / (2 * coord_max)
    x_speed = (observation[1] + 2) / (2 * 2)
    angle = (observation[2] + angle_max) / (2 * angle_max)
    angle_speed = (observation[3] + 2) / (2 * 2)
    solution = nn.get_solution([coord, x_speed, angle, angle_speed])[0]
    action = 0 if solution < 0.5 else 1
    observation, _, done, _ = env.step(action)
    time.sleep(0.01)
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