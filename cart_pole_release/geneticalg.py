import random
import network
from keras import models
from keras import layers
from utils import Weights

def get_partner(partner_1, num_networks):
    allowed_partners = list(range(num_networks))
    allowed_partners.remove(partner_1)
    partner_2 = random.randint(0, len(allowed_partners))
    
    return partner_2


def crossingover(model1, model2):
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
    idx = random.sample(range(len(weights) - 1), num_weights)
    
    for i in idx:
        direction = random.randint(0, 1)
        if direction == 0:
            weights[i] -= mutation_value
        else:
            weights[i] += mutation_value


def selection(num_networks, rewards, num_selected, num_random, num_new_random, tesnsor_size):
    result = []
    nn_name_tmpl = 'nn{}.h5'

    for i in range(num_selected - num_random - num_new_random):
        best_network_idx = rewards.index(max(rewards))
        nn = models.load_model(nn_name_tmpl.format(best_network_idx))
        nn.save(nn_name_tmpl.format(i))
        rewards[best_network_idx] = 0

    for i in range(num_random):
        random_idx = random.randint(0, num_networks - 1)
        nn = models.load_model(nn_name_tmpl.format(random_idx))
        nn.save(nn_name_tmpl.format(num_selected - num_random - num_new_random + i))

    for i in range(num_new_random):
        new_model = network.generate_model(tesnsor_size)
        nn.save(nn_name_tmpl.format(num_selected - num_new_random + i))

    return result


def generate_child(model1, model2, tesnsor_size, layers_info, cfg):
    child_weights = crossingover(model1, model2)
    mutation(child_weights, cfg.NUM_MUTATION_WEIGHTS, cfg.MUTATION_FACTOR)
    child_model = network.generate_model(tesnsor_size)
    
    for layer_idx in range(len(child_model.layers)):
        weights_mtrx = layers_info[layer_idx].get_weights_mtrx(
                        child_weights[:layers_info[layer_idx].size()])
        child_model.layers[layer_idx].set_weights(weights_mtrx)
        child_weights = child_weights[layers_info[layer_idx].size():]
    
    return child_model