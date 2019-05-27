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
        
        for weight_idx in range(0, len(m1_weights_list)):
            t = random.randint(0, 1)
            if t == 0:
                result.append(m1_weights_list[weight_idx])
            else:
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


def generate_child_from_arch(model1, model2, tesnsor_size, layers_info, cfg, arch):
    child_weights = crossingover(model1, model2)
    mutation(child_weights, cfg.NUM_MUTATION_WEIGHTS, cfg.MUTATION_FACTOR)
    child_model = network.generate_model_from_list(arch, tesnsor_size)
    
    for layer_idx in range(len(child_model.layers)):
        weights_mtrx = layers_info[layer_idx].get_weights_mtrx(
                        child_weights[:layers_info[layer_idx].size()])
        child_model.layers[layer_idx].set_weights(weights_mtrx)
        child_weights = child_weights[layers_info[layer_idx].size():]
    
    return child_model


def generate_start_architectures(tesnsor_size, cfg):
    num_layers = random.randint(cfg.RANGE_LAYERS[0], cfg.RANGE_LAYERS[1])
    
    arch = list()
    for layer_id in range(num_layers):
        num_neurons = random.randint(cfg.RANGE_NEURONS[0], cfg.RANGE_NEURONS[1])
        arch.append(num_neurons)

    return arch


def crossingover_architectures(arch1, arch2):
    result = []

    min_layers = min(len(arch1), len(arch2))
    max_layers = max(len(arch1), len(arch2))
    
    num_layers = random.randint(min_layers, max_layers)

    for layer_id in range(num_layers):
        neurons_1 = 0
        neurons_2 = 0

        if len(arch1) > layer_id:
            neurons_1 = arch1[layer_id]
        
        if len(arch2) > layer_id:
            neurons_2 = arch2[layer_id]

        min_neurons = min(neurons_1, neurons_2)
        max_neurons = max(neurons_1, neurons_2)

        num_neurons = random.randint(min_neurons, max_neurons)

        result.append(num_neurons)
    
    return result


def mutation_architecture(arch, cfg):
    for i in range(cfg.NUM_MUTATION_LAYERS):
        direction = random.randint(0, 1)
        if direction == 0:
            if len(arch) > 0:
                layer_id = random.randint(0, len(arch))
                if layer_id > 0:
                    layer_id -= 1
                arch.pop(layer_id)
        else:
            num_neurons = random.randint(cfg.RANGE_NEURONS[0], cfg.RANGE_NEURONS[1])
            position = 0
            if len(arch) > 0:
                position = random.randint(0, len(arch))
    
    num_mut_neurons = cfg.NUM_MUTATION_NEURONS
    if num_mut_neurons > (len(arch) - 1):
        num_mut_neurons = (len(arch) - 1)

    if num_mut_neurons > 0:
        idx = random.sample(range(len(arch) - 1), num_mut_neurons)
        for i in idx:
            direction = random.randint(0, 1)
            if direction == 0:
                arch[i] -= cfg.MUTATION_ARCHITECTURE_FACTOR
            else:
                arch[i] += cfg.MUTATION_ARCHITECTURE_FACTOR


def normilize_architecture(acrch):
    return list(filter(lambda x: x > 0, acrch))

def generate_child_architecture(arch1, arch2, tesnsor_size, cfg):
    child_arch = crossingover_architectures(arch1, arch2)
    mutation_architecture(child_arch, cfg)
    child_arch = normilize_architecture(child_arch)
    
    return child_arch