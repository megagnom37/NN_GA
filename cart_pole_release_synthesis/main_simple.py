import os
import utils
import config
import network
import cartpole
import geneticalg
import numpy as np
from keras import models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
NN_NAME_TMPL = 'nn{}.h5'

alpha = 10
betta = 1

def main():
    # Loading configuration
    cfg = config.Config('ga_cfg.json')
    cfg.update()

    # Creating the environment
    env = cartpole.CartPole(img_mode=False)

    archs = list()
    for i in range(cfg.NUM_PARENT_ARCHITECTURES):
        arch = geneticalg.generate_start_architectures(env.tensor_shape, cfg)
        archs.append(arch)

    for arch_gen_id in range(cfg.NUM_ARCHITECTURES_GENERATIONS):
        print('ARCH_GENERATION: {}'.format(arch_gen_id))
        
        for arch_idx in range(cfg.NUM_PARENT_ARCHITECTURES):
            for child_arch_idx in range(cfg.NUM_CHILD_ARCHITECTURES):
                partner_idx = geneticalg.get_partner(arch_idx, cfg.NUM_PARENT_ARCHITECTURES)

                arch_parent1 = archs[arch_idx]
                arch_parent2 = archs[partner_idx]
                
                child_arch = geneticalg.generate_child_architecture(arch_parent1, 
                                                                    arch_parent2, 
                                                                    env.tensor_shape,
                                                                    cfg)
                archs.append(child_arch)

        print('ALL_ARCHS: {}'.format(archs))
        arch_rewards = list()

        for curr_arch in archs:
            print('CURRENT_ARCH: {}'.format(curr_arch))

            # Generation of the start population
            for i in range(cfg.NUM_START_POPULATION):
                nn = network.generate_model_from_list(curr_arch, env.tensor_shape)
                nn.save(NN_NAME_TMPL.format(i))
                utils.clear_session()

            # Download the first model of the neural network
            nn = models.load_model(NN_NAME_TMPL.format(0))

            # Creating information about network layers
            layers_info = []
            for i in range(len(nn.layers)):
                layers_info.append(utils.Weights(nn.layers[i]))
            
            # Maximum reward for all epochs
            max_reward = 0

            # The main cycle of epochs
            for gen_idx in range(cfg.NUM_GENERATION): 
                print('NN_GENERATION: {}'.format(gen_idx))
                # Read updated configuration
                cfg.update()

                # If the first epocha, then do not generate children
                if gen_idx == 0:
                    num_networks = cfg.NUM_START_POPULATION
                # Else generate children
                else:
                    num_tasks = cfg.NUM_PARENT_NETWORKS * cfg.CHILDREN_PER_PARENT
                    
                    for net_idx in range(cfg.NUM_PARENT_NETWORKS):
                        for child_idx in range(cfg.CHILDREN_PER_PARENT):
                            partner_idx = geneticalg.get_partner(net_idx, cfg.NUM_PARENT_NETWORKS)
                            
                            nn_parent1 = models.load_model(NN_NAME_TMPL.format(net_idx))
                            nn_parent2 = models.load_model(NN_NAME_TMPL.format(partner_idx))
                            
                            child_model = geneticalg.generate_child_from_arch(nn_parent1, 
                                                                    nn_parent2, 
                                                                    env.tensor_shape, 
                                                                    layers_info, 
                                                                    cfg,
                                                                    curr_arch)

                            safe_idx = (cfg.NUM_PARENT_NETWORKS + child_idx 
                                        + net_idx * cfg.CHILDREN_PER_PARENT)

                            child_model.save(NN_NAME_TMPL.format(safe_idx))
                            utils.clear_session()
                        utils.clear_session()

                    num_networks = (cfg.NUM_PARENT_NETWORKS 
                        + cfg.CHILDREN_PER_PARENT * cfg.NUM_PARENT_NETWORKS)

                # Estimates for the current epoch
                gen_rewards = [0 for i in range(num_networks)]

                # Cycle to test each neural network
                for network_idx in range(num_networks):
                    current_nn = models.load_model(NN_NAME_TMPL.format(network_idx))

                    # Estimates for different tests of current neural network
                    nn_rewards = np.array([])
                    # Cycle to test different attempts
                    for start_id in range(cfg.NUM_STARTS_FOR_AVRG):
                        env.prepare_env()

                        while not env.is_done():
                            obs = env.get_obs()
                            obs = obs.reshape((1, 4))

                            predict = current_nn.predict(obs)
                            action = 0 if predict[0][0] < 0.5 else 1

                            env.step(action)

                        nn_rewards = np.append(nn_rewards, env.get_reward())
                    
                    # Save the average estimate for the current neural network
                    gen_rewards[network_idx] = int(np.mean(nn_rewards))
                    # Update and save the best estimate and network for all epochs
                    if max_reward < gen_rewards[network_idx]:
                        max_reward = gen_rewards[network_idx]
                        # with open("max_reward.txt", "w") as f:
                        #     f.writelines(['MAX REWARD COMMON: {}'.format(max_reward)])
                        # current_nn.save('best_network.h5')

                    utils.clear_session()

                print(max(gen_rewards))
                if max(gen_rewards) > 199:
                    break

                # Selection of the best neural networks
                nnetworks = geneticalg.selection(num_networks,
                                    gen_rewards,
                                    cfg.NUM_PARENT_NETWORKS,
                                    cfg.RANDOM_SELECTED_NETWORKS,
                                    cfg.NEW_GENERATED_RANDOM_NETWORK,
                                    env.tensor_shape)
                
                utils.clear_session()
        
            arch_reward = (cfg.NUM_GENERATION * alpha + (max(cfg.RANGE_LAYERS) * max(cfg.RANGE_NEURONS) * betta) 
                           - (gen_idx + 1) * alpha - sum(curr_arch) * betta)

            print('ARCH_REWARD: {}'.format(arch_reward))
            arch_rewards.append(arch_reward)

        new_archs = list()
        for _ in range(cfg.NUM_PARENT_ARCHITECTURES):
            best_arch_idx = arch_rewards.index(max(arch_rewards))
            new_archs.append(archs[best_arch_idx])
            arch_rewards[best_arch_idx] = -1
        archs = new_archs[:]
        
        del new_archs

    
if __name__ == '__main__':
    main()