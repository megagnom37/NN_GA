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


def main():
    # Loading configuration
    cfg = config.Config('ga_cfg.json')
    cfg.update()

    # Creating the environment
    env = cartpole.CartPole(img_mode=True, 
                            img_size=(100, 150), 
                            num_prev_states = cfg.NUM_PREVIOUS_USING_STATES)

    # Generation of the start population
    for i in range(cfg.NUM_START_POPULATION):
        nn = network.generate_model(env.tensor_shape)
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
        print('#### GENERATION {} ####'.format(gen_idx))
        
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
                    
                    child_model = geneticalg.generate_child(nn_parent1, 
                                                            nn_parent2, 
                                                            env.tensor_shape, 
                                                            layers_info, 
                                                            cfg)

                    safe_idx = (cfg.NUM_PARENT_NETWORKS + child_idx 
                                + net_idx * cfg.CHILDREN_PER_PARENT)

                    child_model.save(NN_NAME_TMPL.format(safe_idx))
                    
                    print('Generating: {}%\r'.format((safe_idx - cfg.NUM_PARENT_NETWORKS) 
                                                     / num_tasks * 100),  end='')

                    utils.clear_session()
                utils.clear_session()
            print('')
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

                    predict = current_nn.predict(obs)
                    action = 0 if predict[0][0] < 0.5 else 1

                    env.step(action)

                nn_rewards = np.append(nn_rewards, env.get_reward())
            
            # Save the average estimate for the current neural network
            gen_rewards[network_idx] = int(np.mean(nn_rewards))
            # Update and save the best estimate and network for all epochs
            if max_reward < gen_rewards[network_idx]:
                max_reward = gen_rewards[network_idx]
                with open("max_reward.txt", "w") as f:
                    f.writelines(['MAX REWARD COMMON: {}'.format(max_reward)])
                current_nn.save('best_network.h5')
            
            print('Network {}: {}'.format(network_idx, gen_rewards[network_idx]))
            utils.clear_session()
        
        # Information on the results of the current epoch
        print('-'*40)
        print('MAX REWARD CURRENT: {}'.format(max(gen_rewards)))
        print('MAX REWARD COMMON: {}'.format(max_reward))
        print('-'*40)

        # Selection of the best neural networks
        nnetworks = geneticalg.selection(num_networks,
                              gen_rewards,
                              cfg.NUM_PARENT_NETWORKS,
                              cfg.RANDOM_SELECTED_NETWORKS,
                              cfg.NEW_GENERATED_RANDOM_NETWORK,
                              env.tensor_shape)
        
        utils.clear_session()

if __name__ == '__main__':
    main()