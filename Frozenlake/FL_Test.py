import sys

import gym

# from learning.ddqn_agent import DDQN_Learning_Agent
from Network_Weights.DQN_3x3.ddqn_agent_FL import DDQN_Learning_Agent_FL
import numpy

from MCTS.frozenlake.fl_PAMCTS_Outside import FL_PAMCTS_Outside

from MCTS.frozenlake.stochasticMCTS import StochasticMCTS
# from MCTS.HybridMCTS import HybridMCTS
# from learning.random_agent import RandomAgent
from Simulation import run_simulation
# from learning.random_agent_sample import RandomAgentSample
import numpy as np

class RandomAgentSample():

    def get_action(self, observation, env):

        return env.action_space.sample()





if __name__ == '__main__':

    ENV_NAME = 'FrozenLake-v1'
    gamma = 0.9  # discount factor
    env = gym.make(ENV_NAME, is_slippery=True, map_name="4x4", desc=None)

    ## Random Agent
    # search_agent = RandomAgentSample()

    ## DDQN Agent
    # nb_actions = env.action_space.n
    # weights_filepath = '../learning/frozenlake/duel_dqn_FrozenLake-v1_slip_weights.h5f'
    # # weights_filepath = '../learning/learning/duel_dqn_FrozenLake-v1_nslip_weights.h5f'
    # # weights_filepath = 'learning/duel_dqn_FrozenLake-v1_slip_weights_2.h5f'
    # search_agent = DDQN_Learning_Agent_FL(number_of_actions = nb_actions,
    #                                 env_obs_space_shape=env.observation_space.shape)
    # search_agent.load_saved_weights(weights_filepath)


    ### pure_mcts_agent
    # search_agent = StochasticMCTS(gamma=gamma,
    #                               learning_agent=None,
    #                               alpha=0.0,
    #                               num_iter=300,
    #                               max_depth=100,
    #                               c=1.4142)

    ### pamcts outside agent
    # nb_actions = env.action_space.n
    # weights_filepath = '../learning/frozenlake/duel_dqn_FrozenLake-v1_slip_weights.h5f'
    # # weights_filepath = 'learning/duel_dqn_FrozenLake-v1_slip_weights_2.h5f'
    # learning_agent = DDQN_Learning_Agent_FL(number_of_actions = nb_actions,
    #                                 env_obs_space_shape=env.observation_space.shape)
    # learning_agent.load_saved_weights(weights_filepath)
    # search_agent = FL_PAMCTS_Outside(
    #     gamma=gamma,
    #     learning_agent=learning_agent,
    #     alpha=0.75,
    #     num_iter=300,
    #     max_depth=100,
    #     verbose=False
    # )

    ### pamcts inside agent
    nb_actions = env.action_space.n
    weights_filepath = '../learning/frozenlake/duel_dqn_FrozenLake-v1_slip_weights.h5f'
    # weights_filepath = 'learning/duel_dqn_FrozenLake-v1_slip_weights_2.h5f'
    learning_agent = DDQN_Learning_Agent_FL(number_of_actions = nb_actions,
                                    env_obs_space_shape=env.observation_space.shape)
    learning_agent.load_saved_weights(weights_filepath)
    search_agent = StochasticMCTS(gamma=gamma,
                                  learning_agent=learning_agent,
                                  alpha=0.9,
                                  num_iter=200,
                                  max_depth=100,
                                  c=1.4142)


    res = []
    for i in range(1000):

        env.reset()
        res.append(run_simulation(search_agent=search_agent,
                             env=env,
                             visualize=False,
                             verbose=0)[0])
        sys.stdout.write('\r'+str(i) + ' | ' + str(numpy.sum(res)))
        sys.stdout.flush()

    mean_reward = np.mean(res)
    print()
    print('done!')
    print(mean_reward)

    # print(res[0], res[1])