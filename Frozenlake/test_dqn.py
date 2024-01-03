import gc
import os
import traceback
import gym
import logging
from build_model import build_model
from utils import index_to_array, array_to_index
from MCTS.alphazero.MCTS import MCTS
from Network_Weights.DQN_3x3.ddqn_agent_FL import DDQN_Learning_Agent_FL

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO,
                    handlers=[logging.FileHandler(f"logs/frozenlake_test", mode='w')])
logger = logging.getLogger()

custom_map = [
    "SHF",
    "FFF",
    "HFG"
]

if __name__ == '__main__':
    try:
        while True:
            env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=True)
            state_size = env.observation_space.n
            env_obs_space_shape = env.observation_space.shape
            curr_state = env.reset()
            env._max_episode_steps = 500
            dqn_network = DDQN_Learning_Agent_FL(number_of_actions=4, env_obs_space_shape=env_obs_space_shape)
            dqn_network.load_saved_weights("Network_Weights/DQN_3x3/duel_dqn_FrozenLake-v1_slip_weights_3.h5f")
            terminated = False
            while not terminated:
                action = dqn_network.get_greedy_action(curr_state)
                curr_state, reward, terminated, _ = env.step(action)
                print(os.getpid(), curr_state, reward, terminated, action)
    except:
        logging.critical("Exception occurred")
        traceback.print_exc()
