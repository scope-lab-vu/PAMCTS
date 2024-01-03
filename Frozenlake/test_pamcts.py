import gc
import os
import traceback
from multiprocessing import Pool
import gym
import logging
import time
import pandas as pd
from build_model import build_model
from utils import index_to_array, array_to_index
from Network_Weights.DQN_3x3.ddqn_agent_FL import DDQN_Learning_Agent_FL
from MCTS.pauct.fl_PAMCTS_Outside import FL_PAMCTS_Outside
from env.customized_frozen_lake import Customized_FrozenLakeEnv
from env.frozen_lake_9_10_1_20 import FrozenLakeEnv_9_10_1_20

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO,
                    handlers=[logging.FileHandler(f"logs/frozenlake_test", mode='w')])
logger = logging.getLogger()

custom_map = [
    "SHF",
    "FFF",
    "HFG"
]

num_cpus = 100


def test_pamcts(args):
    try:
        start_time = time.time()
        prob_distribution = args["prob_distribution"]
        sample_id = args["sample_id"]
        alpha = args["alpha"]
        simulations = args["simulations"]
        env = Customized_FrozenLakeEnv(desc=custom_map, is_slippery=True, customized_prob1=prob_distribution[0],
                                       customized_prob2=prob_distribution[1], customized_prob3=prob_distribution[2])
        # env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=True)
        # env = FrozenLakeEnv_9_10_1_20(desc=custom_map, is_slippery=True)
        nb_actions = env.action_space.n
        dqn_learning_agent = DDQN_Learning_Agent_FL(number_of_actions=nb_actions,
                                                    env_obs_space_shape=env.observation_space.shape)
        dqn_learning_agent.load_saved_weights("Network_Weights/DQN_3x3/duel_dqn_FrozenLake-v1_slip_weights_3.h5f")
        search_agent = FL_PAMCTS_Outside(gamma=0.95, learning_agent=dqn_learning_agent, alpha=alpha,
                                         num_iter=simulations, max_depth=500, verbose=False)
        curr_state = env.reset()
        # env._max_episode_steps = 3
        terminated = False
        cumulative_reward = 0
        step_counter = 0
        while not terminated:
            action = search_agent.get_action(curr_state, env, prob_distribution)
            curr_state, reward, terminated, _ = env.step(action)
            step_counter += 1
            if step_counter >= 200:
                terminated = True
            print(os.getpid(), curr_state, reward, terminated, action)
            cumulative_reward += reward

        result = [[cumulative_reward, prob_distribution, alpha, simulations, sample_id, step_counter, time.time() - start_time]]

        if not os.path.exists("frozenlake_pamcts_optimal_alpha_part2.csv"):
            with open("frozenlake_pamcts_optimal_alpha_part2.csv", "w") as f:
                pd.DataFrame(result).to_csv(f, header=["cumulative_reward", "prob_distribution", "alpha", "simulations",
                                                       "sample_id", "step_counter", "computation_time"], index=False)
        else:
            with open("frozenlake_pamcts_optimal_alpha_part2.csv", "a") as f:
                pd.DataFrame(result).to_csv(f, header=False, index=False)
    except:
        logging.critical("Exception occurred")
        traceback.print_exc()


if __name__ == "__main__":
    args = []
    alpha_list = [[], [0.9], [0.85, 0.80], [0.80, 0.70], [0.60], [], [0.40], [0.35, 0.30]]
    alpha_index = 0
    for prob_distribution in [[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], [13.0 / 30.0, 17.0 / 60.0, 17.0 / 60.0],
                              [8.0 / 15.0, 7.0 / 30.0, 7.0 / 30.0], [19.0 / 30.0, 11.0 / 60.0, 11.0 / 60.0],
                              [11.0 / 15.0, 2.0 / 15.0, 2.0 / 15.0], [5.0 / 6.0, 1.0 / 12.0, 1.0 / 12.0],
                              [14.0 / 15.0, 1.0 / 30.0, 1.0 / 30.0], [1.0, 0.0, 0.0]]:
    # for prob_distribution in [[19.0/30.0, 11.0/60.0, 11.0/60.0]]:
        for alpha in alpha_list[alpha_index]:
        # for alpha in [0.50]:
            for simulations in [25, 50, 100, 200, 500, 1000, 1500, 2000, 5000, 10000, 12000, 15000]:
            # for simulations in [10]:
                for sample_id in range(200):
                    args.append({"prob_distribution": prob_distribution, "alpha": alpha, "simulations": simulations,
                                 "sample_id": sample_id})
        alpha_index += 1

    with Pool(processes=num_cpus) as pool:
        pool.map(test_pamcts, args)

    print(f"experiments completed")
