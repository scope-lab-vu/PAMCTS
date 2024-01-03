import gc
import os
import traceback
import time
import gym
import logging
import pandas as pd
from multiprocessing import Pool
from build_model import build_model
from utils import index_to_array, array_to_index
from MCTS.alphazero.MCTS import MCTS
from env.customized_frozen_lake import Customized_FrozenLakeEnv
from env.frozen_lake_9_10_1_20 import FrozenLakeEnv_9_10_1_20
from rl.agents.dqn import DQNAgent


logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO,
                    handlers=[logging.FileHandler(f"logs/frozenlake_test_alphazero", mode='w')])
logger = logging.getLogger()

custom_map = [
    "SHF",
    "FFF",
    "HFG"
]

num_cpus = 100


def test_alphazero(args):
    try:
        # env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=True)
        # env = FrozenLakeEnv_2_3_1_6(desc=custom_map, is_slippery=True)
        start_time = time.time()
        prob_distribution = args["prob_distribution"]
        simulations = args["simulations"]
        sample_id = args["sample_id"]
        env = Customized_FrozenLakeEnv(desc=custom_map, is_slippery=True, customized_prob1=prob_distribution[0],
                                       customized_prob2=prob_distribution[1], customized_prob3=prob_distribution[2])
        state_size = env.observation_space.n
        curr_state = env.reset()
        # env._max_episode_steps = 3
        curr_state = index_to_array(curr_state, state_size)
        search_agent = MCTS(current_state=curr_state, gamma=0.95, num_iter=simulations)
        terminated = False
        network = build_model(5, state_size)
        network.load_weights("Network_Weights/Alphazero_3x3/weights_reorder.h5f").expect_partial()
        cumulative_reward = 0
        step_counter = 0
        while not terminated:
            action_node, action = search_agent.run_mcts(network=network, env=env, map_name="4x4", is_slippery=True,
                                                        prob_distribution=prob_distribution)
            curr_state, reward, terminated, _ = env.step(action)
            step_counter += 1
            if step_counter >= 200:
                terminated = True
            curr_state = index_to_array(curr_state, state_size)
            cumulative_reward += reward
            search_agent.mcts_update_root(curr_state)
            curr_state_idx = array_to_index(curr_state, state_size)
            print(os.getpid(), curr_state_idx, reward, terminated, action)
        _ = gc.collect()
        discounted_cumulative_reward = cumulative_reward * (0.95 ** (step_counter - 1))
        prob_distribution = [str(round(x, 3)) for x in prob_distribution]
        result = [[discounted_cumulative_reward, cumulative_reward, prob_distribution, sample_id, step_counter,
                   time.time() - start_time]]

        if not os.path.exists("frozenlake_alphazero_results.csv"):
            with open("frozenlake_alphazero_results.csv", "w") as f:
                pd.DataFrame(result).to_csv(f, header=["discount_cumulative_reward", "cumulative_reward",
                                                       "prob_distribution", "sample_id", "step_counter",
                                                       "computation_time"], index=False)
        else:
            with open("frozenlake_alphazero_results.csv", "a") as f:
                pd.DataFrame(result).to_csv(f, header=False, index=False)

    except:
        logging.critical("Exception occurred")
        traceback.print_exc()


if __name__ == "__main__":
    args = []
    for prob_distribution in [[1.0, 0.0, 0.0], [14.0 / 15.0, 1.0 / 30.0, 1.0 / 30.0],
                              [5.0 / 6.0, 1.0 / 12.0, 1.0 / 12.0], [11.0 / 15.0, 2.0 / 15.0, 2.0 / 15.0],
                              [19.0 / 30.0, 11.0 / 60.0, 11.0 / 60.0], [8.0 / 15.0, 7.0 / 30.0, 7.0 / 30.0],
                              [13.0 / 30.0, 17.0 / 60.0, 17.0 / 60.0], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]:
        for simulations in [25, 50, 100, 200, 500, 1000, 1500, 2000, 5000, 10000, 12000, 15000]:
        # for simulations in [150000]:
            for sample_id in range(200):
                args.append({"prob_distribution": prob_distribution, "simulations": simulations, "sample_id": sample_id})

    with Pool(processes=num_cpus) as pool:
        pool.map(test_alphazero, args)

    print(f"experiments completed")
