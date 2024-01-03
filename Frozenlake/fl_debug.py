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
import time

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO,
                    handlers=[logging.FileHandler(f"logs/frozenlake_test", mode='w')])
logger = logging.getLogger()

custom_map = [
    "SHF",
    "FFF",
    "HFG"
]

num_cpus = 90


def test_pamcts(args):
    try:
        start_time = time.time()
        prob_distribution = args["prob_distribution"]
        sample_id = args["sample_id"]
        alpha = args["alpha"]
        simulations = args["simulations"]
        c_puct = args["c_puct"]
        gamma = args["gamma"]
        env = Customized_FrozenLakeEnv(desc=custom_map, is_slippery=True, customized_prob1=prob_distribution[0],
                                       customized_prob2=prob_distribution[1], customized_prob3=prob_distribution[2])
        # env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=True)
        # env = FrozenLakeEnv_9_10_1_20(desc=custom_map, is_slippery=True)
        nb_actions = env.action_space.n
        dqn_learning_agent = DDQN_Learning_Agent_FL(number_of_actions=nb_actions,
                                                    env_obs_space_shape=env.observation_space.shape)
        dqn_learning_agent.load_saved_weights("Network_Weights/DQN_3x3/duel_dqn_FrozenLake-v1_slip_weights_reorder.h5f")
        search_agent = FL_PAMCTS_Outside(gamma=gamma, learning_agent=dqn_learning_agent, alpha=alpha,
                                         num_iter=simulations, max_depth=500, verbose=True, c_puct=c_puct)
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

        discounted_cumulative_reward = cumulative_reward * (gamma ** (step_counter-1))
        # convert prob distribution to string, for each element, keep 3 digits after decimal point
        prob_distribution = [str(round(x, 3)) for x in prob_distribution]
        result = [[discounted_cumulative_reward, cumulative_reward, prob_distribution, alpha, simulations, sample_id,
                   step_counter, time.time() - start_time, gamma, c_puct]]

        if not os.path.exists("frozenlake_sanity_check23.csv"):
            with open("frozenlake_sanity_check23.csv", "w") as f:
                pd.DataFrame(result).to_csv(f, header=["discounted_cumulative_reward", "cumulative_reward",
                                                       "prob_distribution", "alpha", "simulations",
                                                       "sample_id", "step_counter", "computation_time", "gamma",
                                                       "c_puct"], index=False)
        else:
            with open("frozenlake_sanity_check23.csv", "a") as f:
                pd.DataFrame(result).to_csv(f, header=False, index=False)
    except:
        logging.critical("Exception occurred")
        traceback.print_exc()


if __name__ == "__main__":
    begin_time = time.time()
    args = []
    # for prob_distribution in [[1.0, 0.0, 0.0], [14.0 / 15.0, 1.0 / 30.0, 1.0 / 30.0],
    #                           [5.0 / 6.0, 1.0 / 12.0, 1.0 / 12.0], [11.0 / 15.0, 2.0 / 15.0, 2.0 / 15.0],
    #                           [19.0 / 30.0, 11.0 / 60.0, 11.0 / 60.0], [8.0 / 15.0, 7.0 / 30.0, 7.0 / 30.0],
    #                           [13.0 / 30.0, 17.0 / 60.0, 17.0 / 60.0], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]:
    for prob_distribution in [[19.0 / 30.0, 11.0 / 60.0, 11.0 / 60.0], [8.0 / 15.0, 7.0 / 30.0, 7.0 / 30.0]]:
        for alpha in [0.0]:
            for simulations in [3000, 5000]:
                for c_puct in [1.414]:
                    for gamma in [1.0, 0.99]:
                        for sample_id in range(50):
                            args.append({"prob_distribution": prob_distribution, "alpha": alpha,
                                         "simulations": simulations, "c_puct": c_puct, "gamma": gamma,
                                         "sample_id": sample_id})

    with Pool(processes=num_cpus) as pool:
        pool.map(test_pamcts, args)

    print(f"experiments completed")
    with open("execution_time.txt", "w") as f:
        f.write(f"experiments execution time: {time.time() - begin_time}")

    df = pd.read_csv("frozenlake_sanity_check23.csv")
    # print the average cumulative reward for each prob distribution, simulations, gamma
    for prob_distribution in ["[0.633, 0.183, 0.183]", "[0.533, 0.233, 0.233]"]:
        for simulations in [3000, 5000]:
            for gamma in [1.0, 0.99]:
                print(f"prob_distribution: {prob_distribution}, simulations: {simulations}, gamma: {gamma}")
                print(f"average: {df[(df['prob_distribution'] == prob_distribution) & (df['simulations'] == simulations) & (df['gamma'] == gamma)]['cumulative_reward'].mean()}")
