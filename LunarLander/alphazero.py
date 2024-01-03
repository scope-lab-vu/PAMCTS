import logging
import time
import gymnasium as gym
import torch
import os
import pandas as pd
from multiprocessing import Pool
from warnings import filterwarnings
from network_weights.Alphazero_Networks.build_network import Alphazero_Network
from mcts.alphazero.mcts import MCTS


logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO,
                    handlers=[logging.FileHandler(f"logs/lunar_lander_alphazero.log", mode='w')])
logger = logging.getLogger()


def alphazero_simulations(args):
    start_time = time.time()
    # unwrap args
    gravity = args["gravity"]
    wind_power = args["wind_power"]
    turbulence_power = args["turbulence_power"]
    iterations = args["iterations"]
    sample_id = args["sample_id"]
    file_name = args["file_name"]

    # create environment
    env = gym.make("LunarLander-v2", gravity=gravity,
                    enable_wind=True, wind_power=wind_power, turbulence_power=turbulence_power)
    env._max_episode_steps = 3000
    nb_actions = env.action_space.n

    # build alphazero network
    network = Alphazero_Network()
    network.load_state_dict(torch.load("lunar_lander_alphazero_state_dict.pth"))
    curr_state, _ = env.reset()
    terminated = False
    cumulative_reward = 0
    step_counter = 0
    search_agent = MCTS(gamma=0.99, c_puct=50, num_iter=iterations, max_depth=500, weight_file="lunar_lander_alphazero_state_dict.pth", num_hidden_layers=5)

    while not terminated:
        prior_state = curr_state
        action = search_agent.run_mcts(network=network, env=env, observation=curr_state)
        curr_state, reward, terminated, _, _ = env.step(action)
        print(f"current state: {curr_state}, action: {action}, reward: {reward}, terminated: {terminated}, "
            f"step_counter: {step_counter}")
        step_counter += 1
        cumulative_reward += reward
        search_agent.clear_tree()
        if step_counter > env._max_episode_steps:
            terminated = True

    result = [[cumulative_reward, gravity, wind_power, turbulence_power, iterations, sample_id, step_counter, time.time() - start_time]]
    print(f"episode ends: {result}")

    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            pd.DataFrame(result).to_csv(f, header=["cumulative_reward", "gravity", "wind_power", "turbulence_power", "iterations",
                                                "sample_id", "step_counter", "computation_time"], index=False)
    else:
        with open(file_name, "a") as f:
            pd.DataFrame(result).to_csv(f, header=False, index=False)


if __name__ == "__main__":
    filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool8` is a deprecated alias')

    num_cpus = 90
    file_name = "lunar_lander_alphazero.csv"
    args = []

    # ! gravity range [0, -12], default -10
    # ! wind_power range [0, 20.0], default 0
    # ! turbulence_power range [0, 2.0], default 0
    # & recommend wind_power, range [0, 10.0, 15.0, 20.0]
    # & recommend iterations, range [10, 25, 50]
    for gravity in [-10.0]:
        for wind_power in [0.0, 10.0, 15.0, 20.0]:
            for turbulence_power in [0.0]:
                for iterations in [10, 25, 50, 75, 100, 200]:
                    for sample_id in range(30):
                        args.append({"gravity": gravity, "iterations": iterations, "sample_id": sample_id,
                                    "wind_power": wind_power, "turbulence_power": turbulence_power, 
                                    "file_name" : file_name})
                        
    with Pool(processes=num_cpus) as pool:
        pool.map(alphazero_simulations, args)
    # alphazero_simulations(args[0])

    print(f"experiments completed")
