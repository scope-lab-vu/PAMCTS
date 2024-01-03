import logging
import time
import gymnasium as gym
import os
import pandas as pd
from multiprocessing import Pool
from warnings import filterwarnings
from network_weights.DDQN.ddqn_agent import DDQN_Learning_Agent
from mcts.pamcts.mcts import MCTS
from mcts.pamcts.pamcts import PA_MCTS


logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO,
                    handlers=[logging.FileHandler(f"logs/lunar_lander_pamcts.log", mode='w')])
logger = logging.getLogger()


def pamcts_simulations(args):
    start_time = time.time()
    # unwrap args
    gravity = args["gravity"]
    wind_power = args["wind_power"]
    turbulence_power = args["turbulence_power"]
    alpha = args["alpha"]
    iterations = args["iterations"]
    sample_id = args["sample_id"]
    file_name = args["file_name"]

    # create environment
    env = gym.make("LunarLander-v2", gravity=gravity,
                    enable_wind=True, wind_power=wind_power, turbulence_power=turbulence_power)
    env._max_episode_steps = 3000

    nb_actions = env.action_space.n
    ddqn_learning_agent = DDQN_Learning_Agent(number_of_actions=nb_actions,
                                            env_obs_space_shape=env.observation_space.shape)
    ddqn_learning_agent.load_saved_weights(
        "network_weights/DDQN/lunar_lander_ddqn_weights.h5f")
    search_agent = PA_MCTS(gamma=0.99, learning_agent=ddqn_learning_agent, num_iter=iterations, max_depth=500,
                        c_puct=50, alpha=alpha)
    # search_agent = MCTS(gamma=0.99, learning_agent=None, num_iter=iterations, max_depth=500, c_puct=50)
    # Standard MCTS only
    curr_state, _ = env.reset()
    terminated = False
    cumulative_reward = 0
    step_counter = 0

    while not terminated:
        action = search_agent.get_action(observation=curr_state, env=env)
        curr_state, reward, terminated, _, _ = env.step(action)
        print(f"current state: {curr_state}, action: {action}, reward: {reward}, terminated: {terminated}, "
            f"step_counter: {step_counter}")
        step_counter += 1
        cumulative_reward += reward
        search_agent.clear_tree()

        if step_counter > env._max_episode_steps:
            terminated = True

    result = [[cumulative_reward, gravity, wind_power, turbulence_power, alpha, iterations, sample_id, step_counter,
            time.time() - start_time]]
    print(f"episode ends: {result}")

    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            pd.DataFrame(result).to_csv(f, header=["cumulative_reward", "gravity", "wind_power", "turbulence_power", "alpha", "iterations",
                                                "sample_id", "step_counter", "computation_time"], index=False)
    else:
        with open(file_name, "a") as f:
            pd.DataFrame(result).to_csv(f, header=False, index=False)


if __name__ == "__main__":
    filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool8` is a deprecated alias')

    num_cpus = 90
    file_name = "lunar_lander_alpha_selection.csv"

    args = []

    # ! gravity range [0, -12], default -10
    # ! wind_power range [0, 20.0], default 0
    # ! turbulence_power range [0, 2.0], default 0
    # & recommend wind_power, range [0, 10.0, 15.0, 20.0]
    # & recommend alpha, range [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    # & recommend iterations, range [5, 10, 15]
    # 30 samples
    for gravity in [-10.0]:
        for wind_power in [0, 10.0, 15.0, 20.0]:
            for turbulence_power in [0.0]:
                for alpha in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
                    for iterations in [5, 10, 15]:
                        for sample_id in range(30):
                            args.append({"gravity": gravity, "alpha": alpha, "iterations": iterations,
                                        "sample_id": sample_id, "wind_power": wind_power, 
                                        "turbulence_power": turbulence_power, "file_name": file_name})

    with Pool(processes=num_cpus) as pool:
        pool.map(pamcts_simulations, args)

    print(f"experiments completed")