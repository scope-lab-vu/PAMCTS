import gc
import os
import traceback
from multiprocessing import Pool
import gym
import logging
import time
import pandas as pd
from pauct.cartpole.MCTS.ddqn_agent import DDQN_Learning_Agent
from pauct.cartpole.MCTS.PAMCTS_Outside import PAMCTS_Outside

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO,
                    handlers=[logging.FileHandler(f"logs/mountain_car_test", mode='w')])
logger = logging.getLogger()

num_cpus = 100


def test_pamcts(args):
    try:
        start_time = time.time()
        gravity = args["gravity"]
        sample_id = args["sample_id"]
        alpha = args["alpha"]
        simulations = args["simulations"]
        env = gym.make("CartPole-v1")
        curr_state = env.reset()
        env.env.gravity = gravity
        env._max_episode_steps = 2500
        nb_actions = env.action_space.n
        dqn_learning_agent = DDQN_Learning_Agent(number_of_actions=nb_actions,
                                                env_obs_space_shape=env.observation_space.shape)
        dqn_learning_agent.load_saved_weights("Network_Files/DQN/duel_dqn_MountainCar-v0_slip_weights.h5f")
        search_agent = PAMCTS_Outside(gamma=0.999, learning_agent=dqn_learning_agent, alpha=alpha,
                                    num_iter=simulations, max_depth=500, verbose=False, c_puct=50.0)
        # curr_state = env.reset()
        # env._max_episode_steps = 3
        terminated = False
        cumulative_reward = 0
        step_counter = 0
        while not terminated:
            action = search_agent.get_action(curr_state, env)
            curr_state, reward, terminated, _ = env.step(action)
            step_counter += 1
            print(os.getpid(), curr_state, reward, terminated, action)
            print(f"step_counter: {step_counter}")
            cumulative_reward += reward

        result = [[cumulative_reward, gravity, alpha, simulations, sample_id, step_counter, time.time() - start_time]]

        if not os.path.exists("mountain_car_pamcts_60.csv"):
            with open("mountain_car_pamcts_60.csv", "w") as f:
                pd.DataFrame(result).to_csv(f, header=["cumulative_reward", "gravity", "alpha", "simulations",
                                                    "sample_id", "step_counter", "computation_time"], index=False)
        else:
            with open("mountain_car_pamcts_60.csv", "a") as f:
                pd.DataFrame(result).to_csv(f, header=False, index=False)
    except:
        logging.critical("Exception occurred")
        traceback.print_exc()


if __name__ == "__main__":
    args = []
    alpha_list = [[], [0.98], [0.94], [0.35]]
    for graivty in [9.8, 20.0, 50.0, 500.0]:
        for alpha in [0.0, 0.25, 0.50, 0.75, 1.0]:
        # for alpha in [0.0]:
            for simulations in [25, 50, 100, 200, 300, 400, 500, 1000]:
            # for simulations in [500]:
                for sample_id in range(60):
                    args.append({"gravity": graivty, "alpha": alpha, "simulations": simulations,
                                "sample_id": sample_id})

    with Pool(processes=num_cpus) as pool:
        pool.map(test_pamcts, args)

    print(f"experiments completed")
