import logging
import os
import time
import sys
import torch
import random
import multiprocessing
import gymnasium as gym
import numpy as np
from multiprocessing import Process, Queue, Manager, Lock
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from warnings import filterwarnings
from build_network import Alphazero_Network
BASE_DIR = "../../"
sys.path.append(BASE_DIR)
from mcts.alphazero.mcts_alphazero import MCTS


logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO,
                    handlers=[logging.FileHandler(f"../../logs/lunar_lander_alphazero_training.log", mode='w')])
logger = logging.getLogger()


def data_generation(args):
    """
    generate data for training
    """
    print(f"data generation: {os.getpid()}")

    # unwrap args
    # TODO: delete unused args
    q = args["multiprocessing queue"]
    lock = args["lock"]
    num_hidden_layers = args["num_hidden_layers"]
    file_name = args["file_name"]
    weights_file = args["weights_file"]
    num_iterations = args["num_iterations"]
    num_episodes = args["num_episodes"]
    gravity = args["gravity"]
    wind_power = args["wind_power"]
    turbulence_power = args["turbulence_power"]
    shared_results = args["shared_results"]
    shared_signal_status = args["shared_signal_status"]
    c_puct = args["c_puct"]
    start_time = time.time()
    average_cumulative_reward_of_last_100_episodes_file = "lunar_lander_alphazero_average_cumulative_reward_of_last_100_episodes.csv"

    # create environment
    env = gym.make("LunarLander-v2", gravity=gravity, enable_wind=True, wind_power=wind_power, turbulence_power=turbulence_power)
    env._max_episode_steps = 3000
    nb_actions = env.action_space.n
    
    # build alphazero network
    network = Alphazero_Network()
    # network = network.cpu()

    for episode in range(num_episodes):
        logger.critical(f"episode: {i}")
        lock = multiprocessing.Lock()
        lock.acquire()
        if os.path.exists(weights_file):
            print(f"loading weights from {weights_file}")
            network.load_state_dict(torch.load(weights_file))
        lock.release()
        curr_state, _ = env.reset()
        terminated = False
        cumulative_reward = 0
        step_counter = 0
        search_agent = MCTS(gamma=0.99, c_puct=c_puct, num_iter=num_iterations, max_depth=500, weight_file=weights_file, num_hidden_layers=num_hidden_layers)

        while not terminated:
            prior_state = curr_state
            action = search_agent.run_mcts(network=network, env=env, observation=curr_state, lock=lock)
            curr_state, reward, terminated, _, _ = env.step(action)
            root_state, prob_priors = search_agent.get_training_data(prior_state)
            training_data = [root_state, prob_priors, reward]
            with lock:
                q.put(training_data)
            print(f"current state: {curr_state}, action: {action}, reward: {reward}, terminated: {terminated}, "
            f"step_counter: {step_counter}")
            step_counter += 1
            cumulative_reward += reward
            search_agent.clear_tree()
            if step_counter > env._max_episode_steps:
                terminated = True

        result = [[cumulative_reward, gravity, wind_power, turbulence_power, num_iterations, episode, step_counter, time.time() - start_time]]
        print(f"episode ends: {result}")
        # add result to shared results
        shared_results.append(result)
        if len(shared_results) % 100 == 0:
            # calculate average cumulative reward of last 100 episodes
            average_cumulative_reward_of_last_100_episodes = np.mean([result[0] for result in shared_results[-100:]])
            print(f"average cumulative reward of last 100 episodes: {average_cumulative_reward_of_last_100_episodes}")
            with open(average_cumulative_reward_of_last_100_episodes_file, "a") as f:
                pd.DataFrame([[average_cumulative_reward_of_last_100_episodes, gravity, wind_power, turbulence_power, num_iterations, episode, step_counter, time.time() - start_time]]).to_csv(f, header=False, index=False)
        
        if not os.path.exists(file_name):
            with open(file_name, "w") as f:
                pd.DataFrame(result).to_csv(f, header=["cumulative_reward", "gravity", "wind_power", "turbulence_power", "num_iterations",
                                                    "episode", "step_counter", "computation_time"], index=False)
        else:
            with open(file_name, "a") as f:
                pd.DataFrame(result).to_csv(f, header=False, index=False)
        

    # for i in range(5):
    #     time.sleep(0.5)  # Simulate some computation
    #     with lock:
    #         shared_results.append(i)
    #         print(f"Process 1: Wrote {i} to shared list")
    #         q.put(i)  # Notify the other process

    print(f"data generation completed")
    shared_signal_status.append("done")


def NN_training(args):
    """
    train neural network
    """
    print(f"NN training: {os.getpid()}")

    # unwrap args
    # TODO: delete unused args
    q = args["multiprocessing queue"]
    lock = args["lock"]
    num_hidden_layers = args["num_hidden_layers"]
    file_name = args["file_name"]
    weights_file = args["weights_file"]
    num_iterations = args["num_iterations"]
    num_episodes = args["num_episodes"]
    gravity = args["gravity"]
    wind_power = args["wind_power"]
    turbulence_power = args["turbulence_power"]
    shared_results = args["shared_results"]
    shared_signal_status = args["shared_signal_status"]
    num_generators = args["num_generators"]
    minibatch_size = args["minibatch_size"]

    # for _ in range(5):
    #     i = q.get()  # Wait until we get a notification
    #     with lock:
    #         print(f"Process 2: Shared list is currently {shared_results}")

    replay_buffer = []
    network = Alphazero_Network()
    if os.path.exists(weights_file):
        print(f"loading weights from {weights_file}")
        network.load_state_dict(torch.load(weights_file))
    criterion = nn.MSELoss()  # Mean Squared Error loss for regression tasks
    optimizer = optim.Adam(network.parameters(), lr=0.001)

    while len(shared_signal_status) < num_generators:
        while len(replay_buffer) < minibatch_size:
            with lock:
                training_data = q.get()
                replay_buffer.append(training_data)
        print(f"replay buffer: {replay_buffer}")
        
        for m in range(10):
            for i in range(100): # ^ 100
                if len(replay_buffer) > 10000:
                    with lock:
                        training_data = q.get()
                    replay_buffer.pop(0)
                    replay_buffer.append(training_data)
                else:
                    with lock:
                        training_data = q.get()
                    replay_buffer.append(training_data)

            for n in range(5):
                states = []
                pi_hats = []
                v_hats = []

                minibatch = random.sample(replay_buffer, minibatch_size)

                for training_data in minibatch:
                    states.append(training_data[0])
                    pi_hats.append(training_data[1])
                    v_hats.append(training_data[2])
                
                states = np.array(states).reshape((minibatch_size, 8))
                pi_hats = np.array(pi_hats).reshape((minibatch_size, 4))
                v_hats = np.array(v_hats).reshape((minibatch_size, 1))

                states = torch.from_numpy(states).float()
                pi_hats = torch.from_numpy(pi_hats).float()
                v_hats = torch.from_numpy(v_hats).float()

                predicted_pi_hats, predicted_v_hats = network.forward(x=states)
                loss = criterion(predicted_pi_hats, pi_hats) + criterion(predicted_v_hats, v_hats)
                print(f"loss: {loss}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                torch.save(network.state_dict(), weights_file)
                print(f"saved weights to {weights_file}")

    print(f"NN Training completed")


if __name__ == "__main__":
    filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool8` is a deprecated alias')

    num_generators = 80
    file_name = "lunar_lander_alphazero_training_iterations_1000.csv"
    weights_file = "lunar_lander_alphazero_state_dict.pth"
    q = Queue()
    lock = Lock()
    data_generation_processes = []
    num_hidden_layers = 5
    num_iterations = 1000
    num_episodes = 400
    gravity = -10.0
    wind_power = 0.0
    turbulence_power = 0.0
    c_puct = 50
    minibatch_size = 200

    args = {"multiprocessing queue": q, "lock": lock, "num_hidden_layers": num_hidden_layers, "file_name": file_name, 
            "weights_file": weights_file, "num_iterations": num_iterations, "num_episodes": num_episodes, "gravity": gravity,
            "wind_power": wind_power, "turbulence_power": turbulence_power, "c_puct": c_puct, "minibatch_size": minibatch_size}

    with Manager() as manager:
        shared_results = manager.list([])
        shared_signal_status = manager.list([])
        args["shared_results"] = shared_results
        args["shared_signal_status"] = shared_signal_status
        args["num_generators"] = num_generators

        for i in range(num_generators):
            process = Process(target=data_generation, args=(args,))
            process.start()
            data_generation_processes.append(process)

        process_nn = Process(target=NN_training, args=(args,))
        process_nn.start()

        # should train infinitely until converges to maximum cumulative reward
        for process in data_generation_processes:
            process.join()
        process_nn.terminate()
