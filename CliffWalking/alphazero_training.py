import os
import time
import torch
import math
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from multiprocessing import Pool, Process, Queue, Manager, Lock
from ddqn_agent import DDQNAgent
from stochastic_cliff_walking import StochasticCliffWalkingEnv
from build_alphazero_network import Alphazero_Network
from utils import convert_to_one_hot_encoding
from build_model_tf import build_model_tf


ENV_WIDTH = 12
ENV_HEIGHT = 4
START_STATE = 36
GOAL_STATE = 47
HOLES = [37, 38, 39, 40, 41, 42, 43, 44, 45, 46]

MOVE_RIGHT = 1
MOVE_DOWN = 0
# MOVE_LEFT = 0
MOVE_UP = 2
MAX_DEPTH = 500
STEP_LIMIT = 100


def transition(state, action, slippery, step_count):
    x = int(state / ENV_WIDTH)
    y = state % ENV_WIDTH
    shape = [4, 12]
    if action == 2:  # UP
        if np.random.uniform() < slippery:
            x = min(x + 1, shape[0] - 1)
        else:
            x = max(x - 1, 0)
    elif action == 1:  # RIGHT
        if np.random.uniform() < slippery:
            x = min(x + 1, shape[0] - 1)
        else:
            y = min(y + 1, shape[1] - 1)
    elif action == 0:  # DOWN
        x = min(x + 1, shape[0] - 1)
    # elif action == 0:  # LEFT
    #     if np.random.uniform() < slippery:
    #         x = min(x + 1, shape[0] - 1)
    #     else:
    #         y = max(y - 1, 0)

    state = x * ENV_WIDTH + y
    state = int(state)

    # Check if agent is in the cliff region
    if state in HOLES or step_count > STEP_LIMIT:
        return state, 0, True

    # Check if agent reached the goal
    if state == GOAL_STATE:
        return state, (100-step_count)/100, True

    return state, 0, False


def select_action(C, state, n, v, p):
    state_n = n[state]
    state_v = v[state]
    state_p = p[state]
    N = sum(state_n)
    # if state == 36:
    #     print(f"four values: {state_v / state_n + C * np.sqrt(math.log(N) / state_n)}")
    for i in range(3):
        if state_n[i] <= 1:
            return i
    return np.argmax(state_v / state_n + C * state_p * np.sqrt(math.log(N) / state_n))


def random_policy(state):
    if state == 36:
        return MOVE_UP
    elif 24 <= state < 35:
        return MOVE_RIGHT
    elif state == 35:
        return MOVE_DOWN
    elif state == 11 or state == 23:
        return MOVE_DOWN
    else:
        return MOVE_RIGHT
    

def select_action_rollout(state):
    return random_policy(state)


def MCTS(root_state, network, slippery, iterations, C, gamma, episode):
    n = np.ones((ENV_WIDTH * ENV_HEIGHT, 3))
    v = np.zeros((ENV_WIDTH * ENV_HEIGHT, 3))
    p = np.zeros((ENV_WIDTH * ENV_HEIGHT, 3))
    r = np.zeros((ENV_WIDTH * ENV_HEIGHT, 3))
    if episode < 100000:
        tutorial = False
    else:
        tutorial = False
    for state in range(ENV_WIDTH * ENV_HEIGHT):
        # ^ pytorch
        # with torch.no_grad():
            # state_input = np.zeros(ENV_HEIGHT * ENV_WIDTH)
            # state_input[state] = 1
            # state_vec = convert_to_one_hot_encoding(state)
            # state_tensor = torch.tensor([state_vec])
            # (p[state], r[state]) = network(state_tensor)
            # print(f"state_tensor: {state_tensor}")
            # (priors, value) = network.forward(state_tensor)
            # p[state] = priors[0].detach().numpy()
            # r[state] = value[0].detach().numpy()
        # ^ tensorflow
        state_vec = convert_to_one_hot_encoding(state)
        outputs_combo = network.predict(x=np.array(state_vec).reshape((1, ENV_WIDTH * ENV_HEIGHT)), batch_size=1, verbose=0)
        prob_priors = outputs_combo[0][0]
        value = outputs_combo[1][0]
        for action in range(3):
            p[state, action] = prob_priors[action] 
            r[state, action] = value[action]
    np.set_printoptions(suppress=True, precision=10)
    # print(f"whole p 35: {p[35]}")
    # print(f"whole p 34: {p[34]}")
    # print(f"whole r 35: {r[35]}")
    # print(f"whole r 34: {r[34]}")
    np.set_printoptions(suppress=True, precision=10)
    # print(f"priors: {p}")
    # print(f"values: {r}")
    # if root_state == 23 or root_state == 35 or root_state == 11 or root_state == 10:
    #     print(f"root_state: {root_state}")
    #     print(f"priors: {p[root_state]}")
    #     print(f"values: {r[root_state]}")
        # r[root_state][0] = 3000
        # p[root_state][0] += 1
        # # normalize priors
        # p[root_state] = p[root_state] / sum(p[root_state])
    for i in range(iterations):
        state = root_state
        done = False
        reward = None
        sa_trajectory = []
        depth = 0
        step_count = 0
        while not done:
            action = select_action(C, state, n, v, p)
            if tutorial:
                action = select_action_rollout(state)
            sa_trajectory.append((state, action))
            (next_state, reward, done) = transition(state, action, slippery=slippery, step_count=step_count)
            step_count += 1
            depth += 1
            if done or depth > MAX_DEPTH:
                break
            if n[state, action] == 1:
                reward = r[state, action]
                break
            state = next_state
        for (state, action) in sa_trajectory:
            n[state, action] += 1
            v[state, action] += reward
        # if root_state == 35:
        #     print(f"after MCTS, v: {v[root_state]}")
    if not tutorial:
        return np.argmax(v[root_state] / n[root_state]), n[root_state] / sum(n[root_state]), v[root_state]
    else:
        return select_action_rollout(root_state), n[root_state] / sum(n[root_state]), v[root_state]


def simulate_episode(policy, network, iterations, C, slippery, gamma, q, lock, episode):
    steps = 0
    state = START_STATE
    # state = 34
    # random sample state, but not Goal STATE, NOT HOLES
    # state = random.choice(range(ENV_WIDTH * ENV_HEIGHT))
    while state == GOAL_STATE or state in HOLES:
        state = random.choice(range(ENV_WIDTH * ENV_HEIGHT))
    step_count = 0
    training_data_seq = []
    while True:
        action, priors, values = policy(state, network, slippery, iterations, C, gamma, episode)
        (next_state, reward, done) = transition(state, action, slippery=slippery, step_count=step_count)
        # state_input = np.zeros(ENV_HEIGHT * ENV_WIDTH)
        # state_input[state] = 1
        training_data = [state, priors, values]
        training_data_seq.append(training_data)
        # q.put(training_data)
        step_count += 1
        print(f"State: {state}; Move: {action}; Next state: {next_state}; Reward: {reward}; Done: {done}")
        steps += 1
        if done or steps > MAX_DEPTH:
            if reward > 0:
                print(f"training_data_seq: {training_data_seq}")
                for training_data in training_data_seq:
                    q.put(training_data)
            return steps, reward
        state = next_state


def evaluate_policy(policy, verbose, network, iterations, C, slippery, gamma, file_name, q, lock):
    start_time = time.time()
    lengths = np.zeros(1)
    outcomes = np.zeros(1)
    for episode in range(1):
        (length, outcome) = simulate_episode(policy, network, iterations, C, slippery, gamma, q, lock, episode)
        lengths[episode] = length
        outcomes[episode] = outcome
        if verbose:
            print(f"Episode: {episode}; Length: {length}; Outcome: {outcome}.")
        results = [[outcomes[0], slippery, iterations, time.time() - start_time, lengths[0], C, gamma]]
        if not os.path.exists(file_name):
            with open(file_name, "w") as f:
                pd.DataFrame(results).to_csv(f, header=["cumulative_reward", "slippery", "iterations", "computation time", "step_counter", "C", "gamma"], index=False)
        else:
            with open(file_name, "a") as f:
                pd.DataFrame(results).to_csv(f, header=False, index=False)


def run_simulations(args):
    print(f"Run Simulations: {os.getpid()}")
    slippery = args["slippery"]
    iterations = args["iterations"]
    C = args["C"]
    num_episodes = args["num_episodes"]
    gamma = args["gamma"]
    file_name = args["file_name"]
    weights_file = args["weights_file"]
    q = args["multiprocessing queue"]
    lock = args["lock"]
    start_time = time.time()
    # ^ pytorch
    # network = Alphazero_Network()
    # ^ tensorflow
    network = build_model_tf(num_hidden_layers=5, state_size=ENV_WIDTH * ENV_HEIGHT)
    for episode in range(num_episodes):
        if episode % 20 == 0:
            print(f"Episode: {episode}/{num_episodes}")
        lock.acquire()
        if os.path.exists("cliff_walking_alphazero_weights.h5f.index"):
            print(f"loading weights from {weights_file}")
            # ^ pytorch
            # network.load_state_dict(torch.load(weights_file))
            # ^ tensorflow
            network.load_weights(weights_file)
        lock.release()
        evaluate_policy(MCTS, verbose=False, network=network, iterations=iterations, C=C, slippery=slippery, 
                        gamma=gamma, file_name=file_name, q=q, lock=lock)
        if episode % 20 == 0:
            print(f"Episode {episode} completed")
    print(f"Run Simulations completed")


def nn_training(args):
    print(f"NN training: {os.getpid()}")
    q = args["multiprocessing queue"]
    lock = args["lock"]
    weights_file = args["weights_file"]
    minibatch_size = args["minibatch_size"]
    shared_signal_status = args["shared_signal_status"]
    num_generators = args["num_generators"]
    possible_states = [i for i in range(37)]

    replay_buffer = []
    minibatch_size_per_state = int(minibatch_size / 2) + 1
    for i in range(37):
        # create 37 empty lists
        replay_buffer_state = []
        replay_buffer.append(replay_buffer_state)
    # ^ pytorch
    # network = Alphazero_Network()
    # ^ tensorflow
    network = build_model_tf(num_hidden_layers=5, state_size=ENV_WIDTH * ENV_HEIGHT)
    if os.path.exists(weights_file):
        print(f"loading weights from {weights_file}")
        # ^ pytorch
        # network.load_state_dict(torch.load(weights_file))
        # ^ tensorflow
        network.load_weights(weights_file)
    # ^ pytorch
    # criterion = nn.MSELoss()  # Mean Squared Error loss for regression tasks
    # optimizer = optim.Adam(network.parameters(), lr=0.001)
    while True:
        for state in possible_states:
            while len(replay_buffer[state]) < minibatch_size_per_state:
                training_data = q.get()
                state = training_data[0]
                replay_buffer[state].append(training_data)
        for m in range(10):
            for i in range(100): # ^ 100
                training_data = q.get()
                state = training_data[0]
                if len(replay_buffer[state]) > 10000:
                    # training_data = q.get()
                    replay_buffer[state].pop(0)
                    replay_buffer[state].append(training_data)
                else:
                    # training_data = q.get()
                    replay_buffer[state].append(training_data)
            for n in range(5):
                states = []
                pi_hats = []
                v_hats = []
                # minibatch = random.sample(replay_buffer, minibatch_size)
                minibatch_list = []
                for state in possible_states:
                    for i in range(minibatch_size_per_state):
                        minibatch_list.append(random.sample(replay_buffer[state], 1)[0])
                # ^ pytorch
                # for training_data in minibatch:
                #     state = training_data[0]
                #     states.append(state)
                #     pi_hats.append(training_data[1])
                #     v_hats.append(training_data[2])
                # states = np.array(states).reshape((minibatch_size, 1))
                # pi_targets = np.array(pi_hats).reshape((minibatch_size, 3))
                # v_targets = np.array(v_hats).reshape((minibatch_size, 3))
                # states_tensor = torch.tensor(states, dtype=torch.float32)
                # pi_targets_tensor = torch.tensor(pi_targets, dtype=torch.float32)
                # v_targets_tensor = torch.tensor(v_targets, dtype=torch.float32)
                # dataset = TensorDataset(states_tensor, pi_targets_tensor, v_targets_tensor)
                # dataloader = DataLoader(dataset, batch_size=10, shuffle=True)  # adjust batch_size as necessary
                # for batch_states, batch_pi_targets, batch_v_targets in dataloader:
                #     print(f"batch_states: {batch_states}, batch_pi_targets: {batch_pi_targets}, batch_v_targets: {batch_v_targets}")
                #     optimizer.zero_grad()
                #     batch_states = batch_states.to(torch.int64)
                #     # batch_states_one_hot = F.one_hot(batch_states, num_classes=ENV_WIDTH*ENV_HEIGHT).float()
                #     pi_hats, v_hats = network(batch_states)
                #     state_index = np.argmax(batch_states.detach().numpy())
                #     loss_pi = criterion(pi_hats, batch_pi_targets)
                #     loss_v = criterion(v_hats, batch_v_targets)      
                #     loss = loss_pi + loss_v
                #     loss.backward()
                #     optimizer.step()
                #     print(f"34: {network(torch.tensor([34], dtype=torch.int64))}")
                #     print(f"35: {network(torch.tensor([35], dtype=torch.int64))}")
                #     print(f"44: {network(torch.tensor([66], dtype=torch.int64))}")
                # lock.acquire()
                # torch.save(network.state_dict(), weights_file)
                # lock.release()

                # ^ tensorflow
                minibatch = pd.DataFrame(minibatch_list, columns=["root_state", "prob_priors", "q"])
                states = []
                pi_hats = []
                v_hats = []
                root_state = minibatch["root_state"].tolist()
                prob_priors = minibatch["prob_priors"].tolist()
                qv = minibatch["q"].tolist()

                for j in range(len(root_state)):
                    state_vec = convert_to_one_hot_encoding(root_state[j])
                    states.append(state_vec)
                    pi_hats.append(prob_priors[j])
                    v_hats.append(qv[j])

                states = np.array(states).reshape((len(root_state), ENV_HEIGHT * ENV_WIDTH))
                pi_hats = np.array(pi_hats).reshape((len(root_state), 3))
                v_hats = np.array(v_hats).reshape((len(root_state), 3))

                lock.acquire()
                network.fit(x=states, y=[pi_hats, v_hats], batch_size=64, epochs=1)
                network.save_weights(weights_file)
                lock.release()

                print(f"saved weights to {weights_file}")
    print(f"NN Training completed")


if __name__ == "__main__":
    num_generators = 1
    file_name = "cliff_walking_alphazero_training.csv"
    # ^ pytorch
    # weights_file = "cliff_walking_alphazero_state_dict.pth"
    # ^ tensorflow
    weights_file = "cliff_walking_alphazero_weights.h5f"
    if os.path.exists(file_name):
        os.remove(file_name)
    q = Queue()
    lock = Lock()
    data_generation_processes = []
    iterations = 500
    num_episodes = 10000
    C = 50.0
    minibatch_size = 200
    slippery = 0
    gamma = 0.99

    args = {"multiprocessing queue": q, "lock": lock, "file_name": file_name, "weights_file": weights_file, "iterations": iterations, 
            "num_episodes": num_episodes,  "C": C, "minibatch_size": minibatch_size, "slippery": slippery, "gamma": gamma}
    
    with Manager() as manager:
        shared_results = manager.list([])
        shared_signal_status = manager.list([])
        args["shared_results"] = shared_results
        args["num_generators"] = num_generators
        args["shared_signal_status"] = shared_signal_status

        for i in range(num_generators):
            process = Process(target=run_simulations, args=(args,))
            process.start()
            data_generation_processes.append(process)

        # process_nn = Process(target=nn_training, args=(args,))
        # process_nn.start()

        # should train infinitely until converges to maximum cumulative reward
        for process in data_generation_processes:
            process.join()
        # process_nn.terminate()
