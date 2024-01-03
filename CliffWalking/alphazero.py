import os
import time
import math
import numpy as np
import pandas as pd
from multiprocessing import Pool
from ddqn_agent import DDQNAgent
from stochastic_cliff_walking import StochasticCliffWalkingEnv
from build_model_tf import build_model_tf
from utils import convert_to_one_hot_encoding


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


def MCTS(root_state, network, slippery, iterations, C, gamma):
    n = np.ones((ENV_WIDTH * ENV_HEIGHT, 3))
    v = np.zeros((ENV_WIDTH * ENV_HEIGHT, 3))
    p = np.zeros((ENV_WIDTH * ENV_HEIGHT, 3))
    r = np.zeros((ENV_WIDTH * ENV_HEIGHT, 3))
    for state in range(ENV_WIDTH * ENV_HEIGHT):
        # ^ tensorflow
        state_vec = convert_to_one_hot_encoding(state)
        outputs_combo = network.predict(x=np.array(state_vec).reshape((1, ENV_WIDTH * ENV_HEIGHT)), batch_size=1, verbose=0)
        prob_priors = outputs_combo[0][0]
        value = outputs_combo[1][0]
        for action in range(3):
            p[state, action] = prob_priors[action] 
            r[state, action] = value[action]
    for i in range(iterations):
        state = root_state
        done = False
        reward = None
        sa_trajectory = []
        depth = 0
        step_count = 0
        while not done:
            action = select_action(C, state, n, v, p)
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
    return np.argmax(v[root_state] / n[root_state])


def simulate_episode(policy, network, iterations, C, slippery, gamma):
    steps = 0
    state = START_STATE
    step_count = 0
    while True:
        action = policy(state, network, slippery, iterations, C, gamma)
        (next_state, reward, done) = transition(state, action, slippery=slippery, step_count=step_count)
        step_count += 1
        print(f"State: {state}; Move: {action}; Next state: {next_state}; Reward: {reward}; Done: {done}")
        steps += 1
        if done or steps > MAX_DEPTH:
            return steps, reward
        state = next_state


def evaluate_policy(policy, verbose, network, iterations, C, slippery, sample_id, gamma, file_name):
    start_time = time.time()
    lengths = np.zeros(1)
    outcomes = np.zeros(1)
    for episode in range(1):
        (length, outcome) = simulate_episode(policy, network, iterations, C, slippery, gamma)
        lengths[episode] = length
        outcomes[episode] = outcome
        if verbose:
            print(f"Episode: {episode}; Length: {length}; Outcome: {outcome}.")
        results = [[outcomes[0], slippery, iterations, sample_id, time.time() - start_time, lengths[0], C, gamma]]
        if not os.path.exists(file_name):
            with open(file_name, "w") as f:
                pd.DataFrame(results).to_csv(f, header=["cumulative_reward", "slippery", "iterations",
                                                        "sample id", "computation time", "step_counter", "C", "gamma"], index=False)
        else:
            with open(file_name, "a") as f:
                pd.DataFrame(results).to_csv(f, header=False, index=False)


def run_simulations(args):
    slippery = args["slippery"]
    iterations = args["iterations"]
    C = args["C"]
    sample_id = args["sample_id"]
    gamma = args["gamma"]
    file_name = args["file_name"]
    weights_file = args["weights_file"]
    start_time = time.time()
    network = build_model_tf(num_hidden_layers=5, state_size=ENV_WIDTH * ENV_HEIGHT)
    network.load_weights(weights_file)
    evaluate_policy(MCTS, verbose=True, network=network, iterations=iterations, C=C,
                    slippery=slippery, sample_id=sample_id, gamma=gamma, file_name=file_name)
    print(f"Sample {sample_id} Time taken: {time.time() - start_time} seconds.")


if __name__ == "__main__":
    num_cpus = 90
    file_name = "cliff_walking_alphazero.csv"
    weights_file = "cliff_walking_alphazero_weights.h5f"
    begin_time = time.time()
    args = []
    for slippery in [0.0, 0.1, 0.2, 0.3]:
        for iterations in [25, 50, 75, 100, 200, 500, 1000]:
            for C in [50.0]:
                for gamma in [0.99]:
                    for sample_id in range(100):
                        args.append({"slippery": slippery, "iterations": iterations, "weights_file": weights_file,
                                    "sample_id": sample_id, "C": C, "gamma": gamma, "file_name": file_name})
    
    with Pool(processes=num_cpus) as pool:
        pool.map(run_simulations, args)
    # run_simulations(args[0])

    print(f"experiments completed")
    with open("cliff_walking_pamcts_execution_time.txt", "w") as f:
        f.write(f"cliff walking pamcts experiments execution time: {time.time() - begin_time}")
