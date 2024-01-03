import gc
import os
import random
import traceback
from multiprocessing import Pool
import gym
import logging
import time
import numpy as np
import pandas as pd
# from build_model import build_model
# from utils import index_to_array, array_to_index
# from Network_Weights.DQN_3x3.ddqn_agent_FL import DDQN_Learning_Agent_FL
# from MCTS.pauct.fl_PAMCTS_Outside import FL_PAMCTS_Outside
# from env.customized_frozen_lake import Customized_FrozenLakeEnv
# from env.frozen_lake_9_10_1_20 import FrozenLakeEnv_9_10_1_20
from MCTS.alphazero.mcts import run_simulations
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

# 0 H 2
# 3 4 5
# H 7 G
ENV_WIDTH = 3
ENV_HEIGHT = 3
HOLES = [1, 6]
START_STATE = 0
GOAL_STATE = 8

MOVE_LEFT = 0
MOVE_DOWN = 1
MOVE_RIGHT = 2
MOVE_UP = 3

COUNTER_CLOCKWISE = -1
STRAIGHT = 0
CLOCKWISE = 1

TEMPORAL_DISCOUNT = 0.999
# STRAIGHT_PROBABILITY = 0.4333
# STRAIGHT_PROBABILITY = 1.0
# CLOCKWISE_PROBABILITY = (1 - STRAIGHT_PROBABILITY) / 2


def clip_move(state, move):
    if move == MOVE_RIGHT:
        if (state % ENV_WIDTH) == (ENV_WIDTH - 1):
            return state
        else:
            return state + 1
    elif move == MOVE_DOWN:
        if state >= (ENV_WIDTH * (ENV_HEIGHT - 1)):
            return state
        else:
            return state + ENV_WIDTH
    elif move == MOVE_LEFT:
        if (state % ENV_WIDTH) == 0:
            return state
        else:
            return state - 1
    elif move == MOVE_UP:
        if state < ENV_WIDTH:
            return state
        else:
            return state - ENV_WIDTH
    return None


def transition(state, move, STRAIGHT_PROBABILITY=None, CLOCKWISE_PROBABILITY=None):
    rnd = random.random()
    if rnd > STRAIGHT_PROBABILITY:
        # slide left or right
        if rnd < STRAIGHT_PROBABILITY + CLOCKWISE_PROBABILITY:
            move = (move + 1) % 4  # turn clockwise
        else:
            move = (move - 1 + 4) % 4  # turn counter-clockwise (note the positive modulo)
    next_state = clip_move(state, move)
    if next_state in HOLES:
        return next_state, 0, True
    elif next_state == GOAL_STATE:
        return next_state, 1, True
    return next_state, 0, False


# Intuitive policy
INTUITIVE_POLICY = {0: MOVE_LEFT,
                    2: MOVE_RIGHT,
                    3: MOVE_UP,
                    4: MOVE_DOWN,
                    5: MOVE_DOWN,
                    7: MOVE_RIGHT}


def intuitive_policy(state):
    return INTUITIVE_POLICY[state]


# Random policy
def random_policy(state):
    if state == 0:
        return MOVE_LEFT
    elif state == 2:
        return MOVE_RIGHT
    elif state == 3:
        return MOVE_UP
    elif state == 4:
        return MOVE_DOWN
    elif state == 5:
        chance = random.random()
        if chance <= 0.33:
            return MOVE_RIGHT
        elif chance <= 0.5:
            return MOVE_UP
        elif chance <= 0.75:
            return MOVE_LEFT
        else:
            return MOVE_DOWN
    elif state == 7:
        return MOVE_RIGHT


def simulate_episode(policy, network, iterations, c_puct, prob_distribution):
    steps = 0
    state = START_STATE
    while True:
        move = policy(root_state=state, network=network, prob_distribution=prob_distribution, iterations=iterations,
                      c_puct=c_puct)
        (next_state, reward, done) = transition(state, move, STRAIGHT_PROBABILITY=prob_distribution[0], CLOCKWISE_PROBABILITY=prob_distribution[1])
        steps += 1
        # logging.warning(f"State: {state}; Move: {move}; Next state: {next_state}; Reward: {reward}; Done: {done}.")
        if done or steps > max_depth:
            return steps, reward
        state = next_state


EVALUATION_EPISODES = 1


def evaluate_policy(policy, verbose, network, iterations, c_puct, prob_distribution, sample_id, file_name):
    start_time = time.time()
    lengths = np.zeros(EVALUATION_EPISODES)
    outcomes = np.zeros(EVALUATION_EPISODES)
    for episode in range(EVALUATION_EPISODES):
        (length, outcome) = simulate_episode(policy, network, iterations, c_puct, prob_distribution)
        lengths[episode] = length
        outcomes[episode] = outcome
        if verbose:
            print(f"Episode: {episode}; Length: {length}; Outcome: {outcome}.")
        _ = gc.collect()
    # results = Table().with_columns("Length", lengths, "Outcome", outcomes)
    # results.hist("Length", bins=10)
    # convert prob_distribution to string, only keep 3 digits after decimal point
    prob_distribution = [f"{prob:.3f}" for prob in prob_distribution]
    results = [[outcomes[0], prob_distribution, iterations, sample_id, time.time() - start_time, lengths[0], c_puct]]
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            pd.DataFrame(results).to_csv(f, header=["cumulative_reward", "prob_distribution", "iterations",
                                                    "sample id", "computation time", "step_counter", "c_puct"], index=False)
    else:
        with open(file_name, "a") as f:
            pd.DataFrame(results).to_csv(f, header=False, index=False)
    print(f"Average episode length: {np.mean(lengths)}")
    print(f"Success rate: {np.mean(outcomes)}")
    print(f"Average discounted reward: {np.mean(outcomes * (TEMPORAL_DISCOUNT ** (lengths - 1)))}")


# MCTS_ITERATIONS = 3000
max_depth = 200


def select_action(C, state, n, v, p):
    import math
    state_n = n[state]
    state_v = v[state]
    state_p = p[state]
    N = sum(state_n)
    return np.argmax(state_v / state_n + C * state_p * np.sqrt(math.log(N) / state_n))


def select_action_rollout(state):
    return random_policy(state)


def rollout(state, prob_distribution):
    done = False
    while not done:
        action = select_action_rollout(state)
        (next_state, reward, done) = transition(state, action, STRAIGHT_PROBABILITY=prob_distribution[0], CLOCKWISE_PROBABILITY=prob_distribution[1])
        state = next_state
    return reward


def MCTS(root_state, network, prob_distribution, iterations=3000, c_puct=1.414):
    from keras import backend as K
    import tensorflow as tf
    from tensorflow import keras
    from keras.layers import Input, Dense
    from keras.models import Model
    from tensorflow.keras.optimizers import Adam
    n = np.ones((ENV_WIDTH * ENV_HEIGHT, 4))
    # v = np.zeros((ENV_WIDTH * ENV_HEIGHT, 4))
    v = np.zeros((ENV_WIDTH * ENV_HEIGHT, 4))
    p = np.zeros((ENV_WIDTH * ENV_HEIGHT, 4))
    r = np.zeros((ENV_WIDTH * ENV_HEIGHT, 1))
    for state in range(ENV_WIDTH * ENV_HEIGHT):
        state_arr = np.zeros(ENV_WIDTH * ENV_HEIGHT)
        state_arr[state] = 1
        state_arr = np.reshape(state_arr, [1, ENV_WIDTH * ENV_HEIGHT])
        outputs_combo = network.predict(x=np.array(state_arr).reshape((1, ENV_WIDTH * ENV_HEIGHT)),
                                        batch_size=1, verbose=0)
        prob_priors = outputs_combo[0][0]
        value = outputs_combo[1][0]
        r[state] = value
        for action in range(4):
            p[state, action] = prob_priors[action]
    # Debug: initialize every value in p to 0.25
    # for i in range(ENV_WIDTH * ENV_HEIGHT):
    #     for j in range(4):
    #         p[i, j] = 0.25
    for i in range(iterations):
        # logging.warning(f"MCTS iteration: {i}")
        state = root_state
        done = False
        reward = None
        trajectory = []
        depth = 0
        while not done:
            # logging.warning(f"State: {state}")
            action = select_action(c_puct, state, n, v, p)
            trajectory.append((state, action))
            (next_state, reward, done) = transition(state, action, STRAIGHT_PROBABILITY=prob_distribution[0], CLOCKWISE_PROBABILITY=prob_distribution[1])
            depth += 1
            if done or depth > max_depth:
                break
            if n[state, action] == 1:
                # reward = rollout(next_state)
                reward = r[state]
                break
            state = next_state
        for (state, action) in trajectory:
            n[state, action] += 1
            v[state, action] += reward
    return select_action(0, root_state, n, v, p)


def run_simulations(args):
    from keras import backend as K
    import tensorflow as tf
    from tensorflow import keras
    from keras.layers import Input, Dense
    from keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from build_model import build_model
    # tensors
    global layer
    print("-= MCTS policy =-")
    start_time = time.time()
    prob_distribution = args["prob_distribution"]
    iterations = args["iterations"]
    c_puct = args["c_puct"]
    sample_id = args["sample_id"]
    file_name = args["file_name"]
    network = build_model(num_hidden_layers=5, state_size=9)
    # check if weights file exists
    # if not os.path.exists("Network_Weights/Alphazero_final/weights_final.h5f"):
    #     print("weights_final.h5f not found, please run train.py first.")
    #     return
    network.load_weights("Network_Weights/Alphazero_final/weights_final.h5f").expect_partial()
    evaluate_policy(MCTS, verbose=True, network=network, iterations=iterations, c_puct=c_puct, prob_distribution=prob_distribution, sample_id=sample_id, file_name=file_name)
    print(f"Time taken: {time.time() - start_time} seconds.")


if __name__ == "__main__":
    num_cpus = 90
    file_name = "frozenlake_alphazero_final_part2.csv"
    begin_time = time.time()
    args = []
    for prob_distribution in [[1.0, 0.0, 0.0], [14.0 / 15.0, 1.0 / 30.0, 1.0 / 30.0],
                              [5.0 / 6.0, 1.0 / 12.0, 1.0 / 12.0], [11.0 / 15.0, 2.0 / 15.0, 2.0 / 15.0],
                              [19.0 / 30.0, 11.0 / 60.0, 11.0 / 60.0], [8.0 / 15.0, 7.0 / 30.0, 7.0 / 30.0],
                              [13.0 / 30.0, 17.0 / 60.0, 17.0 / 60.0], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]:
    # for prob_distribution in [[1.0, 0.0, 0.0]]:
        for iterations in [4000, 5000]:
        # for iterations in [3000]:
            for c_puct in [1.414]:
                for sample_id in range(100):
                    # network = build_model(num_hidden_layers=5, state_size=9)
                    # network.load_weights("Network_Weights/Alphazero_final/weights_final.h5f")
                    args.append({"prob_distribution": prob_distribution, "iterations": iterations,
                                 "c_puct": c_puct, "sample_id": sample_id, "file_name": file_name})

    with Pool(processes=num_cpus) as pool:
        pool.map(run_simulations, args)

    print(f"experiments completed")
    with open("fl_alphazero_execution_time_final.txt", "w") as f:
        f.write(f"fl final alphazero experiments execution time : {time.time() - begin_time}")
