import os
import time
import torch
import math
import numpy as np
import pandas as pd
from multiprocessing import Pool
from ddqn_agent import DDQNAgent
from stochastic_cliff_walking import StochasticCliffWalkingEnv


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


def select_action(C, state, n, v):
    state_n = n[state]
    state_v = v[state]
    N = sum(state_n)
    # if state == 36:
    #     print(f"four values: {state_v / state_n + C * np.sqrt(math.log(N) / state_n)}")
    temp = state_v / state_n + C * np.sqrt(math.log(N) / state_n)
    for i in range(3):
        if state_n[i] <= 1:
            return i
    return np.argmax(state_v / state_n + C * np.sqrt(math.log(N) / state_n))


def random_policy(state):
    if state == 36:
        return MOVE_UP
    elif 24 <= state < 35:
        return MOVE_RIGHT
    elif state == 35:
        return MOVE_DOWN
    elif state & 12 == 11:
        return MOVE_DOWN
    else:
        return MOVE_RIGHT

def select_action_rollout(state):
    return random_policy(state)


def rollout(state, slippery, depth, gamma, step_count):
    done = False
    depth = depth
    while not done and depth < MAX_DEPTH:
        action = select_action_rollout(state)
        (next_state, reward, done) = transition(state, action, slippery=slippery, step_count=step_count)
        step_count += 1
        depth += 1
        state = next_state
    return reward


def MCTS(root_state, slippery, iterations, C, gamma):
    n = np.ones((ENV_WIDTH * ENV_HEIGHT, 3))
    v = np.zeros((ENV_WIDTH * ENV_HEIGHT, 3))
    for i in range(iterations):
        # logging.warning(f"MCTS iteration: {i}")
        step_count = 0
        state = root_state
        done = False
        reward = None
        sa_trajectory = []
        # r_trajectory = []
        depth = 0
        while not done:
            # logging.warning(f"State: {state}")
            action = select_action(C=C, state=state, n=n, v=v)
            sa_trajectory.append((state, action))
            (next_state, reward, done) = transition(state, action, slippery=slippery, step_count=step_count)
            step_count += 1
            # r_trajectory.append(reward)
            depth += 1
            if done or depth > MAX_DEPTH:
                break
            if n[state, action] == 1:
                reward = rollout(next_state, slippery, depth, gamma, step_count)
                # print(f"leaf_rollout_return: {leaf_rollout_return}")
                # r_trajectory.append(leaf_rollout_return)
                break
            state = next_state
        # sa_idx = len(sa_trajectory) - 1
        # r_idx = len(r_trajectory) - 1
        # discounted_return = 0
        # while sa_idx >= 0:
        #     discounted_return = gamma * discounted_return + r_trajectory[r_idx]
        #     (state, action) = sa_trajectory[sa_idx]
        #     n[state, action] += 1
        #     v[state, action] += 1 / n[state, action] * (discounted_return - v[state, action])
        #     sa_idx -= 1
        #     r_idx -= 1
        # print(f"iteration: {i}; sa_trajectory: {sa_trajectory}; r_trajectory: {r_trajectory}")
        for (state, action) in sa_trajectory:
            n[state, action] += 1
            v[state, action] += reward
    # return select_action(0, root_state, n, v)
    # return the list of four v[root_state, for all four action] values
    return v[root_state] / n[root_state]


def get_pa_uct_score(alpha, policy_value, mcts_return):
    hybrid_node_value = (alpha * policy_value) + ((1.0 - alpha) * mcts_return)

    # TODO | might want to include ucb1?
    return hybrid_node_value  # + self.C * np.sqrt(np.log(self.parent.visits)/self.visits)


def simulate_episode(policy, network, iterations, C, slippery, alpha, gamma):
    steps = 0
    state = START_STATE
    step_count = 0
    while True:
        policy_value_list = network.get_q_values(state)
        policy_value_list = policy_value_list[1:]
        mcts_return_list = policy(root_state=state, slippery=slippery, iterations=iterations, C=C, gamma=gamma)
        best_action = None
        best_score = float('-inf')
        for i in range(3):
            score = get_pa_uct_score(alpha=alpha, policy_value=policy_value_list[i], mcts_return=mcts_return_list[i])
            if score > best_score:
                best_score = score
                best_action = i
        (next_state, reward, done) = transition(state, best_action, slippery=slippery, step_count=step_count)
        step_count += 1
        print(f"State: {state}; Move: {best_action}; Next state: {next_state}; Reward: {reward}; Done: {done}")
        steps += 1
        # logging.warning(f"State: {state}; Move: {move}; Next state: {next_state}; Reward: {reward}; Done: {done}.")
        if done or steps > MAX_DEPTH:
            return steps, reward
        state = next_state


def evaluate_policy(policy, verbose, network, iterations, C, slippery, sample_id, gamma, alpha, file_name):
    start_time = time.time()
    lengths = np.zeros(1)
    outcomes = np.zeros(1)
    for episode in range(1):
        (length, outcome) = simulate_episode(policy, network, iterations, C, slippery, alpha, gamma)
        lengths[episode] = length
        outcomes[episode] = outcome
        if verbose:
            print(f"Episode: {episode}; Length: {length}; Outcome: {outcome}.")
        results = [[outcomes[0], slippery, iterations, sample_id, time.time() - start_time, lengths[0], C, gamma, alpha]]
        if not os.path.exists(file_name):
            with open(file_name, "w") as f:
                pd.DataFrame(results).to_csv(f, header=["cumulative_reward", "slippery", "iterations",
                                                        "sample id", "computation time", "step_counter", "C", "gamma", "alpha"], index=False)
        else:
            with open(file_name, "a") as f:
                pd.DataFrame(results).to_csv(f, header=False, index=False)


def run_simulations(args):
    slippery = args["slippery"]
    iterations = args["iterations"]
    C = args["C"]
    alpha = args["alpha"]
    sample_id = args["sample_id"]
    gamma = args["gamma"]
    file_name = args["file_name"]
    start_time = time.time()
    env = StochasticCliffWalkingEnv(slipperiness=slippery)
    ddqn_learning_agent = DDQNAgent(state_size=2, action_size=4)
    ddqn_learning_agent.q_network.load_state_dict(torch.load("cliff_walking_ddqn_weights.pth"))
    evaluate_policy(MCTS, verbose=True, network=ddqn_learning_agent, iterations=iterations, C=C,
                    slippery=slippery, sample_id=sample_id, gamma=gamma, alpha=alpha, file_name=file_name)
    print(f"Sample {sample_id} Time taken: {time.time() - start_time} seconds.")


if __name__ == "__main__":
    num_cpus = 50
    file_name = "cliff_walking_alpha_selection.csv"
    begin_time = time.time()
    args = []
    # ^ slippery = [0.0, 0.1, 0.2, 0.3]
    # ^ alpha = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
    # ^ iterations = [5, 10, 15, 20, 25, 50, 100]
    for slippery in [0.0, 0.1, 0.2, 0.3]:
        for alpha in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]:
            for iterations in [5, 10, 15, 20, 25, 50, 100]:
                for C in [50.0]:
                    for gamma in [0.99]:
                        for sample_id in range(100):
                            args.append({"slippery": slippery, "alpha": alpha, "iterations": iterations,
                                        "sample_id": sample_id, "C": C, "gamma": gamma, "file_name": file_name})
                            
    with Pool(processes=num_cpus) as pool:
        pool.map(run_simulations, args)

    print(f"experiments completed")
    with open("cliff_walking_alpha_selection_execution_time.txt", "w") as f:
        f.write(f"cliff walking alpha selection experiments execution time: {time.time() - begin_time}")