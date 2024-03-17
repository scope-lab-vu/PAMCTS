import math
import os
import traceback
from multiprocessing import Pool
import random
import gym
import logging
import time
import numpy as np
import pandas as pd

# from build_model import build_model
# from utils import index_to_array, array_to_index
# from MCTS.pauct.fl_PAMCTS_Outside import FL_PAMCTS_Outside
# from env.customized_frozen_lake import Customized_FrozenLakeEnv
# from env.frozen_lake_9_10_1_20 import FrozenLakeEnv_9_10_1_20
import time

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.FileHandler(f"logs/frozenlake_test", mode="w")],
)
logger = logging.getLogger()

custom_map = ["SHF", "FFF", "HFG"]

# 0 H 2
# 3 4 5
# H 7 G
ENV_WIDTH = 3
ENV_HEIGHT = 3
HOLES = [1, 6]
START_STATE = 0
GOAL_STATE = 8

MOVE_RIGHT = 2
MOVE_DOWN = 1
MOVE_LEFT = 0
MOVE_UP = 3

COUNTER_CLOCKWISE = -1
STRAIGHT = 0
CLOCKWISE = 1
MAX_DEPTH = 200


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


def transition(state, move, STRAIGHT_PROBABILITY=None, CLOCKWISE_PROBABILITY=None):
    rnd = random.random()
    if rnd > STRAIGHT_PROBABILITY:
        # slide left or right
        if rnd < STRAIGHT_PROBABILITY + CLOCKWISE_PROBABILITY:
            move = (move + 1) % 4  # turn clockwise
        else:
            move = (
                move - 1 + 4
            ) % 4  # turn counter-clockwise (note the positive modulo)
    next_state = clip_move(state, move)
    if next_state in HOLES:
        return next_state, 0, True
    elif next_state == GOAL_STATE:
        return next_state, 1, True
    return next_state, 0, False


def select_action(C, state, n, v):
    state_n = n[state]
    state_v = v[state]
    N = sum(state_n)
    return np.argmax(state_v / state_n + C * np.sqrt(math.log(N) / state_n))


def select_action_rollout(state):
    return random_policy(state)


def rollout(state, prob_distribution, depth):
    done = False
    depth = depth
    while not done and depth < MAX_DEPTH:
        action = select_action_rollout(state)
        (next_state, reward, done) = transition(
            state,
            action,
            STRAIGHT_PROBABILITY=prob_distribution[0],
            CLOCKWISE_PROBABILITY=prob_distribution[1],
        )
        depth += 1
        state = next_state
    return reward


def MCTS(root_state, prob_distribution, iterations, c_puct):
    n = np.ones((ENV_WIDTH * ENV_HEIGHT, 4))
    v = np.zeros((ENV_WIDTH * ENV_HEIGHT, 4))
    for i in range(iterations):
        # logging.warning(f"MCTS iteration: {i}")
        state = root_state
        done = False
        reward = None
        trajectory = []
        depth = 0
        while not done:
            # logging.warning(f"State: {state}")
            action = select_action(C=c_puct, state=state, n=n, v=v)
            trajectory.append((state, action))
            (next_state, reward, done) = transition(
                state,
                action,
                STRAIGHT_PROBABILITY=prob_distribution[0],
                CLOCKWISE_PROBABILITY=prob_distribution[1],
            )
            depth += 1
            if done or depth > MAX_DEPTH:
                break
            if n[state, action] == 1:
                reward = rollout(next_state, prob_distribution, depth)
                break
            state = next_state
        for state, action in trajectory:
            n[state, action] += 1
            v[state, action] += reward
    # return select_action(0, root_state, n, v)
    # return the list of four v[root_state, for all four action] values
    return v[root_state] / n[root_state]


def get_pa_uct_score(alpha, policy_value, mcts_return):
    hybrid_node_value = (alpha * policy_value) + ((1.0 - alpha) * mcts_return)

    # TODO | might want to include ucb1?
    return (
        hybrid_node_value  # + self.C * np.sqrt(np.log(self.parent.visits)/self.visits)
    )


def simulate_episode(policy, network, iterations, c_puct, prob_distribution, alpha):
    steps = 0
    state = START_STATE
    while True:
        policy_value_list = network.get_q_values(state)
        mcts_return_list = policy(
            root_state=state,
            prob_distribution=prob_distribution,
            iterations=iterations,
            c_puct=c_puct,
        )
        best_action = None
        best_score = float("-inf")
        for i in range(4):
            score = get_pa_uct_score(
                alpha=alpha,
                policy_value=policy_value_list[i],
                mcts_return=mcts_return_list[i],
            )
            if score > best_score:
                best_score = score
                best_action = i
        (next_state, reward, done) = transition(
            state,
            best_action,
            STRAIGHT_PROBABILITY=prob_distribution[0],
            CLOCKWISE_PROBABILITY=prob_distribution[1],
        )
        steps += 1
        # logging.warning(f"State: {state}; Move: {move}; Next state: {next_state}; Reward: {reward}; Done: {done}.")
        if done or steps > MAX_DEPTH:
            return steps, reward
        
        # ^ print state, best_action, next_state
        print(f"State: {state}; Best action: {best_action}; Next state: {next_state}, Score: {best_score}")

        state = next_state


def evaluate_policy(
    policy,
    verbose,
    network,
    iterations,
    c_puct,
    prob_distribution,
    sample_id,
    gamma,
    alpha,
    file_name,
):
    start_time = time.time()
    lengths = np.zeros(1)
    outcomes = np.zeros(1)
    for episode in range(1):
        (length, outcome) = simulate_episode(
            policy, network, iterations, c_puct, prob_distribution, alpha
        )
        lengths[episode] = length
        outcomes[episode] = outcome
        if verbose:
            print(f"Episode: {episode}; Length: {length}; Outcome: {outcome}.")
    prob_distribution = [f"{prob:.3f}" for prob in prob_distribution]
    results = [
        [
            outcomes[0],
            prob_distribution,
            iterations,
            sample_id,
            time.time() - start_time,
            lengths[0],
            c_puct,
            gamma,
            alpha,
        ]
    ]
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            pd.DataFrame(results).to_csv(
                f,
                header=[
                    "cumulative_reward",
                    "prob_distribution",
                    "iterations",
                    "sample id",
                    "computation time",
                    "step_counter",
                    "c_puct",
                    "gamma",
                    "alpha",
                ],
                index=False,
            )
    else:
        with open(file_name, "a") as f:
            pd.DataFrame(results).to_csv(f, header=False, index=False)
    # results = Table().with_columns("Length", lengths, "Outcome", outcomes)
    # results.hist("Length", bins=10)
    print(f"Average episode length: {np.mean(lengths)}")
    print(f"Success rate: {np.mean(outcomes)}")
    print(f"Average discounted reward: {np.mean(outcomes * (gamma ** (lengths - 1)))}")


def run_simulations(args):
    from Network_Weights.DQN_3x3.ddqn_agent_FL import DDQN_Learning_Agent_FL

    # print("-= MCTS policy =-")
    env = gym.make("FrozenLake-v1", desc=custom_map)
    number_of_actions = env.action_space.n
    env_obs_space_shape = env.observation_space.shape
    start_time = time.time()
    prob_distribution = args["prob_distribution"]
    iterations = args["iterations"]
    c_puct = args["c_puct"]
    sample_id = args["sample_id"]
    gamma = args["gamma"]
    alpha = args["alpha"]
    file_name = args["file_name"]
    dqn_learning_agent = DDQN_Learning_Agent_FL(
        number_of_actions=number_of_actions, env_obs_space_shape=env_obs_space_shape
    )
    dqn_learning_agent.load_saved_weights(
        "Network_Weights/DQN_3x3/duel_dqn_FrozenLake-v1_slip_weights_reorder.h5f"
    )
    evaluate_policy(
        MCTS,
        verbose=True,
        network=dqn_learning_agent,
        iterations=iterations,
        c_puct=c_puct,
        prob_distribution=prob_distribution,
        sample_id=sample_id,
        gamma=gamma,
        alpha=alpha,
        file_name=file_name,
    )
    print(f"Time taken: {time.time() - start_time} seconds.")


if __name__ == "__main__":
    num_cpus = 1
    saved_file_name = "frozenlake_pamcts_results_final.csv"
    # saved_file_name = "sanity_check2.csv"
    begin_time = time.time()
    args = []
    # for prob_distribution in [
    #     [1.0, 0.0, 0.0],
    #     [14.0 / 15.0, 1.0 / 30.0, 1.0 / 30.0],
    #     [5.0 / 6.0, 1.0 / 12.0, 1.0 / 12.0],
    #     [11.0 / 15.0, 2.0 / 15.0, 2.0 / 15.0],
    #     [19.0 / 30.0, 11.0 / 60.0, 11.0 / 60.0],
    #     [8.0 / 15.0, 7.0 / 30.0, 7.0 / 30.0],
    #     [13.0 / 30.0, 17.0 / 60.0, 17.0 / 60.0],
    #     [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    # ]:
    for prob_distribution in [[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]:
        for alpha in [1.0]:
            # for alpha in [0.0]:
            # for iterations in [
            #     25,
            #     50,
            #     100,
            #     200,
            #     500,
            #     1000,
            #     1500,
            #     2000,
            #     3000,
            #     4000,
            #     5000,
            # ]:
            for iterations in [1000]:
                for c_puct in [1.414]:
                    for gamma in [0.999]:
                        for sample_id in range(1):
                            args.append(
                                {
                                    "prob_distribution": prob_distribution,
                                    "alpha": alpha,
                                    "gamma": gamma,
                                    "iterations": iterations,
                                    "c_puct": c_puct,
                                    "sample_id": sample_id,
                                    "file_name": saved_file_name,
                                }
                            )

    with Pool(processes=num_cpus) as pool:
        pool.map(run_simulations, args)

    print(f"experiments completed")
    with open("frozenlake_pamcts_execution_time_final.txt", "w") as f:
        f.write(
            f"fl final pamcts experiments execution time: {time.time() - begin_time}"
        )
