import random
import logging
import numpy as np
import math

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

TEMPORAL_DISCOUNT = 0.95
# STRAIGHT_PROBABILITY = 0.4333
# STRAIGHT_PROBABILITY = 0.90
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


def simulate_episode(policy, iterations, c_puct, prob_distribution):
    steps = 0
    state = START_STATE
    while True:
        move = policy(state, iterations, c_puct)
        (next_state, reward, done) = transition(state, move, STRAIGHT_PROBABILITY=prob_distribution[0], CLOCKWISE_PROBABILITY=prob_distribution[1])
        steps += 1
        # logging.warning(f"State: {state}; Move: {move}; Next state: {next_state}; Reward: {reward}; Done: {done}.")
        if done:
            return steps, reward
        state = next_state


EVALUATION_EPISODES = 100


def evaluate_policy(policy, verbose=False):
    lengths = np.zeros(EVALUATION_EPISODES)
    outcomes = np.zeros(EVALUATION_EPISODES)
    for episode in range(EVALUATION_EPISODES):
        (length, outcome) = simulate_episode(policy)
        lengths[episode] = length
        outcomes[episode] = outcome
        if verbose:
            print(f"Episode: {episode}; Length: {length}; Outcome: {outcome}.")
    # results = Table().with_columns("Length", lengths, "Outcome", outcomes)
    # results.hist("Length", bins=10)
    print(f"Average episode length: {np.mean(lengths)}")
    print(f"Success rate: {np.mean(outcomes)}")
    print(f"Average discounted reward: {np.mean(outcomes * (TEMPORAL_DISCOUNT ** (lengths - 1)))}")


MCTS_ITERATIONS = 5000


def select_action(C, state, n, v):
    state_n = n[state]
    state_v = v[state]
    N = sum(state_n)
    return np.argmax(state_v / state_n + C * np.sqrt(math.log(N) / state_n))


def select_action_rollout(state):
    return random_policy(state)


def rollout(state):
    done = False
    while not done:
        action = select_action_rollout(state)
        (next_state, reward, done) = transition(state, action)
        state = next_state
    return reward


def MCTS(root_state):
    n = np.ones((ENV_WIDTH * ENV_HEIGHT, 4))
    v = np.zeros((ENV_WIDTH * ENV_HEIGHT, 4))
    for i in range(MCTS_ITERATIONS):
        # logging.warning(f"MCTS iteration: {i}")
        state = root_state
        done = False
        reward = None
        trajectory = []
        while not done:
            # logging.warning(f"State: {state}")
            action = select_action(10, state, n, v)
            trajectory.append((state, action))
            (next_state, reward, done) = transition(state, action)
            if done:
                break
            if n[state, action] == 1:
                reward = rollout(next_state)
                break
            state = next_state
        for (state, action) in trajectory:
            n[state, action] += 1
            v[state, action] += reward
    return select_action(0, root_state, n, v)


if __name__ == "__main__":
    print("-= MCTS policy =-")
    evaluate_policy(MCTS, verbose=True)
