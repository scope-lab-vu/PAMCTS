import sys
import numpy as np
from .mcts_node import MCTSNode
import copy
import os
import pickle
import json
import math
from gym import spaces
import itertools
import time
from .discrete_w_start import DiscreteWS

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class MCTS:

    def __init__(self,
                gamma = 0.99,
                learning_agent=None,
                num_iter=10000,
                max_depth=500,
                c_puct=50.0):

        self.num_iterations = num_iter
        self.max_depth = max_depth
        self.learning_agent = learning_agent
        self.gamma = gamma
        self.c_puct = c_puct
        self.tree = []

    def selection_uct(self, curr_state):
        """
        select the child node with the highest UCT value
        """

        state_n = None
        state_v = None
        for node in self.tree:
            # compare two numpy arrays
            if np.array_equal(node["obs"], curr_state):
                state_n = np.array(node["num_visits"])
                state_v = np.array(node["value"])
                break
        N = sum(state_n)
        return np.argmax(state_v / state_n + self.c_puct * np.sqrt(math.log(N) / state_n))

    def get_action(self, observation, env):
        """
        return the action to take after running MCTS for num_iter iterations
        """
        starting_state = observation
        # add the root node
        self.tree.append({"obs": starting_state, "value": [0, 0, 0, 0], "num_visits": [1, 1, 1, 1]})

        for iteration in range(self.num_iterations):
            sa_trajectory = []
            reward_trajectory = []
            parent_pid = os.getpid()
            child_pid = os.fork()

            if child_pid > 0:
                # the parent process
                process = os.waitpid(child_pid, 0)

                # unpickle the tree
                with open(f"tree_{child_pid}_{iteration}.pkl", "rb") as f:
                    self.tree = pickle.load(f)

                # delete the pickle file
                # comment for debug
                os.remove(f"tree_{child_pid}_{iteration}.pkl")

            else:
                # the child process
                curr_state = starting_state
                terminated = False
                depth = 0
                expand_bool = False

                # select
                while not terminated and depth < self.max_depth:
                    action = self.selection_uct(curr_state)
                    tmp_state = curr_state
                    curr_state, reward, terminated, _, _ = env.step(action)
                    sa_trajectory.append((tmp_state, action))
                    reward_trajectory.append(reward)
                    depth += 1
                    if terminated or depth == self.max_depth:
                        break
                    # check if the curr_state is in the tree
                    # curr_state is a numpy array
                    found = False
                    for node in self.tree:
                        if np.array_equal(node["obs"], curr_state):
                            found = True
                            break
                    if not found:
                        expand_bool = True
                        break

                # debug use
                rollout_computation_time = time.time()

                # if curr_state not in self.tree:
                # leaf node, expand and rollout
                if expand_bool:
                    # expand
                    self.tree.append({"obs": curr_state, "value": [0, 0, 0, 0], "num_visits": [1, 1, 1, 1]})

                    leaf_rollout_return = 0
                    leaf_rollout_depth = 0
                    # rollout
                    # for _ in range(self.max_depth - depth):
                    while not terminated and depth < self.max_depth:
                        action = env.action_space.sample()
                        curr_state, reward, terminated, _, _ = env.step(action)
                        leaf_rollout_return += reward * self.gamma ** leaf_rollout_depth
                        leaf_rollout_depth += 1
                        depth += 1
                        if terminated or depth >= self.max_depth:
                            # ^ print(f"current depth: {depth}")
                            break
                    reward_trajectory.append(leaf_rollout_return)
                # backup the Monte carlo return till root
                reward_idx = len(reward_trajectory) - 1
                sa_idx = len(sa_trajectory) - 1
                discounted_return = 0

                # debug use
                # ^ print(f"rollout computation time: {time.time() - rollout_computation_time}")

                while sa_idx >= 0:  # backup the Monte carlo return till root
                    discounted_return = self.gamma * discounted_return + reward_trajectory[reward_idx]
                    (state, action) = sa_trajectory[sa_idx]
                    # update the value and num_visits of the node
                    node = None
                    for n in self.tree:
                        if np.array_equal(n["obs"], state):
                            node = n
                            break
                    node["num_visits"][action] += 1
                    visit_times = node["num_visits"][action]
                    node["value"][action] += 1/visit_times * (discounted_return - node["value"][action])
                    # move to parent node
                    reward_idx -= 1
                    sa_idx -= 1

                # pickle the tree
                with open(f"tree_{os.getpid()}_{iteration}.pkl", "wb") as f:
                    pickle.dump(self.tree, f)

                sys.exit(0)
        # return the starting_state's action with the highest value/num_visits
        node = [node for node in self.tree if np.array_equal(node["obs"], starting_state)][0]
        return np.argmax(np.array(node["value"]) / np.array(node["num_visits"]))

    def get_action_with_value(self, observation, env):
        """
        return the value for the starting state
        """
        _ = self.get_action(observation=observation, env=env)
        # ^ print(f"self tree: {self.tree}")
        # return the starting_state's node
        node = [node for node in self.tree if np.array_equal(node["obs"], observation)][0]
        return node

    def clear_tree(self):
        """
        clear the MCTS
        """
        self.tree = []
