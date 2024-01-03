import copy
import os
import sys
import pickle
import math
import torch
# BASE_DIR = "../../mcts/alphazero"
# sys.path.append(BASE_DIR)
# from treenode import TreeNode
import numpy as np
import tensorflow as tf
import gc

class MCTS(object):
    """
    An implementation of monte carlo tree search for Alphazero
    """

    def __init__(self, gamma=0.999, c_puct=50, num_iter=500, max_depth=5000, weight_file=None, num_hidden_layers=5):
        self.num_iterations = num_iter
        self.max_depth = max_depth
        self.gamma = gamma
        self.c_puct = c_puct
        self.weights_file = weight_file
        self.num_hidden_layers = num_hidden_layers
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
                state_p = np.array(node["prob_priors"])
                break
        N = sum(state_n)
        return np.argmax(state_v + self.c_puct * state_p * np.sqrt(N) / (1 + state_n))


    def single_iter(self, network, env, starting_state):
        """
        run a single iteration for child process
        """
        import sys
        BASE_DIR2 = "../../network_weights/Alphazero_Networks"
        sys.path.append(BASE_DIR2)

        cum_reward = 0
        terminated = False
        curr_state = starting_state
        curr_node = None
        for node in self.tree:
            if np.array_equal(node["obs"], curr_state):
                curr_node = node
                break
        sa_trajectory = []
        reward_trajectory = []
        expand_bool = False
        depth = 0

        while not terminated and depth < self.max_depth:
            action = self.selection_uct(curr_state)
            tmp_state = curr_state
            curr_state, reward, terminated, _, _ = env.step(action)
            sa_trajectory.append((tmp_state, action))
            reward_trajectory.append(reward)
            depth += 1
            if terminated or depth == self.max_depth:
                expand_bool = False
                break
            found = False
            for node in self.tree:
                if np.array_equal(node["obs"], curr_state):
                    found = True
                    curr_node = node
                    break
            if not found:
                expand_bool = True
                break

        if expand_bool:
            input_tensor = torch.from_numpy(np.array(curr_state).reshape((1, env.observation_space.shape[0])))
            prob_priors, leaf_rollout_return = network.forward(x=input_tensor)
            _ = gc.collect()
            prob_priors = prob_priors.detach().numpy()[0]
            leaf_rollout_return = leaf_rollout_return.detach().numpy()[0][0]
            self.tree.append({"obs": curr_state, "value": [0, 0, 0, 0], "num_visits": [0, 0, 0, 0], "prob_priors": prob_priors})
            reward_trajectory.append(leaf_rollout_return)
        reward_idx = len(reward_trajectory) - 1
        sa_idx = len(sa_trajectory) - 1
        discounted_return = 0

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


    def run_mcts(self, network, env, observation):
        """
        Run the MCTS tree search with num_iterations
        Finally, return the action that the root node select
        return  state ,action-priors, Q value root node
        """
        iteration = 0
        starting_state = observation
        while iteration < self.num_iterations:
            parent_pid = os.getpid()
            child_pid = os.fork()
            if child_pid > 0:
                # the parent process
                process = os.waitpid(child_pid, 0)
                with open(f"tree_{child_pid}_{iteration}.pkl", "rb") as f:
                    self.tree = pickle.load(f)
                os.remove(f"tree_{child_pid}_{iteration}.pkl")
            else:
                if iteration == 0:
                    input_tensor = torch.from_numpy(np.array(starting_state).reshape((1, env.observation_space.shape[0])))
                    prob_priors, leaf_rollout_return = network.forward(x=input_tensor)
                    prob_priors = prob_priors.detach().numpy()[0]
                    self.tree.append({"obs": starting_state, "value": [0, 0, 0, 0], "num_visits": [0, 0, 0, 0], "prob_priors": prob_priors})
                # the child process
                self.single_iter(network, env, starting_state)
                with open(f"tree_{os.getpid()}_{iteration}.pkl", "wb") as f:
                    pickle.dump(self.tree, f)
                sys.exit(0)
            iteration += 1
        node = [node for node in self.tree if np.array_equal(node["obs"], starting_state)][0]
        return np.argmax(np.array(node["num_visits"]))


    # def mcts_update_root(self, current_state):
    #     """
    #     update self.root
    #     """
    #     self.root = TreeNode(parent=None, action=None, state=current_state,
    #                         policy_value=1.0, depth=0, reward=0.0)
        
    def clear_tree(self):
        """
        clear the MCTS
        """
        self.tree = []


    def get_training_data(self, starting_state):
        """
        return root state , action probs, root state value
        """
        node = [node for node in self.tree if np.array_equal(node["obs"], starting_state)][0]
        return node["obs"], node["prob_priors"]
    