"""
Monte Carlo Tree Search for deterministic environment
"""

"""
import modules
"""
import copy
import numpy as np
import tensorflow as tf
import gc
from .TreeNode import TreeNode


class MCTS:
    def __init__(self, current_state, gamma=0.999, c_puct=50, num_iter=500, max_depth=500):
        self.num_iterations = num_iter
        self.max_depth = max_depth
        self.gamma = gamma
        self.c_puct = c_puct
        self.root = TreeNode(parent=None, action=None, state=current_state, policy_value=1.0, depth=0, reward=0.0)

    def single_iter(self, network, env):
        """
        run a single iteration
        """
        cum_reward = 0
        # deepcopy at beginning of each iteration
        env_iter = copy.deepcopy(env)
        node = self.root
        terminated = False

        while (not terminated) and node.depth < self.max_depth and (not node.is_leaf()):
            new_node = node.select(c_puct=self.c_puct, is_exploration=True)

            action = new_node.action
            next_state, reward, terminated, _ = env_iter.step(action)

            cum_reward += reward
            new_node.reward = reward
            node = new_node

        if (not terminated) and node.depth < self.max_depth:
            # expand the leaf node
            # get prob_priors from the neural network
            # graph = tf.compat.v1.get_default_graph()
            # with graph.as_default():
            outputs_combo = network.predict(x=np.array(node.state).reshape((1, 4)), batch_size=1, verbose=0)

            # clear memory leak
            # tf.keras.backend.clear_session()
            # _ = gc.collect()

            prob_priors = outputs_combo[0][0]
            leaf_value = outputs_combo[1][0][0]
            # action_priors = []
            # action_priors.append((0, outputs_combo[0].flatten()[0]))
            # action_priors.append((1, outputs_combo[0].flatten()[1]))
            # leaf_value = outputs_combo[1]
            node.expand(prob_priors, env_iter)  # expand

            node.update_recursive(leaf_value, self.gamma)  # update recursive

        else:
            # if it is terminated or depth already be max depth
            # for depth be max depth part, later will try still evaluate neural network
            leaf_value = 0
            node.update_recursive(leaf_value, self.gamma)

        # env_iter.close()
        return cum_reward

    def run_mcts(self, is_modified_reward, network, env, gravity=9.8, masscart=1.0, masspole=0.1):
        """
        Run the MCTS tree search with num_iterations
        Finally, return the action that the root node select
        return  state ,action-priors, Q value root node
        """
        iteration = 0
        if not is_modified_reward:
            env_mcts = copy.deepcopy(env)
            env_mcts.env.gravity = gravity
            env_mcts.env.masscart = masscart
            env_mcts.env.masspole = masspole
            env_mcts.env.totalmass = env.env.masscart + env.env.masspole
            env_mcts._max_episode_steps = 2500
        else:
            env_mcts = copy.deepcopy(env)
            env_mcts.wrapped_env.env.gravity = gravity
            env_mcts.wrapped_env.env.masscart = masscart
            env_mcts.wrapped_env.env.masspole = masspole
            env_mcts.wrapped_env.env.totalmass = env.wrapped_env.env.masscart + env.wrapped_env.env.masspole
            env_mcts.wrapped_env._max_episode_steps = 2500
        while iteration < self.num_iterations:
            self.single_iter(network, env_mcts)
            iteration += 1
        # env_mcts.close()
        return self.root.select(c_puct=self.c_puct, is_exploration=False).action

    def mcts_update_root(self, current_state):
        """
        update self.root
        """
        self.root = TreeNode(parent=None, action=None, state=current_state, policy_value=1.0, depth=0, reward=0.0)

    def get_training_data(self):
        """
        return root state , action probs, root state value
        """
        visited_frac = self.root.children[0].n_visit / self.root.n_visit
        visited_frac2 = 1.0 - visited_frac
        prob_priors = [visited_frac, visited_frac2]

        return [self.root.state, prob_priors, self.root.q]
