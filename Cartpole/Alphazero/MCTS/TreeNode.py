"""
TreeNode class for deterministic environment of Monte Carlo Tree Search
"""

"""
import modules
"""
import numpy as np
import copy


class TreeNode:
    def __init__(self, parent, action, state, policy_value, depth, reward):
        self.parent = parent
        self.children = []
        self.policy_value = policy_value
        self.n_visit = 0
        self.u = 0
        self.q = 0
        self.action = action  # action taken to get to this from parent
        self.state = state  # state for current step
        self.depth = depth  # depth in the tree
        # whenever creating the node, reward should be updated
        self.reward = reward  # reward received from parent state step to current state

    def get_value(self, c_puct):
        """
        get the value of U + Q of the node
        """
        if self.n_visit == 0:
            return 10000000000 + np.random.rand(1).item()
        self.u = c_puct * self.policy_value * np.sqrt(self.parent.n_visit) / (1 + self.n_visit)
        return self.q + self.u

    def select(self, c_puct, is_exploration):
        """
        select children node that has the max(U+Q) among all children node
        """
        if is_exploration:
            return max(self.children, key=lambda node: node.get_value(c_puct))
        else:
            return max(self.children, key=lambda node: node.n_visit)

    def expand(self, prob_priors, env):
        """
        expand tree by creating new children
        action_priors: a list of tuples of actions and their prior probability according to the
                        policy function
        env: passed the openAI gym env (deep copy version)
        """
        # get the number of actions from the environment
        num_children = env.action_space.n
        for i in range(num_children):
            temp_env = copy.deepcopy(env)
            expanded_state, reward, _, _ = temp_env.step(i)

            self.children.append(TreeNode(self, i, expanded_state, prob_priors[i], self.depth+1, reward))

    def update(self, children_value, gamma):
        """
        update node values from leaf evaluation
        update in a recursive way
        number of visited times +1
        assume leaf_value is derived from neural network
        gamma: discount factor
        reward: individual reward to get to the state
        return the update value
        """
        self.n_visit += 1
        self.q += (children_value - self.q) / self.n_visit
        update = gamma * children_value + self.reward
        return update

    def update_recursive(self, leaf_value, gamma):
        """
        update recursively for all ancestors
        update from the leaf and trace back
        """
        update = self.update(leaf_value, gamma)
        if self.parent:
            self.parent.update_recursive(update, gamma)

    def is_leaf(self):
        """
        check if the node is leaf node
        """
        return self.children == []

    def is_root(self):
        """
        check if the node is root node
        """
        return self.parent is None
