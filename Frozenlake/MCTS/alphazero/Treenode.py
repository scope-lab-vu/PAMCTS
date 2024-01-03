"""
MCTS chance nodes and deterministic nodes
"""
import sys

"""
import modules
"""
import numpy as np


class StateNode:
    """
    StateNode class for FrozenLake
    """

    def __init__(self, parent_action, state, depth, reward, network):
        self.parent_action = parent_action
        self.n_visit = 0
        self.u = 0
        # self.q = 0
        self.state = state  # state for current step
        self.depth = depth  # depth in the tree
        self.reward = reward  # reward received from parent state step to current state
        # outputs_combo = network.predict(x=np.array(state[0]).reshape((1, 1)),
        #                                 batch_size=1, verbose=0)
        # prob_priors = outputs_combo[0][0]
        self.prob_priors = [0.25, 0.25, 0.25, 0.25]
        self.children_actions = []
        self.expand(prob_priors=self.prob_priors)

    def select(self, c_puct, is_exploration, q_values, prob_priors, visit_times):
        """
        select children node that has the max(U+Q) among all children node
        """
        if is_exploration:
            return max(self.children_actions, key=lambda node: node.get_value(c_puct, q_values, prob_priors,
                                                                              visit_times))
        else:
            # return max(self.children_actions, key=lambda node: node.n_visit)
            # for visit_times[self.state, every action], find the max
            print(f"four actions: {visit_times[self.state.tolist()[0].index(1), :]}")
            return np.argmax(visit_times[self.state.tolist()[0].index(1), :])

    def expand(self, prob_priors):
        """
        expand tree by creating four new action chance nodes
        """
        if not self.children_actions:
            for i in range(4):
                self.children_actions.append(ActionNode(curr_state=self,
                                                        action=i, prob_prior=prob_priors[i], depth=self.depth + 1))

    def update(self, update_value, gamma):
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
        if self.parent_action:
            self.parent_action.update(update_value, gamma, self.reward)
        # else:
        #     for action_node in self.children_actions:
        #         if action_node.n_visit < 20:
        #             # write to a txt file
        #             with open("mcts.txt", "a") as f:
        #                 f.write(f"action id: {action_node.action}, n_visit: {action_node.n_visit}, q: {action_node.q}\n")
        #             print(f"action id: {action_node.action}, n_visit: {action_node.n_visit}, q: {action_node.q}")

    def is_leaf(self):
        """
        check if the node is a leaf node
        """
        return self.children_actions == []


class ActionNode:
    """
    ActionNode class for FrozenLake
    """

    def __init__(self, curr_state, action, prob_prior, depth):
        self.parent_state = curr_state
        # children are StateNodes
        self.possible_states = []
        self.prob_prior = prob_prior
        self.action = action
        self.depth = depth
        self.n_visit = 0
        self.u = 0
        self.q = 0

    def get_value(self, c_puct, q_values, prob_priors, visit_times):
        """
        get the value of U + Q of the node
        """
        # if self.n_visit == 0:
        if visit_times[self.parent_state.state.tolist()[0].index(1), self.action] == 0:
            return 10000000000 + np.random.rand(1).item()
        parent_visit_times = np.sum(visit_times[self.parent_state.state.tolist()[0].index(1), :])
        self.u = c_puct * prob_priors[self.parent_state.state.tolist()[0].index(1), self.action] * np.sqrt(
            parent_visit_times) / (1 + visit_times[self.parent_state.state.tolist()[0].index(1), self.action])
        # self.u = c_puct * self.prob_prior * np.sqrt(self.parent_state.n_visit) / (1 + self.n_visit)
        return q_values[self.parent_state.state.tolist()[0].index(1), self.action] + self.u

    def select(self, next_state):
        """
        select next state node as return env.step(action) from parent state
        """
        for possible_state_node in self.possible_states:
            # if possible_state_node.state == next_state:
            if np.array_equal(possible_state_node.state, next_state):
                return possible_state_node

    def expand(self, state, reward, network):
        """
        expand only one state for a given state
        """
        new_state = StateNode(parent_action=self, state=state, depth=self.depth, reward=reward, network=network)
        self.possible_states.append(new_state)
        return new_state

    def update(self, children_value, gamma, q_values, visit_times):
        """
        update the value of the action node
        """
        self.n_visit += 1
        self.q += (children_value - self.q) / self.n_visit
        visit_times[self.parent_state.state.tolist()[0].index(1), self.action] += 1
        q_values[self.parent_state.state.tolist()[0].index(1), self.action] = \
            (q_values[self.parent_state.state.tolist()[0].index(1), self.action] *
             (visit_times[self.parent_state.state.tolist()[0].index(1), self.action] - 1) + children_value) / \
            visit_times[self.parent_state.state.tolist()[0].index(1), self.action]
        # print(f"action id: {self.action}, reward: {reward}")
        # with open("mcts.txt", "a") as f:
        #     f.write(f"action id: {self.action}, reward: {reward} children_value: {children_value} q: {self.q}\n")
        update = gamma * children_value
        if self.parent_state.parent_action:
            self.parent_state.parent_action.update(update, gamma, q_values, visit_times)
