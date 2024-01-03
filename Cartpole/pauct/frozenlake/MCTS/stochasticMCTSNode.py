import numpy as np


class StateNode:

    def __init__(self,
                 parent_action_node,
                 state,
                 # alpha=None,
                 # policy_value=None,
                 depth=0,
                 c=1.4142):

        # self.policy_value = policy_value
        # self.alpha = alpha
        self.parent_action_node = parent_action_node
        self.child_action_nodes = []
        self.visits = 0
        # self.value = 0
        self.depth = depth
        self.state = state
        self.c = c


    @property
    def most_visited_action_child(self):
        """return child with most visits. if no visited child return None"""
        visited_children = [child for child in self.child_action_nodes if child.visits > 0]
        if not visited_children:  # none of them visited
            return None
        return max(visited_children, key=lambda child : child.visits)


    def update_value(self):

        # running average update
        self.visits += 1
        # self.value += 1/(self.visits) * (monte_carlo_return - self.value)


class ActionNode:
    def __init__(self,
                 parent_state_node,
                 action,
                 alpha=None,
                 policy_value=None,
                 depth=0,
                 c=1.4142):

        # TODO the q score should just be added on exploration
        self.policy_value = policy_value
        self.alpha = alpha
        self.parent_state_node = parent_state_node
        self.action = action  # action taken to get to this from the parent
        self.possible_child_state_nodes = []
        self.visits = 0
        self.value = 0
        self.depth = depth
        self.c = c

    @property
    def ucb_score(self):
        if self.visits == 0:
            # TODO | shouldn't this just be float('inf') ?
            return 10000000000 + np.random.rand(1).item()  # arbitrarly large plus noise for tie-break

        if self.policy_value is None:
            return self.value + (self.c * np.sqrt(np.log(self.parent_state_node.visits)/self.visits))
        else:
            hybrid_node_value = (self.alpha * self.policy_value) + ((1.0 - self.alpha) * self.value)
            return hybrid_node_value + (self.c * np.sqrt(np.log(self.parent_state_node.visits)/self.visits))

    def update_value(self, monte_carlo_return):

        # running average update
        self.visits += 1
        self.value += 1/(self.visits) * (monte_carlo_return - self.value)
