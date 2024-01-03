import numpy as np


class MCTSNode:
    # C = 50 # UCB weighing term. Increasing it results in exploring for longer time.
    def __init__(self, parent, action, c_puct, alpha=None, policy_value=None):
        # TODO the q score should just be added on exploration
        self.policy_value = policy_value
        self.alpha = alpha
        self.parent = parent
        self.action = action  # action taken to get to this from the parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.C = c_puct


    @property
    def ucb_score(self):
        if self.visits == 0:
            # TODO | shouldn't this just be float('inf') ?
            return 10000000000 + np.random.rand(1).item()  # arbitrarly large plus noise for tie-break

        if self.policy_value is None:
            return self.value + self.C * np.sqrt(np.log(self.parent.visits)/self.visits)
        else:
            hybrid_node_value = (self.alpha * self.policy_value) + ((1.0 - self.alpha) * self.value)
            return hybrid_node_value + self.C * np.sqrt(np.log(self.parent.visits)/self.visits)


    @property
    def best_child(self):
        """return child with highest value. if no visited child return None"""
        visited_children = [child for child in self.children if child.visits > 0]
        if not visited_children:  # none of them visited
            return None
        return max(visited_children, key=lambda child : child.value)

    @property
    def most_visited_child(self):
        """return child with most visits. if no visited child return None"""
        visited_children = [child for child in self.children if child.visits > 0]
        if not visited_children:  # none of them visited
            return None
        return max(visited_children, key=lambda child : child.visits)

    def update_value(self, monte_carlo_return):

        # running average update
        self.visits += 1
        self.value += 1/(self.visits) * (monte_carlo_return - self.value)
