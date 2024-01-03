

class GreedyPolicyAgent():

    def __init__(self, policy):
        self.policy = policy

    def get_action(self, observation, env):

        return self.policy.get_greedy_action(observation=observation)
