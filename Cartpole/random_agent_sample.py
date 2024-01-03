import random

class RandomAgentSample():

    def get_action(self, observation, env):

        return env.action_space.sample()
