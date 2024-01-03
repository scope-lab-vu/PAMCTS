import random

class RandomAgent():

    ## Note: assumes that all actions are always possible
    def __init__(self, possible_actions):

        self.possible_actions = possible_actions


    def get_action(self, observation, env):

        return random.choice(self.possible_actions)
