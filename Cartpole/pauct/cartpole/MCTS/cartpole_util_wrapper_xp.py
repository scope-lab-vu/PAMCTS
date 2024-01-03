import math

import gym

class ModifiedCartpoleWrapperXP():

    def __init__(self, x_threshold, max_steps=500):
        self.wrapped_env = gym.make('CartPole-v1')
        self.x_threshold = x_threshold
        self.action_space = self.wrapped_env.action_space
        self.wrapped_env._max_episode_steps = max_steps


    def reset(self):
        return self.wrapped_env.reset()

    def step(self, action):

        curr_state, base_reward, done, info = self.wrapped_env.step(action)

        # reward based on how close to center
        modified_reward = 1.0 - (abs(curr_state[0]) / self.x_threshold)

        return curr_state, modified_reward, done, info

    def render(self):
        self.wrapped_env.render()