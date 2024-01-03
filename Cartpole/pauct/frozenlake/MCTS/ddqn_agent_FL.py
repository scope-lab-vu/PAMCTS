from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

import numpy as np

class DDQN_Learning_Agent_FL():

    def __init__(self,
                 number_of_actions,
                 env_obs_space_shape):

        model = Sequential()
        model.add(Flatten(input_shape=(1,) + env_obs_space_shape))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        # model.add(Dense(16))
        # model.add(Activation('relu'))
        model.add(Dense(number_of_actions, activation='linear'))
        print(model.summary())

        memory = SequentialMemory(limit=10000, window_length=1)
        # policy = BoltzmannQPolicy()
        policy = EpsGreedyQPolicy(eps=0.1)
        self.dqn_agent = DQNAgent(model=model,
                       nb_actions=number_of_actions,
                       memory=memory,
                       nb_steps_warmup=500,
                       # enable_dueling_network=True,
                       # dueling_type='avg',
                       target_model_update=1e-2,
                       policy=policy,
                       gamma=.9,
                       train_interval=4,
                       enable_double_dqn=True,
                       batch_size=64)
        self.dqn_agent.compile(Adam(learning_rate=1e-3), metrics=['mae'])

    def train_agent(self, env, nb_steps=1000000, visualize=False, verbose=2):
        self.dqn_agent.fit(env, nb_steps=nb_steps, visualize=visualize, verbose=verbose)

    def save_model_weights(self, save_file_path, overwrite=True):
        self.dqn_agent.save_weights(save_file_path, overwrite=overwrite)

    def load_saved_weights(self, saved_weights_filepath):
        self.dqn_agent.load_weights(saved_weights_filepath) # (f'duel_dqn_{ENV_NAME}_weights_2500.h5f)

    def get_q_values(self, observation):
        return self.dqn_agent.compute_q_values([observation])

    def get_greedy_action(self, observation):
        q_values = self.get_q_values(observation)
        action = np.argmax(q_values)
        return action

    def get_action(self, observation, env=None):
        return self.get_greedy_action(observation)

    def run_tests(self, env, nb_episodes=20, visualize=True):
        self.dqn_agent.test(env=env, nb_episodes=nb_episodes, visualize=visualize)
