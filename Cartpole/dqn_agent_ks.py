from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Input
from tensorflow.keras.optimizers import Adam, SGD

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, GreedyQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger

import json
import keras
import numpy as np
import statistics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class DQN_Learning_Agent_KS():

    def __init__(self,
                 number_of_actions,
                 env_obs_space_shape,
                 graph_size):

        size_hidden_layer = round(statistics.mean([number_of_actions, graph_size]))

        # model = Sequential()
        # model.add(Input(shape=env_obs_space_shape))
        # model.add(Dense(24, activation='relu'))
        # model.add(Dense(24, activation='relu'))
        # model.add(Dense(number_of_actions, activation='linear'))

        model = Sequential()
        model.add(Flatten(input_shape=(1,) + env_obs_space_shape))
        # model.add(Dense(512, activation='relu'))
        # model.add(Dense(512, activation='relu'))
        model.add(Dense(1024, activation='relu')) # 1024 for toy
        # model.add(Dense(1024, activation='sigmoid'))
        # model.add(Dense(256, activation='relu'))
        model.add(Dense(number_of_actions, activation='linear'))
        print(model.summary())

        ''''
        sigmoid results (current): 
        Episode 1: reward: -676.000, steps: 500
        Episode 2: reward: -725.000, steps: 500
        Episode 3: reward: -721.000, steps: 500
        Episode 4: reward: -721.000, steps: 500
        Episode 5: reward: -669.000, steps: 500
        Episode 6: reward: -721.000, steps: 500
        Episode 7: reward: -682.000, steps: 500
        Episode 8: reward: -714.000, steps: 500
        Episode 9: reward: -707.000, steps: 500
        Episode 10: reward: -719.000, steps: 500
        Episode 11: reward: -721.000, steps: 500
        Episode 12: reward: -660.000, steps: 500
        Episode 13: reward: -711.000, steps: 500
        Episode 14: reward: -715.000, steps: 500
        Episode 15: reward: -734.000, steps: 500
        Episode 16: reward: -730.000, steps: 500
        Episode 17: reward: -737.000, steps: 500
        Episode 18: reward: -681.000, steps: 500
        Episode 19: reward: -708.000, steps: 500
        Episode 20: reward: -680.000, steps: 500
        '''

        # model = Sequential()
        # model.add(Flatten(input_shape=(1,) + env_obs_space_shape))
        # model.add(Dense(64))
        # model.add(Activation('sigmoid'))
        # model.add(Dense(number_of_actions, activation='linear'))
        # print(model.summary())

        # model = Sequential()
        # model.add(Flatten(input_shape=(1,) + env_obs_space_shape))
        # model.add(Dense(16))
        # model.add(Activation('relu'))
        # # model.add(Activation('sigmoid'))
        # model.add(Dense(16))
        # model.add(Activation('relu'))
        # model.add(Dense(16))
        # model.add(Activation('relu'))
        # model.add(Dense(number_of_actions, activation='linear'))
        # print(model.summary())


        '''
        one hot encoding and embedding 
        
        https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/:
         - "if you wish to connect a Dense layer directly to an Embedding layer, you must first flatten the 2D output matrix to a 1D vector using the Flatten layer."
        
        https://keras.io/api/layers/core_layers/embedding/
        https://stats.stackexchange.com/questions/270546/how-does-keras-embedding-layer-work
        https://towardsdatascience.com/deep-learning-4-embedding-layers-f9a02d55ac12
        
        
        general q stuff: 
        https://towardsdatascience.com/reinforcement-learning-explained-visually-part-5-deep-q-networks-step-by-step-5a5317197f4b
        https://towardsdatascience.com/dueling-deep-q-networks-81ffab672751
        https://www.tensorflow.org/agents/tutorials/0_intro_rl
        https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc
        
        '''
        # memory = SequentialMemory(limit=5000, window_length=1)
        memory = SequentialMemory(limit=100000, window_length=1)
        policy = EpsGreedyQPolicy(eps=0.2) # GreedyQPolicy() # BoltzmannQPolicy()
        self.dqn_agent = DQNAgent(model=model,
                                  nb_actions=number_of_actions,
                                  memory=memory,
                                  nb_steps_warmup=1000, # 1000
                                  # enable_dueling_network=True,
                                  # dueling_type='avg',
                                  target_model_update=100, #1e-3, # 100
                                  policy=policy,
                                  gamma=.9, #.99
                                  # train_interval=4,
                                  enable_double_dqn=True,
                                  batch_size=30) # 30
        # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=1e-2,
        #     decay_steps=10000,
        #     decay_rate=0.9)

        # self.dqn_agent.compile(Adam(learning_rate=lr_schedule), metrics=['mae'])

        self.dqn_agent.compile(Adam(learning_rate=1e-3), metrics=['mae'])
        # self.dqn_agent.compile(Adam(learning_rate=1e-2), metrics=['mae'])

        # decay set to (learning rate / num episodes): 0.01/200000
        # self.dqn_agent.compile(SGD(learning_rate=0.01, decay=0.00000005, momentum=0.9, nesterov=False), metrics=['mae'])

    def train_agent(self, env, nb_steps=100000, visualize=False, verbose=2):
        callbacks = [FileLogger('testing_log_file.json', interval=10)]

        a = self.dqn_agent.fit(env, nb_steps=nb_steps, visualize=visualize, verbose=verbose, callbacks=callbacks)

        # logs = json.load('testing_log_file.json')
        logs = pd.read_json('testing_log_file.json')
        logs.reset_index(inplace=True)
        sns.lineplot(data=logs, x='episode', y='loss')
        plt.show()

        sns.lineplot(data=logs, x='episode', y='mean_q')
        plt.show()

        # print('a')

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

    def run_tests(self, env, nb_episodes=20, visualize=False):
        self.dqn_agent.test(env=env, nb_episodes=nb_episodes, visualize=visualize)

    def make_thread_safe(self):
        self.dqn_agent.make_predict_function()
