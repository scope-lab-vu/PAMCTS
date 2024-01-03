from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import numpy as np


class DDQN_Learning_Agent():
    def __init__(self, number_of_actions, env_obs_space_shape):
        # general info
        self.optimizer = Adam(learning_rate=0.0001)
        self.batch_size = 64
        self.state_size = env_obs_space_shape[0]

        # build policy network
        self.brain_policy = Sequential()
        self.brain_policy.add(Dense(128, input_dim=self.state_size, activation="relu"))
        self.brain_policy.add(Dense(128, activation="relu"))
        self.brain_policy.add(Dense(number_of_actions, activation="linear"))
        self.brain_policy.compile(loss="mse", optimizer=self.optimizer)

    def load_saved_weights(self, weights_path):
        self.brain_policy.load_weights(weights_path)

    def get_q_values(self, observation):
        input_state = np.reshape(observation, (1, 8))
        return self.brain_policy.predict(input_state)

