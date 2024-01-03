import time

import numpy as np
import gym

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from customized_frozen_lake import Customized_FrozenLakeEnv


start_time = time.time()
ENV_NAME = 'FrozenLake-v1'
custom_map = [
    "SHF",
    "FFF",
    "HFG"
]

env = Customized_FrozenLakeEnv(desc=custom_map, is_slippery=True, customized_prob1=1.0/3.0,
                               customized_prob2=1.0/3.0, customized_prob3=1.0/3.0)
env._max_episode_steps = 200
env.seed(0)
nb_actions = env.action_space.n

# Next, we build a very simple model regardless of the dueling architecture
# if you enable dueling network in DQN , DQN will build a dueling network base on your model automatically
# Also, you can build a dueling network by yourself and turn off the dueling network in DQN.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
model.add(Dense(nb_actions, activation='linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
# even the metrics!
# memory = SequentialMemory(limit=50000, window_length=1)
memory = SequentialMemory(limit=10000, window_length=1)
policy = BoltzmannQPolicy()
# policy = EpsGreedyQPolicy(eps=0.1)
# enable the dueling network

# you can specify the dueling_type to one of {'avg','max','naive'}
dqn = DQNAgent(model=model,
               nb_actions=nb_actions,
               memory=memory,
               nb_steps_warmup=500,
               # enable_dueling_network=True,
               # dueling_type='avg',
               target_model_update=1e-2,
               policy=policy,
               gamma=.95,
               train_interval=4,
               enable_double_dqn=True,
               batch_size=64)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.load_weights("duel_dqn_FrozenLake-v1_slip_weights_reorder.h5f")
dqn.fit(env, nb_steps=1000000, visualize=False, verbose=1)

# After training is done, we save the final weights.
dqn.save_weights(f'duel_dqn_{ENV_NAME}_slip_weights_retrain_reorder.h5f', overwrite=True)

end_time = time.time()
# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=100, visualize=False)

print('computation time: {}'.format(end_time - start_time))
print('done')
# write computation time to file
with open('reorder_retrain_computation_time.txt', 'a') as f:
    f.write('reorder retrain computation time: {}'.format(end_time - start_time))
