import time

import numpy as np
import gym

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

# from rl.callbacks import WandbLogger

start_time = time.time()
ENV_NAME = 'FrozenLake-v1'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME, is_slippery=True, map_name="4x4", desc=None)
# env._max_episode_steps = 2500# 2500
# env.reward_threshold = 2500

# env.env.gravity = 9.8
# env.env.force_mag = 75.0 # 10.0
# env.env.length = 0.5
# env.env.masscart = 3.0 #1.0
# env.env.masspole = 0.1
# env.env.total_mass = 3.1 #1.1
# env.env.polemass_length = 0.05
# env.env.tau = 0.05 #0.02
# env.env.theta_threshold_radians = 0.20943951023931953
# env.env.x_threshold = 1.0 #2.4


# np.random.seed(123)
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
               gamma=.9,
               train_interval=4,
               enable_double_dqn=True,
               batch_size=64)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=5000000, visualize=False, verbose=1)

# After training is done, we save the final weights.
dqn.save_weights(f'duel_dqn_{ENV_NAME}_slip_weights_2.h5f', overwrite=True)
# dqn.load_weights(f'duel_dqn_{ENV_NAME}_slip_weights_2.h5f')

# dqn.load_weights(f'duel_dqn_{ENV_NAME}_weights_2500_computime.h5f')
# dqn.save_weights(f'duel_dqn_{ENV_NAME}_weights_500.h5f')

end_time = time.time()
# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=40, visualize=False)


# print(dqn/.compute_q_values(env.reset()))

# a = dqn.compute_q_values([env.reset()])
# best_action = np.argmax(a)
print('computation time: {}'.format(end_time - start_time))
print('done')