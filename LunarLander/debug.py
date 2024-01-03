import os
import sys
import gymnasium as gym
import numpy as np
from network_weights.Alphazero_Networks.build_network import build_network

parent_pid = os.getpid()
child_pid = os.fork()
env = gym.make("LunarLander-v2", gravity=-10.0, enable_wind=True, wind_power=0.0, turbulence_power=0.0)
curr_state, _ = env.reset()
print(f"checkpoint 0")
network = build_network(num_hidden_layers=5, state_size=env.observation_space.shape[0], action_dim=env.action_space.n)
predictions = network.predict(x=np.array(curr_state).reshape((1, env.observation_space.shape[0])), batch_size=1, verbose=0)
print(f"checkpoint 1")
network.predict(x=np.array(curr_state).reshape((1, env.observation_space.shape[0])), batch_size=1, verbose=0)

if child_pid > 0:
    process = os.waitpid(child_pid, 0)
else:
    print(f"checkpoint 2")
    predictions = network.predict(x=np.array(curr_state).reshape((1, env.observation_space.shape[0])), batch_size=1, verbose=0)
    print(f"predictions: {predictions}")
    network2 = build_network(num_hidden_layers=5, state_size=env.observation_space.shape[0], action_dim=env.action_space.n)
    print(f"checkpoint 3")
    sys.exit(0)
