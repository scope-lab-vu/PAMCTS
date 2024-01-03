import numpy as np
import gymnasium as gym
import torch
from stochastic_cliff_walking import StochasticCliffWalkingEnv
from ddqn_agent import DDQNAgent
from mcts.pauct.mcts import MCTS


if __name__ == "__main__":
    env = StochasticCliffWalkingEnv(shape=[4,12], slipperiness=0.0)
    state = env.reset()
    terminated = False
    print(f"initial state: {state}")
    env.render()
    ddqn_agent = DDQNAgent(state_size=2, action_size=4)
    ddqn_agent.q_network.load_state_dict(torch.load("cliff_walking_ddqn_weights.pth"))
    while not terminated:
        action = ddqn_agent.select_action(state, epsilon=0.0)
        state, reward, terminated = env.step(action)
        print(f"state: {state}, action: {action}, reward: {reward}, terminated: {terminated}")
        env.render()
