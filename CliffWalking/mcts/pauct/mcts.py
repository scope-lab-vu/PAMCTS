import math
import numpy as np
from copy import deepcopy


# class MCTS:
#     def __init__(self, gamma = 0.99, num_iter=10000, max_depth=500, C=50.0):
#         self.gamma = gamma
#         self.num_iterations = num_iter
#         self.max_depth = max_depth
#         self.C = C
#         self.tree = []

#     def selection_uct(self, curr_state):
#         state_n = None
#         state_v = None
#         for node in self.tree:
#             # compare two numpy arrays
#             if node["obs"] == curr_state:
#                 state_n = np.array(node["num_visits"])
#                 state_v = np.array(node["value"])
#                 break
#         N = sum(state_n)
#         return np.argmax(state_v + self.c_puct * np.sqrt(math.log(N) / state_n))
    
#     def get_action(self, env):
#         starting_state = env.state
#         self.tree.append({"obs": starting_state, "value": [0, 0, 0, 0], "num_visits": [1, 1, 1, 1]})
#         for iteration in range(self.num_iterations):
#             sa_trajectory = []
#             reward_trajectory = []
#             state = starting_state
#             terminated = False
#             depth = 0
#             env_copy = deepcopy(env)
#             expand_bool = False
#             while not terminated and depth < self.max_depth:
#                 action = self.selection_uct(curr_state=state)
#                 sa_trajectory.append((state, action))
#                 state, reward, terminated = env_copy.step(action)
#                 reward_trajectory.append(reward)
#                 depth += 1
#                 found = False
#                 for node in self.tree:
#                     if node["obs"] == state:
#                         found = True
#                         break
#                 if not found:
#                     expand_bool = True
#                     break

#             if expand_bool:
#                 self.tree.append({"obs": state, "value": [0, 0, 0, 0], "num_visits": [1, 1, 1, 1]})
#                 leaf_rollout_return = 0
#                 leaf_rollout_depth = 0
#                 if 
#                 while not terminated and depth < self.max_depth:
#                     action = np.random.randint(0, 4)
#                     state, reward, terminated, _, _ = env.step(action)



