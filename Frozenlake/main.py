import logging
import gym
from env.customized_frozen_lake import FrozenLakeEnv_2_3_1_6
from env.frozen_lake_9_10_1_20 import FrozenLakeEnv_9_10_1_20

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO,
                    handlers=[logging.FileHandler(f"logs/frozenlake_test", mode='w')])
logger = logging.getLogger()

custom_map = [
    "SHF",
    "FFF",
    "HFG"
]

if __name__ == '__main__':
    counter = 0
    counter_4 = 0
    counter_2 = 0
    counter_8 = 0
    while True:
        counter += 1
        # env = FrozenLakeEnv_2_3_1_6(desc=custom_map, is_slippery=True)
        env = FrozenLakeEnv_9_10_1_20(desc=custom_map, is_slippery=True)
        env.reset()
        terminated = False
        env.s = 5
        action = 0
        curr_state, reward, _, _ = env.step(action)
        if curr_state == 4:
            counter_4 += 1
        elif curr_state == 2:
            counter_2 += 1
        elif curr_state == 8:
            counter_8 += 1
        if counter == 200:
            print(f"4: {counter_4}, 2: {counter_2}, 8: {counter_8}")
