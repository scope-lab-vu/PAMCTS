import numpy as np


class StochasticCliffWalkingEnv:
    def __init__(self, shape=[4, 12], slipperiness=0):
        self.shape = shape
        self.start = [3, 0]
        self.goal = [3, 11]
        self.slipperiness = slipperiness
        self.state = self.start
        self.terminated = False

    def reset(self):
        self.state = self.start
        self.terminated = False
        return self.state
    
    def render(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if [i, j] == self.state:
                    print("A ", end='')
                elif [i, j] == self.goal:
                    print("G ", end='')
                elif i == 3 and 1 <= j <= 10:
                    print("C ", end='')
                else:
                    print(". ", end='')
            print()  # New line for next row

    def step(self, action):
        if self.terminated:
            return self.state, 0, True

        x, y = self.state
        if action == 3:  # UP
            if np.random.uniform() < self.slipperiness:
                x = min(x + 1, self.shape[0] - 1)
            else:
                x = max(x - 1, 0)
        elif action == 2:  # RIGHT
            if np.random.uniform() < self.slipperiness:
                x = min(x + 1, self.shape[0] - 1)
            else:
                y = min(y + 1, self.shape[1] - 1)
        elif action == 1:  # DOWN
            x = min(x + 1, self.shape[0] - 1)
        elif action == 0:  # LEFT
            if np.random.uniform() < self.slipperiness:
                x = min(x + 1, self.shape[0] - 1)
            else:
                y = max(y - 1, 0)

        self.state = [x, y]

        # Check if agent is in the cliff region
        if x == 3 and 1 <= y <= 10:
            self.terminated = True
            return self.state, -100, True

        # Check if agent reached the goal
        if [x, y] == self.goal:
            self.terminated = True
            return self.state, 1, True

        return self.state, -1, False
    
    def get_state_index(self, state):
        return state[0] * self.shape[1] + state[1]
