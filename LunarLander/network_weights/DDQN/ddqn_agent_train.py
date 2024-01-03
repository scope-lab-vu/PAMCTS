# Landing pad is always at coordinates (0,0). Coordinates are the first
# two numbers in state vector. Reward for moving from the top of the screen
# to landing pad and zero speed is about 100..140 points. If lander moves
# away from landing pad it loses reward back. Episode finishes if the lander
# crashes or comes to rest, receiving additional -100 or +100 points.
# Each leg ground contact is +10. Firing main engine is -0.3 points each frame.
# Solved is 200 points. Landing outside landing pad is possible. Fuel is
# infinite, so an agent can learn to fly and then land on its first attempt.
# Four discrete actions available: do nothing, fire left orientation engine,
# fire main engine, fire right orientation engine.
import os
import platform
import gym
import random
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.activations import relu, linear

import numpy as np
import pandas as pd

if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)


class DQN:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1.0
        self.gamma = .99
        self.batch_size = 64
        self.epsilon_min = .01
        self.lr = 0.001
        self.epsilon_decay = .996
        self.memory = deque(maxlen=1000000)
        self.model = self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Dense(150, input_dim=self.state_space, activation=relu))
        model.add(Dense(120, activation=relu))
        model.add(Dense(self.action_space, activation=linear))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        history = self.model.fit(states, targets_full, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        return loss

    def save_model(self):
        # save the model to disk
        self.model.save_weights("ddqn_LunarLander-v2_weights.h5f")

    def load_model(self):
        # load the model from disk
        # if the file exists, load the model
        if os.path.exists("ddqn_LunarLander-v2_weights.h5f") or os.path.isfile("ddqn_LunarLander-v2_weights.h5f.index"):
            print(f"load the weights from file")
            self.model.load_weights("ddqn_LunarLander-v2_weights.h5f")


def train_dqn(episode, csv_file="ddqn_LunarLander-v2.csv"):
    loss_list = []
    validation_scores = []
    total_steps_count = 0
    step_losses = []
    validation_losses = []
    agent = DQN(env.action_space.n, env.observation_space.shape[0])
    agent.load_model()
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, 8))
        score = 0
        max_steps = 1000
        steps_in_episode = 0
        for i in range(max_steps):
            action = agent.act(state)
            # env.render()
            next_state, reward, done, _ = env.step(action)
            total_steps_count += 1
            steps_in_episode += 1
            score += reward
            next_state = np.reshape(next_state, (1, 8))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                step_losses.append((total_steps_count, loss))
            if total_steps_count % 5000 == 0:
                # Perform validation and record validation loss
                validation_loss = perform_validation(agent, env)
                validation_losses.append((total_steps_count, validation_loss))
            if done or steps_in_episode >= max_steps:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                # save episode e and total steps count into csv file
                if os.path.exists(csv_file):
                    with open(csv_file, "a") as f:
                        pd.DataFrame([[e, total_steps_count, score]]).to_csv(f, header=False, index=False)
                else:
                    with open(csv_file, "w") as f:
                        pd.DataFrame([[e, total_steps_count, score]]).to_csv(f, header=["episode", "total_steps_count", "score"], index=False)
                break
        loss_list.append(score)

        # Average score of last 100 episode
        is_solved = np.mean(loss_list[-100:])
        if total_steps_count > 300000:
            print('\n Task Completed! \n')
            break
        # if is_solved > 200:
        #     print('\n Task Completed! \n')
        #     break
        print(f"total_steps_count: {total_steps_count}")
        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
        agent.save_model()
    return loss_list, step_losses, validation_losses


# Add a function for performing validation
def perform_validation(agent, env, validation_episodes=1):
    total_loss = 0
    target_score = 200  # Define a target score
    for _ in range(validation_episodes):
        state = env.reset()
        state = np.reshape(state, (1, 8))
        score = 0
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, 8))
            state = next_state
            if done:
                break
        # Calculate the loss as the deviation from the target score
        loss = abs(score - target_score)
        total_loss += loss
    return total_loss / validation_episodes


if __name__ == '__main__':

    print(env.observation_space)
    print(env.action_space)
    episodes = 400
    loss, step_losses, validation_scores = train_dqn(episodes)
    plt.plot([i+1 for i in range(0, len(loss), 2)], loss[::2])
    plt.savefig('ddqn_LunarLander-v2.png')

    # Extract step numbers and loss values for plotting
    steps, losses = zip(*step_losses)
    validation_steps, validation_losses = zip(*validation_scores)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label='Training Loss')
    plt.plot(validation_steps, validation_losses, label='Validation Loss')
    plt.xlabel('Number of Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Steps')
    plt.legend()
    plt.savefig('ddqn_LunarLander-v2_performance.png')