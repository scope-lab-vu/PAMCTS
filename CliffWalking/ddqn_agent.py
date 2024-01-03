import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import random
from stochastic_cliff_walking import StochasticCliffWalkingEnv


# Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
# Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
# DDQN Agent
class DDQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, batch_size=64, memory_size=10000, lr=0.001, gamma=0.99, tau=0.05):
        self.q_network = QNetwork(state_size, action_size, hidden_size).float()
        self.target_network = QNetwork(state_size, action_size, hidden_size).float()
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.action_size = action_size
        self.training_losses = []  # List to store training losses

    def select_action(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.choice(range(self.action_size))
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(dim=1).item()
            
    def get_q_values(self, state):
        x = int(state / 12)
        y = state % 12
        state = [x, y]
        # Convert the state to a tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Use the q_network to get the Q-values
        with torch.no_grad():
            q_values_tensor = self.q_network(state_tensor)
        
        # Convert the Q-values tensor to a Python list
        q_values_list = q_values_tensor[0].tolist()
        
        return q_values_list

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(batch.state)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1)
        next_state_batch = torch.FloatTensor(batch.next_state)
        done_batch = torch.BoolTensor(batch.done)

        q_values = self.q_network(state_batch).gather(1, action_batch)
        next_q_values = self.q_network(next_state_batch)
        best_actions = next_q_values.argmax(dim=1).unsqueeze(1)
        next_q_values_target = self.target_network(next_state_batch).gather(1, best_actions)
        expected_q_values = reward_batch + self.gamma * next_q_values_target * (~done_batch)

        loss = nn.functional.mse_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        loss = nn.functional.mse_loss(q_values, expected_q_values)
        self.training_losses.append(loss.item())  # Store the loss

    def validate(self, env, max_t=10):
        total_loss = 0.0
        for t_ in range(max_t):
            state = env.reset()
            done = False
            max_steps = 500
            count = 0
            while not done and count < max_steps:
                action = self.select_action(state, 0)  # Use epsilon = 0 for validation
                next_state, reward, done = env.step(action)
                # Compute loss (similar to train method, but without backward step)
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_tensor = torch.LongTensor([action]).unsqueeze(1)
                reward_tensor = torch.FloatTensor([reward]).unsqueeze(1)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                done_tensor = torch.BoolTensor([done])

                q_value = self.q_network(state_tensor).gather(1, action_tensor)
                next_q_value = self.q_network(next_state_tensor)
                best_action = next_q_value.argmax(dim=1).unsqueeze(1)
                next_q_value_target = self.target_network(next_state_tensor).gather(1, best_action)
                expected_q_value = reward_tensor + self.gamma * next_q_value_target * (~done_tensor)

                loss = nn.functional.mse_loss(q_value, expected_q_value)
                total_loss += loss.item()

                state = next_state
                count += 1

        return total_loss / max_t  # Return average loss


# Training loop
def train_ddqn(agent, env, num_episodes=10000, epsilon_start=1.0, epsilon_end=0.0001, epsilon_decay=0.9995, max_t=20):
    scores = []
    epsilons = []
    validation_losses = []
    epsilon = epsilon_start

    for episode in range(num_episodes):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.select_action(state, epsilon)
            next_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            score += reward
            if done:
                break

        scores.append(score)
        epsilons.append(epsilon)
        epsilon = max(epsilon_end, epsilon_decay * epsilon)  # decrease epsilon

        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes} | Score: {score} | Epsilon: {epsilon:.4f}")

        # print(f"episode: {episode}")
        # Perform validation every 1000 episodes
        if episode % 200 == 0:
            validation_loss = agent.validate(env, max_t=20)
            validation_losses.append(validation_loss)
            print(f"Validation Loss: {validation_loss:.4f}")
    
    # Save the weights
    torch.save(agent.q_network.state_dict(), "cliff_walking_ddqn_weights.pth")

    return scores, epsilons, agent.training_losses, validation_losses


if __name__ == "__main__":
    env = StochasticCliffWalkingEnv(slipperiness=0.0)
    agent = DDQNAgent(state_size=2, action_size=4)
    agent.q_network.load_state_dict(torch.load("cliff_walking_ddqn_weights.pth"))
    scores, epsilons, training_losses, validation_losses = train_ddqn(agent, env)

    # Optional: Plot scores and epsilons over episodes
    import matplotlib.pyplot as plt

    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(scores)
    # plt.title("Scores over Episodes")
    # plt.xlabel("Episode")
    # plt.ylabel("Score")

    # plt.subplot(1, 2, 2)
    # plt.plot(epsilons)
    # plt.title("Epsilon over Episodes")
    # plt.xlabel("Episode")
    # plt.ylabel("Epsilon")
    # plt.tight_layout()
    # # save the figure
    # plt.savefig("cliff_walking_ddqn_scores_epsilons.png")

    # # Plot for training and validation losses
    # plt.subplot(1, 3, 3)
    # plt.plot(training_losses, label='Training Loss')
    # plt.plot(np.linspace(0, len(training_losses), len(validation_losses)), validation_losses, label='Validation Loss')
    # plt.title("Training and Validation Losses")
    # plt.xlabel("Training Steps")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("cliff_walking_ddqn_performance.png")

    # Plot for scores and epsilons
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title("Scores over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Score")

    plt.subplot(1, 2, 2)
    plt.plot(epsilons)
    plt.title("Epsilon over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.tight_layout()
    plt.savefig("cliff_walking_ddqn_scores_epsilons.png")

    # Separate plot for training and validation losses
    plt.figure(figsize=(6, 4))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(np.linspace(0, len(training_losses), len(validation_losses)), validation_losses, label='Validation Loss')
    plt.title("Training and Validation Losses")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cliff_walking_ddqn_training_validation_losses.png")
