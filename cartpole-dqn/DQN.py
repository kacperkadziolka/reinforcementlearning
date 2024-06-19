import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from tqdm import tqdm
from model import DQN


# Define the DQN agent
class DQNAgent:
    def __init__(self, state_n=4, action_n=2, hidden_n=64, learning_rate=0.001, gamma=0.99, memory_size=10000,
                 batch_size=32, target_update_interval=100, episodes=1000, policy='e-greedy', epsilon=1.0,
                 eval_interval=100, temperature=0.2, use_memory=True, use_target_network=True):

        # NN Model hyperparameters
        self.state_n = state_n
        self.action_n = action_n
        self.hidden_n = hidden_n
        self.learning_rate = learning_rate

        # DQN hyperparameters
        self.policy = policy
        self.gamma = gamma
        self.epsilon = epsilon
        self.temperature = temperature
        self.target_update_interval = target_update_interval
        self.episodes = episodes
        self.eval_interval = eval_interval

        # Boolean values to run experiments
        self.use_memory = use_memory
        self.use_target_network = use_target_network

        # Torch settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_model_nn = DQN(state_n, action_n, hidden_n).to(self.device)
        if self.use_target_network:
            self.target_model_nn = DQN(state_n, action_n, hidden_n).to(self.device)
            self.target_model_nn.load_state_dict(self.policy_model_nn.state_dict())
            self.target_model_nn.eval()
        self.optimizer = optim.Adam(self.policy_model_nn.parameters(), lr=self.learning_rate)

        # Experience replay settings
        self.memory_size = memory_size
        if self.use_memory:
            self.memory = deque(maxlen=self.memory_size)
        self.batch_size = batch_size
        self.batch = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # state = torch.tensor(state, dtype=torch.float32, device=self.device)
        q_values = self.policy_model_nn(state)

        if self.policy == 'e-greedy':
            if np.random.rand() < self.epsilon:
                action = np.random.randint(self.action_n)
            else:
                action = torch.argmax(q_values).item()
            return action

        if self.policy == 'softmax':
            # action = torch.softmax(q_values / self.temperature, dim=-1).data.numpy().squeeze().to(self.device)
            action = torch.softmax(q_values / self.temperature, dim=-1).cpu().data.numpy().squeeze()
            action = np.random.choice(self.action_n, p=action)
            return action

    def update(self):
        if self.use_memory and len(self.memory) < self.batch_size:
            return

        if self.use_memory:
            batch = random.sample(self.memory, self.batch_size)
        else:
            batch = self.batch

        states, actions, rewards, next_states, dones = zip(*batch)

        "Create from numpy arrays, torch tensor and make them 2D to work with batches"
        states = torch.FloatTensor(np.float32(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.float32(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        """
        The gather operation is used here to index the output tensor along the first dimension (the action dimension),
        using the indices provided by action. 
        This results in a tensor of Q-values specifically for the actions that were actually taken, 
        which can be used for calculating the loss during training and updating the network.
        """
        current_Q_values = self.policy_model_nn(states).gather(1, actions)
        if self.use_target_network:
            # find the maximum Q-value among all possible actions in the next state,
            # providing the best action's value according to the target network.
            next_Q_values = self.target_model_nn(next_states).max(1)[0].unsqueeze(1)
        else:
            # find the maximum Q-value among all possible actions in the next state,
            # providing the best action's value according to the current network.
            next_Q_values = self.policy_model_nn(next_states).max(1)[0].unsqueeze(1)

        target_Q_values = rewards + self.gamma * next_Q_values * (1 - dones)

        # Update the policy network
        loss = nn.functional.smooth_l1_loss(current_Q_values, target_Q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_memory(self, state, action, reward, next_state, done):
        if self.use_memory:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.batch = [(state, action, reward, next_state, done)]

    def train(self, env):
        """
        Train the network only for single episode
        """
        current_state, _ = env.reset()
        done = False

        total_reward = 0
        total_steps = 0

        while not done:
            # Select the action using an epsilon-greedy policy
            action = self.select_action(current_state)

            # Take the action and observe the next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store the experience in replay memory, and update policy and target networks
            self.update_memory(current_state, action, reward, next_state, done)
            self.update()

            current_state = next_state
            total_reward += reward
            total_steps += 1

        return total_reward, total_steps

    def test(self, env):
        """
        Evaluate the model
        """
        current_state, _ = env.reset()
        done = False

        total_reward = 0
        total_steps = 0

        while not done:
            action = self.select_action(current_state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            current_state = next_state
            total_reward += reward
            total_steps += 1

        return total_reward, total_steps

    def fit(self, env):
        """
        Fit the model on given number of episodes and test it after each test_evaluation interval
        """
        best_test_reward = 0
        episode_rewards = []

        for i in tqdm(range(self.episodes)):
            total_reward, total_steps = self.train(env)

            # print(f"Episode {i}: total reward = {total_reward}, total steps = {total_steps}")

            episode_rewards.append(total_reward)

            if i % self.target_update_interval == 0 and self.use_target_network:
                self.target_model_nn.load_state_dict(self.policy_model_nn.state_dict())
                self.target_model_nn.eval()

            if i % self.eval_interval == 0:
                avg_reward, avg_steps = self.test(env)
                print(f"Episode {i}: average reward test = {avg_reward}, average steps test = {avg_steps}")

                if avg_reward > best_test_reward:
                    print(f"New best test reward: {best_test_reward}. Saving model...")

                    best_test_reward = avg_reward
                    torch.save(self.policy_model_nn.state_dict(), 'best_model.pth')

        print(f"Average reward after training: {sum(episode_rewards) / self.episodes}")
        print(f"Best test reward: {best_test_reward}")
        return episode_rewards
