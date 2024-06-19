import torch
from torch import optim

from agents.base_agent import BaseAgent
from torch.distributions import Categorical
from tqdm import tqdm

from models.policy_network import PolicyNetwork


class ReinforceAgent(BaseAgent):
    def __init__(self, state_n: int, action_n: int, hidden_n: int, learning_rate: float, gamma: float,
                 entropy_coef: float, episodes: int, eval_interval: int):
        super().__init__(state_n, action_n, hidden_n, learning_rate, gamma, entropy_coef, episodes, eval_interval)

        # Policy and optimizer settings
        self.policy = PolicyNetwork(self.state_n, self.action_n, self.hidden_n).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        # Entropy regularization
        self.entropies = []

        # Set up for saving and loading actions and rewards
        self.saved_log_probabilities = []
        self.rewards = []

    def select_action(self, state, test=False):
        """
        - Generates a probability distribution over possible actions
        - Samples an action based on this distribution
        - Keeps track of the decision-making progress
        """
        self.policy.eval()  # Evaluate the policy
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probabilities = self.policy(state)
        m = Categorical(action_probabilities)
        action = m.sample()

        'When we evaluate the model we do not want to save any other information'
        if not test:
            self.saved_log_probabilities.append(m.log_prob(action))
            self.entropies.append(m.entropy())
        return action.item()

    def calculate_returns(self):
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        return returns

    def policy_update(self, returns):
        policy_loss = []
        entropy_term = []

        for log_probability, R, entropy in zip(self.saved_log_probabilities, returns, self.entropies):
            policy_loss.append(-log_probability * R) # minus for gradient ASCENT
            entropy_term.append(entropy)

        # Update the policy network taking into account the entropy
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        entropy_loss = torch.cat(entropy_term).sum()
        total_loss = policy_loss - entropy_loss * self.entropy_coef
        total_loss.backward()
        self.optimizer.step()

        # we clear the log_probabilities, the rewards and entropies vectors
        self.saved_log_probabilities = []
        self.rewards = []
        self.entropies = []

    def finish_episode(self):
        returns = self.calculate_returns()
        self.policy_update(returns)

    def train(self, env):
        """
        Train for only one episode
        """
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            'Select action'
            action = self.select_action(state)

            'Take the action and observe the next state and reward'
            next_state, reward, done, _ = env.step(action)

            state = next_state
            self.rewards.append(reward)
            episode_reward += reward
        self.finish_episode()

        return episode_reward

    def fit(self, env):
        """"
        Fit the model on the given number of episodes and test it after each test_evaluation interval
        """
        total_rewards = []
        best_test_reward = 0

        for i in tqdm(range(1, self.episodes + 1)):
            episode_reward = self.train(env)
            total_rewards.append(episode_reward)

            if i % self.eval_interval == 0:
                test_reward = self.test(env)
                print(f"Episode {i}: average training rewards until now: {sum(total_rewards)/ i}")
                print(f"Episode {i}: the test reward is = {test_reward}")

                if best_test_reward < test_reward:
                    best_test_reward = test_reward

        print(f"Average reward after training: {sum(total_rewards)/ self.episodes}")
        print(f"Best test reward: {best_test_reward}")
        return total_rewards
