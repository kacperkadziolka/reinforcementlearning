from collections import namedtuple

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from agents.base_agent import BaseAgent
from models.actor_critic_network import ActorCriticModel


SavedAction = namedtuple('SavedAction', ['log_prob', 'value', 'entropy'])

class ActorCriticAgent(BaseAgent):
    def __init__(self, state_n: int, action_n: int, hidden_n: int, learning_rate: float, gamma: float,
                 entropy_coef: float, episodes: int, eval_interval: int, use_bootstrap: bool, n_step: int,
                 use_baseline_subtraction: bool):
        """
        Initializes an Actor-Critic agent with specified hyperparameters, training parameters,
        and architecture settings.

        :param use_bootstrap: Flag to use bootstrapping for approximating the n-step returns.
        :param n_step: The number of steps to look ahead for calculating returns.
        :param use_baseline_subtraction: Flag to use baseline subtraction to reduce variance.

        :attribute model: The combined actor-critic model.
        :attribute optimizer: The optimizer for training the model.
        """
        super().__init__(state_n, action_n, hidden_n, learning_rate, gamma, entropy_coef, episodes, eval_interval)

        # Actor / Critic networks and optimizer settings
        self.model = ActorCriticModel(state_n, action_n, hidden_n)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Bootstrapping and baseline subtraction parameters
        self.use_bootstrap = use_bootstrap
        self.n_step = n_step
        self.use_baseline_subtraction = use_baseline_subtraction

    def select_action(self, state, test=False):
        """
        Selects an action based on the current state using the policy learned by the agent.

        This function converts the state to a PyTorch tensor and passes it through the model
        to obtain action probabilities and a state value estimate. It then samples an action
        using these probabilities, facilitated by a categorical distribution.
        """
        state = torch.from_numpy(state).float().to(self.device)
        probs, state_value = self.model(state)

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

        # sample an action using the distribution
        action = m.sample()
        entropy = m.entropy()

        # save it to the action buffer
        self.model.saved_actions.append(SavedAction(m.log_prob(action), state_value, entropy))

        # the action to take (left or right)
        return action.item()

    def optimize(self):
        """
        Calculates actor and critic loss and performs backprop.

        Allows optional n-step bootstrapping, with following parameters:
        - use_bootstrap: Boolean, if True uses n-step bootstrapping, otherwise uses standard calculation.
        - n: The number of steps to bootstrap, used only when use_bootstrap is True.

        - use_baseline: Boolean, if True uses baseline subtraction, otherwise uses standard calculation.
        """
        saved_actions = self.model.saved_actions
        policy_losses = []
        value_losses = []
        returns = []
        entropy_losses = []

        # extract rewards and values, which are used for bootstrapping(values)
        rewards = self.model.rewards
        values = [sa.value for sa in saved_actions]

        if self.use_bootstrap:
            # calculate bootstrapped n-step returns
            for i in range(len(rewards)):
                R = sum(self.gamma ** k * rewards[i + k] for k in range(self.n_step) if i + k < len(rewards))
                if i + self.n_step < len(rewards):
                    R += self.gamma ** self.n_step * values[i + self.n_step].item()
                returns.append(R)
        else:
            # calculate the discounted value without bootstrapping
            R = 0
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.append(R)
            returns.reverse()

        # normalize rewards
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value, entropy), R in zip(saved_actions, returns):
            if self.use_baseline_subtraction:
                # Using value as baseline
                advantage = R - value.item()
            else:
                # No baseline subtraction
                advantage = R

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

            # Accumulate entropy for regularization
            entropy_losses.append(entropy)

        # Include entropy regularization in the total loss
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum() - torch.stack(entropy_losses).sum() * self.entropy_coef

        # reset gradients, sum losses, perform backpropagation, and reset buffers
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        del self.model.rewards[:]
        del self.model.saved_actions[:]

    def train(self, env):
        """
        Train for only one episode
        """
        state = env.reset()
        episode_reward = 0

        # Run for maximum 9999 steps, to avoid infinite loop error
        for i in range(1, 10000):
            action = self.select_action(state)
            state, reward, done, _ = env.step(action)

            self.model.rewards.append(reward)
            episode_reward += reward

            if done:
                break

        return episode_reward

    def fit(self, env):
        """
        Fit the model on the given number of episodes and test it after each evaluation interval.

        The Lunar Lander problem is considered solved when an average of 200 points is
        achieved over 100 consecutive runs.
        """
        running_reward = 10
        total_rewards = []
        mean_rewards = []

        for i_episode in tqdm(range(1, self.episodes + 1)):
            episode_reward = self.train(env)
            total_rewards.append(episode_reward)

            # update cumulative reward with putting more weight on the most recent episodes
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

            # perform network backpropagation
            self.optimize()

            # print the training values
            if i_episode % self.eval_interval == 0:
                #test_reward = self.test(env)
                print('\nEpisode {}\tLast episode reward: {:.2f}\tAverage reward: {:.2f}'
                      '\tAverage training reward until now: {:.2f}'
                      .format(i_episode, episode_reward, running_reward, sum(total_rewards) / i_episode))

            # break if we solved the problem
            # if len(total_rewards) >= 100:
            #     mean_last_100 = sum(total_rewards[-100:]) / 100
            #     mean_rewards.append(mean_last_100)
            #
            #     if mean_last_100 > 200:
            #         print(f'\nMean of last 100 episode rewards exceeds 200 ({mean_last_100}). '
            #               'Problem solved, stopping training.')
            #         break

        return total_rewards
