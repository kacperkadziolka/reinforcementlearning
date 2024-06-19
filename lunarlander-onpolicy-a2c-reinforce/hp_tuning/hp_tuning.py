import os

import numpy as np
import gym
from functools import reduce

from agents.actor_critic_agent import ActorCriticAgent
from agents.reinforce_agent import ReinforceAgent

bestAvgReward = float('-inf')
bestParameter = None

# Defining env
env = gym.make('LunarLander-v2')
state_space = env.observation_space.shape
state_n = reduce(lambda x, y: x * y, state_space)

# Defining fixed HP
action_n = env.action_space.n
hidden_n = 64
gamma = 0.99
episodes = 10000
eval_interval = 100
entropy_coefficient = 0.001
learning_rate = 0.005

# A2C HPs
use_bootstrap = True
#n_step = 3
use_baseline_subtraction = False

# Defining HP to tune
'We are first tuning the lr; the best lr will be used to tune the entropy coefficient.'
#learning_rate = [0.003]

'We are now tuning the n-step for bootstrapping on the best performed lr.'
n_step = [30, 50]

'We are now tuning the entropy_coef on the best performed lr.'
#entropy_coef = [0.001, 0.005, 0.01]

# Directory
directory = 'plot_vectors'
if not os.path.exists(directory):
    os.makedirs(directory)

print("Tuning hyperparameters...")
for parameter in n_step:
    print("The entropy_coef is:", parameter)

    agent = ActorCriticAgent(state_n, action_n, hidden_n, learning_rate, gamma, entropy_coefficient, episodes,
                             eval_interval, use_bootstrap, parameter, use_baseline_subtraction)
    #agent = ReinforceAgent(state_n, action_n, hidden_n, parameter, gamma, entropy_coefficient, episodes, eval_interval)

    episode_rewards = agent.fit(env)

    # Save the episode rewards vector on the computer for future usage
    episode_rewards = np.array(episode_rewards)

    file_name_reward = f'episode_rewards_nstep_{parameter}.npy'
    np.save(os.path.join(directory, file_name_reward), episode_rewards)

    #file_name_entropy = f'episode_rewards_entropy_{parameter}.npy'
    #np.save(os.path.join(directory, file_name_entropy), episode_rewards)

    # Evaluate and save results
    avg_episode_reward = np.mean(episode_rewards)

    if avg_episode_reward > bestAvgReward:
        bestAvgReward = avg_episode_reward
        bestParameter = parameter

print("Best parameter after hyperparameter tuning:")
print(bestParameter)
