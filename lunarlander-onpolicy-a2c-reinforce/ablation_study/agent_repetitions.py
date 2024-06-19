import numpy as np
import time
from agents.reinforce_agent import ReinforceAgent
from agents.actor_critic_agent import ActorCriticAgent
import gym
from functools import reduce
import os


def run_repetitions_REINFORCE(env, state_n, action_n, hidden_n, learning_rate, gamma,
                              episodes, eval_interval, entropy_coefficient, n_repetitions):
    now = time.time()

    directory = 'figures/reinforce'
    if not os.path.exists(directory):
        os.makedirs(directory)

    print("Running REINFORCE...")
    for rep in range(n_repetitions):  # Loop over repetitions

        agent = ReinforceAgent(state_n, action_n, hidden_n, learning_rate, gamma, entropy_coefficient, episodes,
                               eval_interval)
        episode_rewards = agent.fit(env)

        # Save the episode rewards vector on the computer for future usage
        episode_rewards = np.array(episode_rewards)
        file_name_reward = f'episode_rewards_REINFORCE_rep_{rep + 1}.npy'
        np.save(os.path.join(directory, file_name_reward), episode_rewards)

    print("All experiments were saved.")
    print("This experiment took {} minutes".format((time.time() - now) / 60))


def run_repetitions_ActorCritic(env, state_n, action_n, hidden_n, learning_rate, gamma, entropy_coefficient, episodes,
                                eval_interval, use_bootstrapping, n_step, use_baseline, n_repetitions):
    now = time.time()

    directory = 'figures/actor_critic'
    if not os.path.exists(directory):
        os.makedirs(directory)

    print("Running ActorCritic...")
    for rep in range(n_repetitions):  # Loop over repetitions

        if use_bootstrapping and not use_baseline:
            print("ActorCritic with Bootstrapping repetition {}".format(rep + 1))
            agent = ActorCriticAgent(state_n, action_n, hidden_n, learning_rate, gamma, entropy_coefficient, episodes,
                                     eval_interval, use_bootstrapping, n_step, use_baseline)
            episode_rewards = agent.fit(env)

            # Save the episode rewards vector on the computer for future usage
            episode_rewards = np.array(episode_rewards)
            file_name_reward = f'episode_rewards_AC_BootstrappingOnly_rep_{rep + 1}.npy'
            np.save(os.path.join(directory, file_name_reward), episode_rewards)

        if not use_bootstrapping and use_baseline:
            print("ActorCritic with Baseline repetition {}".format(rep + 1))
            agent = ActorCriticAgent(state_n, action_n, hidden_n, learning_rate, gamma, entropy_coefficient, episodes,
                                     eval_interval, use_bootstrapping, n_step, use_baseline)
            episode_rewards = agent.fit(env)

            # Save the episode rewards vector on the computer for future usage
            episode_rewards = np.array(episode_rewards)
            file_name_reward = f'episode_rewards_AC_BaselineOnly_rep_{rep + 1}.npy'
            np.save(os.path.join(directory, file_name_reward), episode_rewards)

        if use_bootstrapping and use_baseline:
            print("ActorCritic with Bootstrapping and Baseline repetition {}".format(rep + 1))
            agent = ActorCriticAgent(state_n, action_n, hidden_n, learning_rate, gamma, entropy_coefficient, episodes,
                                     eval_interval, use_bootstrapping, n_step, use_baseline)
            episode_rewards = agent.fit(env)

            # Save the episode rewards vector on the computer for future usage
            episode_rewards = np.array(episode_rewards)
            file_name_reward = f'episode_rewards_AC_rep_{rep + 1}.npy'
            np.save(os.path.join(directory, file_name_reward), episode_rewards)

        if not use_bootstrapping and not use_baseline:
            print("Plain ActorCritic repetition {}".format(rep + 1))
            agent = ActorCriticAgent(state_n, action_n, hidden_n, learning_rate, gamma, entropy_coefficient, episodes,
                                     eval_interval, use_bootstrapping, n_step, use_baseline)
            episode_rewards = agent.fit(env)

            # Save the episode rewards vector on the computer for future usage
            episode_rewards = np.array(episode_rewards)
            file_name_reward = f'episode_rewards_plain_AC_rep_{rep + 1}.npy'
            np.save(os.path.join(directory, file_name_reward), episode_rewards)

    print("All experiments were saved.")
    print("This experiment took {} minutes".format((time.time() - now) / 60))


def experiment():
    n_repetitions = 3
    env = gym.make('LunarLander-v2')
    state_space = env.observation_space.shape
    state_n = reduce(lambda x, y: x * y, state_space)
    action_n = env.action_space.n
    print("Number of states is: ", state_n, " and the number of actions is: ", action_n)

    # The fine-tuned hyperparameters for Actor Critic algorithm
    hidden_n = 64
    learning_rate = 0.005
    gamma = 0.99
    entropy_coefficient = 0.001
    episodes = 10000
    eval_interval = 100
    n_step = 50

    # REINFORCE
    run_repetitions_REINFORCE(env, state_n, action_n, hidden_n, learning_rate, gamma, episodes,
                              eval_interval, entropy_coefficient, n_repetitions)

    # ActorCritic with Bootstrapping and Baseline
    use_bootstrapping = True
    use_baseline = True
    run_repetitions_ActorCritic(env, state_n, action_n, hidden_n, learning_rate, gamma, entropy_coefficient, episodes,
                                eval_interval, use_bootstrapping, n_step, use_baseline, n_repetitions)
    #
    # ActorCritic with Bootstrapping
    use_bootstrapping = True
    use_baseline = False
    run_repetitions_ActorCritic(env, state_n, action_n, hidden_n, learning_rate, gamma, entropy_coefficient, episodes,
                                eval_interval, use_bootstrapping, n_step, use_baseline, n_repetitions)

    # ActorCritic with Baseline
    use_bootstrapping = False
    use_baseline = True
    run_repetitions_ActorCritic(env, state_n, action_n, hidden_n, learning_rate, gamma, entropy_coefficient, episodes,
                                eval_interval, use_bootstrapping, n_step, use_baseline, n_repetitions)

    # Plain ActorCritic without baseline or bootstrapping
    use_bootstrapping = False
    use_baseline = False
    run_repetitions_ActorCritic(env, state_n, action_n, hidden_n, learning_rate, gamma, entropy_coefficient, episodes,
                                eval_interval, use_bootstrapping, n_step, use_baseline, n_repetitions)


if __name__ == '__main__':
    experiment()
