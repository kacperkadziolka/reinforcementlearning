import numpy as np
import time
from DQN import DQNAgent
import gym
from LearningCurve import LearningCurvePlot, smooth_curve


def average_over_repetitions(state_n, action_n, hidden_n, learning_rate, gamma, memory_size, batch_size,
                             target_update_interval, episodes, policy, epsilon, eval_interval, temperature, use_memory,
                             use_target_network, n_repetitions, smoothing_window):
    returns_over_repetitions = []
    now = time.time()
    env = gym.make('CartPole-v1')

    print("Running Policy Experiment...")
    for rep in range(n_repetitions):  # Loop over repetitions

        if policy == 'e-greedy':
            print("DQN e-greedy repetition {}/{}".format(rep, n_repetitions + 1))
            agent = DQNAgent(state_n, action_n, hidden_n, learning_rate, gamma, memory_size, batch_size,
                             target_update_interval, episodes, policy, epsilon, eval_interval, temperature, use_memory,
                             use_target_network)

            episode_rewards = agent.fit(env)

        elif policy == 'softmax':
            print("DQN softmax repetition {}/{}".format(rep, n_repetitions))
            agent = DQNAgent(state_n, action_n, hidden_n, learning_rate, gamma, memory_size, batch_size,
                             target_update_interval, episodes, policy, epsilon, eval_interval, temperature, use_memory,
                             use_target_network)

            episode_rewards = agent.fit(env)

        returns_over_repetitions.append(episode_rewards)

    print("This experiment took {} minutes".format((time.time() - now) / 60))
    learning_curve = np.mean(np.array(returns_over_repetitions), axis=0)
    learning_curve = smooth_curve(learning_curve, smoothing_window)
    return learning_curve


def experiment():
    n_repetitions = 20
    smoothing_window = 9
    state_n = 4
    action_n = 2
    target_update_interval = 100
    memory_size = 10000
    episodes = 1000
    eval_interval = 100
    use_memory = True
    use_target_network = True

    # 2 implicit values for epsilon and temperature, needed for function to work. They are overwritten later.
    epsilon = 0.1
    temperature = 0.2

    "Tuned HPs for e-greedy"
    policy = 'e-greedy'

    learning_rate = 0.001
    hidden_n = 256
    batch_size = 64
    epsilon = 0.1
    gamma = 0.95

    Plot = LearningCurvePlot(title='Policy Analysis')

    # DQN e-greedy
    learning_curve = average_over_repetitions(state_n, action_n, hidden_n, learning_rate, gamma, memory_size,
                                              batch_size, target_update_interval, episodes, policy, epsilon,
                                              eval_interval, temperature, use_memory, use_target_network, n_repetitions,
                                              smoothing_window)
    Plot.add_curve(learning_curve, label="DQN e-greedy")

    "Tuned HPs for softmax"
    policy = 'softmax'

    learning_rate = 0.01
    hidden_n = 256
    batch_size = 32
    temperature = 0.01
    gamma = 1

    # DQN softmax
    learning_curve = average_over_repetitions(state_n, action_n, hidden_n, learning_rate, gamma, memory_size,
                                              batch_size, target_update_interval, episodes, policy, epsilon,
                                              eval_interval, temperature, use_memory, use_target_network, n_repetitions,
                                              smoothing_window)
    Plot.add_curve(learning_curve, label="DQN softmax")

    Plot.save('experiment_policy.png')


if __name__ == '__main__':
    experiment()
