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

    print("Running Ablation Experiment")
    for rep in range(n_repetitions):  # Loop over repetitions

        if use_memory and use_target_network:
            print("DQN repetition {}/{}".format(rep, n_repetitions + 1))
            agent = DQNAgent(state_n, action_n, hidden_n, learning_rate, gamma, memory_size, batch_size,
                             target_update_interval, episodes, policy, epsilon, eval_interval, temperature, use_memory,
                             use_target_network)

            episode_rewards = agent.fit(env)

        elif not use_memory and use_target_network:
            print("DQN-ER repetition {}/{}".format(rep, n_repetitions))
            agent = DQNAgent(state_n, action_n, hidden_n, learning_rate, gamma, memory_size, batch_size,
                             target_update_interval, episodes, policy, epsilon, eval_interval, temperature, use_memory,
                             use_target_network)

            episode_rewards = agent.fit(env)

        elif use_memory and not use_target_network:
            print("DQN-TN repetition {}/{}".format(rep, n_repetitions))
            agent = DQNAgent(state_n, action_n, hidden_n, learning_rate, gamma, memory_size, batch_size,
                             target_update_interval, episodes, policy, epsilon, eval_interval, temperature, use_memory,
                             use_target_network)

            episode_rewards = agent.fit(env)

        elif not use_memory and not use_target_network:
            print("DQN-ER-TN repetition {}/{}".format(rep, n_repetitions))
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
    memory_size = 10000
    target_update_interval = 100
    episodes = 1000
    eval_interval = 100

    # 2 implicit values for epsilon and temperature, needed for function to work. They are overwritten later.
    epsilon = 0.1
    temperature = 0.2

    # Comment/Uncomment based on what policy is experimented on
    'HP tuned for e-greedy'
    policy = 'e-greedy'

    learning_rate = 0.001
    hidden_n = 256
    batch_size = 64
    epsilon = 0.1
    gamma = 0.95

    'HP tuned for softmax'
    # policy = 'softmax'
    #
    # learning_rate = 0.01
    # hidden_n = 256
    # batch_size = 32
    # temperature = 0.01
    # gamma = 1

    Plot = LearningCurvePlot(title='Ablation Study')

    # DQN
    use_memory = True
    use_target_network = True
    learning_curve = average_over_repetitions(state_n, action_n, hidden_n, learning_rate, gamma, memory_size,
                                              batch_size, target_update_interval, episodes, policy, epsilon,
                                              eval_interval, temperature, use_memory, use_target_network, n_repetitions,
                                              smoothing_window)
    Plot.add_curve(learning_curve, label="DQN")

    # DQN without experience replay -ER
    use_memory = False
    use_target_network = True
    learning_curve = average_over_repetitions(state_n, action_n, hidden_n, learning_rate, gamma, memory_size,
                                              batch_size, target_update_interval, episodes, policy, epsilon,
                                              eval_interval, temperature, use_memory, use_target_network, n_repetitions,
                                              smoothing_window)
    Plot.add_curve(learning_curve, label="DQN-ER")

    # DQN without target network - TN
    use_memory = True
    use_target_network = False

    learning_curve = average_over_repetitions(state_n, action_n, hidden_n, learning_rate, gamma, memory_size,
                                              batch_size, target_update_interval, episodes, policy, epsilon,
                                              eval_interval, temperature, use_memory, use_target_network, n_repetitions,
                                              smoothing_window)

    Plot.add_curve(learning_curve, label="DQN-TN")

    # DQN without target network and experience replay -ER-TN
    use_memory = False
    use_target_network = False

    learning_curve = average_over_repetitions(state_n, action_n, hidden_n, learning_rate, gamma, memory_size,
                                              batch_size, target_update_interval, episodes, policy, epsilon,
                                              eval_interval, temperature, use_memory, use_target_network, n_repetitions,
                                              smoothing_window)

    Plot.add_curve(learning_curve, label="DQN-ER-TN")

    Plot.save('experiment_DQN_ER_TN.png')


if __name__ == '__main__':
    experiment()
