from DQN import DQNAgent
import numpy as np
import itertools
import gym


def search(env, search_space):
    bestAvgReward = float('-inf')
    bestParameters = None

    print("Tuning hyperparameters...")
    count_combination = 1
    # Iterate over each combination
    for parametersCombination in itertools.product(*search_space.values()):
        print(f"DQN runs now for HP combination: {count_combination}")
        parameters = dict(zip(search_space.keys(), parametersCombination))
        print("The HP combination is:", parameters)

        agent = DQNAgent(state_n=4, action_n=2, episodes=1000, policy='e-greedy', **parameters)
        episode_rewards = agent.fit(env)

        # Evaluate and save results
        avg_episode_reward = np.mean(episode_rewards)

        if avg_episode_reward > bestAvgReward:
            bestAvgReward = avg_episode_reward
            bestParameters = parameters

        count_combination += 1

    return bestParameters


def main():
    # Defining search space
    HP_search_space = {
        'learning_rate': [0.001, 0.01, 0.1],
        'hidden_n': [64, 128, 256],
        'batch_size': [32, 64, 128],
        'epsilon': [0.1, 0.3, 1.0],
        # 'temperature': [0.01, 0.1, 1.0],
        'gamma': [0.9, 0.95, 1]
    }

    env = gym.make('CartPole-v1')

    # Generate all possible combinations of hyperparameters
    bestParameters = search(env, HP_search_space)
    print("Best parameters after hyperparameter tuning:")
    print(bestParameters)


if __name__ == '__main__':
    main()
