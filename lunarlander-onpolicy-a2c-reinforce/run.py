import argparse

import gym

from agents.actor_critic_agent import ActorCriticAgent
from agents.reinforce_agent import ReinforceAgent


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Agent Type Selector")

    # Add an argument for the type of agent
    parser.add_argument('--type', type=str, choices=['actorcritic', 'reinforce'],
                        help='Specify the type of agent: actorcritic or reinforce')

    # Parse the arguments
    args = parser.parse_args()

    # Initialize env
    env = gym.make('LunarLander-v2')

    # Agent Hyperparameters
    state_n = 8
    action_n = 4
    hidden_n = 64
    learning_rate = 0.005
    gamma = 0.99
    entropy_coef = 0.001
    episodes = 10000
    eval_interval = 100
    use_bootstrap = True
    n_step = 50
    use_baseline_subtraction = True

    if args.type == 'actorcritic':
        print("Agent type is set to Actor-Critic.")
        agent = ActorCriticAgent(state_n, action_n, hidden_n, learning_rate, gamma, entropy_coef, episodes,
                                 eval_interval, use_bootstrap, n_step, use_baseline_subtraction)
    elif args.type == 'reinforce':
        print("Agent type is set to REINFORCE.")
        agent = ReinforceAgent(state_n, action_n, hidden_n, learning_rate, gamma, entropy_coef, episodes,
                               eval_interval)
    else:
        print("No agent type specified, or specified type is not recognized.")

    agent.fit(env)


if __name__ == '__main__':
    main()
