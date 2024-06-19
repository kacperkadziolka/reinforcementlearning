import gym

from DQN import DQNAgent


def main():
    env = gym.make("CartPole-v1", render_mode="human")

    agent = DQNAgent()
    agent.fit(env)


if __name__ == "__main__":
    main()
