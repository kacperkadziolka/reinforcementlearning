import numpy as np
import torch
from gym import Env


class BaseAgent:
    def __init__(self, state_n: int, action_n: int, hidden_n: int, learning_rate: float, gamma: float,
                 entropy_coef: float, episodes: int, eval_interval: int):
        """
        Base agent implementation for reinforce and A2C agents.

        :param state_n: The size of the state space.
        :param action_n: The size of the action space.
        :param hidden_n: The number of hidden units in each layer of the model.
        :param learning_rate: The learning rate for the optimizer.
        :param gamma: The discount factor for future rewards.
        :param entropy_coef: The coefficient for entropy regularization to encourage exploration.
        :param episodes: The total number of episodes for training.
        :param eval_interval: The interval at which the policy is evaluated.

        :attribute device: The device (CPU or GPU) to run the computations on.
        :attribute eps: A small epsilon value to prevent division by zero in computations.
        """
        # Hyperparameters
        self.state_n = state_n
        self.action_n = action_n
        self.hidden_n = hidden_n
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        # Training parameters
        self.episodes = episodes
        self.eval_interval = eval_interval

        # Torch settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Auxiliary epsilon parameter to avoid division by zero
        self.eps = np.finfo(np.float32).eps.item()

    def select_action(self, state: np.ndarray, test: bool = False):
        """
        Implementation is done in concrete agents
        """
        pass

    def test(self, env: Env):
        """
        Evaluate the model
        """
        state = env.reset()
        done = False

        test_reward = 0

        while not done:
            action = self.select_action(state, test=True)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            test_reward += reward

        return test_reward



