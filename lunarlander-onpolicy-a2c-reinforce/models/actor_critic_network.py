from typing import Tuple

from torch import nn, Tensor
import torch.nn.functional as F


class ActorCriticModel(nn.Module):
    """
    Neural network that implements both actor and critic in a single model.
    It facilitates the simultaneous training of both the policy, which determines the best action to take,
    and the value, which estimates the expected return from a given state.

    Attributes:
        fl1 (nn.Linear): A fully connected layer.
        action_head (nn.Linear): A fully connected layer that outputs action probabilities given state features.
        value_head (nn.Linear): A fully connected layer that outputs value prediction of the state.
        saved_actions (list): A list used to store action probabilities and value predictions during episodes.
        rewards (list): A list used to store rewards obtained from the environment during episodes.
    """

    def __init__(self, state_n: int, action_n: int, hidden_n: int) -> None:
        super(ActorCriticModel, self).__init__()
        self.fl1 = nn.Linear(state_n, hidden_n)

        # outputs probabilities for each of the 4 possible actions
        self.action_head = nn.Linear(hidden_n, action_n)

        # outputs a single value estimating the state's value
        self.value_head = nn.Linear(hidden_n, 1)

        # buffers to store actions taken and rewards received for each episode
        self.saved_actions = []
        self.rewards = []

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        The actor part computes a softmax distribution over possible actions, and the critic part evaluates the
        expected return from the current state.

        Returns:
            tuple: A tuple containing:
                - action_prob (Tensor): The softmax probability distribution over actions.
                - state_values (Tensor): The predicted value of the input state from the critic.
        """
        x = F.relu(self.fl1(x))

        # compute action probabilities using softmax to ensure they sum to 1
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # compute state value estimation from state features
        state_values = self.value_head(x)

        return action_prob, state_values
