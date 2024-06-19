import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, state_n: int, action_n: int, hidden_n: int):
        """
        The network is designed to output a probability distribution over actions based on the input state.

        fc1: The first fully connected layer which maps from state space to hidden space.
        fc2: The second fully connected layer which maps from hidden space to action space.
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_n, hidden_n)
        self.fc2 = nn.Linear(hidden_n, action_n)

    def forward(self, x):
        """
        Defines the computation performed at every call of the network, which outputs a probability distribution
        over actions. A softmax function is applied to the output of fc2 to get a probability distribution over actions.
        """
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)
