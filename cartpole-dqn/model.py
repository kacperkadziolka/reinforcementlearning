import torch
import torch.nn as nn

class DQN(nn.Module):

    def __init__(self, state_n, action_n, hidden_n):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_n, hidden_n)
        self.fc2 = nn.Linear(hidden_n, hidden_n)
        self.fc3 = nn.Linear(hidden_n, action_n)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
