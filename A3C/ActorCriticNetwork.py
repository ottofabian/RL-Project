import torch
import torch.nn as nn
import torch.nn.functional as F

# code from https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py
from gym.spaces import Discrete


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        m.bias.data.fill_(0)


class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, n_inputs, action_space):
        super(ActorCriticNetwork, self).__init__()

        self.fc1 = nn.Linear(n_inputs, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(0.5)

        n_outputs = action_space.n if isinstance(action_space, Discrete) else action_space.shape[0]
        # n_outputs = action_space.n
        self.critic_linear = nn.Linear(1024, 1)
        self.actor_linear = nn.Linear(1024, n_outputs)

        self.apply(init_weights)

        self.train()

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs.float()))
        x = self.dropout1(x)
        # x = F.relu(self.fc2(x))
        # x = self.dropout2(x)

        return self.critic_linear(x), self.actor_linear(x)
