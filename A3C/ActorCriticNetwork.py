import torch
import torch.nn as nn
import torch.nn.functional as F


# code from https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal(m.weight.data)
        m.bias.data.fill_(0)


class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, n_inputs, action_space):
        super(ActorCriticNetwork, self).__init__()

        self.fc1 = nn.Linear(n_inputs, 512)
        self.dropout = nn.Dropout(0.5)

        n_outputs = action_space.n
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, n_outputs)

        self.apply(init_weights)

        self.train()

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs.float()))
        # x = self.dropout(x)

        return self.critic_linear(x), self.actor_linear(x)
