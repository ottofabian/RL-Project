import torch
import torch.nn as nn
import torch.nn.functional as F


# code from https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py


def init_weights(m):
    if isinstance(m, nn.Linear):
        # nn.init.normal_(m.weight.data, 0, .1)
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)


# for loading stab policy:

# init:
#         n_hidden = 200
#
#         self.n_inputs = self.n_inputs
#         self.hidden_action1 = nn.Linear(self.n_inputs, n_hidden)
#         self.hidden_action2 = nn.Linear(n_hidden, n_hidden)
#         self.mu = nn.Linear(n_hidden, self.n_outputs)
#         self.sigma = nn.Linear(n_hidden, self.n_outputs)
#
#         self.apply(init_weights)
#         self.train()

# forward:
# action_hidden = F.relu(self.hidden_action1(inputs))
# action_hidden = F.relu(self.hidden_action2(action_hidden))
# mu = 5 * torch.tanh(self.mu(action_hidden))
# sigma = F.softplus(self.sigma(action_hidden)) + 1e-5

load_stab = True #True

class CriticNetwork(torch.nn.Module):
    def __init__(self, n_inputs, action_space, is_discrete=False):
        super(CriticNetwork, self).__init__()

        self.is_discrete = is_discrete
        self.action_space = action_space

        self.n_outputs = action_space.shape[0]
        self.n_inputs = n_inputs

        if load_stab:
            n_hidden = 200

            self.n_inputs = self.n_inputs
            self.hidden_value1 = nn.Linear(self.n_inputs, n_hidden)
            self.hidden_value2 = nn.Linear(n_hidden, n_hidden)
            self.value = nn.Linear(n_hidden, 1)

            self.apply(init_weights)
            self.train()
        else:
            n_hidden = 100

            self.n_inputs = self.n_inputs

            self.inputs = nn.Linear(self.n_inputs, n_hidden)
            # self.hidden_value1 = nn.Linear(n_hidden, n_hidden)
            # self.hidden_value2 = nn.Linear(n_hidden, n_hidden)
            # self.hidden_value3 = nn.Linear(n_hidden, n_hidden)

            self.value = nn.Linear(n_hidden, 1)

            self.apply(init_weights)
            self.train()

    def forward(self, x):
        """
        Defines the forward pass of the network.

        :param x: Input array object which sufficiently represents the full state of the environment.
        :return: value
        """

        x = x.float()

        if load_stab:
            x = F.relu(self.hidden_value1(x))
            x = F.relu(self.hidden_value2(x))
        else:
            x = F.relu(self.inputs(x))
            # x = F.relu(self.hidden_value1(x))
            # x = F.relu(self.hidden_value2(x))
            # x = F.relu(self.hidden_value3(x))

        return self.value(x)
