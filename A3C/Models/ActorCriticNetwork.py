import torch
import torch.nn as nn
import torch.nn.functional as F


# code from https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0, .01)
        # nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)


class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, n_inputs, n_actions, max_action):
        super(ActorCriticNetwork, self).__init__()

        self.max_action = max_action

        self.n_actions = n_actions
        self.n_inputs = n_inputs
        self.n_hidden = 20

        self.input = nn.Linear(self.n_inputs, self.n_hidden)
        self.hidden_1 = nn.Linear(self.n_hidden, self.n_hidden)
        # self.hidden_2 = nn.Linear(self.n_hidden, self.n_hidden)
        # self.hidden_3 = nn.Linear(self.n_hidden, self.n_hidden)

        self.value = nn.Linear(self.n_hidden, 1)

        self.mu = nn.Linear(self.n_hidden, self.n_actions)
        self.sigma = nn.Linear(self.n_hidden, self.n_actions)

        self.apply(init_weights)
        self.train()

    def forward(self, inputs):
        """
        Defines the forward pass of the network.

        :param inputs: Input array object which sufficiently represents the full state of the environment.
        :return: In the discrete case: value, policy
                 In the continuous case: value, mu, sigma
        """

        x = F.relu(self.input(inputs.float()))
        x = F.relu(self.hidden_1(x))
        # x = F.leaky_relu(self.hidden_2(x), .0)
        # x = F.leaky_relu(self.hidden_3(x), .0)

        mu = self.max_action * torch.tanh(self.mu(x))
        sigma = F.softplus(self.sigma(x)) + 1e-5

        value = self.value(x)

        return value, mu, sigma
