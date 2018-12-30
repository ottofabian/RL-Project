import torch
import torch.nn as nn
import torch.nn.functional as F


# code from https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0, .1)
        m.bias.data.fill_(0)


class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, n_inputs, action_space, n_hidden, max_action):
        super(ActorCriticNetwork, self).__init__()

        self.action_space = action_space
        self.max_action = max_action

        self.n_outputs = action_space.shape[0]
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden

        self.n_inputs = self.n_inputs

        self.input = nn.Linear(self.n_inputs, self.n_hidden)
        self.hidden_1 = nn.Linear(self.n_hidden, self.n_hidden)
        self.hidden_2 = nn.Linear(self.n_hidden, self.n_hidden)
        # self.hidden_3 = nn.Linear(self.n_hidden, self.n_hidden)

        self.value = nn.Linear(self.n_hidden, 1)

        self.mu = nn.Linear(self.n_hidden, self.n_outputs)
        self.sigma = nn.Linear(self.n_hidden, self.n_outputs)

        self.apply(init_weights)

    def forward(self, inputs):
        """
        Defines the forward pass of the network.

        :param inputs: Input array object which sufficiently represents the full state of the environment.
        :return: In the discrete case: value, policy
                 In the continuous case: value, mu, sigma
        """

        x = F.relu(self.input(inputs.float()))
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        # x = F.relu(self.hidden_3(x))

        mu = self.max_action * torch.tanh(self.mu(x))
        sigma = F.softplus(self.sigma(x)) + 1e-5

        value = self.value(x)

        return value, mu, sigma
