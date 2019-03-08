import logging

import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        # nn.init.normal_(m.weight.data, 0, .1)
        # nn.init.kaiming_normal_(m.weight.data, nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
        # nn.init.orthogonal_(m.weight.data, gain=1)
        # m.bias.data.fill_(0)


class CriticNetwork(torch.nn.Module):
    def __init__(self, n_inputs):
        super(CriticNetwork, self).__init__()

        self.n_inputs = n_inputs
        # self.n_hidden = n_hidden
        self.n_hidden = 100

        # act = nn.LeakyReLU()
        act = nn.ReLU()

        self.model = nn.Sequential(
            nn.Linear(self.n_inputs, self.n_hidden),
            act,
            nn.Linear(self.n_hidden, self.n_hidden),
            act,
            nn.Linear(self.n_hidden, 1)
        )

        logging.info(self.model)

        self.apply(init_weights)
        self.train()

    def forward(self, x):
        """
        Defines the forward pass of the network.

        :param x: Input array object which sufficiently represents the full state of the environment.
        :return: value
        """

        x = x.float()

        return self.model(x)
