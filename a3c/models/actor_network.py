import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, nn.Linear):
        # nn.init.normal_(m.weight.data, 0, .1)
        # nn.init.kaiming_normal_(m.weight.data, nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
        # nn.init.orthogonal_(m.weight.data, gain=1)
        # m.bias.data.fill_(0)


class ActorNetwork(torch.nn.Module):
    def __init__(self, n_inputs, n_actions):
        super(ActorNetwork, self).__init__()

        self.n_outputs = n_actions
        self.n_inputs = n_inputs

        self.n_hidden = 200

        # act = nn.LeakyReLU()
        act = nn.ReLU()

        self.body = nn.Sequential(
            nn.Linear(self.n_inputs, self.n_hidden),
            act,
            nn.Linear(self.n_hidden, self.n_hidden),
            act,
        )

        self.mu = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_outputs)
        )

        self.sigma = nn.Parameter(torch.zeros(self.n_outputs))

        logging.info(self.body)
        logging.info(self.mu)
        logging.info(self.sigma)

        self.apply(init_weights)
        self.train()

    def forward(self, x):
        x = x.float()

        body = self.body(x)
        mu = self.mu(body)
        return mu, F.softplus(self.sigma)
