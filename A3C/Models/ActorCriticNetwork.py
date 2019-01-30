from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# code from https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py


def init_weights(m):
    if isinstance(m, nn.Linear):
        # nn.init.normal_(m.weight.data, 0, .01)
        nn.init.kaiming_normal_(m.weight.data, nonlinearity="leaky_relu")
        # nn.init.orthogonal_(m.weight.data, gain=1)
        m.bias.data.fill_(0)


class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, n_inputs, n_actions):
        super(ActorCriticNetwork, self).__init__()

        self.n_actions = n_actions
        self.n_inputs = n_inputs
        self.n_hidden = 200

        # act = nn.LeakyReLU()
        act = nn.ReLU()

        self.body = nn.Sequential(
            nn.Linear(self.n_inputs, self.n_hidden),
            act,
            # nn.Linear(self.n_hidden, self.n_hidden),
            # act,
            # nn.Linear(self.n_hidden, self.n_hidden),
        )

        self.mu = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_actions)
        )

        self.sigma = nn.Parameter(torch.zeros(self.n_actions))

        self.value = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_actions)
        )

        self.apply(init_weights)
        self.train()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Defines the forward pass of the network.

        :param x: Input array object which sufficiently represents the full state of the environment.
        :return: value, mu, sigma
        """

        body = self.body(x.float())

        return self.value(body), self.mu(body), F.softplus(self.sigma)
