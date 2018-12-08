import torch
import torch.nn as nn
import torch.nn.functional as F


# code from https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0, .1)
        m.bias.data.fill_(0)


class ActorNetwork(torch.nn.Module):
    def __init__(self, n_inputs, action_space, is_discrete=False):
        super(ActorNetwork, self).__init__()

        self.is_discrete = is_discrete
        self.action_space = action_space

        self.n_outputs = action_space.shape[0]
        self.n_inputs = n_inputs

        n_hidden = 200

        self.n_inputs = self.n_inputs
        self.hidden_action1 = nn.Linear(self.n_inputs, n_hidden)
        self.hidden_action2 = nn.Linear(n_hidden, n_hidden)
        # self.hidden_action3 = nn.Linear(n_hidden, n_hidden)
        self.mu = nn.Linear(n_hidden, self.n_outputs)
        self.sigma = nn.Linear(n_hidden, self.n_outputs)

        self.apply(init_weights)
        self.train()

    def forward(self, inputs):
        inputs = inputs.float()
        action_hidden = F.relu(self.hidden_action1(inputs))
        # action_hidden = F.relu(self.hidden_action2(action_hidden))
        # action_hidden = F.relu(self.hidden_action3(action_hidden))
        mu = torch.from_numpy(self.action_space.high) * torch.tanh(self.mu(action_hidden))
        # mu = 2 * torch.tanh(self.mu(action_hidden))
        sigma = F.softplus(self.sigma(action_hidden)) + 1e-5  # avoid 0

        return mu, sigma
