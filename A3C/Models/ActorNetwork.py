import torch
import torch.nn as nn
import torch.nn.functional as F


# code from https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py


def init_weights(m):
    if isinstance(m, nn.Linear):
        # nn.init.normal_(m.weight.data, 0, .1)
        # nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
        nn.init.orthogonal_(m.weight.data)
        m.weight.data.mul_(1e-3)
        m.bias.data.fill_(0)


# for loading stab policy:

# init:
#         n_hidden = 200
#
#         self.n_inputs = self.n_inputs
#         self.hidden_action1 = nn.Linear(self.n_inputs, n_hidden)
#         self.mu = nn.Linear(n_hidden, self.n_outputs)
#         self.sigma = nn.Linear(n_hidden, self.n_outputs)
#
#         self.apply(init_weights)
#         self.train()

# forward:
# action_hidden = F.relu(self.hidden_action1(inputs))
# mu = 5 * torch.tanh(self.mu(action_hidden))
# sigma = F.softplus(self.sigma(action_hidden)) + 1e-5


# swing up:

# __init__
# n_hidden = 64
#
# self.n_inputs = self.n_inputs
# self.inputs = nn.Linear(self.n_inputs, n_hidden)
# self.hidden_action1 = nn.Linear(n_hidden, n_hidden)
# self.hidden_action2 = nn.Linear(n_hidden, n_hidden)
# self.hidden_action3 = nn.Linear(n_hidden, n_hidden)
# self.mu = nn.Linear(n_hidden, self.n_outputs)
# self.sigma = nn.Linear(n_hidden, self.n_outputs)

# forward:

# x = x.float()
#         x = F.relu(self.inputs(x))
#         x = F.relu(self.hidden_action1(x))
#         x = F.relu(self.hidden_action2(x))
#         x = F.relu(self.hidden_action3(x))
#         # mu = torch.from_numpy(self.action_space.high) * torch.tanh(self.mu(x))
#         mu = 10 * torch.tanh(self.mu(x))
#         sigma = F.softplus(self.sigma(x)) + 1e-5  # avoid 0

load_stab = False  # True


class ActorNetwork(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, max_action):
        super(ActorNetwork, self).__init__()

        self.n_outputs = n_outputs
        self.n_inputs = n_inputs

        self.n_hidden = 20

        self.max_action = max_action

        if load_stab:
            # n_hidden = 200

            self.n_inputs = self.n_inputs
            self.fc2 = nn.Linear(self.n_inputs, self.n_hidden)
            self.mu = nn.Linear(self.n_hidden, self.n_outputs)
            # self.sigma = nn.Linear(self.n_hidden, self.n_outputs)
            self.sigma = nn.Parameter(torch.zeros(n_outputs))

            self.apply(init_weights)
            self.train()

        else:
            # n_hidden = 200

            self.fc1 = nn.Linear(self.n_inputs, self.n_hidden)
            # self.fc2 = nn.Linear(self.n_hidden, self.n_hidden)
            # self.fc3 = nn.Linear(self.n_hidden, self.n_hidden)
            # self.fc4 = nn.Linear(self.n_hidden, self.n_hidden)
            self.mu = nn.Linear(self.n_hidden, self.n_outputs)
            self.sigma = nn.Linear(self.n_hidden, self.n_outputs)

            self.apply(init_weights)
            self.train()

    def forward(self, x):

        if load_stab:
            x = x.float()
            x = F.relu(self.fc2(x))
            mu = self.max_action * torch.tanh(self.mu(x))
            sigma = F.softplus(self.sigma(x)) + 1e-5
        else:
            x = F.relu(self.fc1(x.float()))
            # x = F.relu(self.fc2(x))
            # x = F.relu(self.fc3(x))
            # x = F.relu(self.fc4(x))
            # TODO test if no tanh is better
            mu = torch.tanh(self.mu(x)) * self.max_action
            sigma = F.softplus(self.sigma(x))
            # + 1e-5  # avoid 0

        return mu, sigma
