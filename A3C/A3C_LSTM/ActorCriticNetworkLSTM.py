import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# code from https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0., .1)
        m.bias.data.fill_(0)


# def normalized_columns_initializer(weights, std=1.0):
#     out = torch.randn(weights.size())
#     out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
#     return out

def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x ** 2).sum(1, keepdim=True))
    return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data /= torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ActorCriticNetworkLSTM(torch.nn.Module):
    def __init__(self, n_inputs, action_space, n_hidden, n_frames=1):
        super(ActorCriticNetworkLSTM, self).__init__()

        self.action_space = action_space

        self.n_outputs = action_space.shape[0]
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden

        self.n_inputs = self.n_inputs

        self.input = nn.Linear(self.n_inputs, self.n_hidden)
        self.fc1 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_hidden)
        # self.fc3 = nn.Linear(self.n_hidden, self.n_hidden)

        self.lstm_stacks = n_frames * self.n_hidden
        self.lstm = nn.LSTMCell(self.lstm_stacks, self.n_hidden)

        self.value = nn.Linear(self.n_hidden, 1)

        self.mu = nn.Linear(self.n_hidden, self.n_outputs)
        self.sigma = nn.Linear(self.n_hidden, self.n_outputs)

        self.apply(init_weights)

        lrelu = nn.init.calculate_gain('leaky_relu')
        self.input.weight.data.mul_(lrelu)
        self.fc1.weight.data.mul_(lrelu)
        self.fc2.weight.data.mul_(lrelu)
        # self.fc3.weight.data.mul_(lrelu)

        self.mu.weight.data = norm_col_init(self.mu.weight.data, 0.01)
        self.mu.bias.data.fill_(0)
        self.sigma.weight.data = norm_col_init(self.sigma.weight.data, 0.01)
        self.sigma.bias.data.fill_(0)
        self.value.weight.data = norm_col_init(self.value.weight.data, 1.0)
        self.value.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs

        x = F.leaky_relu_(self.input(inputs.float()), .1)
        x = F.leaky_relu_(self.fc1(x), .1)
        x = F.leaky_relu_(self.fc2(x), .1)
        # x = F.leaky_relu_(self.fc3(x), .1)

        x = x.view(-1, self.lstm_stacks)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        value = self.value(x)

        # mu = torch.from_numpy(self.action_space.high) * torch.tanh(self.mu(x))
        # mu = 5 * torch.tanh(self.mu(x))
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x)) + 1e-5  # avoid 0

        return value, mu, sigma, (hx, cx)
