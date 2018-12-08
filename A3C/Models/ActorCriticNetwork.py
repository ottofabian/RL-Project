import torch
import torch.nn as nn
import torch.nn.functional as F


# code from https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        m.bias.data.fill_(0)


class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, n_inputs, action_space, n_hidden):
        super(ActorCriticNetwork, self).__init__()

        self.action_space = action_space

        self.n_outputs = action_space.shape[0]
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden

        self.n_inputs = self.n_inputs

        self.input = nn.Linear(self.n_inputs, self.n_hidden)
        self.hidden_1 = nn.Linear(self.n_hidden, self.n_hidden)
        self.hidden_2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.hidden_3 = nn.Linear(self.n_hidden, self.n_hidden)

        self.value = nn.Linear(self.n_hidden, 1)

        self.mu = nn.Linear(self.n_hidden, self.n_outputs)
        self.sigma = nn.Linear(self.n_hidden, self.n_outputs)

        self.apply(init_weights)

        # network architecture specification
        # n_hidden_units = 256

        # self.fc1 = nn.Linear(n_inputs, fc1_out)

        # self.stem = nn.Sequential(
        #     nn.Linear(n_inputs, n_hidden_units),
        #     # nn.Linear(n_hidden_units, n_hidden_units),
        #     nn.ReLU(),
        # )

        # Define the two heads of the network
        # -----------------------------------

        # * Value head
        # The value head has only 1 output
        # self.critic_value = nn.Sequential(
        #     nn.Linear(n_hidden_units, n_hidden_units),
        #     nn.ReLU(),
        #     nn.Linear(n_hidden_units, 1),
        #     nn.Tanh(),
        # )

        # * Policy head
        # Define the number of output for the policy
        # n_outputs = action_space.n if isinstance(action_space, Discrete) else action_space.shape[0]

        # if self.is_discrete:
        #     # in the discrete case it uses as many outputs as there are discrete actions
        #     self.actor_policy = nn.Sequential(
        #         nn.Linear(n_hidden_units, n_hidden_units),
        #         nn.ReLU(),
        #         nn.Linear(n_hidden_units, n_outputs),
        #         # note: The softmax activation will be applied during training
        #     )
        # else:
        #     # in the continuous case it has one output for the mu and one for the sigma variable
        #     # later the workers can sample from a normal distribution
        #     # the test worker takes action according to the mu value
        #     self.actor_mu = nn.Sequential(
        #         nn.Linear(n_hidden_units, n_outputs),
        #         nn.Tanh(),
        #     )
        #
        #     self.actor_variance = nn.Sequential(
        #         nn.Linear(n_hidden_units, n_outputs),
        #         nn.Softplus(),
        #     )
        #
        # # initialize the weights
        # self.apply(init_weights)
        # self.train()

    def forward(self, inputs):
        """
        Defines the forward pass of the network.

        :param inputs: Input array object which sufficiently represents the full state of the environment.
        :return: In the discrete case: value, policy
                 In the continuous case: value, mu, sigma
        """

        x = F.relu(self.input(inputs.float()))
        x = F.relu(self.hidden_1(x))
        # x = F.relu(self.hidden_2(x))
        # x = F.relu(self.hidden_3(x))

        mu = torch.from_numpy(self.action_space.high) * torch.tanh(self.mu(x))
        sigma = F.softplus(self.sigma(x)) + 1e-5

        value = self.value(x)

        return value, mu, sigma

