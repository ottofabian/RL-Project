import torch
import torch.nn as nn
import torch.nn.functional as F


# code from https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0, .1)
        m.bias.data.fill_(.1)


class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, n_inputs, action_space, is_discrete=False):
        super(ActorCriticNetwork, self).__init__()

        self.is_discrete = is_discrete
        self.action_space = action_space

        self.n_outputs = action_space.shape[0]
        self.n_inputs = n_inputs

        self.n_inputs = self.n_inputs
        self.hidden_action = nn.Linear(self.n_inputs, 100)
        self.mu = nn.Linear(100, self.n_outputs)
        self.sigma = nn.Linear(100, self.n_outputs)

        self.hidden_value = nn.Linear(self.n_inputs, 100)
        self.value = nn.Linear(100, 1)

        self.apply(init_weights)

        # set_init([self.a1, self.mu, self.sigma, self.c1, self.v])
        self.distribution = torch.distributions.Normal

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
        # stem_out = self.stem(inputs)

        # if self.is_discrete:
        #     # x = self.fc1(inputs)
        #     # x = F.relu6(x)
        #     # x = F.relu(x)
        #     # x = self.fc2(x)
        #     # x = F.relu(x)
        #
        #     # return torch.tanh(self.critic_linear(x)), self.actor_linear(x)
        #     return self.critic_value(stem_out), self.actor_policy(stem_out)
        #
        # else:

        # # x = self.fc1(inputs)
        # # x = F.relu(x)
        # bound = torch.from_numpy(self.action_space.high)
        # # mu = bound * torch.tanh(self.mu(x))
        # # print("Mu scaled:", mu.data)
        # # sigma = F.softplus(self.sigma(x)) + 1e-5
        # # return torch.tanh(self.critic_linear(x)), mu, sigma
        # # value = torch.tanh(self.critic_linear(x))
        # # value = self.critic_linear(F.relu6(self.fc2(inputs)))
        # # return value, mu, sigma
        #
        # return self.critic_value(stem_out), bound * self.actor_mu(stem_out), self.actor_variance(stem_out)

        x = inputs.float()
        action_hidden = F.relu(self.hidden_action(x))
        mu = torch.from_numpy(self.action_space.high) * torch.tanh(self.mu(action_hidden))
        sigma = F.softplus(self.sigma(action_hidden)) + 1e-3  # avoid 0
        value_hidden = F.relu(self.hidden_value(x))
        value = self.value(value_hidden)
        return value, mu, sigma

    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mu.view(1, ).data, sigma.view(1, ).data)
        return m.sample().numpy()
