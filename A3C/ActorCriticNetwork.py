import torch
import torch.nn as nn
import torch.nn.functional as F

# code from https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py
from gym.spaces import Discrete


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight.data, 0, .1)
        m.bias.data.fill_(0)


class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, n_inputs, action_space, is_discrete=False):
        super(ActorCriticNetwork, self).__init__()

        self.is_discrete = is_discrete
        self.action_space = action_space

        # network architecture specification
        fc1_out = 100
        fc2_out = 200
        # fc_tower_out defines the number of units after the tower forward pass
        fc_tower_out = fc1_out

        self.fc1 = nn.Linear(n_inputs, fc1_out)
        # self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(n_inputs, fc2_out)
        # self.dropout2 = nn.Dropout(0.5)

        # Define the two heads of the network
        # -----------------------------------

        # * Value head
        # The value head has only 1 output
        self.critic_linear = nn.Linear(fc1_out, 1)

        # * Policy head
        # Define the number of output for the policy
        n_outputs = action_space.n if isinstance(action_space, Discrete) else action_space.shape[0]

        if self.is_discrete:
            # in the dicrete case it has
            self.actor_linear = nn.Linear(fc2_out, n_outputs)
        else:
            # in the continuous case it has one output for the mu and one for the sigma variable
            # later the workers can sample from a normal distribution
            self.mu = nn.Linear(fc2_out, n_outputs)
            self.sigma = nn.Linear(fc2_out, n_outputs)

        # initialize the weights using Xavier initialization
        self.apply(init_weights)
        self.train()

    def forward(self, inputs):
        """
        Defines the forward pass of the network.

        :param inputs: Input array object which sufficiently represents the full state of the environment.
        :return: In the discrete case: value, policy
                 In the continuous case: value, mu, sigma
        """
        inputs = inputs.float()
        critic_nn = F.relu(self.fc1(inputs))
        # critic_nn = self.dropout1(critic_nn)
        actor_nn = F.relu(self.fc2(inputs))
        # actor_nn = self.dropout2(actor_nn)

        if self.is_discrete:
            return self.critic_linear(critic_nn), self.actor_linear(actor_nn)
        else:
            # assuming action space is centered at 0
            scale = torch.Tensor(self.action_space.high)
            mu = scale * torch.tanh(self.mu(actor_nn))
            sigma = F.softplus(self.sigma(actor_nn)) + 1e-5
            return self.critic_linear(critic_nn), mu, sigma
