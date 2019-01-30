#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import torch

from torch import nn, Tensor
import torch.functional as F


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class GaussianActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, phi_body=None, actor_body=None, critic_body=None):
        super(GaussianActorCriticNet, self).__init__()

        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.std = nn.Parameter(torch.zeros(action_dim))
        self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        obs = Tensor(obs)
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        mean = F.tanh(self.network.fc_action(phi_a))
        v = self.network.fc_critic(phi_v)

        dist = torch.distributions.Normal(mean, F.softplus(self.std))

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)

        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'mean': mean,
                'v': v}


class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, phi_body, actor_body, critic_body):
        super(ActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())


class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim
