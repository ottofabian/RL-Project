from torch import nn


class SplitWrapper(nn.Module):

    def __init__(self, actor_model: nn.Module, critic_model: nn.Module):
        super().__init__()
        self.actor_model = actor_model
        self.critic_model = critic_model

    def forward(self, x):
        mu, sigma = self.actor_model
        value = self.critic_model

        return value, mu, sigma

    def _load_from_state_dict(self, state_dict: tuple, prefix, metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        self.actor_model.load_state_dict(state_dict[0])
        self.critic_model.load_state_dict(state_dict[1])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.actor_model.state_dict(), self.critic_model.state_dict()

    def parameters(self):
        return self.actor_model.parameters(), self.critic_model.parameters()
