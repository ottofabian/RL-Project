import torch

min_state = None
max_state = None


# TODO Adapt this to api
# TODO Why do we need unnormalize, we only do this before feeding it to the NN
class MinMaxScaler(BaseException):

    def __init__(self, min_state: torch.Tensor, max_state: torch.Tensor):
        """
        Constructor
        :param min_state: Representation of the minimum values for all feature of a state
        :param max_state: Representation of the maximum values for all feature of a state
        """
        self.min_state = min_state
        self.max_state = max_state

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Applies min-/max-scaling to to the state features into range [0,1] for all features
        :param state: Pytorch tensor which defines the state
        :return: Normalized version of the state in which all entries are within the range [0,1]
        """
        return (state - self.min_state) / (self.max_state - self.min_state)

    def unormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Reverts the min-/max-scaling back to the original state representation
        :param state: Pytorch tensor which defines the state
        :return: Original state representation of the environment
        """
        return state * (self.max_state - self.min_state) + self.min_state

# Call in class ,so it is not deleted:
#     if self.args.env_name in ["CartpoleStabShort-v0", "CartpoleStabRR-v0",
#                               "CartpoleSwingShort-v0", "CartpoleSwingRR-v0"]:
#         min_state = env.observation_space.low
#         max_state = env.observation_space.high
#         # set the minimum and maximum for x_dot and theta_dot manually because
#         # they are set to infinity by default
#         min_state[3] = -3
#         max_state[3] = 3
#         min_state[4] = -80
#         max_state[4] = 80
#         min_state = torch.from_numpy(min_state).double()
#         max_state = torch.from_numpy(max_state).double()
#         min_max_scaler = MinMaxScaler(min_state, max_state)
#     else:
#         logging.warning("You're given environment %s isn't supported for normalization" % self.args.env_name)
