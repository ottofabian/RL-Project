import torch

min_state = None
max_state = None


class MinMaxScaler:

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
