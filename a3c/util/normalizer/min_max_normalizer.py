import torch

from a3c.util.normalizer.base_normalizer import BaseNormalizer
import numpy as np

min_state = None
max_state = None


class MinMaxScaler(BaseNormalizer):

    def __init__(self, min_state: np.ndarray, max_state: np.ndarray, read_only):
        """
        Constructor
        :param min_state: Representation of the minimum values for all feature of a state
        :param max_state: Representation of the maximum values for all feature of a state
        """
        super().__init__(read_only)
        self.min_state = min_state
        self.max_state = max_state

    def __call__(self, x):
        x = np.asarray(x)
        return torch.from_numpy((x - self.min_state) / (self.max_state - self.min_state))
