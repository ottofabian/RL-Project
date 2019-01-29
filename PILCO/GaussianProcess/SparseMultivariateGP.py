import numpy as np

from typing import Union, Type

from PILCO.GaussianProcess import GaussianProcess, RBFNetwork
from PILCO.GaussianProcess.MultivariateGP import MultivariateGP


class SparseMultivariateGP(MultivariateGP):

    def __init__(self, n_targets: int, container: Union[Type[GaussianProcess], Type[RBFNetwork]],
                 length_scales: np.ndarray):
        super().__init__(n_targets, container, length_scales)
