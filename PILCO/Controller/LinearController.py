from autograd import numpy as np

from PILCO.Controller.Controller import Controller


class LinearController(Controller):
    """Linear Controller/Policy"""

    def __init__(self, W):
        self.W = W

    def choose_action(self, X):
        # TODO Check axis dim
        return np.sum(self.W @ X, axis=1)

    def optimize_params(self, W):
        self.W = W
