import autograd.numpy as np


# From: https://github.com/cryscan/pilco-learner/blob/c0444d02c5df8358ee3358b5d36f79b4224ea2d3/pilco/gp.py


class Kernel(object):
    def __init__(self):
        self.n_hyperparams = None
        self.sub = None

    def __add__(self, other):
        kernel = Kernel()
        kernel.sub = self, other
        kernel.n_hyperparams = lambda x: self.n_hyperparams(x) + other.n_hyperparams(x)

        return kernel

    def __call__(self, log_hyperparams, x, z=None):
        log_hyperparams = np.atleast_2d(log_hyperparams)

        left, right = self.sub
        split = left.n_hyperparams(x)

        return left(log_hyperparams[:, :split], x, z) + right(log_hyperparams[:, split:], x, z)
