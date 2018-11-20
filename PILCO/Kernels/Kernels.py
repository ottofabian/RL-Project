from autograd import numpy as np


# From: https://github.com/cryscan/pilco-learner/blob/c0444d02c5df8358ee3358b5d36f79b4224ea2d3/pilco/gp.py
# TODO rewrite this nicer and more clear


class Kernel:
    def __init__(self):
        pass

    def __add__(self, other):
        sum = Kernel()
        sum.sub = self, other
        sum.num_hyp = lambda x: self.num_hyp(x) + other.num_hyp(x)
        return sum

    def __call__(self, loghyp, x, z=None):
        loghyp = np.atleast_2d(loghyp)
        left, right = self.sub
        L = left.num_hyp(x)
        return left(loghyp[:, :L], x, z) + right(loghyp[:, L:], x, z)


class RBFKernel(Kernel):
    """
    Squared exponential covariance function.
    """

    def __init__(self):
        super(RBFKernel, self).__init__()
        self.num_hyp = lambda x: np.size(x, 1) + 1

    def __call__(self, loghyp, x, z=None):
        loghyp = np.atleast_2d(loghyp)
        n, D = x.shape
        ell = np.exp(loghyp[:, :D])  # [1, D]
        sf2 = np.exp(2 * loghyp[:, D])
        sf2 = sf2.reshape(-1, 1, 1)

        x_ell = np.expand_dims(x, 0) / np.expand_dims(ell, 1)  # [n, D]
        if z is None:
            diff = np.expand_dims(x_ell, 1) - np.expand_dims(x_ell, 2)
        else:
            z_ell = np.expand_dims(z, 0) / np.expand_dims(ell, 1)
            diff = np.expand_dims(x_ell, 1) - np.expand_dims(z_ell, 2)

        K = sf2 * np.exp(-0.5 * np.sum(diff ** 2, axis=3))  # [n, n]
        return K


class WhiteNoiseKernel(Kernel):
    """
    White noise.
    """

    def __init__(self):
        super(WhiteNoiseKernel, self).__init__()
        self.num_hyp = lambda x: 1

    def __call__(self, loghyp, x, z=None):
        loghyp = np.atleast_2d(loghyp)
        n, D = x.shape
        s2 = np.exp(2 * loghyp)  # [E, 1]
        s2 = s2.reshape(-1, 1, 1)

        if z is None:
            K = s2 * np.expand_dims(np.identity(n), 0)
        else:
            K = 0
        return K
