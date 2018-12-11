import autograd.numpy as np


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
        split = left.num_hyp(x)

        return left(loghyp[:, :split], x, z) + right(loghyp[:, split:], x, z)
