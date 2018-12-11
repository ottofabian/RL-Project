import autograd.numpy as np

from PILCO.Kernels.Kernels import Kernel


# From: https://github.com/cryscan/pilco-learner/blob/c0444d02c5df8358ee3358b5d36f79b4224ea2d3/pilco/gp.py

class WhiteNoiseKernel(Kernel):

    def __init__(self):
        super(WhiteNoiseKernel, self).__init__()
        self.num_hyp = lambda x: 1

    def __call__(self, log_hyperparam, x, z=None):

        log_hyperparam = np.atleast_2d(log_hyperparam)

        sigma_eps = np.exp(2 * log_hyperparam)
        sigma_eps = sigma_eps.reshape(-1, 1, 1)

        if z is None:
            K = sigma_eps * np.expand_dims(np.identity(x.shape[0]), 0)
        else:
            K = 0
        return K
