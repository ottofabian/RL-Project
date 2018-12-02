from autograd import numpy as np

from PILCO.Kernels.Kernels import Kernel


# From: https://github.com/cryscan/pilco-learner/blob/c0444d02c5df8358ee3358b5d36f79b4224ea2d3/pilco/gp.py

class RBFKernel(Kernel):

    def __init__(self):
        super(RBFKernel, self).__init__()
        self.num_hyp = lambda x: np.size(x, 1) + 1

    def __call__(self, log_hyperparam, x, z=None):

        log_hyperparam = np.atleast_2d(log_hyperparam)

        length_scales = np.exp(log_hyperparam[:, :x.shape[1]])
        sigma_f = np.exp(2 * log_hyperparam[:, x.shape[1]])
        sigma_f = sigma_f.reshape(-1, 1, 1)

        # datapoints x dimension
        scaled_x = np.expand_dims(x, 0) / np.expand_dims(length_scales, 1)
        if z is None:
            diff = np.expand_dims(scaled_x, 1) - np.expand_dims(scaled_x, 2)
        else:
            scaled_z = np.expand_dims(z, 0) / np.expand_dims(length_scales, 1)
            diff = np.expand_dims(scaled_x, 1) - np.expand_dims(scaled_z, 2)

        return sigma_f * np.exp(-0.5 * np.sum(diff ** 2, axis=3))
