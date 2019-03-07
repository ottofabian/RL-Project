import autograd.numpy as np

from pilco.kernel.kernel import Kernel


# From: https://github.com/cryscan/pilco-learner/blob/c0444d02c5df8358ee3358b5d36f79b4224ea2d3/pilco/gp.py

class RBFKernel(Kernel):

    def __init__(self):
        """
        Initalize RBFKernel
        """
        super(RBFKernel, self).__init__()
        self.n_hyperparams = lambda x: np.size(x, 1) + 1

    def __call__(self, log_hyperparams: np.ndarray, x: np.ndarray, z: np.ndarray = None) -> np.ndarray:
        """
        returns value for RBF kernel, if no z is given x is evaluated against itself
        :param log_hyperparams: hyperparameter set [[sigma_f, length scales, sigma_eps] x input dimension]
        :param x: samples of shape [n samples x dimensionality]
        :param z: samples of shape [n samples x dimensionality]
        :return: RBF values for each x or z
        """
        log_hyperparams = np.atleast_2d(log_hyperparams)

        length_scales = np.exp(log_hyperparams[:, :x.shape[1]])
        sigma_f = np.exp(2 * log_hyperparams[:, x.shape[1]]).reshape(-1, 1, 1)

        scaled_x = np.expand_dims(x, 0) / np.expand_dims(length_scales, 1)
        if z is None:
            diff_a = np.expand_dims(scaled_x, 1)
            diff_b = np.expand_dims(scaled_x, 2)
        else:
            scaled_z = np.expand_dims(z, 0) / np.expand_dims(length_scales, 1)
            diff_a = np.expand_dims(scaled_x, 1)
            diff_b = np.expand_dims(scaled_z, 2)

        # The maha call without Q just computes the squared error
        # mahalanobis_dist = np.expand_dims(np.sum(diff_a * diff_a, axis=-1), axis=-1) + np.expand_dims(
        #     np.sum(diff_b * diff_b, axis=-1), axis=-2) - 2 * np.einsum('...ij, ...kj->...ik', diff_a, diff_b)
        # return sigma_f * np.exp(-.5 * np.sum(mahalanobis_dist, axis=3))

        return sigma_f * np.exp(-.5 * np.sum((diff_a - diff_b) ** 2, axis=3))
