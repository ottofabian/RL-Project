import autograd.numpy as np

from pilco.kernel.kernel import Kernel


# From: https://github.com/cryscan/pilco-learner/blob/c0444d02c5df8358ee3358b5d36f79b4224ea2d3/pilco/gp.py

class WhiteNoiseKernel(Kernel):

    def __init__(self):
        """
        Initalize WhiteNoiseKernel
        """
        super(WhiteNoiseKernel, self).__init__()
        self.n_hyperparams = lambda x: 1

    def __call__(self, log_hyperparams: np.ndarray, x: np.ndarray, z: np.ndarray = None):
        """
        adds white noise to kernel value, if z is given no noise is added
        :param log_hyperparams: hyperparameter set [[sigma_eps] x input dimension]
        :param x: samples of shape [n samples x dimensionality]
        :param z: samples of shape [n samples x dimensionality]
        :return: noise matrix or 0
        """

        K = 0
        if z is None:
            log_hyperparams = np.atleast_2d(log_hyperparams)

            sigma_eps = np.exp(2 * log_hyperparams).reshape(-1, 1, 1)
            K = sigma_eps * np.expand_dims(np.identity(x.shape[0]), 0)

        return K
