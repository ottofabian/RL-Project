import logging
from typing import Union

import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize

from pilco.kernel.rbf_kernel import RBFKernel
from pilco.kernel.white_noise_kernel import WhiteNoiseKernel


class GaussianProcess(object):

    def __init__(self, length_scales: np.ndarray, sigma_f: Union[np.ndarray, float] = 1,
                 sigma_eps: Union[np.ndarray, float] = 1, length_scale_pen: float = 100, signal_to_noise: float = 500):
        """
        Gaussian Process Regression
        penalty parameters are for the modified log-likelhihood optimization as in Deisenroth(2010)
        :param length_scales: prior for length scale values
        :param sigma_f: prior for signal variance
        :param sigma_eps: prior for noise variance
        :param length_scale_pen: penalty for lengthscales
        :param signal_to_noise: signal to noise ratio in order to trade off signal and noise variance
        """

        # kernel definition as in Deisenroth(2010), p.10
        self.kernel = RBFKernel() + WhiteNoiseKernel()

        # data of GP
        self.x = None
        self.y = None

        # hyperparameters of GP
        self.length_scales = length_scales
        self.sigma_f = np.atleast_1d(sigma_f)
        self.sigma_eps = np.atleast_1d(sigma_eps)

        # dimensionality of data points
        self.n_targets = None
        self.state_dim = None

        # params to penalize bad hyperparams choices when optimizing log likelihood GP,
        # this is only required for the dynamics model
        self.length_scale_pen = length_scale_pen
        self.signal_to_noise = signal_to_noise

        # container for caching gram matrix, betas and inv of gram matrix
        self.K = None
        self.betas = None
        self.K_inv = None

    def set_XY(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        set x and y
        :param x: input variables [n_samples, sample dim]
        :param y: target variables [n_samples, 1]
        :return:
        """

        # ensure they are 2d
        y = y.reshape(y.shape[0], -1)

        self.x = x
        self.y = y

        self.n_targets = y.shape[1]
        self.state_dim = x.shape[1]

    def _optimize_hyperparams(self, params: np.ndarray) -> float:
        """
        function handle for scipy optimizer
        :param params: vector of [length scales, signal variance, noise variance]
        :return: penalized marginal log likelihood
        """
        likelihood = -self.log_marginal_likelihood(params)

        # penalty computation
        p = 30
        length_scales, sigma_f, sigma_eps = self.unwrap_params(params)
        std = np.std(self.x, axis=0)

        likelihood = likelihood + (((length_scales - np.log(std)) / np.log(self.length_scale_pen)) ** p).sum()
        likelihood = likelihood + (((sigma_f - sigma_eps) / np.log(self.signal_to_noise)) ** p).sum()

        return likelihood

    def optimize(self) -> None:
        """
        This is used to optimize the hyperparams for the GP
        :return: None
        """

        # start optimizing with current parameter set
        params = self._wrap_kernel_hyperparams()

        logging.debug("Length scales before: {}".format(np.array2string(np.exp(self.length_scales))))
        logging.debug("Sigma_f before: {}".format(np.array2string(np.exp(self.sigma_f))))
        logging.debug("Sigma_eps before: {}".format(np.array2string(np.exp(self.sigma_eps))))

        try:
            logging.info("Optimization with L-BFGS-B started.")
            res = minimize(value_and_grad(self._optimize_hyperparams), params, jac=True, method='L-BFGS-B')
        except Exception:
            # use CG if numerical instabilities occur during optimization
            logging.info("Optimization with CG started.")
            res = minimize(value_and_grad(self._optimize_hyperparams), params, jac=True, method='CG')

        best_params = res.x

        self.length_scales, self.sigma_f, self.sigma_eps = self.unwrap_params(best_params)

        logging.debug("Length scales after: {}".format(np.array2string(np.exp(self.length_scales))))
        logging.debug("Sigma_f after: {}".format(np.array2string(np.exp(self.sigma_f))))
        logging.debug("Sigma_eps after: {}".format(np.array2string(np.exp(self.sigma_eps))))

        # compute betas and K_inv which is required for later predictions
        self.compute_matrices()

    def _wrap_kernel_hyperparams(self) -> np.ndarray:
        """
        wraps GP hyperparams to vector.
        Required for optimization.
        :return: vector of [length scales, signal variance, noise variance]
        """
        return np.concatenate([self.length_scales, self.sigma_f, self.sigma_eps])

    def log_marginal_likelihood(self, hyperparams: np.ndarray) -> float:
        """
        compute log marginal likelihood for given hyperparameter set
        :param hyperparams: vector of [length scales, signal variance, noise variance]
        :return: log marginal likelihood
        """

        K = self.kernel(hyperparams, self.x)[0]

        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(K, self.y)

        return -.5 * self.x.shape[0] * self.n_targets * np.log(2 * np.pi) \
               - .5 * np.dot(self.y.flatten(order="F"), alpha) - (np.log(np.diag(L))).sum()

    def compute_matrices(self) -> None:

        """
        recomputes kernel matrix, the inverse of the kernel matrix and betas,
        which is requires after updating the parameters.
        This is essentially caching these values.
        :return: None
        """

        params = self._wrap_kernel_hyperparams()
        self.K = self.kernel(params, self.x)[0]  # [1,n,n]

        self.K_inv = np.linalg.solve(self.K, np.identity(self.K.shape[0]))
        self.betas = np.linalg.solve(self.K, self.y).T

    def unwrap_params(self, params) -> tuple:
        """
        unwrap vector of hyperparams into separate values for gp
        Required for optimization
        :param params: vector of [length scales, signal variance, noise variance]
        :return: length scales, sigma_f, sigma_eps
        """

        split1 = self.state_dim
        split2 = self.n_targets + split1

        length_scales = params[:split1]
        sigma_f = params[split1:split2]
        sigma_eps = params[split2:]
        return length_scales, sigma_f, sigma_eps
