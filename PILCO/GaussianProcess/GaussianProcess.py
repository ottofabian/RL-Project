import logging

import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize

from PILCO.Kernels.RBFKernel import RBFKernel
from PILCO.Kernels.WhiteNoiseKernel import WhiteNoiseKernel


class GaussianProcess(object):
    """
    Gaussian Process Regression
    """

    def __init__(self, length_scales, sigma_f=1, sigma_eps=1, is_policy=False):

        self.kernel = RBFKernel() + WhiteNoiseKernel()

        self.X = None
        self.y = None
        self.sigma_eps = np.atleast_1d(sigma_eps)
        self.length_scales = length_scales
        self.sigma_f = np.atleast_1d(sigma_f)

        self.n_targets = None
        self.state_dim = None

        self.is_policy = is_policy

        # params to penalize high hyperparams
        # TODO tune those hyperparams
        self.length_scale_pen = 100
        self.signal_to_noise = 500
        self.std_pen = 1

        self.K = None
        self.betas = None
        self.K_inv = None

        self.logger = logging.getLogger(__name__)

    def fit(self, X, y):

        # ensure they are 2d
        y = y.reshape(y.shape[0], -1)

        self.X = X
        self.y = y

        self.n_targets = y.shape[1]
        self.state_dim = X.shape[1]

        if not self.is_policy:
            self.std_pen = np.std(X, 0)

    def _optimize_hyperparams(self, params):
        """
        function handle for scipy optimizer
        :param params: vector of [length_scales, sigma_f, sigma_eps]
        :return: penalized marginal log likelihood
        """
        p = 30
        length_scales, sigma_f, sigma_eps = self.unwrap_params(params)

        L = self.log_marginal_likelihood(params)

        L = L + np.sum(((length_scales - np.log(self.std_pen)) / np.log(self.length_scale_pen)) ** p)
        L = L + np.sum(((sigma_f - sigma_eps) / np.log(self.signal_to_noise)) ** p)
        # print(L)
        return L

    def optimize(self) -> None:
        """
        This is used to optimize the hyperparams for the GP
        :return: None
        """

        # start optimizing with current parameter set
        params = self._wrap_kernel_hyperparams()

        self.logger.debug("Log Length scales before: {}".format(np.array2string(self.length_scales)))
        self.logger.debug("Log Sigma_f before: {}".format(np.array2string(self.sigma_f)))
        self.logger.debug("Log Sigma_eps before: {}".format(np.array2string(self.sigma_eps)))

        try:
            self.logger.info("Optimization with L-BFGS-B started.")
            res = minimize(value_and_grad(self._optimize_hyperparams), params, jac=True, method='L-BFGS-B')
        except Exception:
            # use CG if numerical instablities occur during optimization
            self.logger.info("Optimization with CG started.")
            res = minimize(value_and_grad(self._optimize_hyperparams), params, jac=True, method='CG')

        best_params = res.x

        self.length_scales, self.sigma_f, self.sigma_eps = self.unwrap_params(best_params)

        self.logger.debug("Log Length scales after: {}".format(np.array2string(self.length_scales)))
        self.logger.debug("Log Sigma_f after: {}".format(np.array2string(self.sigma_f)))
        self.logger.debug("Log Sigma_eps after: {}".format(np.array2string(self.sigma_eps)))

        # compute betas and K_inv which is required for later predictions
        self.compute_matrices()

    def _wrap_kernel_hyperparams(self) -> np.ndarray:
        """
        wraps GPS hyperparams to vector.
        Required for optimization.
        :return: vector of [length scales, sigma_f, sigma_eps]
        """
        return np.concatenate([self.length_scales, self.sigma_f, self.sigma_eps])

    def predict(self, x: np.ndarray, normalize=False) -> np.ndarray:
        """
        Computes point predictions from GPs
        :param x: points to predict th man for
        :return: point prediction for the corresponing state dim
        """
        # K_trans = self.kernel(self._wrap_kernel_hyperparams(), x)
        # y_mean = K_trans.dot(self.betas)
        y_mean = np.mean(self.y, axis=0) if normalize else 0
        return (y_mean + self.betas @ self.kernel(self._wrap_kernel_hyperparams(), x, self.X)[0]).flatten()

    def log_marginal_likelihood(self, hyperparams: np.ndarray) -> float:
        """
        compute log marginal likelihood for given hyperparameter set
        :param hyperparams: vector of [length scales, sigma_f, sigma_eps]
        :return: negative log marginal likelihood
        """

        K = self.kernel(hyperparams, self.X)[0]
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(K, self.y)

        return 0.5 * self.X.shape[0] * self.n_targets * np.log(2 * np.pi) \
               + 0.5 * np.dot(self.y.flatten(order="F"), alpha) + np.sum(np.log(np.diag(L)))

    def compute_matrices(self) -> None:

        """
        recomputes kernel matrix, the inverse of the kernel matrix and betas,
        which is requires after updating the parameters.
        This can be seen as caching those values.
        :return: None
        """

        params = self._wrap_kernel_hyperparams()
        self.K = self.kernel(params, self.X)[0]  # [1,n,n]

        # noise is already added with WhiteNoiseKernel
        # noise = np.identity(self.X.shape[0]) * self.sigma_eps
        # self.K_inv = np.linalg.solve(self.K + noise, np.identity(self.K.shape[0]))

        self.K_inv = np.linalg.solve(self.K, np.identity(self.K.shape[0]))
        self.betas = np.linalg.solve(self.K, self.y).T

        # -------------------------------
        # TODO: Prob better, but autograd has no solve_triangular
        # L = np.linalg.cholesky(K + noise)
        # self.K_inv = solve_triangular(L, np.identity(self.X.shape[0]))
        # self.betas = solve_triangular(L, self.y)

    # def compute_mu(self, mu, sigma):
    #     """
    #     Returns the new mean value of the predictive dist, e.g. p(u), given x~N(mu, sigma) via Moment matching
    #     :param mu: mean of input, e.g. state-action distribution
    #     :param sigma: covar of the input, e.g joint state-action distribution to represent uncertainty
    #     :return: mu of predictive distribution p(u)
    #     """
    #
    #     # Numerically more stable???
    #     precision = np.diag(np.exp(self.length_scales))
    #
    #     diff = (self.X - mu) @ precision
    #
    #     # TODO: This is going to give nan for negative det
    #     B = precision @ sigma @ precision + np.identity(precision.shape[0])
    #     t = diff @ B
    #
    #     coefficient = np.exp(2 * self.sigma_f) * np.linalg.det(B) ** -.5
    #     mean = coefficient * self.betas.T @ np.exp(-.5 * np.sum(diff * t, 1))
    #
    #     return mean

    def unwrap_params(self, params) -> tuple:
        """
        unwrap vector of hyperparams into seperate values for gp
        Required for optimization
        :param params: vector of [length scales, sigma_f, sigma_eps]
        :return: length scales, sigma_f, sigma_eps
        """

        split1 = self.state_dim
        split2 = self.n_targets + split1

        length_scales = params[:split1]
        sigma_f = params[split1:split2]
        sigma_eps = params[split2:]
        return length_scales, sigma_f, sigma_eps
