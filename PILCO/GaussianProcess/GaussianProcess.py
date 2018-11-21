import logging

import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize

from PILCO.Kernels.Kernels import RBFKernel, WhiteNoiseKernel


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
        self.length_scale_pen = 100
        self.signal_to_noise = 500
        self.std_pen = 1

        self.betas = None
        self.K_inv = None
        self.qs = None

        self.logger = logging.getLogger(__name__)

    def fit(self, X, y):

        # ensure they are 2d
        y = y.reshape(y.shape[0], -1)
        # X = X.reshape(X.shape[0], -1)

        self.X = X
        self.y = y

        self.n_targets = y.shape[1]
        self.state_dim = X.shape[1]

        if not self.is_policy:
            self.std_pen = np.std(X, 0)

    def _optimize_hyperparams(self, params):
        p = 30
        length_scales, sigma_f, sigma_eps = self._unwrap_params(params)

        L = self.log_marginal_likelihood(params)
        L = L + np.sum(((length_scales - np.log(self.std_pen)) / np.log(self.length_scale_pen)) ** p)
        L = L + np.sum(((sigma_f - sigma_eps) / np.log(self.signal_to_noise)) ** p)
        # print(L)
        return L

    def optimize(self):
        """
        This is used to optimize the hyperparams for the GP
        :return:
        """
        self.logger.info("Optimization for GP started.")
        params = self._wrap_kernel_hyperparams()

        try:
            self.logger.info("Optimization with L-BFGS-B started.")
            res = minimize(value_and_grad(self._optimize_hyperparams), params, jac=True, method='L-BFGS-B')
        except Exception:
            self.logger.info("Optimization with CG started.")
            res = minimize(value_and_grad(self._optimize_hyperparams), params, jac=True, method='CG')

        best_params = res.x

        self.length_scales, self.sigma_f, self.sigma_eps = self._unwrap_params(best_params)
        self.compute_params()

    def _wrap_kernel_hyperparams(self):

        # split1 = self.state_dim
        # split2 = self.n_targets + split1
        #
        # params = np.zeros([self.state_dim + 2])
        # params[:split1] = np.log(self.length_scales)
        # params[split1:split2] = np.log(self.sigma_f)
        # params[split2:] = np.log(self.sigma_eps)
        #
        # return params

        return np.concatenate([self.length_scales, self.sigma_f, self.sigma_eps])

    def log_marginal_likelihood(self, hyp):

        K = self.kernel(hyp, self.X)[0]  # [n, n]
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(K, self.y)

        return 0.5 * self.X.shape[0] * self.n_targets * np.log(2 * np.pi) + 0.5 * np.dot(self.y.flatten(), alpha) + \
               np.sum(np.log(np.diag(L)))

    def compute_params(self):

        params = self._wrap_kernel_hyperparams()
        K = self.kernel(params, self.X)[0]

        # learned variance from evidence maximization
        noise = np.identity(self.X.shape[0]) * self.sigma_eps
        self.K_inv = np.linalg.solve(K + noise, np.identity(K.shape[0]))
        self.betas = K @ self.y

        # -------------------------------
        # TODO: Prob better, but autograd has no solve_triangular
        # L = np.linalg.cholesky(K + noise)
        # self.K_inv = solve_triangular(L, np.identity(self.X.shape[0]))
        # self.betas = solve_triangular(L, self.y)

    def compute_mu(self, mu, sigma):
        """
        Returns the new mean value of the predictive dist, e.g. p(u), given x~N(mu, sigma) via Moment matching
        :param mu: mean of input, e.g. state-action distribution
        :param sigma: covar of the input, e.g joint state-action distribution to represent uncertainty
        :return: mu of predictive distribution p(u)
        """

        # Numerically more stable???
        precision = np.diag(np.exp(self.length_scales))

        diff = (self.X - mu) @ precision

        # TODO: This is going to give nan for negative det
        B = precision @ sigma @ precision + np.identity(precision.shape[0])
        t = diff @ B

        coefficient = np.exp(2 * self.sigma_f) * np.linalg.det(B) ** -.5
        mean = coefficient * self.betas.T @ np.exp(-.5 * np.sum(diff * t, 1))

        return mean

    def _unwrap_params(self, params):

        split1 = self.state_dim
        split2 = self.n_targets + split1

        length_scales = params[:split1]
        sigma_f = params[split1:split2]
        sigma_eps = params[split2:]
        return length_scales, sigma_f, sigma_eps
