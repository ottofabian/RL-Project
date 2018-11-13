import numpy as np
from numpy.linalg import solve
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


class GaussianProcessRegressorOverDistribution(GaussianProcessRegressor):
    """
    Multivariate Gaussian Process Regression
    """

    def __init__(self, length_scales, optimizer="fmin_l_bfgs_b", sigma_f=1, sigma_eps=1e-6, alpha=1e-10,
                 n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None, is_fixed=False):
        """

        :param optimizer:
        :param sigma_f:
        :param sigma_eps:
        :param alpha:
        :param n_restarts_optimizer:
        :param normalize_y:
        :param copy_X_train:
        :param random_state:
        """

        # TODO: Change the length scales to a good initial value
        self.length_scales = length_scales

        # TODO: Check if WhiteKernel is used correctly
        if is_fixed:
            kernel = ConstantKernel(sigma_f, constant_value_bounds=(sigma_f, sigma_f)) * RBF(
                length_scale=length_scales) + WhiteKernel(sigma_eps, noise_level_bounds=(sigma_eps, sigma_eps))
        else:
            kernel = ConstantKernel(sigma_f) * RBF(length_scale=length_scales) + WhiteKernel(sigma_eps)

        super(GaussianProcessRegressorOverDistribution, self).__init__(kernel, alpha, optimizer, n_restarts_optimizer,
                                                                       normalize_y, copy_X_train,
                                                                       random_state)
        self.X = None
        self.y = None
        self.sigma_eps = sigma_eps
        self.length_scales = length_scales
        self.sigma_f = sigma_f
        self.is_fixed = is_fixed

        self.betas = None
        self.K_inv = None
        self.qs = None

    def fit(self, X, y):
        super(GaussianProcessRegressorOverDistribution, self).fit(X, y)
        self.X = X
        self.y = y

        # retrieve optimized hyperparams
        self.sigma_eps = np.exp(self.kernel_.theta[-1])
        self.sigma_f = np.exp(self.kernel_.theta[0])
        self.length_scales = np.exp(self.kernel_.theta[1:-1])

    def predict_from_dist(self, mu, sigma):
        """
        Returns
        :param mu: mean of joint state-action distribution
        :param sigma: covar of joint state-action distribution to represent uncertainty
        :return: mu, sigma of predictive distribution p(u)
        """

        # TODO: Maybe use this, but it adds no noise
        # L = self.kernel_.L_
        # L_inv = solve_triangular(L.T, np.eye(self.L_.shape[0]))
        # K_inv = L_inv.dot(L_inv.T)
        # ---------------------------------

        # This applies noise, but requires computing L
        K = self.kernel_(self.X)
        # learned variance from evidence maximization
        noise = np.identity(self.X.shape[0]) * self.sigma_eps
        # inv of K only using lower part of matrix
        # L = cholesky(K + noise, lower=True)
        # self.K_inv = solve_triangular(L, np.identity(self.X.shape[0]))

        # self.betas = solve_triangular(L, self.y)[:, 0]

        if self.is_fixed:
            # this is required for the RBF controller
            self.K_inv = np.zeros(K.shape)
        else:
            self.K_inv = solve(K + noise, np.identity(K.shape[0]))

        self.betas = K @ self.y

        return self.compute_mu(mu, sigma)

    def compute_mu(self, mu, sigma):
        """
        compute expectation of kernel(x_i, x*), where x*~N(mu, sigma) given the dist of the state  x~N(mu, sigma)
        :param mu:
        :param sigma:
        :return: mu of p(delta_t+1)
        """

        precision = np.diag(self.length_scales)
        precision_inv = solve(precision, np.identity(len(precision)))

        coefficient = self.sigma_f * np.linalg.det(sigma @ precision_inv + np.identity(len(precision_inv))) ** -.5

        sigma_plus_precision_inv = solve(sigma + precision, np.identity(len(precision)))

        diff = self.X - mu
        self.qs = np.array([coefficient * np.exp(-.5 * d.T @ sigma_plus_precision_inv @ d) for d in diff])

        return self.betas.T @ self.qs
