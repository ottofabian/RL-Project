import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


class GaussianProcessRegressorOverDistribution(GaussianProcessRegressor):
    """
    Gaussian Process Regression given an input Gaussian
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
        # self.sigma_eps = sigma_eps
        # self.length_scales = length_scales
        # self.sigma_f = sigma_f
        self.is_fixed = is_fixed

        self.betas = None
        self.K_inv = None
        self.qs = None

    def fit(self, X, y):
        if not self.is_fixed:
            super(GaussianProcessRegressorOverDistribution, self).fit(X, y)
        self.X = X
        self.y = y

    def compute_params(self):

        # TODO: Maybe use this, but it adds no noise
        # L = self.kernel_.L_
        # L_inv = np.linalg.solve_triangular(L.T, np.eye(self.L_.shape[0]))
        # K_inv = L_inv.dot(L_inv.T)
        # ---------------------------------

        # This applies noise, but requires computing L
        K = self.kernel(self.X)

        if self.is_fixed:
            # this is required for the RBF controller
            self.K_inv = np.zeros(K.shape)
            self.betas = K @ self.y
        else:
            # learned variance from evidence maximization
            noise = np.identity(self.X.shape[0]) * self.sigma_eps
            self.K_inv = np.linalg.solve(K + noise, np.identity(K.shape[0]))
            self.betas = K @ self.y
            # -------------------------------
            # TODO: Prob better
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

        # TODO det is still nan, error somwhere else?
        # Numerically more stable???
        # precision = np.diag(self.length_scales)
        # precision_inv = np.linalg.solve(precision, np.identity(len(precision)))

        # diff = (self.X - mu) @ precision
        #
        # # TODO: This is going to give nan for negative det
        # B = precision @ sigma @ precision + np.identity(precision.shape[0])
        # t = diff @ B
        #
        # if np.any(np.isnan(B)):
        #     print("Nan is B")
        #
        # coefficient = 2 * self.sigma_f * np.linalg.det(B) ** -.5
        #
        # # sigma_plus_precision_inv = np.linalg.solve(sigma + precision, np.identity(len(precision)))
        #
        # mean = coefficient * self.betas.T @ np.exp(-.5 * np.sum(diff * t, 1))
        #
        # return mean

        # As seen in paper implementation
        # precision = np.diag(self.length_scales)
        # precision_inv = np.linalg.solve(precision, np.identity(len(precision)))
        #
        # TODO: This is going to give nan for negative det
        precision = np.diag(self.length_scales)
        precision_inv = np.diag(1 / self.length_scales)
        coefficient = self.sigma_f * np.linalg.det(sigma @ precision_inv + np.identity(len(precision_inv))) ** -.5

        sigma_plus_precision_inv = np.linalg.solve(sigma + precision, np.identity(len(precision)))

        diff = self.X - mu
        self.qs = np.array([coefficient * np.exp(-.5 * d.T @ sigma_plus_precision_inv @ d) for d in diff])

        return self.betas.T @ self.qs

    @property
    def length_scales(self):
        return self.kernel.get_params()['k1__k2__length_scale']

    @length_scales.setter
    def length_scales(self, length_scales: np.ndarray) -> None:
        self.kernel.set_params(k1__k2__length_scale=length_scales)

    @property
    def sigma_f(self):
        return self.kernel.get_params()['k1__k1__constant_value']

    @property
    def sigma_eps(self):
        return self.kernel.get_params()['k2__noise_level']
