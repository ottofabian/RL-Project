import numpy as np
from numpy.linalg import solve
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from PILCO.Controller.Controller import Controller


class RBFController(Controller):
    """RBF Controller/Policy"""

    def __init__(self, X, var: float, l: np.ndarray):
        """

        :param l: lengthscale of RBF Kernel
        :param X: training samples
        :param var: noise for WhiteKernel and training targets
        """

        # Hyperparams to optimize are y, length-scales, X and noise/var

        self.X = X
        self.kernel = RBF(length_scale=l) + WhiteKernel(var)

        # TODO var has to have shape n_sampels or 1 not sure yet
        self.var = var
        self.sigma = self.var * np.identity(X.shape[0])
        self.mu = np.zeros(self.sigma.shape[0])

        # self.beta = np.random.multivariate_normal(0, self.sigma, size=(X.shape[0], 1))
        self.beta = None

        # init taken from Master Thesis
        noise_eps = 0.01 ** 2
        self.y = np.random.multivariate_normal(np.zeros((X.shape[0])), noise_eps * np.identity(X.shape[0]))
        self._set_beta()

    def predict(self, x):
        """
        The predictive mean of p(π̃(x∗)) for a known state x∗ is equivalent to RBF policy in equation which itself is
        identical to the predictive mean of a GP. In contrast to the GP model, both the predictive variance and the
        uncertainty about the underlying function in an RBF network are zero.
        Thus, the predictive distribution p(π̃(x*)) for a given state x* has zero variance.
        :param x: point(s) to predict
        :return:
        """
        # TODO: Return mean and covar of action dist
        # beta should have size of n_samples ?????
        # return self.beta.T @ self.kernel(self.X, x)
        return self.beta.T @ self.kernel(self.X, x), 123

    def update_params(self, X):
        self.X = X

        self.sigma = self.var * np.identity(X.shape[0])
        self.mu = np.zeros(self.sigma.shape[0])

        self._set_beta()

    def _set_y(self):
        pred = self.predict(self.X)
        return pred + self.sample_measurement_noise()

    def _set_beta(self):
        K = self.kernel(self.X)
        self.beta = solve(K + self.sample_measurement_noise(), np.identity(K.shape[0])) @ self.y.T

    def sample_measurement_noise(self):
        return np.random.multivariate_normal(self.mu, self.sigma)

    # def __init__(self, W, sigma, mu):
    #     self.W = W
    #     self.sigma = sigma
    #     self.mu = mu
    #
    # def phi(self, X):
    #     phi = np.zeros((self.mu.shape[0], X.shape[1]))
    #     for i in range(self.mu.shape[0]):
    #         d = X - self.mu[i]
    #         phi[i] = np.exp(-.5 * d @ self.sigma @ d.T)
    #     return phi
    #
    # def predict(self, X):
    #     # TODO Check axis dim
    #     return np.sum(self.W @ self.phi(X), axis=1)
    #
    # def update_params(self, W, sigma, mu):
    #     self.W = W
    #     self.sigma = sigma
    #     self.mu = mu
