import numpy as np
from numpy.linalg import solve
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from PILCO.Controller.Controller import Controller


class RBFController(Controller):
    """RBF Controller/Policy"""

    def __init__(self, X, var: np.ndarray):
        self.X = X
        self.kernel = RBF() + WhiteKernel(var)

        # TODO var has to have shape n_sampels or 1 not sure yet
        self.var = var
        self.sigma = self.var * np.identity(X.shape[0])
        self.mu = np.zeros(self.sigma.shape[0])

        self.beta = np.random.normal(0, var, size=(X.shape[0], 1))

    def predict(self, x):
        """
        The predictive mean of p(π̃(x∗)) for a known state x∗ is equivalent to RBF policy in equation which itself is
        identical to the predictive mean of a GP. In contrast to the GP model, both the predictive variance and the
        uncertainty about the underlying function in an RBF network are zero.
        Thus, the predictive distribution p(π̃(x*)) for a given state x* has zero variance.
        :param x: point(s) to predict
        :return:
        """
        # TODO: add weight parameter beta
        # beta should have size of n_samples ?????
        # return self.beta.T @ self.kernel(self.X, x)
        return self.beta.T @ self.kernel(self.X, x)

    def update_params(self, X):
        self.X = X

        self.sigma = self.var * np.identity(X.shape[0])
        self.mu = np.zeros(self.sigma.shape[0])

        self._set_beta()

    def get_y(self):
        pred = self.predict(self.X)
        return pred + self.sample_measurement_noise()

    def _set_beta(self):
        K = self.kernel(self.X)
        y = self.get_y()
        self.beta = solve(K + self.sample_measurement_noise(), np.identity(K.shape[0])) @ y.T

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
