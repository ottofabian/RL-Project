import numpy as np


class RBFController(object):
    """RBF Controller/Policy"""

    def __init__(self, W, sigma, mu):
        self.W = W
        self.sigma = sigma
        self.mu = mu

    def phi(self, X):
        phi = np.zeros((self.mu.shape[0], X.shape[1]))
        for i in range(self.mu.shape[0]):
            d = X - self.mu[i]
            phi[i] = np.exp(-.5 * d @ self.sigma @ d.T)
        return phi

    def predict(self, X):
        phi = self.phi(X)
        # sum = np.zeros((X.shape[1],))
        # for i in range(self.W.shape[0]):
        #     sum += self.W[i] * phi[i]
        # return sum
        return np.sum(self.W @ phi)

    def update_params(self, W, sigma, mu):
        self.W = W
        self.sigma = sigma
        self.mu = mu
