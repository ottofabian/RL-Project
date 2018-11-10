import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


class MGPR(GaussianProcessRegressor):
    """
    Multivariate Gaussian Process Regression
    """

    def __init__(self, dim, optimizer="fmin_l_bfgs_b", length_scale=1., sigma_f=1, sigma_eps=1, alpha=1e-10,
                 n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None):

        kernel = sigma_f * RBF(length_scale=length_scale, length_scale_bounds=(1e-05, 100000.0)) \
                 + WhiteKernel(noise_level=sigma_eps, noise_level_bounds=(1e-05, 100000.0))

        super(MGPR, self).__init__(kernel, alpha, optimizer, n_restarts_optimizer, normalize_y, copy_X_train,
                                   random_state)
        self.dim = dim
        # For a D-dimensional state space, we use D separate GPs, one for each state dimension.
        # - Efficient Reinforcement Learning using Gaussian Processes, Marc Peter Deisenroth
        self.gp_container = [
            GaussianProcessRegressor(kernel, alpha, optimizer, n_restarts_optimizer, normalize_y, copy_X_train,
                                     random_state) for _ in range(dim)]

    def fit(self, X, Y):
        for i in range(self.dim):
            self.gp_container[i].fit(X, Y[:, i])

    def predict(self, X, return_std=False, return_cov=False):
        Y = np.empty((X.shape[0], self.dim))
        stat = []

        for i in range(self.dim):
            if return_cov or return_std:
                Y[:, i], tmp = self.gp_container[i].predict(X, return_std=return_std, return_cov=return_cov)
                stat.append(tmp)
            else:
                Y[:, i] = self.gp_container[i].predict(X, return_std=return_std, return_cov=return_cov)
        if stat:
            return Y, stat

        return Y

    def y_train_mean(self):
        means = []
        for i in range(self.dim):
            means.append(self.gp_container[i].y_train_mean())
        return means

    def sample_y(self, X, n_samples=1, random_state=0):
        samples = []
        for i in range(self.dim):
            samples.append(self.gp_container[i].sample_y(X, n_samples, random_state))
        return samples

    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        lml = []
        for i in range(self.dim):
            lml.append(self.gp_container[i].log_marginal_likelihood(theta, eval_gradient))
        return lml
