import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


class MGPR(GaussianProcessRegressor):
    """
    Multivariate Gaussian Process Regression
    """

    def __init__(self, dim, optimizer="fmin_l_bfgs_b", length_scale=1., sigma_f=1, sigma_eps=1, alpha=1e-10,
                 n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None):

        kernel = RBF(length_scale=length_scale) #+ WhiteKernel() * sigma_eps
        # TODO: Add WhiteKernel and sigma_f multiplication
        #+ Kernel_C()
        WhiteKernel()
        #[WhiteKernel(noise_level=sigma_eps)] * 4 #, noise_level_bounds=(1e-05, 100000.0)) # * sigma_f #, length_scale_bounds=(1e-05, 100000.0))\

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
                res = self.gp_container[i].predict(X, return_std=return_std, return_cov=return_cov)
                Y[:, i] = res
        if stat:
            return Y, np.array(stat)

        return Y

    def get_kernels(self, X):
        """
        Returns the gram matrix built out of all gaussian process containers.
        """
        gram_matrix = np.zeros((len(self.gp_container), len(X), len(X)))

        # concatenate all individual gram matrices for each gaussian process
        for i in range(len(self.gp_container)):
            gram_matrix[i] = self.gp_container[i].kernel_(X) #.diag(X)

        return gram_matrix

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

    def get_alphas(self):
        return np.array([c.alpha_ for c in self.gp_container])
