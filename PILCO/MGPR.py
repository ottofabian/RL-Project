import numpy as np
from numpy.dual import solve
from sklearn.gaussian_process import GaussianProcessRegressor

from PILCO.GaussianProcessRegressorOverDistribution import GaussianProcessRegressorOverDistribution


class MGPR(GaussianProcessRegressor):
    """
    Multivariate Gaussian Process Regression
    """

    def __init__(self, length_scales, n_targets, optimizer="fmin_l_bfgs_b", sigma_f=1, sigma_eps=1, alpha=1e-10,
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

        super(MGPR, self).__init__(alpha, optimizer, n_restarts_optimizer, normalize_y, copy_X_train, random_state)
        self.X = None
        self.y = None
        self.n_targets = n_targets
        # For a D-dimensional state space, we use D separate GPs, one for each state dimension.
        # - Efficient Reinforcement Learning using Gaussian Processes, Marc Peter Deisenroth
        self.gp_container = [
            GaussianProcessRegressorOverDistribution(length_scales=length_scales, sigma_eps=sigma_eps, sigma_f=sigma_f,
                                                     alpha=alpha, optimizer=optimizer, random_state=random_state,
                                                     is_fixed=is_fixed) for _ in range(self.n_targets)]

    def fit(self, X, y):
        """
        This is essentially used to compute the posterior of the GP, given the trainings samples
        :param X:
        :param y:
        :return:
        """
        self.X = X
        for i in range(self.n_targets):
            self.gp_container[i].fit(X, y[:, i])

    def predict(self, X, return_std=False, return_cov=False):
        y = np.empty((X.shape[0], self.n_targets))
        stat = []

        for i in range(self.n_targets):
            if return_cov or return_std:
                y[:, i], tmp = self.gp_container[i].predict(X, return_std=return_std, return_cov=return_cov)
                stat.append(tmp)
            else:
                res = self.gp_container[i].predict(X, return_std=return_std, return_cov=return_cov)
                y[:, i] = res
        if stat:
            return y, np.array(stat)

        return y

    def predict_from_dist(self, mu, sigma):

        """
        Predict dist given an uncertain input x~N(mu,sigma)
        :param mu: n_targets x n_state + n_actions
        :param sigma: n_targets x (n_state + n_actions) x (n_state + n_actions)
        :return: mu, sigma for each target
        """

        mu_out = np.zeros((self.n_targets,))
        sigma_out = np.zeros((self.n_targets, self.n_targets))
        input_output_cov = np.zeros((self.n_targets, self.X.shape[1]))

        # get the independet mus from the gps
        for i in range(self.n_targets):
            mu_out[i] = self.gp_container[i].predict_from_dist(mu, sigma)

        # The cov or delta x is not diagonal, therefor it is necessary to
        # compute the cross-cov between each output
        # This requires to compute the Expected value of the GP's outputs
        # from xa,xb - the product of the corresponding mean values from above

        # calculate combined Expectation

        # compute zeta before hand, it does not change
        zeta = (self.X - mu).T

        for i in range(self.n_targets):
            beta_a = self.gp_container[i].betas
            input_output_cov[i] = self.compute_input_output_cov(i, beta_a, sigma, zeta)

            for j in range(self.n_targets):

                Q = self.compute_cross_cov(i, j, zeta, sigma)
                beta_b = self.gp_container[j].betas

                # place into cov matrix
                if i == j:
                    cov_ab = beta_a.T @ Q @ beta_b - self.gp_container[i].sigma_f - \
                             np.trace(self.gp_container[i].K_inv @ Q)
                else:
                    cov_ab = beta_a.T @ Q @ beta_b - mu_out[i] * mu_out[j]

                sigma_out[i, j] = cov_ab

        return mu_out, sigma_out, input_output_cov

    def compute_cross_cov(self, i, j, zeta, sigma):
        precision_a_inv = np.diag(1 / self.gp_container[i].length_scales)
        precision_b_inv = np.diag(1 / self.gp_container[j].length_scales)

        # compute R
        R = sigma @ (precision_a_inv + precision_b_inv) + np.identity(sigma.shape[0])
        R_inv = solve(R, np.identity(R.shape[0]))

        # compute z
        z = precision_a_inv @ zeta + precision_b_inv @ zeta

        # compute n using matrix inversion lemma, Appendix A.4 Deisenroth (2010)
        right = (zeta.T @ precision_a_inv @ zeta +
                 zeta.T @ precision_b_inv @ zeta -
                 z.T @ R_inv @ sigma @ z) / 2

        const = 2 * (np.log(self.gp_container[i].sigma_f) + np.log(self.gp_container[j].sigma_f))
        n_square = const - right

        # Compute Q
        Q = np.exp(n_square) / np.sqrt(np.linalg.det(R))

        return Q

    def compute_input_output_cov(self, i, beta, sigma, zeta):

        precision = np.diag(self.gp_container[i].length_scales)
        # print(precision, sigma)
        sigma_plus_precision_inv = solve(sigma @ precision, np.identity(sigma.shape[0]))
        return np.sum(beta @ self.gp_container[i].qs * sigma @ sigma_plus_precision_inv @ zeta, axis=1)

    def get_kernels(self, X):
        """
        Returns the gram matrices given X for all gaussian processes.
        """
        K = np.zeros((len(self.gp_container), len(X), len(X)))

        # concatenate all individual gram matrices for each gaussian process
        for i in range(len(self.gp_container)):
            K[i] = self.gp_container[i].kernel_(X)

        return K

    def y_train_mean(self):
        means = []
        for i in range(self.n_targets):
            means.append(self.gp_container[i].y_train_mean())
        return means

    def sample_y(self, X, n_samples=1, random_state=0):
        samples = []
        for i in range(self.n_targets):
            samples.append(self.gp_container[i].sample_y(X, n_samples, random_state))
        return samples

    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        lml = []
        for i in range(self.n_targets):
            lml.append(self.gp_container[i].log_marginal_likelihood(theta, eval_gradient))
        return lml

    def get_sigma_fs(self):
        return np.array([c.sigma_f for c in self.gp_container])

    def get_sigma_eps(self):
        return np.array([c.sigma_eps for c in self.gp_container])
