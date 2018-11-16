import numpy as np
# from scipy.linalg import np.linalg.solve
from scipy.linalg import solve
from sklearn.gaussian_process import GaussianProcessRegressor

from PILCO.GaussianProcess.GaussianProcessRegressorOverDistribution import GaussianProcessRegressorOverDistribution


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
        self.y = y
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
        Based on the idea of: https://github.com/cryscan/pilco-learner
        :param mu: n_targets x n_state + n_actions
        :param sigma: n_targets x (n_state + n_actions) x (n_state + n_actions)
        :return: mu, sigma for each target
        """

        state_dim = self.X.shape[1]
        target_dim = self.y.shape[1]

        [gp.compute_params() for gp in self.gp_container]

        beta = np.vstack([gp.alpha for gp in self.gp_container]).T
        length_scales = self.get_length_scales()

        # compute mean of predictive dist based on matlab code
        precision_inv = np.stack([np.diag(1 / l) for l in length_scales])
        diff = self.X - mu
        # The precision_inv cancels out later on
        diff_precision = diff @ precision_inv

        B = precision_inv @ sigma @ precision_inv + np.eye(state_dim)

        # diff / B, this has to be done differently in python
        diff_B = np.stack([solve(B[i].T, diff_precision[i].T).T for i in range(target_dim)])

        scaled_beta = np.exp(-np.sum(diff_precision * diff_B, 2) / 2) * beta.T
        # lets hope det(B) is not negative, happend more than once :(
        coefficient = 2 * self.get_sigma_fs() * np.linalg.det(B) ** -.5

        mean = np.sum(scaled_beta, 1) * coefficient

        # compute cross cov between input and output
        diff_B_precision = np.matmul(diff_B, precision_inv)
        input_output_cov = (np.transpose(diff_B_precision, [0, 2, 1]) @ np.expand_dims(scaled_beta, 2)).reshape(
            target_dim, state_dim).T * coefficient
        input_output_cov = sigma @ input_output_cov

        # compute predictive covariance

        k = 2 * self.get_sigma_fs().reshape(target_dim, 1) - np.sum(diff_precision ** 2, 2) * .5

        diff_scaled = np.expand_dims(diff, 0) / np.expand_dims(2 * length_scales, 1)
        diff_a = np.repeat(diff_scaled[:, np.newaxis, :, :], target_dim, 1)
        diff_b = np.repeat(diff_scaled[np.newaxis, :, :, :], target_dim, 0)

        precision_inv = np.stack([np.diag(1 / (2 * l)) for l in length_scales])
        precision_add = np.expand_dims(precision_inv, 0) + np.expand_dims(precision_inv, 1)

        # compute R, which is used for scaling
        R = np.matmul(sigma, precision_add) + np.eye(state_dim)
        denominator = np.linalg.det(R) ** -.5

        R_inv = np.stack([solve(R.reshape(-1, state_dim, state_dim)[i], sigma) for i in range(target_dim ** 2)])
        R_inv = R_inv.reshape(target_dim, target_dim, state_dim, state_dim)

        # compute mahalanobis distance
        aQ = np.matmul(diff_a, R_inv / 2)
        bQ = np.matmul(-diff_b, R_inv / 2)
        mahalanobis_dist = np.expand_dims(np.sum(aQ * diff_a, -1), -1) + np.expand_dims(
            np.sum(bQ * diff_b, -1), -2) - 2 * np.einsum('...ij, ...kj->...ik', aQ, diff_b)

        # compute Q matrix
        Q = np.exp(k[:, np.newaxis, :, np.newaxis] + k[np.newaxis, :, np.newaxis, :] + mahalanobis_dist)

        cov = np.einsum('ji,iljk,kl->il', beta, Q, beta)
        trace = np.hstack([np.sum(Q[i, i] * gp.K_inv) for i, gp in enumerate(self.gp_container)])
        cov = (cov - np.diag(trace)) * denominator + np.diag(2 * length_scales)
        cov = cov - np.matmul(mean[:, np.newaxis], mean[np.newaxis, :])

        return mean, cov, input_output_cov.T

    def predict_from_dist_v2(self, mu, sigma):

        """
        This method should not be used, it is only based on the mathematical description of Deisenroth(2010).
        Use predict_from_dist() in order to get a matlab based method, which is numerically more stable.
        :param mu:
        :param sigma:
        :return:
        """

        mu_out = np.zeros((self.n_targets,))
        sigma_out = np.zeros((self.n_targets, self.n_targets))
        input_output_cov = np.zeros((self.n_targets, self.X.shape[1]))

        # calculate combined Expectation of all gps
        for i in range(self.n_targets):
            mu_out[i] = self.gp_container[i].compute_params(mu, sigma)

        # The cov og e.g. delta x is not diagonal, therefor it is necessary to
        # compute the cross-cov between each output
        # This requires to compute the expected value of the GP's outputs
        # from xa,xb - the product of the corresponding mean values from above

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
                    cov_ab = beta_a.T @ Q @ beta_a - mu_out[i] ** 2 + self.gp_container[i].sigma_f ** 2 - \
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
        R_inv = np.linalg.solve(R, np.identity(R.shape[0]))

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
        sigma_plus_precision_inv = np.linalg.solve(sigma @ precision, np.identity(sigma.shape[0]))
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
        return np.array([gp.y_train_mean for gp in self.gp_container])

    def sample_y(self, X, n_samples=1, random_state=0):
        return np.array([gp.sample_y(X, n_samples, random_state) for gp in self.gp_container])

    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        return np.array([gp.log_marginal_likelihood(theta, eval_gradient) for gp in self.gp_container])

    def get_sigma_fs(self):
        return np.array([c.sigma_f for c in self.gp_container])

    def get_sigma_eps(self):
        return np.array([c.sigma_eps for c in self.gp_container])

    def get_length_scales(self):
        return np.array([c.length_scales for c in self.gp_container])
