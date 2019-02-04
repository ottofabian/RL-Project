import logging

from typing import Union, Type

import dill as pickle
import autograd.numpy as np

from PILCO.GaussianProcess.GaussianProcess import GaussianProcess
from PILCO.GaussianProcess.RBFNetwork import RBFNetwork


class MultivariateGP(object):

    def __init__(self, n_targets: int, container: Union[Type[GaussianProcess], Type[RBFNetwork]],
                 length_scales: np.ndarray, sigma_f: np.ndarray, sigma_eps: np.ndarray, is_policy: bool = False):
        """
        Multivariate Gaussian Process Regression
        :param n_targets: amount of target, each dimension of data inputs requires one target
        :param container: submodel type for each target, this depends on whether this should model dynamics of RBF policy
        :param length_scales: prior for length scales
        :param sigma_f: prior for signal variance
        :param sigma_eps: prior for noise variance
        :param is_policy: is this instanced used as RBF policy or not,
                          the moment matching is consequently computed differently
        """

        self.X = None
        self.y = None
        self.n_targets = n_targets
        self.is_policy = is_policy

        # TODO make this in matrix vector form
        # For a D-dimensional state space, we use D separate GPs, one for each state dimension. Deisenroth (2010)
        self.gp_container = [
            container(length_scales=length_scales[i], sigma_eps=sigma_eps[i], sigma_f=sigma_f[i]) for i in
            range(self.n_targets)]

        self.logger = logging.getLogger(__name__)

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
            self.gp_container[i].fit(X, y[:, i:i + 1])

    def predict_from_dist(self, mu: np.ndarray, sigma: np.ndarray) -> tuple:

        """
        Predict dist given an uncertain input x~N(mu,sigma) from gaussian process
        Based on the idea of: https://github.com/cryscan/pilco-learner
        :param mu: n_targets x n_state + n_actions
        :param sigma: n_targets x (n_state + n_actions) x (n_state + n_actions)
        :return: mu, sigma and inv(sigma) @ input_output_cov
        """

        mu = np.atleast_2d(mu)

        state_dim = self.X.shape[1]
        target_dim = self.y.shape[1]

        [gp.compute_matrices() for gp in self.gp_container]

        # ----------------------------------------------------------------------------------------------------
        # Helper

        beta = np.vstack(np.array([gp.betas for gp in self.gp_container])).T
        length_scales = self.length_scales()
        sigma_f = self.sigma_fs().reshape(self.n_targets)

        precision_inv = np.stack(np.array([np.diag(np.exp(-l)) for l in length_scales]))
        precision_inv2 = np.stack(np.array([np.diag(np.exp(-2 * l)) for l in length_scales]))

        # centralized inputs
        diff = self.X - mu
        diff_scaled = np.expand_dims(diff, axis=0) / np.expand_dims(np.exp(2 * length_scales), axis=1)

        # ----------------------------------------------------------------------------------------------------
        # compute mean of predictive dist based on matlab code

        # The precision_inv cancels out later on
        zeta_a = diff @ precision_inv

        B = precision_inv @ sigma @ precision_inv + np.identity(state_dim)

        # t = zeta_a / B
        # B[i] is symmetric, so B[i].T=B[i]
        t = np.stack(np.array([np.linalg.solve(B[i], zeta_a[i].T).T for i in range(target_dim)]))

        scaled_beta = np.exp(-.5 * np.sum(zeta_a * t, axis=2)) * beta.T

        coefficient = np.exp(2 * sigma_f) * np.linalg.det(B) ** -.5

        mean = np.sum(scaled_beta, axis=1) * coefficient

        # compute cross cov between input and output times inv(S)
        zeta_b = t @ precision_inv
        input_output_cov = (np.transpose(zeta_b, [0, 2, 1]) @ np.expand_dims(scaled_beta, axis=2)).reshape(
            target_dim, state_dim).T * coefficient

        # ----------------------------------------------------------------------------------------------------
        # compute predictive covariance, non-central moments

        k = 2 * sigma_f.reshape(target_dim, 1) - .5 * np.sum(zeta_a ** 2, axis=2)

        diff_a = np.repeat(diff_scaled[:, np.newaxis, :, :], target_dim, axis=1)
        diff_b = -np.repeat(diff_scaled[np.newaxis, :, :, :], target_dim, axis=0)

        precision_add = np.expand_dims(precision_inv2, 0) + np.expand_dims(precision_inv2, 1)

        # compute R, which is used for scaling
        R = sigma @ precision_add + np.identity(state_dim)
        scaling_factor = np.linalg.det(R) ** -.5

        R_inv = np.stack(np.array(
            [np.linalg.solve(R.reshape(-1, state_dim, state_dim)[i], sigma) for i in range(target_dim ** 2)]))
        R_inv = R_inv.reshape(target_dim, target_dim, state_dim, state_dim) / 2

        # compute squared mahalanobis distance
        diff_a_Q = diff_a @ R_inv
        diff_b_Q = diff_b @ R_inv
        mahalanobis_dist = np.expand_dims(np.sum(diff_a_Q * diff_a, axis=-1), axis=-1) + np.expand_dims(
            np.sum(diff_b_Q * diff_b, axis=-1), axis=-2) - 2 * np.einsum('...ij, ...kj->...ik', diff_a_Q, diff_b)

        # compute Q matrix
        Q = np.exp(k[:, np.newaxis, :, np.newaxis] + k[np.newaxis, :, np.newaxis, :] + mahalanobis_dist)

        if self.is_policy:
            # noise for numerical reasons/ridge term
            cov = scaling_factor * np.einsum('ji,iljk,kl->il', beta, Q, beta) + 1e-6 * np.identity(target_dim)
        else:
            cov = np.einsum('ji,iljk,kl->il', beta, Q, beta)
            trace = np.hstack(np.array([np.sum(Q[i, i] * gp.K_inv) for i, gp in enumerate(self.gp_container)]))
            cov = (cov - np.diag(trace)) * scaling_factor + np.diag(np.exp(2 * sigma_f))

        # Centralize moments
        cov = cov - mean[:, np.newaxis] @ mean[np.newaxis, :]

        return mean, cov, input_output_cov

    def optimize(self):
        for i, gp in enumerate(self.gp_container):
            self.logger.info("Optimization for GP (output={}) started.".format(i))
            gp.optimize()

    def predict(self, X: np.ndarray):
        """
        Computes point predictions from GPs
        :param X: points to predict th man for
        :return:
        """
        # compute K_inv and betas
        [gp.compute_matrices() for gp in self.gp_container]
        return np.array([gp.predict(X) for gp in self.gp_container])

    # def compute_cross_cov(self, i, j, zeta, sigma):
    #     precision_a_inv = np.diag(1 / self.gp_container[i].length_scales)
    #     precision_b_inv = np.diag(1 / self.gp_container[j].length_scales)
    #
    #     # compute R
    #     R = sigma @ (precision_a_inv + precision_b_inv) + np.identity(sigma.shape[0])
    #     R_inv = np.linalg.solve(R, np.identity(R.shape[0]))
    #
    #     # compute z
    #     z = precision_a_inv @ zeta + precision_b_inv @ zeta
    #
    #     # compute n using matrix inversion lemma, Appendix A.4 Deisenroth (2010)
    #     right = (zeta.T @ precision_a_inv @ zeta +
    #              zeta.T @ precision_b_inv @ zeta -
    #              z.T @ R_inv @ sigma @ z) / 2
    #
    #     const = 2 * (np.log(self.gp_container[i].sigma_f) + np.log(self.gp_container[j].sigma_f))
    #     n_square = const - right
    #
    #     # Compute Q
    #     Q = np.exp(n_square) / np.sqrt(np.linalg.det(R))
    #
    #     return Q
    #
    # def predict_from_dist_v2(self, mu, sigma):
    #     """
    #     This method should not be used, it is only based on the mathematical description of Deisenroth(2010).
    #     Use predict_from_dist() in order to get a matlab based method, which is numerically more stable.
    #     :param mu:
    #     :param sigma:
    #     :return:
    #     """
    #
    #     mu_out = np.zeros((self.n_targets,))
    #     sigma_out = np.zeros((self.n_targets, self.n_targets))
    #     input_output_cov = np.zeros((self.n_targets, self.X.shape[1]))
    #
    #     # calculate combined Expectation of all gps
    #     for i in range(self.n_targets):
    #         mu_out[i] = self.gp_container[i].compute_matrices(mu, sigma)
    #
    #     # The cov og e.g. delta x is not diagonal, therefor it is necessary to
    #     # compute the cross-cov between each output
    #     # This requires to compute the expected value of the GP's outputs
    #     # from xa,xb - the product of the corresponding mean values from above
    #
    #     # compute zeta before hand, it does not change
    #     zeta = (self.X - mu).T
    #
    #     for i in range(self.n_targets):
    #         beta_a = self.gp_container[i].betas
    #         input_output_cov[i] = self.compute_input_output_cov(i, beta_a, sigma, zeta)
    #
    #         for j in range(self.n_targets):
    #
    #             Q = self.compute_cross_cov(i, j, zeta, sigma)
    #             beta_b = self.gp_container[j].betas
    #
    #             # place into cov matrix
    #             if i == j:
    #                 cov_ab = beta_a.T @ Q @ beta_a - mu_out[i] ** 2 + self.gp_container[i].sigma_f ** 2 - \
    #                          np.trace(self.gp_container[i].K_inv @ Q)
    #             else:
    #                 cov_ab = beta_a.T @ Q @ beta_b - mu_out[i] * mu_out[j]
    #
    #             sigma_out[i, j] = cov_ab
    #
    #     return mu_out, sigma_out, input_output_cov
    #
    # def compute_input_output_cov(self, i, beta, sigma, zeta):
    #
    #     precision = np.diag(self.gp_container[i].length_scales)
    #     # print(precision, sigma)
    #     sigma_plus_precision_inv = np.linalg.solve(sigma @ precision, np.identity(sigma.shape[0]))
    #     return np.sum(beta @ self.gp_container[i].qs * sigma @ sigma_plus_precision_inv @ zeta, axis=1)

    def save(self, reward):
        pickle.dump(self, open(f"dynamics_reward-{reward}.p", "wb"))

    def sigma_fs(self):
        return np.array([c.sigma_f for c in self.gp_container])

    def sigma_eps(self):
        return np.array([c.sigma_eps for c in self.gp_container])

    def length_scales(self):
        return np.array([c.length_scales for c in self.gp_container])
