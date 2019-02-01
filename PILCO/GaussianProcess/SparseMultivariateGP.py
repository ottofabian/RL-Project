import logging

import numpy as np
import GPy

from typing import Union, Type

from GPy import likelihoods

from PILCO.GaussianProcess import GaussianProcess, RBFNetwork
from PILCO.GaussianProcess.MultivariateGP import MultivariateGP


class SparseMultivariateGP(MultivariateGP):

    def __init__(self, X, y,
                 n_targets: int,
                 n_inducing_points: int,
                 container,
                 length_scales: np.ndarray,
                 sigma_f: np.ndarray,
                 sigma_eps: np.ndarray,
                 is_policy: bool = False):

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

        super(SparseMultivariateGP, self).__init__(y.shape[1], container, length_scales, sigma_f, sigma_eps, is_policy)

        self.X = X
        self.y = y
        self.n_targets = n_targets
        self.is_policy = is_policy
        self.n_inducing_points = n_inducing_points

        self.logger = logging.getLogger(__name__)

        self.gp_container = []
        idx = np.random.permutation(X.shape[0])[:min(n_inducing_points, X.shape[0])]
        Z = X.view(np.ndarray)[idx].copy()

        for i in range(self.n_targets):
            input_dim = length_scales.shape[1]
            kernel = GPy.kern.RBF(input_dim=input_dim, lengthscale=np.exp(length_scales[i]), ARD=True,
                                  variance=np.exp(sigma_f[i]))

            kernel = kernel + GPy.kern.White(input_dim=input_dim, variance=np.exp(sigma_eps[i]))
            model = GPy.models.SparseGPRegression(X=X, Y=y[:, i:i + 1], kernel=kernel, Z=Z)
            self.gp_container.append(model)

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
            self.gp_container[i].set_XY(X, y[:, i:i + 1])

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
        induced_dim = np.array(self.gp_container[0].Z).shape[0]

        # TODO move this part somewhere else

        Kmm = np.stack([gp.kern.rbf.K(gp.Z) + 1e-6 * np.identity(induced_dim) for gp in self.gp_container])
        Kmn = np.stack([gp.kern.rbf.K(gp.Z, gp.X) for gp in self.gp_container])

        L = np.linalg.cholesky(Kmm)

        V = np.stack([np.linalg.solve(L[i], Kmn[i]) for i in range(target_dim)])  # inv(sqrt(Kmm)) * Kmn
        G = np.exp(2 * self.sigma_fs()) - np.sum(V ** 2, axis=1)
        G = np.sqrt(1. + G / np.exp(2 * self.sigma_eps()))  # this is nan for theta_dot, fuck this algorithm
        V_scaled = V / G[:, None]

        # noise = np.expand_dims(np.identity(self.n_inducing_points), 0) * np.expand_dims(np.exp(2 * self.sigma_eps()), 1)
        Am = np.linalg.cholesky(np.stack(
            [V_scaled[i] @ V_scaled[i].T + np.identity(induced_dim) * np.exp(2 * self.sigma_eps()[i]) for i
             in range(target_dim)]))

        At = L @ Am
        iAt = np.stack([np.linalg.solve(At[i], np.identity(induced_dim)) for i in range(target_dim)])

        V_scaled = V / G[:, None]
        # one big ugly loopy, because numpy cannot do it differently
        beta = np.stack([np.linalg.solve(L[i], np.linalg.solve(Am[i], (V_scaled[i]) @ gp.Y)) for i, gp in
                         enumerate(self.gp_container)])[:, :, 0]

        iB = np.stack([iAt[i] @ iAt[i].T for i in range(target_dim)]) * self.sigma_eps()[:, None, None]
        iK = np.stack([np.linalg.solve(L[i], np.identity(induced_dim)) for i in range(target_dim)]) - iB

        # ----------------------------------------------------------------------------------------------------
        # Helper

        # beta = np.vstack([gp.betas for gp in self.gp_container]).T
        length_scales = self.length_scales()
        sigma_f = self.sigma_fs().reshape(self.n_targets)

        precision_inv = np.stack([np.diag(np.exp(-l)) for l in length_scales])
        precision_inv2 = np.stack([np.diag(np.exp(-2 * l)) for l in length_scales])

        # centralized inputs
        diff = np.array(self.gp_container[0].Z) - mu
        diff_scaled = np.expand_dims(diff, axis=0) / np.expand_dims(np.exp(2 * length_scales), axis=1)

        # ----------------------------------------------------------------------------------------------------
        # compute mean of predictive dist based on matlab code

        # The precision_inv cancels out later on
        zeta_a = diff @ precision_inv

        B = precision_inv @ sigma @ precision_inv + np.identity(state_dim)

        # t = zeta_a / B
        # B[i] is symmetric, so B[i].T=B[i]
        t = np.stack([np.linalg.solve(B[i], zeta_a[i].T).T for i in range(target_dim)])

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

        R_inv = np.stack(
            [np.linalg.solve(R.reshape(-1, state_dim, state_dim)[i], sigma) for i in range(target_dim ** 2)])
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
            trace = np.hstack([np.sum(Q[i, i] * iK[i]) for i in range(target_dim)])
            cov = (cov - np.diag(trace)) * scaling_factor + np.diag(np.exp(2 * sigma_f))

        # Centralize moments
        cov = cov - mean[:, np.newaxis] @ mean[np.newaxis, :]

        return mean, cov, input_output_cov

    def optimize(self):
        for i, gp in enumerate(self.gp_container):
            self.logger.info("Optimization for GP (output={}) started.".format(i))
            try:
                self.logger.info("Optimization with L-BFGS-B started.")
                gp.optimize("lbfgsb", messages=1)
            except Exception:
                self.logger.info("Optimization with SCG started.")
                gp.optimize('scg', messages=1)

            print(gp)

    def sigma_fs(self):
        return np.log(np.array([gp.kern.rbf.variance for gp in self.gp_container]))

    def sigma_eps(self):
        return np.log(np.array([gp.kern.white.variance for gp in self.gp_container]))

    def length_scales(self):
        return np.log(np.array([gp.kern.rbf.lengthscale for gp in self.gp_container]))