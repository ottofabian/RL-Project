import logging
from typing import Union, Type

import dill as pickle
import autograd.numpy as np

from pilco.gaussian_process.gaussian_process import GaussianProcess
from pilco.gaussian_process.rbf_network import RBFNetwork


class MultivariateGP(object):

    def __init__(self, x: np.ndarray, y: np.ndarray, n_targets: int,
                 container: Union[Type[GaussianProcess], Type[RBFNetwork], None], length_scales: np.ndarray,
                 sigma_f: np.ndarray, sigma_eps: np.ndarray, is_policy: bool = False):
        """
        Multivariate Gaussian Process Regression
        :param n_targets: amount of target, each dimension of data inputs requires one target
        :param container: submodel type for each target, this depends on whether this should model dynamics of RBF policy
        :param length_scales: prior for length scales
        :param sigma_f: prior for signal variance
        :param sigma_eps: prior for noise variance
        :param is_policy: is this instance used as RBF policy or not,
                          the moment matching is computed slightly different based on that.
        """

        self.x = x
        self.y = y
        self.n_targets = n_targets
        self.is_policy = is_policy

        self.beta = None
        self.K_inv = None

        self.models = []

        self.make_models(length_scales, sigma_f, sigma_eps, container)

    def make_models(self, length_scales: np.ndarray, sigma_f: np.ndarray, sigma_eps: np.ndarray,
                    container: Union[Type[GaussianProcess], Type[RBFNetwork]]):
        """
        generate models for GP
        :param length_scales: length scale init for models
        :param sigma_f: signal variance init for models
        :param sigma_eps: noise variance init for models
        :param container: container type, depending if this is a rbf policy or a dynamics model
        :return:
        """
        # For a D-dimensional state space, we use D separate GPs, one for each state dimension. Deisenroth (2010)
        for i in range(self.n_targets):
            self.models.append(
                container(length_scales=length_scales[i], sigma_eps=sigma_eps[i], sigma_f=sigma_f[i]))
            self.models[i].set_XY(self.x, self.y[:, i:i + 1])

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        set x and y
        :param x: input variables [n_samples, sample dim]
        :param y: target variables [n_samples, n_targets]
        :return: None
        """
        self.x = x
        self.y = y

        # reset cached matrices when new data is added
        self.K_inv = None
        self.beta = None

        for i in range(self.n_targets):
            self.models[i].set_XY(x, y[:, i:i + 1])

    def cache(self):
        """
        Precomputes the inverse gram matrix and betas for gp
        :return: None
        """
        [gp.compute_matrices() for gp in self.models]
        self.beta = np.vstack(np.array([gp.betas for gp in self.models])).T
        self.K_inv = np.array([gp.K_inv for gp in self.models])

    def predict_from_dist(self, mu: np.ndarray, sigma: np.ndarray) -> tuple:

        """
        Use moment matching to predict dist given an uncertain input x~N(mu,sigma) from gaussian process
        :param mu: n_targets x n_state + n_actions
        :param sigma: n_targets x (n_state + n_actions) x (n_state + n_actions)
        :return: mu, sigma and inv(sigma) @ input_output_cov
        """
        # Adapted from: https://github.com/cryscan/pilco-learner

        mu = np.atleast_2d(mu)

        state_dim = self.x.shape[1]
        target_dim = self.y.shape[1]

        # ----------------------------------------------------------------------------------------------------
        # Helper

        # matrices are not cached already
        if self.K_inv is None or self.beta is None:
            self.cache()

        length_scales = self.length_scales()
        sigma_f = self.sigma_f().reshape(self.n_targets)

        precision_inv = np.stack(np.array([np.diag(np.exp(-l)) for l in length_scales]))
        precision_inv2 = np.stack(np.array([np.diag(np.exp(-2 * l)) for l in length_scales]))

        # centralized inputs
        diff = self.center_inputs(mu)
        diff_scaled = diff / np.expand_dims(np.exp(2 * length_scales), axis=1)

        # ----------------------------------------------------------------------------------------------------
        # compute mean of predictive dist based on matlab code Deisenroth(2010)

        # The precision_inv cancels out later on
        zeta_a = diff @ precision_inv

        B = precision_inv @ sigma @ precision_inv + np.identity(state_dim)

        # B[i] is symmetric, so B[i].T=B[i]
        t = np.stack(np.array([np.linalg.solve(B[i], zeta_a[i].T).T for i in range(target_dim)]))

        scaled_beta = np.exp(-.5 * np.sum(zeta_a * t, axis=2)) * self.beta.T

        # If something is nan, then this line is the problem.
        # Or more precisely, the lengthscale parameters are choosen poorly and cause the determinant to be negative.
        coefficient = np.exp(2 * sigma_f) / np.sqrt(np.linalg.det(B))

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
            # simplified computation for policy as K_inv = 0 anyway
            # only adding of ridge term
            cov = scaling_factor * np.einsum('ji,iljk,kl->il', self.beta, Q, self.beta) + 1e-6 * np.identity(target_dim)
        else:
            cov = np.einsum('ji,iljk,kl->il', self.beta, Q, self.beta)

            trace = np.hstack(np.array([np.sum(Q[i, i] * self.K_inv[i]) for i in range(target_dim)]))
            cov = (cov - np.diag(trace)) * scaling_factor + np.diag(np.exp(2 * sigma_f))

        # Centralize moments
        cov = cov - mean[:, np.newaxis] @ mean[np.newaxis, :]

        return mean, cov, input_output_cov

    def optimize(self) -> None:
        """
        optimizes the hyperparameters for all gaussian process models
        :return: None
        """

        # reset parameters of gps are changing
        self.K_inv = None
        self.beta = None

        for i, gp in enumerate(self.models):
            logging.info("Optimization for GP (output={}) started.".format(i))
            gp.optimize()

    def save(self, save_dir) -> None:
        """
        pickle dumps the gp to hard drive
        :param save_dir: directory where the dynamic model will be saved
        :return:
        """
        pickle.dump(self, open(f"{save_dir}dynamics.p", "wb"))

    def sigma_f(self) -> np.ndarray:
        """
        returns signal variance of gp
        :return: ndarray of [n_targets, 1]
        """
        return np.array([c.sigma_f for c in self.models])

    def sigma_eps(self) -> np.ndarray:
        """
        returns noise variance of gp
        :return: ndarray of [n_targets, 1]
        """
        return np.array([c.sigma_eps for c in self.models])

    def length_scales(self) -> np.ndarray:
        """
        returns length scales of gp
        :return: ndarray of [n_targets, input_dim]
        """
        return np.array([c.length_scales for c in self.models])

    def center_inputs(self, mu):
        return np.expand_dims(self.x - mu, axis=0)
