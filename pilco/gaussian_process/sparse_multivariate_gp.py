import logging
from typing import Union, Type

import autograd.numpy as np
import scipy
import GPy

from pilco.gaussian_process.gaussian_process import GaussianProcess
from pilco.gaussian_process.multivariate_gp import MultivariateGP
from pilco.gaussian_process.rbf_network import RBFNetwork


class SparseMultivariateGP(MultivariateGP):

    def __init__(self, x, y, n_targets, length_scales, sigma_f, sigma_eps, n_inducing_points: int, is_policy=False):

        """
        Sparse Multivariate Gaussian Process Regression
        :param x: inputs [n_samples, state_dim]
        :param y: targets
        :param n_targets: amount of target, each dimension of data inputs requires one target
        :param length_scales: prior for length scales
        :param sigma_f: prior for signal variance
        :param sigma_eps: prior for noise variance
        :param is_policy: is this instanced used as RBF policy or not,
                          the moment matching is consequently computed differently
        """

        self.n_inducing_points = n_inducing_points

        super(SparseMultivariateGP, self).__init__(x, y, n_targets, None, length_scales, sigma_f, sigma_eps,
                                                   is_policy)

    def make_models(self, length_scales: np.ndarray, sigma_f: np.ndarray, sigma_eps: np.ndarray,
                    container: Union[Type[GaussianProcess], Type[RBFNetwork]]):
        """
        generate models for Sparse GP
        :param length_scales: length scale init for models
        :param sigma_f: signal variance init for models
        :param sigma_eps: noise variance init for models
        :param container: container type, depending if this is a rbf policy or a dynamics model
        :return: None
        """

        # generate same initial inducing points for all GPs as subset of all given points
        idx = np.random.permutation(self.x.shape[0])[:min(self.n_inducing_points, self.x.shape[0])]
        z = self.x.view(np.ndarray)[idx].copy()

        for i in range(self.n_targets):
            kernel = GPy.kern.RBF(input_dim=length_scales.shape[1], lengthscale=np.exp(length_scales[i]), ARD=True,
                                  variance=np.exp(sigma_f[i]))

            model = GPy.models.SparseGPRegression(X=self.x, Y=self.y[:, i:i + 1], kernel=kernel, Z=z)

            # set noise variance
            model.likelihood.noise = np.exp(sigma_eps[i])

            # set constraints for length scales and signal noise
            # as we cannot optimize the penalized likelihood with GPy.
            model.kern.lengthscale.constrain_bounded(0, 300)
            model.likelihood.variance.constrain_bounded(1e-15, 1e-3)
            # model.likelihood.variance = 1e-3
            # model.likelihood.variance.fix()
            # prior = GPy.priors.gamma_from_EV(0.5, 1)
            # gp.kern.lengthscale.set_prior(prior, warning=False)

            # set the approximate inference method to FITC as used by Deisenroth(2010)
            model.inference_method = GPy.inference.latent_function_inference.FITC()
            self.models.append(model)

    def cache(self):
        """
        Precomputes the inverse gram matrix and betas for sparse gp
        :return:
        """

        target_dim = self.y.shape[1]
        induced_dim = np.array(self.models[0].Z).shape[0]

        Kmm = np.stack(np.array([gp.kern.K(gp.Z) + 1e-6 * np.identity(induced_dim) for gp in self.models]))
        Kmn = np.stack(np.array([gp.kern.K(gp.Z, gp.X) for gp in self.models]))

        Kmm_cho = np.linalg.cholesky(Kmm)

        # inv(sqrt(Kmm)) * Kmn
        Kmm_sqrt_inv_Kmn = np.stack(np.array([scipy.linalg.solve_triangular(Kmm_cho[i], Kmn[i], lower=True) for i in
                                              range(target_dim)]))
        G = np.exp(2 * self.sigma_f()) - np.sum(Kmm_sqrt_inv_Kmn ** 2, axis=1)

        # this can be nan when no contraints are used for optimizing
        G = np.sqrt(1. + G / np.exp(2 * self.sigma_eps()))
        Kmm_sqrt_inv_Kmn_scaled = Kmm_sqrt_inv_Kmn / G[:, None]

        Am = np.linalg.cholesky(np.stack(np.array(
            [Kmm_sqrt_inv_Kmn_scaled[i] @ Kmm_sqrt_inv_Kmn_scaled[i].T + np.identity(induced_dim) * np.exp(
                2 * self.sigma_eps()[i]) for i
             in range(target_dim)])))

        # chol(sig*B) Deisenroth(2010)
        sig_B_cho = Kmm_cho @ Am
        sig_B_cho_inv = np.stack(np.array(
            [scipy.linalg.solve_triangular(sig_B_cho[i], np.identity(induced_dim), lower=True) for i in
             range(target_dim)]))

        Kmm_sqrt_inv_Kmn_scaled = Kmm_sqrt_inv_Kmn_scaled / G[:, None]

        self.beta = np.stack(
            np.array(
                [(np.linalg.solve(Am[i], Kmm_sqrt_inv_Kmn_scaled[i]).T @ sig_B_cho_inv[i]).T @ gp.Y.flatten() for i, gp
                 in enumerate(self.models)])).T

        B_inv = np.stack(
            np.array(
                [sig_B_cho_inv[i].T @ sig_B_cho_inv[i] * np.exp(2 * self.sigma_eps()[i]) for i in range(target_dim)]))

        # inverse gram matrix
        self.K_inv = np.stack(
            np.array([np.linalg.solve(Kmm[i], np.identity(induced_dim)) for i in range(target_dim)])) - B_inv

    def optimize(self) -> None:
        """
        optimizes the hyperparameters for all sparse gaussian process models
        :return: None
        """

        for i, gp in enumerate(self.models):
            logging.info("Optimization for GP (output={}) started.".format(i))
            try:
                logging.info("Optimization with L-BFGS-B started.")
                msg = gp.optimize("lbfgsb", messages=True)
            except Exception:
                logging.info("Optimization with SCG started.")
                msg = gp.optimize('scg', messages=True)

            logging.info(msg)
            logging.info(gp)
            print("Length scales:", gp.kern.lengthscale.values)

    def sigma_f(self):
        return np.log(np.sqrt(np.array([gp.kern.variance.values for gp in self.models])))

    def sigma_eps(self):
        return np.log(np.sqrt(np.array([gp.likelihood.variance.values for gp in self.models])))

    def length_scales(self):
        return np.log(np.array([gp.kern.lengthscale.values for gp in self.models]))

    def center_inputs(self, mu):
        return np.stack(np.array([self.models[i].Z.values for i in range(self.y.shape[1])])) - mu
