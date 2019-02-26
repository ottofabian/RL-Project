import logging

import autograd.numpy as np
import scipy
import GPy

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

        super(SparseMultivariateGP, self).__init__(n_targets, container, length_scales, sigma_f, sigma_eps, is_policy)

        self.X = X
        self.y = y
        self.n_inducing_points = n_inducing_points

        self.logger = logging.getLogger(__name__)

        self.gp_container = []
        idx = np.random.permutation(X.shape[0])[:min(n_inducing_points, X.shape[0])]
        Z = X.view(np.ndarray)[idx].copy()

        for i in range(self.n_targets):
            input_dim = length_scales.shape[1]
            kernel = GPy.kern.RBF(input_dim=input_dim, lengthscale=np.exp(length_scales[i]), ARD=True,
                                  variance=np.exp(sigma_f[i]))

            # kernel = kernel + GPy.kern.White(input_dim=input_dim, variance=np.exp(sigma_eps[i]))
            model = GPy.models.SparseGPRegression(X=X, Y=y[:, i:i + 1], kernel=kernel, Z=Z)
            model.likelihood.noise = np.exp(sigma_eps[i])
            model.kern.lengthscale.constrain_bounded(0, 300)
            model.likelihood.variance.constrain_bounded(1e-15, 1e-3)
            # model.kern.lengthscale[1].constrain_bounded(0, 10)
            # prior = GPy.priors.gamma_from_EV(0.5, 1)
            # gp.kern.lengthscale.set_prior(prior, warning=False)
            model.inference_method = GPy.inference.latent_function_inference.FITC()
            self.gp_container.append(model)

    def cache(self):
        """
        Precomputes the inverse gram matrix and betas for sparse gp
        :return:
        """

        target_dim = self.y.shape[1]
        induced_dim = np.array(self.gp_container[0].Z).shape[0]

        Kmm = np.stack(np.array([gp.kern.K(gp.Z) + 1e-6 * np.identity(induced_dim) for gp in self.gp_container]))
        Kmn = np.stack(np.array([gp.kern.K(gp.Z, gp.X) for gp in self.gp_container]))

        Kmm_cho = np.linalg.cholesky(Kmm)

        Kmm_sqrt_inv_Kmn = np.stack(np.array([scipy.linalg.solve_triangular(Kmm_cho[i], Kmn[i], lower=True) for i in
                                              range(target_dim)]))  # inv(sqrt(Kmm)) * Kmn
        G = np.exp(2 * self.sigma_fs()) - np.sum(Kmm_sqrt_inv_Kmn ** 2, axis=1)
        G = np.sqrt(1. + G / np.exp(2 * self.sigma_eps()))  # this can be nan when no contraints are used for optimizing
        Kmm_sqrt_inv_Kmn_scaled = Kmm_sqrt_inv_Kmn / G[:, None]

        Am = np.linalg.cholesky(np.stack(np.array(
            [Kmm_sqrt_inv_Kmn_scaled[i] @ Kmm_sqrt_inv_Kmn_scaled[i].T + np.identity(induced_dim) * np.exp(
                2 * self.sigma_eps()[i]) for i
             in range(target_dim)])))

        sig_B_cho = Kmm_cho @ Am  # chol(sig*B) Deisenroth(2010)
        sig_B_cho_inv = np.stack(np.array(
            [scipy.linalg.solve_triangular(sig_B_cho[i], np.identity(induced_dim), lower=True) for i in
             range(target_dim)]))

        Kmm_sqrt_inv_Kmn_scaled = Kmm_sqrt_inv_Kmn_scaled / G[:, None]
        # one big ugly loopy, because numpy cannot do it differently, at least I did not find a way for hours
        self.beta = np.stack(
            np.array(
                [(np.linalg.solve(Am[i], Kmm_sqrt_inv_Kmn_scaled[i]).T @ sig_B_cho_inv[i]).T @ gp.Y.flatten() for i, gp
                 in enumerate(self.gp_container)])).T

        B_inv = np.stack(
            np.array(
                [sig_B_cho_inv[i].T @ sig_B_cho_inv[i] * np.exp(2 * self.sigma_eps()[i]) for i in range(target_dim)]))

        # covariance matrix for predictive variances
        self.K_inv = np.stack(
            np.array([np.linalg.solve(Kmm[i], np.identity(induced_dim)) for i in range(target_dim)])) - B_inv

    def optimize(self):
        """
        optimizes the hyperparameters for all sparse gaussian process models
        :return: None
        """

        for i, gp in enumerate(self.gp_container):
            self.logger.info("Optimization for GP (output={}) started.".format(i))
            try:
                self.logger.info("Optimization with L-BFGS-B started.")
                msg = gp.optimize("lbfgsb", messages=True)
            except Exception:
                self.logger.info("Optimization with SCG started.")
                msg = gp.optimize('scg', messages=True)

            self.logger.info(msg)
            self.logger.info(gp)
            print("Length scales:", gp.kern.lengthscale.values)

    def sigma_fs(self):
        return np.log(np.sqrt(np.array([gp.kern.variance.values for gp in self.gp_container])))

    def sigma_eps(self):
        return np.log(np.sqrt(np.array([gp.likelihood.variance.values for gp in self.gp_container])))

    def length_scales(self):
        return np.log(np.array([gp.kern.lengthscale.values for gp in self.gp_container]))

    def center_inputs(self, mu):
        return np.stack(np.array([self.gp_container[i].Z.values for i in range(self.y.shape[1])])) - mu
