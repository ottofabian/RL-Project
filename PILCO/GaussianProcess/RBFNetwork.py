import copy

import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize

from PILCO.GaussianProcess.GaussianProcess import GaussianProcess


class RBFNetwork(GaussianProcess):
    """
    Gaussian Process Regression given an input Gaussian
    """

    def __init__(self, length_scales):

        sigma_f = 1
        sigma_eps = 0.01

        super(RBFNetwork, self).__init__(length_scales, sigma_f, sigma_eps)
        self.opt_ctr = 0  # optimization counter

    def optimize(self):
        params = self._wrap_kernel_hyperparams()
        options = {'maxiter': 150, 'disp': True}

        try:
            res = minimize(value_and_grad(self._optimize_hyperparams), params, method='L-BFGS-B', jac=True,
                           options=options)
        except Exception:
            res = minimize(value_and_grad(self._optimize_hyperparams), params, method='CG', jac=True,
                           options=options)

        self.opt_ctr = 0
        best_params = res.x
        X, y, l, e = self._unwrap_params(best_params)
        self.X, self.y, self.length_scales, self.sigma_eps = X, y, np.exp(l), np.exp(e)
        self.compute_params()

        # self.logger.debug("Best Params: \n", X, y, length_scales)

    def _optimization_callback(self, policy):

        if self.opt_ctr % 10 == 0:
            print("Policy optimization iteration: {} -- Cost: {}".format(self.opt_ctr, self.rollout(policy)))
        else:
            print("Policy optimization iteration: {}".format(self.opt_ctr))

    def _wrap_policy_hyperparams(self):

        split1 = self.state_dim  # split for RBF centers
        split2 = self.n_targets + split1  # split for training targets/weights

        N = self.X.shape[0]

        params = np.zeros([self.n_targets, self.state_dim * N + self.n_targets * N + self.state_dim + 1])
        params[:, :split1] = self.X.reshape(self.n_targets, -1)
        params[:, split1:split2] = self.y.reshape(self.n_targets, -1)
        params[:, split2:-1] = np.log(self.length_scales)
        params[:, -1] = np.log(self.sigma_eps)

        return params

    def compute_params(self):

        params = self._wrap_kernel_hyperparams()
        K = self.kernel(params, self.X)[0]

        # Interpreting the RBF network as deterministic GP, the inverse of K has to be eliminated
        self.K_inv = np.zeros(K.shape)
        self.betas = K @ self.y

    def _unwrap_params(self, params):

        N = self.X.shape[0]

        split1 = self.state_dim * self.n_targets * N  # split for RBF centers
        split2 = self.n_targets * N + split1  # split for training targets/weights

        X = params[:split1].reshape(self.n_targets, self.state_dim, -1)
        y = params[split1:split2].reshape(self.n_targets, N)
        length_scales = params[split2:-1].reshape(self.n_targets, self.state_dim)
        sigma_eps = params[-1]

        return X, y, length_scales, sigma_eps

    def _optimize_hyperparams(self, params, *args):

        self.opt_ctr += 1
        p = args[0]
        policy = copy.deepcopy(self)
        X, y, l, e = self._unwrap_params(params)

        policy.set_hyper_params(X, y, length_scales)
        self.optimization_callback(policy)
        # self.logger.debug("Best Params: \n", X, y, length_scales)
        return self.rollout(policy)

    # @property
    # def length_scales(self):
    #     return self.kernel.get_params()['k1__k2__length_scale']
    #
    # @property
    # def sigma_f(self):
    #     return self.kernel.get_params()['k1__k1__constant_value']
    #
    # @property
    # def sigma_eps(self):
    #     return self.kernel.get_params()['k2__noise_level']
    #
    # @length_scales.setter
    # def length_scales(self, length_scales: np.ndarray) -> None:
    #     self.kernel.set_params(k1__k2__length_scale=length_scales)
    #
    # @sigma_eps.setter
    # def sigma_eps(self, value):
    #     self._sigma_eps = value
    #
    # @sigma_f.setter
    # def sigma_f(self, value):
    #     self._sigma_f = value
