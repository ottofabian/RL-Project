import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize

from PILCO.GaussianProcess.GaussianProcess import GaussianProcess


class RBFNetwork(GaussianProcess):
    """
    Gaussian Process Regression given an input Gaussian
    """

    def __init__(self, length_scales, sigma_f=1, sigma_eps=.01, is_policy=True):

        super(RBFNetwork, self).__init__(length_scales, sigma_f, sigma_eps, is_policy)

        self.opt_ctr = 0  # optimization counter

    #     self.rollout = None
    #
    # def set_rollout(self, rollout: callable):
    #     self.rollout = rollout

    def wrap_policy_hyperparams(self):

        n_features = self.X.shape[0]

        split1 = self.state_dim * n_features  # split for RBF centers
        split2 = self.n_targets * n_features + split1  # split for training targets/weights

        params = np.zeros([self.state_dim * n_features + self.n_targets * n_features + self.state_dim + 1])

        params[:split1] = self.X.reshape(self.state_dim * n_features)
        params[split1:split2] = self.y.reshape(self.n_targets * n_features)
        params[split2:-1] = self.length_scales
        params[-1] = self.sigma_eps

        return params

    def compute_params(self):

        params = self._wrap_kernel_hyperparams()
        K = self.kernel(params, self.X)[0]

        # Interpreting the RBF network as deterministic GP, the inverse of K has to be eliminated
        self.K_inv = np.zeros(K.shape)
        self.betas = K @ self.y

    def unwrap_params(self, params):

        n_features = self.X.shape[0]

        split1 = self.state_dim * n_features  # split for RBF centers
        split2 = self.n_targets * n_features + split1  # split for training targets/weights

        X = params[:split1].reshape(n_features, self.state_dim)
        y = params[split1:split2].reshape(n_features, self.n_targets)
        length_scales = params[split2:-1]
        sigma_eps = params[-1]

        # ensure noise is an numpy array
        self.X, self.y, self.length_scales, self.sigma_eps = X, y, length_scales, np.atleast_1d(sigma_eps)

    # def _optimize_hyperparams(self, params):
    #     self.opt_ctr += 1
    #     self.X, self.y, self.length_scales, self.sigma_eps = self.unwrap_params(params)
    #     print(params)
    #
    #     # computes beta and K_inv for updated hyperparams
    #     self.compute_params()
    #
    #     # returns cost of trajectory rollout
    #     cost = self.rollout(self, print=False)
    #
    #     # print progress
    #     self.logger.info("Policy optimization iteration: {} -- Cost: {}".format(self.opt_ctr, cost._value[0]))
    #
    #     return cost

    def optimize(self):
        params = self.wrap_policy_hyperparams()
        options = {'maxiter': 150, 'disp': True}

        try:
            # res = minimize(value_and_grad(self._optimize_hyperparams), params, method='L-BFGS-B', jac=True,
            #                options=options)
            res = minimize(self._optimize_hyperparams, params, method='L-BFGS-B', jac=False,
                           options=options)
        except Exception:
            res = minimize(value_and_grad(self._optimize_hyperparams), params, method='CG', jac=True,
                           options=options)

        # Make one more run for plots
        self.rollout(self, print=True)

        self.opt_ctr = 0
        self.X, self.y, self.length_scales, self.sigma_eps = self.unwrap_params(res.x)
        self.compute_params()

        # self.logger.debug("Best Params: \n", self.X, self.y, self.length_scales)
