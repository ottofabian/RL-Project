import autograd.numpy as np
from pandas.io import pickle

from PILCO.GaussianProcess.GaussianProcess import GaussianProcess


class RBFNetwork(GaussianProcess):
    """
    Gaussian Process Regression given an input Gaussian
    """

    def __init__(self, length_scales, sigma_f=1, sigma_eps=.01):
        super(RBFNetwork, self).__init__(length_scales, sigma_f, sigma_eps)

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

    def compute_matrices(self):
        # Interpreting the RBF network as deterministic GP, the inverse of K and is not relevant for the computation,
        # therefore it is set to 0
        super(RBFNetwork, self).compute_matrices()
        self.K_inv = np.zeros(self.K.shape)

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

    def optimize(self):
        self.logger.warning(
            "RBF networks are only optimized during the trajectory rollout, calling this method does nothing.")
        pass
