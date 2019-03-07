import autograd.numpy as np

from pilco.gaussian_process.gaussian_process import GaussianProcess
import logging


class RBFNetwork(GaussianProcess):
    def __init__(self, length_scales, sigma_f=1, sigma_eps=.01):
        """
        RBF Network as deterministic GP
        :param length_scales:
        :param sigma_f:
        :param sigma_eps:
        """
        super(RBFNetwork, self).__init__(length_scales, sigma_f, sigma_eps)

    @property
    def length(self) -> int:
        """
        returns length of flattened features
        :return: int of length
        """
        return self.state_dim * self.x.shape[0] + self.n_targets * self.x.shape[0] + self.state_dim + 1

    def wrap_policy_hyperparams(self):
        """
        get policy parameters as flattened array
        :return: ndarray of flattened params of length self.length
        """
        n_features = self.x.shape[0]

        split1 = self.state_dim * n_features  # split for RBF centers
        split2 = self.n_targets * n_features + split1  # split for training targets/weights

        params = np.zeros([self.length])

        params[:split1] = self.x.reshape(self.state_dim * n_features)
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
        n_features = self.x.shape[0]

        split1 = self.state_dim * n_features  # split for RBF centers
        split2 = self.n_targets * n_features + split1  # split for training targets/weights

        x = params[:split1].reshape(n_features, self.state_dim)
        y = params[split1:split2].reshape(n_features, self.n_targets)
        length_scales = params[split2:-1]
        sigma_eps = params[-1]

        # ensure noise is an numpy array
        self.x, self.y, self.length_scales, self.sigma_eps = x, y, length_scales, np.atleast_1d(sigma_eps)

    def optimize(self):
        raise ValueError(
            "RBF networks are only optimized during the trajectory rollout, calling this method does nothing.")
