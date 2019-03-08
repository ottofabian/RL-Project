import autograd.numpy as np

from pilco.cost_function.loss import Loss


class SaturatedLoss(Loss):

    def __init__(self, state_dim: np.ndarray, target_state: np.ndarray = None, weights: np.ndarray = None,
                 cost_width: np.ndarray = None):
        """
        Initialize saturated loss function
        :param state_dim: state dimensionality
        :param target_state: target state which should be reached
        :param weights: weight matrix
        :param cost_width: TODO what is this
        """

        self.state_dim = state_dim

        # set target state to all zeros if not other specified
        self.target_state = np.atleast_2d(np.zeros(self.state_dim) if target_state is None else target_state)

        # weight matrix
        self.weights = np.identity(self.state_dim) if weights is None else weights

        # -----------------------------------------------------
        # This is only useful if we have any penalties etc.
        self.cost_width = np.array([1]) if cost_width is None else cost_width

    def compute_cost(self, mu: np.ndarray, sigma: np.ndarray) -> tuple:
        """
        Compute cost of current state distribution
        :param mu: mean of state
        :param sigma: covariance of state
        :return: cost distribution of given state distribution
        """
        mu = np.atleast_2d(mu)

        sigma_weighted = np.dot(sigma, self.weights)
        sigma_weighted_inv = np.linalg.solve((np.identity(self.state_dim) + sigma_weighted).T, self.weights.T).T
        diff = mu - self.target_state

        # compute expected cost
        scale = np.sqrt(np.linalg.det(np.identity(self.state_dim) + sigma_weighted))
        cost_mean = -np.exp(-diff @ sigma_weighted_inv @ diff.T / 2) / scale

        # compute variance of cost
        sigma_weighted_inv2 = np.linalg.solve((np.identity(self.state_dim) + 2 * sigma_weighted).T, self.weights.T).T
        scale2 = np.sqrt(np.linalg.det(np.identity(self.state_dim) + 2 * sigma_weighted))
        r2 = np.exp(-diff @ sigma_weighted_inv2 @ diff.T) / scale2
        cost_cov = r2 - cost_mean ** 2

        # compute cross covariance
        t = np.dot(self.weights, self.target_state.T) - sigma_weighted_inv @ (
                np.dot(sigma_weighted, self.target_state.T) + mu.T)

        cost_input_output_cov = sigma @ (cost_mean * t)

        # bring cost to the interval [0,1]
        return 1 + cost_mean, cost_cov, cost_input_output_cov

    def compute_loss(self, mu: np.ndarray, sigma: np.ndarray) -> float:
        """
        compute penalized loss function of state distribution
        :param mu: mean of state distribution
        :param sigma: covariance of state distribution
        :return: loss
        """
        # We do not have information about the env for reasonable penalties or the like.

        cost = 0
        for w in self.cost_width:
            mu, _, _ = self.compute_cost(mu, sigma)
            cost = cost + mu

        return cost / len(self.cost_width)

    @property
    def target_state(self):
        return self._target_state

    @property
    def state_dim(self):
        return self._state_dim

    @target_state.setter
    def target_state(self, value):
        self._target_state = value

    @state_dim.setter
    def state_dim(self, value):
        self._state_dim = value
