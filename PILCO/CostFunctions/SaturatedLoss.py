import numpy as np

from PILCO.CostFunctions.Loss import Loss


class SaturatedLoss(Loss):

    def __init__(self, state_dim, target_state=None, T_inv=None, cost_width=None, p=.5):

        self.state_dim = state_dim

        # set target state to all zeros if not other specified
        self.target_state = np.zeros(self.state_dim) if target_state is None else target_state
        self.target_state = np.atleast_2d(self.target_state)

        # weight matrix
        self.T_inv = np.identity(self.state_dim) if T_inv is None else T_inv

        # -----------------------------------------------------
        # This is only useful if we have any penalties etc.
        self.cost_width = np.array([1]) if cost_width is None else cost_width
        self.p = p

    def compute_cost(self, mu, sigma):
        mu = np.atleast_2d(mu)

        sigma_T_inv = np.dot(sigma, self.T_inv)
        S1 = np.linalg.solve((np.identity(self.state_dim) + sigma_T_inv).T, self.T_inv.T).T
        diff = mu - self.target_state

        # compute expected cost
        mean = -np.exp(-diff @ S1 @ diff.T / 2) / np.sqrt(np.linalg.det(np.identity(self.state_dim) + sigma_T_inv))

        # compute variance of cost
        S2 = np.linalg.solve((np.identity(self.state_dim) + 2 * sigma_T_inv).T, self.T_inv.T).T
        r2 = np.exp(-diff @ S2 @ diff.T) * ((np.linalg.det(np.identity(self.state_dim) + 2 * sigma_T_inv)) ** -.5)
        cov = r2 - mean ** 2

        # for numeric reasons set to 0
        if np.all(cov < 1e-12):
            cov = np.zeros(cov.shape)

        t = np.dot(self.T_inv, self.target_state.T) - S1 @ (np.dot(sigma_T_inv, self.target_state.T) + mu.T)

        cross_cov = sigma @ (mean * t)

        # bring cost to the interval [0,1]
        return 1 + mean, cov, cross_cov

    def compute_loss(self, mu, sigma):
        # TODO: Ask supervisors if we need to do this.
        # We do not have information about the env for penalties or the like.

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
