import autograd.numpy as np

from PILCO.Controller.Controller import Controller
from PILCO.GaussianProcess.MultivariateGP import MultivariateGP


class RBFController(MultivariateGP, Controller):
    """RBF Controller/Policy"""

    def __init__(self, n_actions, ridge=1e-6):
        """

        :param length_scales:
        :param n_actions:
        """

        # Hyperparams to optimize_policy are y, length-scales, X

        # sigma_f is fixed for the RBF Controller, if it is seen as deterministic GP
        # sigma_eps is .01 to ensure a numerical stable computation
        MultivariateGP.__init__(self, length_scales=None, n_targets=n_actions, optimizer="fmin_l_bfgs_b", sigma_f=1,
                                sigma_eps=.01, alpha=1e-10, is_policy=True)

        self.length_scales = None
        self.ridge = ridge

    def get_hyperparams(self):
        concat = np.concatenate([self.X, self.y], axis=1)
        return np.concatenate([concat.T.flatten(), self.get_length_scales().T.flatten()])

    def set_hyper_params(self, X, y, length_scales):
        self.X = X.reshape(X.shape[1], -1)
        self.y = y.reshape(y.shape[1], -1)
        self.length_scales = length_scales
        super(RBFController, self).fit(X, y)
        for i, gp in enumerate(self.gp_container):
            gp.length_scales = length_scales[i]

    def fit(self, X, y):
        self.set_hyper_params(X, y, self.length_scales)

    def choose_action(self, mu, sigma):
        action_mu, action_cov, input_output_cov = self.predict_from_dist(mu, sigma)
        action_cov = action_cov - self.get_sigma_eps()[0] - self.ridge

        return action_mu, action_cov, input_output_cov

    # def choose_action(self, mu, sigma):
    #     """
    #     The predictive mean of p(π̃(x∗)) for a known state x∗ is equivalent to RBF policy in equation which itself is
    #     identical to the predictive mean of a GP. In contrast to the GP model, both the predictive variance and the
    #     uncertainty about the underlying function in an RBF network are zero.
    #     Thus, the predictive distribution p(π̃(x*)) for a given state x* has zero variance.
    #     :param x: point(s) to choose_action
    #     :return:
    #     """
    #     # TODO: Return mean and covar of action dist
    #     mu_action = self.beta.T @ self.kernel(self.X, x), 123
    #
    #     return
    #
    # def optimize_params(self, X):
    #     self.X = X
    #
    #     self.sigma = self.var * np.identity(X.shape[0])
    #     self.mu = np.zeros(self.sigma.shape[0])
    #
    #     self._set_beta()
    #
    # def _set_y(self):
    #     pred = self.choose_action(self.X)
    #     return pred + self.sample_measurement_noise()
    #
    # def _set_beta(self):
    #     K = self.kernel(self.X)
    #     self.beta = solve(K + self.sample_measurement_noise(), np.identity(K.shape[0])) @ self.y.T
    #
    # def sample_measurement_noise(self):
    #     return np.random.multivariate_normal(self.mu, self.sigma)
    #
    # # def __init__(self, W, sigma, mu):
    # #     self.W = W
    # #     self.sigma = sigma
    # #     self.mu = mu
    # #
    # # def phi(self, X):
    # #     phi = np.zeros((self.mu.shape[0], X.shape[1]))
    # #     for i in range(self.mu.shape[0]):
    # #         d = X - self.mu[i]
    # #         phi[i] = np.exp(-.5 * d @ self.sigma @ d.T)
    # #     return phi
    # #
    # # def choose_action(self, X):
    # #     # TODO Check axis dim
    # #     return np.sum(self.W @ self.phi(X), axis=1)
    # #
    # # def optimize_params(self, W, sigma, mu):
    # #     self.W = W
    # #     self.sigma = sigma
    # #     self.mu = mu
