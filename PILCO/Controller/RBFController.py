import autograd.numpy as np

from PILCO.Controller.Controller import Controller
from PILCO.GaussianProcess.MultivariateGP import MultivariateGP
from PILCO.GaussianProcess.RBFNetwork import RBFNetwork


class RBFController(MultivariateGP, Controller):
    """RBF Controller/Policy"""

    def __init__(self, n_actions, rollout, length_scales, ridge=1e-6):
        # sigma_f is fixed for the RBF Controller, if it is seen as deterministic GP
        # sigma_eps is .01 to ensure a numerical stable computation
        sigma_f = 1
        sigma_eps = .01

        MultivariateGP.__init__(self, length_scales=length_scales, n_targets=n_actions, sigma_f=sigma_f,
                                sigma_eps=sigma_eps,
                                container=RBFNetwork, is_policy=True)

        for gp in self.gp_container:
            gp.set_rollout(rollout)

        self.length_scales = None
        self.ridge = ridge

    def fit(self, X, y):
        MultivariateGP.fit(self, X, y)
        [gp.compute_params() for gp in self.gp_container]

    def choose_action(self, mu, sigma, squash=True, bound=1):
        action_mu, action_cov, input_output_cov = self.predict_from_dist(mu, sigma)
        # action_cov = action_cov - self.get_sigma_eps()[0] - self.ridge

        return self.squash_action_dist(action_mu, action_cov, input_output_cov, bound)

    def squash_action_dist(self, mu, sigma, input_output_cov, bound):
        """
        Rescales and squashes the distribution x with sin(x)
        :param input_output_cov:
        :param mu:
        :param sigma:
        :return:
        """

        # p(u)' is squashed distribution over p(u) scaled by action space values,
        # see Deisenroth (2010), page 46, 2a)+b) and Section 2.3.2

        # compute mean of squashed dist
        # See Appendix A.1 for mu of sin(x), where x~N(mu, sigma)
        mu_squashed = bound * np.exp(-sigma / 2) @ np.sin(mu)

        # covar: E[sin(x)^2] - E[sin(x)]^2
        sigma2 = -(sigma.T + sigma) / 2
        sigma2_exp = np.exp(sigma2)
        sigma_squashed = ((np.exp(sigma2 + sigma) - sigma2_exp) * np.cos(mu.T - mu) -
                          (np.exp(sigma2 - sigma) - sigma2_exp) * np.cos(mu.T + mu))
        sigma_squashed = bound.T @ bound * sigma_squashed / 2

        # compute input-output-covariance and squash through sin(x)
        input_output_cov_squashed = np.diag((bound * np.exp(-sigma / 2) * np.cos(mu)).flatten())
        input_output_cov_squashed = input_output_cov_squashed @ input_output_cov

        # compute cross-cov between input and squashed output
        # input_output_cov_squashed = bound * np.diag(np.exp(-np.diag_part(sigma) / 2) * np.cos(mu))

        return mu_squashed, sigma_squashed, input_output_cov_squashed

    # def get_hyperparams(self):
    #     concat = np.concatenate([self.X, self.y], axis=1)
    #     return np.concatenate([concat.T.flatten(), self.get_length_scales().T.flatten()])

    # def set_hyper_params(self, X, y, length_scales):
    #     self.X = X.reshape(X.shape[1], -1)
    #     self.y = y.reshape(y.shape[1], -1)
    #     self.length_scales = length_scales
    #     super(RBFController, self).fit(X, y)
    #     for i, gp in enumerate(self.gp_container):
    #         gp.length_scales = length_scales[i]

    # def fit(self, X, y):
    #     self.set_hyper_params(X, y, self.length_scales)
    #
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
