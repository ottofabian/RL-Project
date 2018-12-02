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
                                sigma_eps=sigma_eps, container=RBFNetwork, is_policy=True)

        for gp in self.gp_container:
            gp.set_rollout(rollout)

        self.length_scales = None
        self.ridge = ridge

    def fit(self, X, y):
        MultivariateGP.fit(self, X, y)
        # compute K_inv and betas for later prediction
        [gp.compute_params() for gp in self.gp_container]

    def choose_action(self, mu, sigma, squash=True, bound=1):
        action_mu, action_cov, input_output_cov = self.predict_from_dist(mu, sigma)

        if squash:
            action_mu, action_cov, input_output_cov = self.squash_action_dist(action_mu, action_cov, input_output_cov,
                                                                              bound)
        # prediction from GP of cross_cov is times inv(s)
        return action_mu, action_cov, input_output_cov @ sigma

    def squash_action_dist(self, mu, sigma, input_output_cov, bound):
        """
        Rescales and squashes the distribution x with sin(x)
        See Deisenroth(2010) Appendix A.1 for mu of sin(x), where x~N(mu, sigma)
        :param mu:
        :param sigma:
        :param input_output_cov:
        :return: mu_squashed, sigma_squashed, input_output_cov_squashed
        """

        # p(u)' is squashed distribution over p(u) scaled by action space values,
        # see Deisenroth (2010), page 46, 2a)+b) and Section 2.3.2

        # compute mean of squashed dist
        mu_squashed = bound * np.exp(-sigma / 2) * np.sin(mu)

        # covar: E[sin(x)^2] - E[sin(x)]^2
        sigma2 = -(sigma.T + sigma) / 2
        sigma2_exp = np.exp(sigma2)
        sigma_squashed = ((np.exp(sigma2 + sigma) - sigma2_exp) * np.cos(mu.T - mu) -
                          (np.exp(sigma2 - sigma) - sigma2_exp) * np.cos(mu.T + mu))
        sigma_squashed = np.dot(bound.T, bound) * sigma_squashed / 2

        # compute input-output-covariance and squash through sin(x)
        input_output_cov_squashed = np.diag((bound * np.exp(-sigma / 2) * np.cos(mu)).flatten())
        input_output_cov_squashed = input_output_cov @ input_output_cov_squashed

        return mu_squashed, sigma_squashed, input_output_cov_squashed.T
