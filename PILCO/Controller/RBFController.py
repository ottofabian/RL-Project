import pickle

from autograd import numpy as np

from PILCO.Controller.Controller import Controller
from PILCO.GaussianProcess.MultivariateGP import MultivariateGP
from PILCO.GaussianProcess.RBFNetwork import RBFNetwork


def squash_action_dist(mu: np.ndarray, sigma: np.ndarray, input_output_cov: np.ndarray, bound: np.ndarray) -> tuple:
    """
    Rescales and squashes the distribution x with sin(x)
    See Deisenroth(2010) Appendix A.1 for mu of sin(x), where x~N(mu, sigma)
    :param bound: max action to take
    :param mu: mean of action distribution
    :param sigma: covariance of actions distribution
    :param input_output_cov: state action input out covariance
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

    return mu_squashed, sigma_squashed, input_output_cov_squashed


class RBFController(MultivariateGP, Controller):
    """RBF Controller/Policy"""

    def __init__(self, n_actions, n_features, length_scales):
        # sigma_f is fixed for the RBF Controller, if it is seen as deterministic GP
        # sigma_eps is .01 to ensure a numerical stable computation
        sigma_f = np.log(np.ones((n_actions,)))
        sigma_eps = np.log(np.ones((n_actions,)) * .01)

        MultivariateGP.__init__(self, length_scales=length_scales, n_targets=n_actions, sigma_f=sigma_f,
                                sigma_eps=sigma_eps, container=RBFNetwork, is_policy=True)

        self.n_features = n_features
        self.opt_ctr = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        set x and y
        :param X: input variables [n_samples, sample dim]
        :param y: target variables [n_samples, 1]
        :return: None
        """

        # TODO this fits all X for all predictions, this does not work for >1D actions
        MultivariateGP.fit(self, X, y)

    def choose_action(self, mu: np.ndarray, sigma: np.ndarray, bound: np.ndarray = None) -> tuple:
        """
        Choose an action based on the current RBF functions
        :param mu: mean of state
        :param sigma: covariance of state
        :param bound: float for squashing action in [-bound, bound] or None when no squashing is needed
        :return: action_mu, action_cov, input_output_cov
        """
        action_mu, action_cov, input_output_cov = self.predict_from_dist(mu, sigma)

        if bound is not None:
            action_mu, action_cov, input_output_cov = squash_action_dist(action_mu, action_cov, input_output_cov, bound)

        # prediction of cross_cov from GP is cross_cov @ inv(sigma)
        return action_mu, action_cov, sigma @ input_output_cov

    def optimize(self) -> None:
        """
        This does nothing, policy needs to be optimized based on rollouts.
        :return: None
        """
        raise ValueError("Cannot be used for optimization")

    def save(self, reward) -> None:
        """
        pickle dumps the policy to hard drive
        :param reward: required for the name of the file.
        :return:
        """
        pickle.dump(self, open(f"policy_reward-{reward}.p", "wb"))
