import dill as pickle

from autograd import numpy as np

from PILCO.Controller.Controller import Controller
from PILCO.GaussianProcess.MultivariateGP import MultivariateGP
from PILCO.GaussianProcess.RBFNetwork import RBFNetwork
from PILCO.util.util import squash_action_dist


class RBFController(MultivariateGP, Controller):

    def __init__(self, X, y, n_actions, length_scales):
        """
        Deisenroth (2010), Nonlinear Model: RBF Network
        :param X: pseudo inputs
        :param y: pseudo targets
        :param n_actions: number of actions/GP models
        :param n_features: number of
        :param length_scales:
        """
        # sigma_f is fixed for the RBF Controller, if it is seen as deterministic GP
        # sigma_eps is .01 to ensure a numerical stable computation
        sigma_f = np.log(np.ones((n_actions,)))
        sigma_eps = np.log(np.ones((n_actions,)) * .01)

        MultivariateGP.__init__(self, X=X, y=y, length_scales=length_scales, n_targets=n_actions, sigma_f=sigma_f,
                                sigma_eps=sigma_eps, container=RBFNetwork, is_policy=True)

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

    def set_params(self, params):
        # reset cached matrices when new params are added
        self.K_inv = None
        self.beta = None

        for i, gp in enumerate(self.gp_container):
            gp.unwrap_params(params[gp.length * i: gp.length * (i + 1)])
            # computes beta and K_inv for updated hyperparams
            gp.compute_matrices()

    def optimize(self) -> None:
        """
        This does nothing, policy needs to be optimized based on rollouts.
        :return: None
        """
        raise ValueError("Cannot be used for optimization")

    def save(self, save_dir) -> None:
        """
        pickle dumps the policy to hard drive
        :param save_dir: directory where the policy will be saved
        :return:
        """
        pickle.dump(self, open(f"{save_dir}policy.p", "wb"))
