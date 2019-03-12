import dill as pickle

from autograd import numpy as np

from pilco.controller.controller import Controller
from pilco.gaussian_process.multivariate_gp import MultivariateGP
from pilco.gaussian_process.rbf_network import RBFNetwork
from pilco.util.util import squash_action_dist


class RBFController(MultivariateGP, Controller):

    def __init__(self, x: np.ndarray, y: np.ndarray, n_actions: int, length_scales: np.ndarray) -> None:
        """
        Deisenroth (2010), Nonlinear Model: RBF Network
        :param x: pseudo input data points, which are optimized
        :param y: pseudo target data points, which are optimized
        :param n_actions: number of actions/GP models
        :param length_scales: initial length_scales to start with
        """
        # sigma_f is fixed for the RBF controller, if it is seen as deterministic GP
        # sigma_eps is .01 to ensure a numerical stable computation,
        # it is also possible to fix the sigma_eps to ensure a fixed signal to noise ratio.
        # However, here we allow to train it as well.
        sigma_f = np.log(np.ones((n_actions,)))
        sigma_eps = np.log(np.ones((n_actions,)) * .01)

        MultivariateGP.__init__(self, x=x, y=y, length_scales=length_scales, n_targets=n_actions, sigma_f=sigma_f,
                                sigma_eps=sigma_eps, container=RBFNetwork, is_policy=True)

    def choose_action(self, mean: np.ndarray, cov: np.ndarray, bound: np.ndarray = None) -> tuple:
        """
        choose an action based on the current RBF functions
        :param mean: mean of state
        :param cov: covariance of state
        :param bound: float for squashing action in [-bound, bound] or None when no squashing is needed
        :return: action_mean, action_cov, input_output_cov
        """
        action_mean, action_cov, input_output_cov = self.predict_from_dist(mean, cov)
        if bound:
            action_mean, action_cov, input_output_cov = squash_action_dist(action_mean, action_cov, input_output_cov,
                                                                           bound)

        # prediction of cross_cov from GP is cross_cov @ inv(sigma)
        return action_mean, action_cov, cov @ input_output_cov

    def set_params(self, params):
        # reset cached matrices when new params are added
        self.K_inv = None
        self.beta = None

        for i, gp in enumerate(self.models):
            gp.unwrap_params(params[gp.length * i: gp.length * (i + 1)])
            # computes beta and K_inv for updated hyperparams of each GP model
            gp.compute_matrices()

    def get_params(self) -> np.ndarray:
        """
        returns parameters to optimize of policy
        :return: array of flattened rbf parameters
        """
        return np.array([gp.wrap_policy_hyperparams() for gp in self.models]).flatten()

    def optimize(self) -> None:
        """
        this does nothing, policy needs to be optimized based on rollouts.
        :return: None
        """
        raise ValueError("RBF policies cannot be optimized individually."
                         "Please use the optimization in the pilco class.")
