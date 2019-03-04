from PILCO.Controller.Controller import Controller
import autograd.numpy as np

from PILCO.util.util import squash_action_dist


class LinearController(Controller):

    def __init__(self, state_dim: int, n_actions: int, W: np.ndarray = None, b: np.ndarray = None):
        """
        Linear controller
        :param state_dim: state dim of env
        :param n_actions: amount of actions required
        :param W: Weight parameters
        :param b: bias parameters
        """
        self.W = W if W else np.random.rand(state_dim, n_actions)
        self.b = b if b else np.random.rand(1, n_actions)

    def set_params(self, params: np.ndarray):
        """
        set parameters of linear policy as flatt array. containing first W then b
        :param params: flat ndarray of params
        :return: None
        """
        idx = len(self.W.flatten())
        self.W = params[:idx].reshape(self.W.shape)
        self.b = params[idx:].reshape(self.b.shape)

    def get_params(self):
        return np.concatenate([self.W.flatten(), self.b.flatten()])

    def choose_action(self, mu, sigma, bound):
        action_mu = mu @ self.W + self.b
        action_sigma = self.W.T @ sigma @ self.W
        action_input_output_cov = self.W

        if bound is not None:
            action_mu, action_sigma, action_input_output_cov = squash_action_dist(action_mu, action_sigma,
                                                                                  action_input_output_cov, bound)

        return action_mu, action_sigma, action_input_output_cov
