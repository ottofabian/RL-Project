from pilco.controller.controller import Controller
import autograd.numpy as np

from pilco.util.util import squash_action_dist


class LinearController(Controller):

    def __init__(self, state_dim: int, n_actions: int, weights: np.ndarray = None, bias: np.ndarray = None):
        """
        Linear controller
        :param state_dim: state dim of env
        :param n_actions: amount of actions required
        :param weights: Weight parameters
        :param bias: bias parameters
        """
        self.weights = weights if weights else np.random.rand(state_dim, n_actions)
        self.bias = bias if bias else np.random.rand(1, n_actions)

    def set_params(self, params: np.ndarray):
        """
        set parameters of linear policy as flatt array. containing first W then b
        :param params: flat ndarray of params
        :return: None
        """
        idx = len(self.weights.flatten())
        self.weights = params[:idx].reshape(self.weights.shape)
        self.bias = params[idx:].reshape(self.bias.shape)

    def get_params(self):
        """
        get parameters of linear policy as flattened array
        :return: ndarray of flat [W,b]
        """
        return np.concatenate([self.weights.flatten(), self.bias.flatten()])

    def choose_action(self, mean: np.ndarray, cov: np.ndarray, bound: np.ndarray = None) -> tuple:
        """
        chooses action based on linear policy from given state distribution
        :param mean: mean of state distribution
        :param cov: covariance of state distribution
        :param bound: max action if required
        :return: action_mean, action_cov, input_output_cov
        """
        action_mean = mean @ self.weights + self.bias
        action_cov = self.weights.T @ cov @ self.weights
        action_input_output_cov = self.weights

        if bound is not None:
            action_mean, action_cov, action_input_output_cov = squash_action_dist(action_mean, action_cov,
                                                                                  action_input_output_cov, bound)

        return action_mean, action_cov, action_input_output_cov
