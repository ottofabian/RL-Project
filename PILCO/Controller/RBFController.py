import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize

from PILCO.Controller.Controller import Controller
from PILCO.GaussianProcess.MultivariateGP import MultivariateGP
from PILCO.GaussianProcess.RBFNetwork import RBFNetwork


def squash_action_dist(mu, sigma, input_output_cov, bound):
    """
    Rescales and squashes the distribution x with sin(x)
    See Deisenroth(2010) Appendix A.1 for mu of sin(x), where x~N(mu, sigma)
    :param bound:
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

    return mu_squashed, sigma_squashed, input_output_cov_squashed


class RBFController(MultivariateGP, Controller):
    """RBF Controller/Policy"""

    def __init__(self, n_actions, n_features, compute_cost, length_scales, ridge=1e-6):
        # sigma_f is fixed for the RBF Controller, if it is seen as deterministic GP
        # sigma_eps is .01 to ensure a numerical stable computation
        sigma_f = np.log(1)
        sigma_eps = np.log(.01)

        MultivariateGP.__init__(self, length_scales=length_scales, n_targets=n_actions, sigma_f=sigma_f,
                                sigma_eps=sigma_eps, container=RBFNetwork, is_policy=True)

        self.length_scales = None
        self.ridge = ridge
        self.n_features = n_features
        self.compute_cost = compute_cost
        self.opt_ctr = 0

    def fit(self, X, y):
        # TODO this fits all X for all predictions, this does not matter for 1D actions
        MultivariateGP.fit(self, X, y)

        # second X shape is for lengthscales
        # self.n_params = X.shape[0] * X.shape[1] + y.shape[0] * y.shape[1] + X.shape[0] + 1

    def choose_action(self, mu, sigma, bound=None):
        action_mu, action_cov, input_output_cov = self.predict_from_dist(mu, sigma)

        if bound is not None:
            action_mu, action_cov, input_output_cov = squash_action_dist(action_mu, action_cov, input_output_cov, bound)

        # prediction from GP of cross_cov is times inv(s)
        return action_mu, action_cov, sigma @ input_output_cov

    def optimize(self):
        # TODO make this working for n_actions > 1
        params = np.array([gp.wrap_policy_hyperparams() for gp in self.gp_container]).flatten()
        options = {'maxiter': 150, 'disp': True}

        try:
            self.logger.info("Starting to optimize policy with L-BFGS-B.")
            res = minimize(fun=value_and_grad(self._optimize_hyperparams), x0=params, method='L-BFGS-B', jac=True,
                           options=options)

            # numerical grad
            # res = minimize(fun=self._optimize_hyperparams, x0=params, method='L-BFGS-B', jac=False,
            #                options=options)
        except Exception:
            self.logger.info("Starting to optimize policy with CG.")
            res = minimize(fun=value_and_grad(self._optimize_hyperparams), x0=params, method='CG', jac=True,
                           options=options)
        self.opt_ctr = 0

        # TODO make this working for n_actions > 1
        for gp in self.gp_container:
            gp.unwrap_params(res.x)
            gp.compute_matrices()

        # Make one more run for plots
        self.compute_cost(self, print_trajectory=True)

    def _optimize_hyperparams(self, params):

        self.opt_ctr += 1

        # TODO make this working for n_actions > 1
        for gp in self.gp_container:
            gp.unwrap_params(params)
            # computes beta and K_inv for updated hyperparams
            gp.compute_matrices()

        self.logger.debug("Params as given from the optimization step:")
        self.logger.debug(np.array2string(params if type(params) == np.ndarray else params._value))

        # returns cost of trajectory rollout
        cost = self.compute_cost(self, print_trajectory=False)

        # print progress
        # ToDo make this a callback
        self.logger.info("Policy optimization iteration: {} -- Cost: {}".format(self.opt_ctr, np.array2string(
            cost if type(cost) == np.ndarray else cost._value)))

        return cost
