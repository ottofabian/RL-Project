import copy
import logging

import autograd.numpy as np
import gym
import quanser_robots
from autograd import value_and_grad
from scipy.optimize import minimize

from PILCO.Controller import Controller
from PILCO.Controller.RBFController import RBFController
from PILCO.GaussianProcess.MultivariateGP import MultivariateGP


class PILCO(object):

    def __init__(self, env_name: str, seed: int, n_features: int, Horizon: int, cost_function: callable,
                 T_inv: np.ndarray = None, target_state: np.ndarray = None, cost_width: np.ndarray = None):
        """

        :param env_name: gym env to work with
        :param seed: random seed for reproduceability
        :param n_features: Amount of features for RBF Controller
        :param Horizon: number of steps for trajectory rollout, also defined as horizon
        :param cost_function: Function handle which defines the cost for the given environment.
                              This function is used for policy optimization.
        """

        # -----------------------------------------------------
        # general
        self.env_name = env_name
        self.seed = seed
        self.n_features = n_features

        # -----------------------------------------------------
        # env setup
        # check if the requested environment is a quanser robot env
        if self.env_name in ['CartpoleStabShort-v0']:
            self.env = quanser_robots.GentlyTerminating(gym.make(self.env_name))
        else:
            # use the official gym env as default
            self.env = gym.make(self.env_name)
        self.env.seed(self.seed)

        # get the number of available action from the environment
        self.state_dim = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        # -----------------------------------------------------
        # training params
        # self.n_training_samples = n_training_samples
        self.gamma = 0.9  # discount factor

        # -----------------------------------------------------
        # dynamics model
        self.dynamics_model = None

        # -----------------------------------------------------
        # Hyperparameter to optimize_policy
        # TODO change to meaning full inits
        self.l = np.random.normal(0, np.ones(self.state_dim))
        self.var_f = 1
        self.var_eps = 0.1  # target noise for prediction of GP and Policy
        self.opt_ctr = 0

        # -----------------------------------------------------
        # policy search
        self.T = Horizon

        # -----------------------------------------------------
        # Value calc
        # TODO: use GP as cost function for unkown cost
        # The cost function has to be learn with a GP as it comes from the environment and is not known.

        # known cost function
        # TODO
        # self.cost_function = cost_function
        self.cost_function = self.saturated_cost
        # weight matix for sat loss
        self.T_inv = np.identity(self.state_dim) if T_inv is None else T_inv
        # set target state to all zeros if not other specified
        self.target_state = np.zeros(self.state_dim) if target_state is None else target_state
        self.cost_width = np.array([1]) if cost_width is None else cost_width

        # -----------------------------------------------------
        # Container for collected experience
        self.state_action_pairs = None
        self.state_delta = None

        # -----------------------------------------------------
        # logging instance
        self.logger = logging.getLogger(__name__)

    def run(self, n_init):

        # TODO maybe change the structure to Deisenroth (2010), page 36
        self.sample_inital_data_set(n_init=n_init)

        policy = self.get_rbf_policy()

        while True:
            self.learn_dynamics_model(self.state_action_pairs, self.state_delta)
            # Deisenroth (2010), page 47, 3.5.4
            self.policy_improvement(policy)

            X_test, y_test, reward_test = self.execute_test_run(policy)
            self.state_action_pairs = np.append(self.state_action_pairs, X_test, axis=0)
            self.state_delta = np.append(self.state_delta, y_test, axis=0)
            rewards = np.append(rewards, reward_test)

    def sample_inital_data_set(self, n_init):
        """
        sample dataset with random actions
        :param n_init: amount of samples to be generated
        :return:
        """
        self.state_action_pairs = np.zeros((n_init, self.state_dim + self.n_actions))
        self.state_delta = np.zeros((n_init, self.state_dim))
        rewards = np.zeros((n_init,))

        i = 0
        while i < n_init:
            state_prev = self.env.reset()
            done = False

            while not done and i < n_init:
                action = self.env.action_space.sample()
                state, reward, done, _ = self.env.step(action)

                state = np.array(state)
                state_prev = np.array(state_prev)

                # state-action pair as input
                self.state_action_pairs[i] = np.concatenate([state_prev, action])

                # delta to following state as output plus some noise
                self.state_delta[i] = state - state_prev + np.random.multivariate_normal(np.ones(state.shape),
                                                                                         self.var_eps * np.identity(
                                                                                             state.shape[0]))

                rewards[i] = reward
                state_prev = state
                i += 1

        # convert to numpy
        self.state_action_pairs = np.array(self.state_action_pairs)
        self.state_delta = np.array(self.state_delta)
        rewards = np.array(rewards)

    def learn_dynamics_model(self, X, y):
        if self.dynamics_model is None:
            l, sigma_f, sigma_eps = self.get_init_hyperparams(X, y)
            self.dynamics_model = MultivariateGP(length_scales=l, n_targets=y.shape[1], sigma_f=sigma_f,
                                                 sigma_eps=sigma_eps)
        self.dynamics_model.fit(X, y)

    def unwrap_rbf_params(self, x):
        split1 = self.state_dim * self.n_features
        split2 = self.n_actions * self.n_features + split1
        X = x[:split1].reshape(self.n_features, self.state_dim)
        y = x[split1:split2].reshape(self.n_features, self.n_actions)
        length_scales = x[split2:].reshape(self.n_actions, self.state_dim)
        return X, y, length_scales

    def optimize_policy(self, x, *args):
        self.opt_ctr += 1
        p = args[0]
        policy = copy.deepcopy(p)
        if isinstance(policy, RBFController):
            X, y, length_scales = self.unwrap_rbf_params(x)
        # X = np.array([np.linspace(-.01, .01, num=self.n_features) for _ in range(self.state_dim)]).T
        # y = x.reshape(self.n_features, self.n_actions)
        else:
            raise NotImplementedError("For this policy no optimization is implemented.")

        policy.set_hyper_params(X, y, length_scales)
        self.optimization_callback(policy)
        # self.logger.debug("Best Params: \n", X, y, length_scales)
        return self.rollout(policy)

    def policy_improvement(self, policy):
        # minimise cost given policy
        args = (policy,)
        # x0 = policy._wrap_kernel_hyperparams()[self.state_dim * self.n_features:]
        x0 = policy._wrap_kernel_hyperparams()
        # For testing only
        # options = {'maxiter': 1, 'disp': True}
        # res = minimize(self.optimize_policy, x0, args, method='L-BFGS-B', jac=None)
        # TODO Autograd
        res = minimize(value_and_grad(self.optimize_policy), x0, args, method='L-BFGS-B', jac=True)

        self.opt_ctr = 0
        X, y, length_scales = self.unwrap_rbf_params(res.x)
        policy.set_hyper_params(X, y, length_scales)
        # self.logger.debug("Best Params: \n", X, y, length_scales)

    def execute_test_run(self, policy):

        X = []
        y = []
        rewards = []

        state_prev = self.env.reset()
        done = False
        t = 0
        while not done:
            self.env.render()
            t += 1
            state_prev = np.array(state_prev)
            # no uncertainty during testing required
            action, _, _ = policy.choose_action(state_prev, 0 * np.identity(len(state_prev)))
            state, reward, done, _ = self.env.step(action)

            # create history and new training instance
            X.append(np.append(state_prev, action))
            epsilon = np.random.normal(0, self.var_eps)
            y.append(state - state_prev + epsilon)
            rewards.append(reward)
            state_prev = state

        print("reward={}, episode_len={}".format(np.sum(rewards), t))
        return np.array(X), np.array(y), np.array(rewards)

    def get_init_hyperparams(self, X, y):
        """
        Compute hyperparams for GPR
        :param i:
        :param X: training vector containing values for [x,u]^T
        :param y: target vector containing deltas of states
        :return:
        """
        l = np.std(X, axis=0)
        sigma_f = np.std(y)
        sigma_eps = np.std(y / 10)

        return l, sigma_f, sigma_eps

    def rollout(self, policy: Controller):

        # TODO select good initial state dist
        # Currently this is taken from the CartPole Problem, Deisenroth (2010)
        state_mu = np.zeros((self.state_dim,))
        # state_mu = np.array([0., 0., 0., np.pi, np.pi])
        state_cov = 1e-2 * np.identity(self.state_dim)
        reward = 0

        # --------------------------------------------------------
        # Alternatives:
        # state_mu = X[:, :self.state_dim].mean(axis=0)

        # state_cov = X[:, :self.state_dim].std(axis=0)
        # state_cov = np.cov(X[:, :self.state_dim], rowvar=False
        # --------------------------------------------------------

        # container
        # TODO: Is this needed??
        # trajectory_mu = np.zeros((self.T + 1, state_mu.shape[0]))
        # trajectory_cov = np.zeros((self.T + 1, state_cov.shape[0], state_cov.shape[1]))
        # trajectory_mu[0] = state_mu
        # trajectory_cov[0] = state_cov

        for t in range(0, self.T):
            # ------------------------------------------------
            # get mean and covar over next action
            # Deisenroth (2010), page 44, Nonlinear Model: RBF Network
            action_mu, action_cov, action_input_output_cov = policy.choose_action(state_mu, state_cov)

            action_squashed_mu, action_squashed_cov, action_squashed_input_output_cov = self.squash_action_dist(
                action_mu, action_cov, action_input_output_cov)

            # ------------------------------------------------
            # get joint dist over successor state p(x,u)
            state_action_mu, state_action_cov = self.get_joint_dist(state_mu, state_cov, action_squashed_mu,
                                                                    action_squashed_cov,
                                                                    action_squashed_input_output_cov)

            # ------------------------------------------------
            # compute new state dist
            delta_mu, delta_cov, delta_input_output_cov = self.dynamics_model.predict_from_dist(state_action_mu,
                                                                                                state_action_cov)
            # Cov(state, delta) is subset of Cov((state, action), delta)
            state_input_output_cov = delta_input_output_cov[:, :self.state_dim]

            state_next_mu = state_mu + delta_mu
            state_next_cov = state_cov + delta_cov + state_input_output_cov + state_input_output_cov.T

            # trajectory_mu[t] = state_next_mu
            # trajectory_cov[t] = state_next_cov
            r, _, _ = self.cost_function(state_next_mu, state_next_cov)
            reward = reward + self.gamma ** t * r.flatten()

            state_mu = state_next_mu
            state_cov = state_next_cov

        return reward

    def get_joint_dist(self, state_mu, state_cov, action_mu, action_cov, input_output_cov):
        """
        This returns the joint gaussian dist of state and action
        :param state_mu:
        :param state_cov:
        :param action_mu:
        :param action_cov:
        :param input_output_cov:
        :return:
        """

        # compute joint Gaussian as in Master Thesis, page 23
        joint_mu = np.concatenate([state_mu, action_mu])

        # Has shape
        # Σxt Σxt,ut
        # (Σxt,ut).T Σut
        top = np.vstack((state_cov, input_output_cov))
        bottom = np.vstack((input_output_cov.T, action_cov))
        joint_cov = np.hstack((top, bottom))

        return joint_mu, joint_cov

    def squash_action_dist(self, mu, sigma, input_output_cov):
        """
        Rescales and squashes the distribution x with sin(x)
        :param input_output_cov:
        :param mu:
        :param sigma:
        :return:
        """

        # mu, sigma = np.array([.01, .5]), np.random.normal(size=(2, 2))

        # p(u)' is squashed distribution over p(u) scaled by action space values,
        # see Deisenroth (2010), page 46, 2a)+b) and Section 2.3.2
        bound = self.env.action_space.high

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

    def get_rbf_policy(self, mu=0, sigma=0.1 ** 2, target_noise=0.1):

        # init model params
        policy_X = np.random.multivariate_normal(np.full(self.state_dim, mu), sigma * np.identity(self.state_dim),
                                                 size=(self.n_features,))
        policy_y = target_noise * np.random.randn((self.n_features, self.n_actions))

        # augmented states would be initalized with .7
        length_scales = np.ones((self.n_actions, self.state_dim))

        policy = RBFController(n_actions=self.n_actions)
        policy.set_hyper_params(policy_X, policy_y, length_scales)
        return policy

    def optimization_callback(self, policy):

        if self.opt_ctr % 2 == 0:
            print("Policy optimization iteration: {} -- Cost: {}".format(self.opt_ctr, self.rollout(policy)))
        else:
            print("Policy optimization iteration: {}".format(self.opt_ctr))

    def saturated_cost(self, mu, sigma):
        mu = np.atleast_2d(mu)
        self.target_state = np.atleast_2d(self.target_state)

        sigma_T_inv = np.dot(sigma, self.T_inv)
        S1 = np.linalg.solve((np.eye(self.state_dim) + sigma_T_inv).T, self.T_inv.T).T
        diff = mu - self.target_state

        # compute expected cost
        mean = -np.exp(-diff @ S1 @ diff.T * .5) * ((np.linalg.det(np.eye(self.state_dim) + sigma_T_inv)) * -.5)

        # compute variance of cost
        S2 = np.linalg.solve((np.eye(self.state_dim) + 2 * sigma_T_inv).T, self.T_inv.T).T
        r2 = np.exp(-diff @ S2 @ diff.T) * ((np.linalg.det(np.eye(self.state_dim) + 2 * sigma_T_inv)) * -.5)
        cov = r2 - mean ** 2

        # for numeric reasons set to 0
        if np.all(cov < 1e-12):
            cov = np.zeros(cov.shape)

        t = np.dot(self.T_inv, self.target_state.T) - S1 @ (np.dot(sigma_T_inv, self.target_state.T) + mu.T)

        cross_cov = sigma @ (mean * t)

        # bring cost to the interval [0,1]
        return 1 + mean, cov, cross_cov

    def saturated_loss(self):
        # TODO: Ask supervisors if we need to do this.
        # We do not have information about the env for penalties or the like.
        for w in self.cost_width:
            return
