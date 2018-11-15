import copy

import autograd.numpy as np
import gym
import quanser_robots
from autograd import grad
from scipy.optimize import minimize

from PILCO.Controller import Controller
from PILCO.Controller.RBFController import RBFController
from PILCO.MGPR import MGPR


class PILCO(object):

    def __init__(self, env_name: str, seed: int, n_features: int, T: int, cost_function: callable,
                 n_training_samples: int):
        """

        :type n_training_samples: object
        :param env_name: gym env to work with
        :param seed: random seed for reproduceability
        :param n_features: Amount of features for RBF Controller
        :param T: number of steps for trajectory rollout, also defined as horizon
        :param cost_function: Function handle which defines the cost for the given environment.
                              This function is used for policy optimization.
        """

        # -----------------------------------------------------
        # general
        self.env_name = env_name
        self.seed = seed
        self.n_features = n_features
        self.noise_var = None

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
        self.n_training_samples = n_training_samples

        # -----------------------------------------------------
        # dynamics model
        self.dynamics_model = None

        # -----------------------------------------------------
        # Hyperparameter to optimize
        # TODO change to meaning full inits
        self.l = np.random.normal(0, np.ones(self.state_dim))
        self.var_f = 1
        self.var_eps = 0.1  # target noise for prediction of GP and Policy
        self.opt_ctr = 0

        # -----------------------------------------------------
        # policy search
        self.T = T

        # -----------------------------------------------------
        # Value calc
        # TODO: use GP as cost function for unkown cost
        # The cost function has to be learn with a GP as it comes from the environment and is not known.

        # known cost function
        self.cost_function = cost_function

        # -----------------------------------------------------
        # Container for collected experience
        self.state_action_pairs = None
        self.state_delta = None

    def run(self, n_init):

        # TODO maybe change the structure to Deisenroth (2010), page 36

        # sample dataset with random actions
        self.state_action_pairs = np.zeros((n_init, self.state_dim + self.n_actions))
        self.state_delta = np.zeros((n_init, self.state_dim))
        rewards = np.zeros((n_init,))

        policy = self.get_rbf_policy()

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

        while True:
            self.learn_dynamics_model(self.state_action_pairs, self.state_delta)
            # Deisenroth (2010), page 47, 3.5.4
            self.policy_improvement(policy)

            X_test, y_test, reward_test = self.execute_test_run(policy)
            self.state_action_pairs = np.append(self.state_action_pairs, X_test)
            self.state_delta = np.append(self.state_delta, y_test)
            np.append(rewards, reward_test)

    def learn_dynamics_model(self, X, y):
        # TODO do we only change params at the beginning?
        l, sigma_f, sigma_eps = self.get_init_hyperparams(X, y)
        self.dynamics_model = MGPR(length_scales=l, n_targets=y.shape[1], sigma_f=sigma_f,
                                   sigma_eps=sigma_eps)
        self.dynamics_model.fit(X, y)

    def optimize_policy(self, x, *args):
        self.opt_ctr += 1
        p = args
        policy = copy.deepcopy(p)[0]
        if isinstance(policy, RBFController):
            split = self.state_dim * self.n_features
            X = x[:split].reshape(self.n_features, self.state_dim)
            y = x[split:].reshape(self.n_features, self.n_actions)

            # X = np.array([np.linspace(-.01, .01, num=self.n_features) for _ in range(self.state_dim)]).T
            # y = x.reshape(self.n_features, self.n_actions)
        else:
            raise NotImplementedError("For this policy no optimization is implemented.")

        policy.fit(X, y)
        self.optimization_callback(policy)

        return self.rollout(policy)

    def optimization_callback(self, policy):
        if self.opt_ctr % 10 == 0:
            print("Policy optimization iteration: {} -- Cost: {}".format(self.opt_ctr, self.rollout(policy)))
        else:
            print("Policy optimization iteration: {}".format(self.opt_ctr))

    def policy_improvement(self, policy):
        # minimise cost given policy
        args = (policy,)
        # x0 = policy.get_hyperparams()[self.state_dim * self.n_features:]
        x0 = policy.get_hyperparams()
        res = minimize(self.optimize_policy, x0, args, method='L-BFGS-B', jac=grad(self.rollout))

        self.opt_ctr = 0
        policy.fit(res)

    def execute_test_run(self, policy):
        X = []
        y = []
        rewards = []

        state_prev = self.env.reset()
        done = False

        while not done:
            action = policy.choose_action(state_prev)
            state, reward, done, _ = self.env.step(action)

            # create history and create new training instance
            X.append(np.append(state_prev, action))
            epsilon = np.random.normal(0, self.noise_var)
            y.append(state - state_prev + epsilon)

            state_prev = state

        return X, y, rewards

    def get_init_hyperparams(self, X, y, i=None):
        """
        Compute hyperparams for GPR
        :param i:
        :param X: training vector containing values for [x,u]^T
        :param y: target vector containing deltas of states
        :return:
        """
        l = np.var(X[:i, :], axis=0)
        sigma_f = np.var(y[:i, :])
        sigma_eps = np.var(y[:i, :] / 10)

        return l, sigma_f, sigma_eps

    def rollout(self, policy: Controller):

        # TODO select good initial state dist
        # Currently this is taken from the CartPole Problem, Deisenroth (2010)
        state_mu = np.zeros((self.state_dim,))
        if np.any(np.isnan(state_mu)):
            print("NaNNaNNaNNaNNaNNaNNaN Batman")
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
            # sample for reward
            action = np.random.multivariate_normal(action_squashed_mu, action_squashed_cov,
                                                   size=self.n_training_samples)
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

            sampled_state = np.random.multivariate_normal(state_mu, state_cov, size=self.n_training_samples)
            reward += np.mean([self.cost_function(s, action[i]) for i, s in enumerate(sampled_state)])

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

            # compute mean and cov of successor state dist
            # # compute precision matrix and different inv
            # precision = np.diag(self.l)
            # state_tilde_cov_inv = np.linalg.solve(state_tilde_cov, np.identity(len(state_tilde_cov)))
            # precision_inv = np.linalg.solve(precision, np.identity(len(precision)))
            # precision_absolute_inv = np.linalg.solve(np.abs(precision), np.identity(len(precision)))
            #
            # exp = -(self.state_dim + self.n_actions) * .5
            #
            # c1_inv = (1 / self.var_f) * (2 * np.pi) ** exp * precision_absolute_inv ** .5
            #
            # const = (2 * np.pi) ** exp
            # # diff = (X - state_tilde_mu)
            # inv = np.linalg.solve(precision + state_tilde_cov, np.identity(len(precision)))
            # c2_inv = np.array([const * np.exp(-.5 * (x - state_tilde_mu).T @ inv @ (x - state_tilde_mu)) for x in X])
            #
            # omega = np.linalg.solve(precision_inv + state_tilde_cov_inv, np.identity(len(precision_inv)))
            # w = (X @ precision_inv.T + state_tilde_cov_inv @ state_tilde_mu) @ omega
            #
            # q = np.array([np.linalg.inv(c1_inv) * c for c in c2_inv])
            # beta = delta_mu @ np.linalg.inv(q)
            #
            # state_tilde_delta_mu = np.sum((1 / c1_inv) * c2_inv) * beta * w

            # betas = self.dynamics_model.get_betas()
            # qs = np.array([betas[i] / mu for i, mu in enumerate(delta_mu)])
            #
            # matrices = state_tilde_cov @ np.linalg.solve(state_tilde_cov + precision, np.identity(state_tilde_cov.shape[0])) @ (
            #         X - state_tilde_mu).T
            #
            # cross_cov_state_tilde = np.array([np.sum(betas[i] * qs[i] * matrices, axis=1) for i in range(len(betas))])
            # cross_cov_state = cross_cov_state_tilde[:, :self.state_dim]

            state_next_mu = state_mu + delta_mu
            state_next_cov = state_cov + delta_cov + state_input_output_cov + state_input_output_cov.T

            # trajectory_mu[t] = state_next_mu
            # trajectory_cov[t] = state_next_cov

            state_cov = state_next_cov
            state_mu = state_next_mu

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

    def get_rbf_policy(self):

        # init model params
        policy_X = np.random.multivariate_normal(np.zeros(self.state_dim),
                                                 np.diag(np.full(self.state_dim, self.var_eps)), size=self.n_features)
        policy_y = np.random.multivariate_normal(np.zeros(self.n_actions),
                                                 np.diag(np.full(self.n_actions, self.var_eps)), size=self.n_features)

        # noise values are fixed for RBF policy
        l, _, _ = self.get_init_hyperparams(policy_X, policy_y)

        policy = RBFController(length_scales=l, n_actions=self.n_actions)
        policy.fit(policy_X, policy_y)
        return policy
