import logging

import autograd.numpy as np
import gym
import matplotlib.pyplot as plt
import quanser_robots

from PILCO.Controller.RBFController import RBFController
from PILCO.GaussianProcess.GaussianProcess import GaussianProcess
from PILCO.GaussianProcess.MultivariateGP import MultivariateGP


class PILCO(object):

    def __init__(self, env_name: str, seed: int, n_features: int, Horizon: int, cost_function: callable,
                 max_episode_steps: int = None, T_inv: np.ndarray = None, target_state: np.ndarray = None,
                 cost_width: np.ndarray = None, squash=True, p=.5):
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

        # -----------------------------------------------------
        # env setup
        # check if the requested environment is a real robot env
        if 'RR' in self.env_name:
            self.env = quanser_robots.GentlyTerminating(gym.make(self.env_name))
        else:
            # use the official gym env as default
            self.env = gym.make(self.env_name)

        if max_episode_steps is not None:
            self.env._max_episode_steps = max_episode_steps
        self.env.seed(self.seed)
        self.state_names = ["x", "sin(theta)", "cos(theta)", "x_dot", "theta_dot"]

        # get the number of available action from the environment
        self.state_dim = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        # -----------------------------------------------------
        # training params
        self.gamma = .99  # discount factor

        # -----------------------------------------------------
        # dynamics
        self.dynamics_model = None

        # -----------------------------------------------------
        # policy
        self.policy = None
        self.squash = squash
        self.n_features = n_features
        # TODO: increase by 25% when successful
        self.T = Horizon

        # -----------------------------------------------------
        # Value calc
        # TODO: use GP as cost function for unknown cost
        # If the cose comes from the environment and is not known,
        #  the cost function has to be learn with a GP or the like.

        # known cost function
        # self.cost_function = cost_function

        # -----------------------------------------------------
        # saturated cost function
        self.cost_function = self.saturated_cost
        # weight matrix
        self.T_inv = np.identity(self.state_dim) if T_inv is None else T_inv
        # set target state to all zeros if not other specified
        self.target_state = np.zeros(self.state_dim) if target_state is None else target_state

        # -----------------------------------------------------
        # This is only useful if we have any penalties etc.
        self.cost_width = np.array([1]) if cost_width is None else cost_width
        self.p = p

        # -----------------------------------------------------
        # Container for collected experience
        self.state_action_pairs = None
        self.state_delta = None
        self.rewards = None

        # -----------------------------------------------------
        # logging instance
        self.logger = logging.getLogger(__name__)

    def run(self, n_samples, n_steps=10):

        self.sample_inital_data_set(n_init=n_samples)

        for _ in range(n_steps):
            self.learn_dynamics_model(self.state_action_pairs, self.state_delta)
            self.learn_policy()

            X_test, y_test, reward_test = self.execute_test_run()
            self.state_action_pairs = np.append(self.state_action_pairs, X_test, axis=0)
            self.state_delta = np.append(self.state_delta, y_test, axis=0)
            self.rewards = np.append(self.rewards, reward_test)

    def sample_inital_data_set(self, n_init):
        """
        sample dataset with random actions
        :param n_init: amount of samples to be generated
        :return:
        """
        self.state_action_pairs = np.zeros((n_init, self.state_dim + self.n_actions))
        self.state_delta = np.zeros((n_init, self.state_dim))
        self.rewards = np.zeros((n_init,))

        i = 0
        while i < n_init:
            state_prev = self.env.reset()
            done = False

            while not done and i < n_init:
                self.env.render()
                action = self.env.action_space.sample()
                state, reward, done, _ = self.env.step(action)

                state = np.array(state)
                state_prev = np.array(state_prev)

                # state-action pair as input
                self.state_action_pairs[i] = np.concatenate([state_prev, action])

                # TODO maybe add some noise to the delta
                # noise = np.random.multivariate_normal(np.ones(state.shape), self.var_eps * np.identity(state.shape[0]))
                self.state_delta[i] = state - state_prev

                self.rewards[i] = reward
                state_prev = state
                i += 1

    def learn_dynamics_model(self, X, y):
        if self.dynamics_model is None:
            l, sigma_f, sigma_eps = self.get_init_hyperparams(X, y)
            self.dynamics_model = MultivariateGP(length_scales=l, n_targets=y.shape[1], sigma_f=sigma_f,
                                                 sigma_eps=sigma_eps, container=GaussianProcess)
        self.dynamics_model.fit(X, y)
        self.dynamics_model.optimize()

    def execute_test_run(self):

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
            action, _, _ = self.policy.choose_action(state_prev, 0 * np.identity(len(state_prev)), squash=True,
                                                     bound=self.env.action_space.high)

            state, reward, done, _ = self.env.step(action)

            # create history and new training instance
            X.append(np.append(state_prev, action))

            # TODO potentially add some noise
            # epsilon = np.random.normal(0, self.var_eps)
            # y.append(state - state_prev + epsilon)

            y.append(state - state_prev)

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
        l = np.log(np.std(X, axis=0))
        sigma_f = np.log(np.std(y))
        sigma_eps = np.log(np.std(y / 10))

        return l, sigma_f, sigma_eps

    def rollout(self, policy, print_trajectory=False):

        # TODO select good initial state dist
        # Currently this is taken from the CartPole Problem, Deisenroth (2010)
        state_mu = np.zeros((self.state_dim,))
        state_cov = 1e-2 * np.identity(self.state_dim)
        # Make s positive semidefinite
        state_cov = state_cov.dot(state_cov.T)

        reward = 0

        # --------------------------------------------------------
        # Alternatives:
        # state_mu = X[:, :self.state_dim].mean(axis=0)
        # state_mu = np.array([0., 0., 0., np.pi, np.pi])

        # state_cov = X[:, :self.state_dim].std(axis=0)
        # state_cov = np.cov(X[:, :self.state_dim], rowvar=False
        # --------------------------------------------------------

        mu_state_container = []
        sigma_state_container = []

        mu_state_container.append(state_mu)
        sigma_state_container.append(state_cov)

        mu_action_container = []
        sigma_action_container = []

        for t in range(0, self.T):
            # ------------------------------------------------
            # get mean and covar over next action, optionally with squashing
            # Deisenroth (2010), page 44, Nonlinear Model: RBF Network
            # TODO check if this is working properly, the successor state is never changing over time. Maybe wrong actions is applied
            action_mu, action_cov, action_input_output_cov = policy.choose_action(state_mu, state_cov,
                                                                                  squash=self.squash,
                                                                                  bound=self.env.action_space.high)

            mu_action_container.append(action_mu)
            sigma_action_container.append(action_cov)

            # ------------------------------------------------
            # get joint dist over successor state p(x,u)
            state_action_mu, state_action_cov, state_action_input_output_cov = self.get_joint_dist(state_mu, state_cov,
                                                                                                   action_mu.flatten(),
                                                                                                   action_cov,
                                                                                                   action_input_output_cov)

            # ------------------------------------------------
            # compute delta and build next state dist
            # TODO check if this is working properly, the successor state is never changing over time. See plots
            delta_mu, delta_cov, delta_input_output_cov = self.dynamics_model.predict_from_dist(state_action_mu,
                                                                                                state_action_cov)

            # cross cov is times inv(s), see matlab code
            delta_input_output_cov = state_action_input_output_cov @ delta_input_output_cov

            # -----------------------------------------------------------------------------------------------------------
            # Debugging code

            # try:
            #     state_action_mu = state_action_mu._value
            # except Exception:
            #     state_action_mu = state_action_mu
            #
            # try:
            #     state_action_cov = state_action_cov._value
            # except Exception:
            #     state_action_cov = state_action_cov

            # sample over dist
            # x = np.random.multivariate_normal(state_action_mu, state_action_cov, size=1000)
            # if np.any(np.isnan(state_action_cov)) or np.any(np.isnan(state_action_mu)):
            #     print(state_action_cov)
            #     print(state_action_mu)
            #     print("nan")

            # x = []
            # for _ in range(100):
            #     # reparametrization trick
            #     x.append(np.random.randn(len(state_action_mu)) @ state_action_cov + state_action_mu)
            # x = np.array(x)
            # # use real env for dynamics
            # pred = []
            # self.env.reset()
            # for elem in x:
            #     print(elem)
            #     if np.any(np.isnan(elem)):
            #         continue
            #     self.env.env.state = elem[:-1]
            #     s, r, d, _ = self.env.step(np.array([elem[-1]])._value)
            #     pred.append(s)
            #
            # pred = np.array(pred).T

            # use deterministic prediction on samples one real GP dynamics
            # pred = self.dynamics_model.predict(x)

            # delta_mu = np.mean(pred, axis=1)
            # diff = pred - delta_mu[:, None]
            # delta_cov = 1 / (diff - 1) * diff @ diff.T

            # delta_cov = np.cov(pred)

            # print("-" * 50)
            # print("Difference between sampling and Moment matching:")
            # print("Mean:\n{}".format(np.mean(pred, axis=1) - delta_mu._value))
            # print("Mean ratio:\n {}".format((np.mean(pred, axis=1) - delta_mu._value) / delta_mu._value))
            # print("Covariance:\n{}".format(np.cov(pred) - delta_cov._value))
            # print("Covariance ratio:\n{}".format((np.cov(pred) - delta_cov._value) / delta_cov._value))

            # ------------------------------------------------
            # compute distribution over next state

            state_next_mu = delta_mu + state_mu
            state_next_cov = delta_cov + state_cov + delta_input_output_cov + delta_input_output_cov.T

            # compute value of current state prediction
            r, _, _ = self.cost_function(state_next_mu, state_next_cov)
            reward = reward + self.gamma ** t * r.flatten()

            mu_state_container.append(state_next_mu)
            sigma_state_container.append(state_next_cov)

            state_mu = state_next_mu
            state_cov = state_next_cov

        if print_trajectory:

            # plot state trajectory
            mu_state_container = np.array(mu_state_container)
            sigma_state_container = np.array(sigma_state_container)

            for i in range(len(state_mu)):
                m = mu_state_container[:, i]
                s = sigma_state_container[:, i, i]

                # TODO: This is stupid and bad
                try:
                    m = m._value
                except Exception:
                    m = m
                try:
                    s = s._value
                except Exception:
                    s = s

                plt.errorbar(np.arange(0, len(m)), m, yerr=s, fmt='-o')
                plt.title("Trajectory prediction for {}".format(self.state_names[i]))
                plt.show()

            # plot action trajectory
            mu_action_container = np.array(mu_action_container)
            sigma_action_container = np.array(sigma_action_container)

            # TODO: This is stupid and bad
            try:
                m = mu_action_container._value
            except Exception:
                m = mu_action_container
            try:
                s = sigma_action_container._value
            except Exception:
                s = sigma_action_container

            plt.errorbar(np.arange(0, len(m)), m, yerr=s, fmt='-o')
            plt.title("Trajectory prediction for actions")
            plt.show()

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
        top = np.hstack((state_cov, input_output_cov))
        bottom = np.hstack((input_output_cov.T, action_cov))
        joint_cov = np.vstack((top, bottom))

        return joint_mu, joint_cov, top

    def learn_policy(self, mu=0, sigma=0.1 ** 2, target_noise=0.1):

        # initialize policy if we do not already have one
        if self.policy is None:
            # init model params
            policy_X = np.random.multivariate_normal(np.full(self.state_dim, mu), sigma * np.identity(self.state_dim),
                                                     size=(self.n_features,))
            policy_y = target_noise * np.random.randn(self.n_features, self.n_actions)
            # A = np.random.rand(self.state_dim, self.n_actions)
            # policy_y = np.sin(policy_X).dot(A) + policy_y - 0.5)

            # augmented states would be initialized with .7, but we already have sin and cos given
            # and do not need to compute this with gaussian_trig
            length_scales = np.ones(self.state_dim)

            self.policy = RBFController(n_actions=self.n_actions, n_features=self.n_features, rollout=self.rollout,
                                        length_scales=length_scales)
            self.policy.fit(policy_X, policy_y)

        self.policy.optimize()
        print()

    def saturated_cost(self, mu, sigma):
        mu = np.atleast_2d(mu)
        self.target_state = np.atleast_2d(self.target_state)

        sigma_T_inv = np.dot(sigma, self.T_inv)
        S1 = np.linalg.solve((np.identity(self.state_dim) + sigma_T_inv).T, self.T_inv.T).T
        diff = mu - self.target_state

        # compute expected cost
        mean = -np.exp(-diff @ S1 @ diff.T / 2) * ((np.linalg.det(np.identity(self.state_dim) + sigma_T_inv)) ** -.5)

        # compute variance of cost
        S2 = np.linalg.solve((np.identity(self.state_dim) + 2 * sigma_T_inv).T, self.T_inv.T).T
        r2 = np.exp(-diff @ S2 @ diff.T) * ((np.linalg.det(np.identity(self.state_dim) + 2 * sigma_T_inv)) ** -.5)
        cov = r2 - mean ** 2

        # for numeric reasons set to 0
        if np.all(cov < 1e-12):
            cov = np.zeros(cov.shape)

        t = np.dot(self.T_inv, self.target_state.T) - S1 @ (np.dot(sigma_T_inv, self.target_state.T) + mu.T)

        cross_cov = sigma @ (mean * t)

        # bring cost to the interval [0,1]
        return 1 + mean, cov, cross_cov

    def saturated_loss(self, mu, sigma):
        # TODO: Ask supervisors if we need to do this.
        # We do not have information about the env for penalties or the like.
        cost = 0
        for w in self.cost_width:
            mu, _, _ = self.saturated_cost(mu, sigma)
            cost = cost + mu
        return cost / len(w)
