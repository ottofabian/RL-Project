import logging

import autograd.numpy as np
import gym
import matplotlib.pyplot as plt
import quanser_robots

from PILCO.Controller.Controller import Controller
from PILCO.Controller.RBFController import RBFController
from PILCO.CostFunctions.Loss import Loss
from PILCO.GaussianProcess.GaussianProcess import GaussianProcess
from PILCO.GaussianProcess.MultivariateGP import MultivariateGP


class PILCO(object):

    def __init__(self, env_name: str, seed: int, n_features: int, Horizon: int, loss: Loss, start_mu: np.ndarray = None,
                 start_cov: np.ndarray = None, gamma=1, max_episode_steps: int = None, bound: np.ndarray = None):
        """

        :param env_name: gym env to work with
        :param seed: random seed for reproduceability
        :param n_features: Amount of features for RBF Controller
        :param Horizon: number of steps for trajectory rollout, also defined as horizon
        :param loss: loss object which defines the cost for the given environment.
                              This function is used for policy optimization.
        :param start_mu: mean of starting state for trajectory rollout
        :param start_cov: covariance of starting state for trajectory rollout
        :param gamma: discount factor
        :param max_episode_steps: maximum steps for one episode
        :param bound: squash action with sin to +-bound or None if no squashing is required
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
        if self.env_name == "Pendulum-v0":
            self.state_names = ["cos($\\theta$)", "sin($\\theta$)", "$\\dot{\\theta}$"]
        elif "Cartpole" in self.env_name:
            # self.state_names = self.env.observation_space.labels
            self.state_names = ["x", "sin($\\theta$)", "cos($\\theta$)", "$\\dot{x}$", "$\\dot{\\theta}$"]

        # get the number of available action from the environment
        self.state_dim = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        # -----------------------------------------------------
        # training params
        self.gamma = gamma  # discount factor

        # -----------------------------------------------------
        # dynamics
        self.dynamics_model = None

        # -----------------------------------------------------
        # policy
        self.policy = None
        self.bound = bound
        self.n_features = n_features
        # TODO: increase by 25% when successful
        self.Horizon = Horizon

        # -----------------------------------------------------
        # rollout variables
        self.loss = loss
        self.start_mu = start_mu
        self.start_cov = start_cov

        # -----------------------------------------------------
        # Container for collected experience
        self.state_action_pairs = None
        self.state_delta = None
        self.rewards = None

        # -----------------------------------------------------
        # logging instance
        self.logger = logging.getLogger(__name__)

    def run(self, n_samples: int, n_steps: int = 10) -> None:
        """
        start pilco training run
        :param n_samples: number of initial samples before first policy optimization
        :param n_steps: maximum umber of learning steps until termination
        :return: None
        """

        self.sample_inital_data_set(n_init=n_samples)

        for _ in range(n_steps):
            self.learn_dynamics_model()
            self.learn_policy()

            X_test, y_test, reward_test = self.execute_test_run()

            # add test history to training data set
            self.state_action_pairs = np.append(self.state_action_pairs, X_test, axis=0)
            self.state_delta = np.append(self.state_delta, y_test, axis=0)
            self.rewards = np.append(self.rewards, reward_test)

    def sample_inital_data_set(self, n_init: int) -> None:
        """
        sample dataset with random actions
        :param n_init: amount of samples to be generated
        :return: None
        """
        self.state_action_pairs = np.zeros((n_init, self.state_dim + self.n_actions))
        self.state_delta = np.zeros((n_init, self.state_dim))
        self.rewards = np.zeros((n_init,))

        i = 0
        state_prev = self.env.reset()

        if self.env_name == "Pendulum-v0":
            theta = (np.arctan2(self.start_mu[1], self.start_mu[0]) + np.random.normal(0, .1, 1))[0]
            self.env.env.state = [theta, 0]
            state_prev = np.array([np.cos(theta), np.sin(theta), 0.])

        while i < n_init:

            self.env.render()

            # take initial random action
            action = self.env.action_space.sample()
            state, reward, done, _ = self.env.step(action)

            # safe state-action pair as input for dynamics GP
            self.state_action_pairs[i] = np.concatenate([state_prev, action])

            # include some noise to reduce data correlations and non semi definite matrices during optimization
            noise = np.random.multivariate_normal(np.zeros(state.shape), 1e-6 * np.identity(state.shape[0]))
            self.state_delta[i] = state - state_prev + noise

            self.rewards[i] = reward

            # reset env if terminal state was reached before max samples were generated
            if done:
                state = self.env.reset()
                if self.env_name == "Pendulum-v0":
                    theta = (np.arctan2(self.start_mu[1], self.start_mu[0]) + np.random.normal(0, .1, 1))[0]
                    self.env.env.state = [theta, 0]
                    state = np.array([np.cos(theta), np.sin(theta), 0.])

            state_prev = state
            i += 1

    def learn_dynamics_model(self) -> None:
        """
        Learn the dynamics model for the given environment
        :return: None
        """

        if self.dynamics_model is None:
            l, sigma_f, sigma_eps = self.get_init_hyperparams()
            self.dynamics_model = MultivariateGP(n_targets=self.state_dim, container=GaussianProcess, length_scales=l,
                                                 sigma_f=sigma_f, sigma_eps=sigma_eps)

        self.dynamics_model.fit(self.state_action_pairs, self.state_delta)
        self.dynamics_model.optimize()

    def learn_policy(self, mu: float = 0, sigma: float = 0.1 ** 2, target_noise: float = 0.1) -> None:
        """
        learn the policy based by trajectory rollouts
        :param mu: mean for sampling pseudo input
        :param sigma: covariance for sampling pseudo input
        :param target_noise: noise for sampling pseudo targets
        :return:
        """

        # initialize policy if we do not already have one
        if self.policy is None:
            # init model params
            policy_X = np.random.multivariate_normal(np.full(self.state_dim, mu), sigma * np.identity(self.state_dim),
                                                     size=(self.n_features,))
            policy_y = target_noise * np.random.randn(self.n_features, self.n_actions)

            # augmented states would be initialized with .7, but we already have sin and cos given
            # and do not need to compute this with gaussian_trig
            if self.env_name == "Pendulum-v0":
                length_scales = np.array([1., 1., 1.])
            else:
                length_scales = np.ones(self.state_dim)

            self.policy = RBFController(n_actions=self.n_actions, n_features=self.n_features,
                                        compute_cost=self.compute_trajectory_cost,
                                        length_scales=length_scales)
            self.policy.fit(policy_X, policy_y)

        self.policy.optimize()

    def compute_trajectory_cost(self, policy: Controller, print_trajectory: bool = False) -> float:
        """
        Compute predicted cost of on trajectory rollout using current policy and dynamics.
        This is used to optimize the policy
        :param policy: policy, which decides on actions
        :param print_trajectory: print the resulting trajectory
        :return: cost of trajectory
        """

        state_mu = self.start_mu
        state_cov = self.start_cov

        cost = 0

        mu_state_container = []
        sigma_state_container = []

        mu_state_container.append(state_mu)
        sigma_state_container.append(state_cov)

        mu_action_container = []
        sigma_action_container = []

        for t in range(0, self.Horizon):
            state_next_mu, state_next_cov, action_mu, action_cov = self.rollout(policy, state_mu, state_cov)

            # compute value of current state prediction
            l = self.loss.compute_loss(state_next_mu, state_next_cov)
            cost = cost + self.gamma ** t * l.flatten()

            mu_state_container.append(state_next_mu)
            sigma_state_container.append(state_next_cov)

            mu_action_container.append(action_mu)
            sigma_action_container.append(action_cov)

            state_mu = state_next_mu
            state_cov = state_next_cov

        if print_trajectory:
            self.print_trajectory(np.array(mu_state_container), np.array(sigma_state_container),
                                  np.array(mu_action_container), np.array(sigma_action_container))

        return cost

    def rollout(self, policy, state_mu, state_cov) -> tuple:
        """
        compute a single rollout given a state mean and covariance
        :param policy: policy object which decides on action
        :param state_mu: current mean to start rollout from
        :param state_cov: current covariance to start rollout from
        :return: state_next_mu, state_next_cov, action_mu, action_cov
        """

        # ------------------------------------------------
        # get mean and covar of next action, optionally with squashing and scaling towards an action bound
        # Deisenroth (2010), page 44, Nonlinear Model: RBF Network
        action_mu, action_cov, action_input_output_cov = policy.choose_action(state_mu, state_cov,
                                                                              bound=self.bound)

        # ------------------------------------------------
        # get joint dist p(x,u)
        state_action_mu, state_action_cov, state_action_input_output_cov = self.get_joint_dist(state_mu, state_cov,
                                                                                               action_mu.flatten(),
                                                                                               action_cov,
                                                                                               action_input_output_cov)

        # ------------------------------------------------
        # compute delta and build next state dist
        delta_mu, delta_cov, delta_input_output_cov = self.dynamics_model.predict_from_dist(state_action_mu,
                                                                                            state_action_cov)

        # cross cov is times inv(s), see matlab code
        delta_input_output_cov = state_action_input_output_cov @ delta_input_output_cov

        # ------------------------------------------------
        # compute distribution over next state
        state_next_mu = delta_mu + state_mu
        state_next_cov = delta_cov + state_cov + delta_input_output_cov + delta_input_output_cov.T

        return state_next_mu, state_next_cov, action_mu, action_cov

    def execute_test_run(self) -> tuple:
        """
        execute test run for max episode steps and return new training samples
        :return: states, state_deltas, rewards
        """

        X = []
        y = []
        rewards = []

        state_prev = self.env.reset()
        # [1,3] is returned and is reduced to 1D
        state_prev = state_prev.flatten()

        if self.env_name == "Pendulum-v0":
            theta = (np.arctan2(self.start_mu[1], self.start_mu[0]) + np.random.normal(0, .1, 1))[0]
            state_prev = np.array([np.cos(theta), np.sin(theta), 0])
            self.env.env.state = [theta, 0]

        done = False
        t = 0
        while not done:
            self.env.render()
            t += 1

            # no uncertainty during testing required
            action, _, _ = self.policy.choose_action(state_prev, 0 * np.identity(len(state_prev)), bound=self.bound)
            action = action.flatten()

            state, reward, done, _ = self.env.step(action)
            state = state.flatten()

            # create history and new training instance
            X.append(np.append(state_prev, action))

            noise = np.random.multivariate_normal(np.zeros(state.shape), 1e-6 * np.identity(state.shape[0]))
            y.append(state - state_prev + noise)

            rewards.append(reward)
            state_prev = state

        print("reward={}, episode_len={}".format(np.sum(rewards), t))
        return np.array(X), np.array(y), np.array(rewards)

    def get_joint_dist(self, state_mu, state_cov, action_mu, action_cov, input_output_cov) -> tuple:
        """
        returns the joint gaussian distributions of state and action distributions
        :param state_mu: mean of state distribution
        :param state_cov: covariance of state distribution
        :param action_mu: mean of action distribution
        :param action_cov: covariance of action distribution
        :param input_output_cov: input output covariance of state-action
        :return: joint_mu, joint_cov, joint_input_output_cov
        """

        # compute joint Gaussian
        joint_mu = np.concatenate([state_mu, action_mu])

        # covariance has shape
        # [[state mean, input_output_cov]
        # [input_output_cov.T, action_cov]]
        top = np.hstack((state_cov, input_output_cov))
        bottom = np.hstack((input_output_cov.T, action_cov))
        joint_cov = np.vstack((top, bottom))

        return joint_mu, joint_cov, top

    def get_init_hyperparams(self) -> tuple:
        """
        Compute initial hyperparameters for dynamics GP
        :return: [length scales, signal variance, noise variance]
        """
        l = np.log(np.std(self.state_action_pairs, axis=0))
        sigma_f = np.log(np.std(self.state_delta))
        sigma_eps = np.log(np.std(self.state_delta / 10))

        return l, sigma_f, sigma_eps

    def print_trajectory(self, mu_states, sigma_states, mu_actions, sigma_actions) -> None:

        """
        Create plot for a given trajectory
        :param mu_states: means of state trajectory
        :param sigma_states: covariance of state trajectory
        :param mu_actions: means of action trajectory
        :param sigma_actions: covariance of action trajectory
        :return: None
        """

        # plot state trajectory
        for i in range(self.state_dim):
            m = mu_states[:, i]
            s = sigma_states[:, i, i]

            x = np.arange(0, len(m))
            plt.errorbar(x, m, yerr=s, fmt='-o')
            # plt.fill_between(x, m - s, m + s)
            plt.title("Trajectory prediction for {}".format(self.state_names[i]))
            plt.show()

        # plot action trajectory
        x = np.arange(0, len(mu_actions))
        plt.errorbar(x, mu_actions, yerr=sigma_actions, fmt='-o')
        # plt.fill_between(x, mu_actions - sigma_actions, mu_actions + sigma_actions)
        plt.title("Trajectory prediction for actions")
        plt.show()
