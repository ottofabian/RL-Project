import logging

import autograd.numpy as np
import gym
import quanser_robots

from PILCO.Controller.RBFController import RBFController
from PILCO.GaussianProcess.GaussianProcess import GaussianProcess
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

        self.env._max_episode_steps = 500
        self.env.seed(self.seed)

        # get the number of available action from the environment
        self.state_dim = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        # -----------------------------------------------------
        # training params
        self.gamma = .99  # discount factor

        # -----------------------------------------------------
        # models
        self.dynamics_model = None
        self.policy = None

        # -----------------------------------------------------
        # policy search Horizon
        # TODO: increase by 25% when successful
        self.T = Horizon

        # -----------------------------------------------------
        # Value calc
        # TODO: use GP as cost function for unknown cost
        # If the cose comes from the environment and is not known,
        #  the cost function has to be learn with a GP or the like.

        # known cost function
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
        self.rewards = None

        # -----------------------------------------------------
        # logging instance
        self.logger = logging.getLogger(__name__)

    def run(self, n_samples, n_steps=10):

        # TODO maybe change the structure to Deisenroth (2010), page 36
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
                action = self.env.action_space.sample()
                state, reward, done, _ = self.env.step(action)

                state = np.array(state)
                state_prev = np.array(state_prev)

                # state-action pair as input
                self.state_action_pairs[i] = np.concatenate([state_prev, action])

                # delta to following state as output plus some noise
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

    def rollout(self):

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
            # get mean and covar over next action, optionally with squashing
            # Deisenroth (2010), page 44, Nonlinear Model: RBF Network
            action_mu, action_cov, action_input_output_cov = self.policy.choose_action(state_mu, state_cov, squash=True,
                                                                                       bound=self.env.action_space.high)
            # self.logger.debug(str(t))
            # self.logger.debug(str("action"))
            # self.logger.debug(str(np.any(np.isnan(action_mu))))
            # self.logger.debug(str(np.any(np.isnan(action_cov))))

            # ------------------------------------------------
            # get joint dist over successor state p(x,u)
            state_action_mu, state_action_cov = self.get_joint_dist(state_mu, state_cov, action_mu,
                                                                    action_cov,
                                                                    action_input_output_cov)
            # self.logger.debug(str("joint state"))
            # self.logger.debug(str(np.any(np.isnan(state_action_mu))))
            # self.logger.debug(str(np.any(np.isnan(state_action_cov))))
            # ------------------------------------------------
            # compute delta and build next state dist
            delta_mu, delta_cov, delta_input_output_cov = self.dynamics_model.predict_from_dist(state_action_mu,
                                                                                                state_action_cov)
            # Cov(state, delta) is subset of Cov((state, action), delta)
            state_input_output_cov = delta_input_output_cov[:, :self.state_dim]

            state_next_mu = state_mu + delta_mu
            state_next_cov = state_cov + delta_cov + state_input_output_cov + state_input_output_cov.T
            # self.logger.debug(str("next state"))
            # self.logger.debug(str(np.any(np.isnan(state_next_mu))))
            # self.logger.debug(str(np.any(np.isnan(state_next_cov))))

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

    def learn_policy(self, mu=0, sigma=0.1 ** 2, target_noise=0.1):

        if self.policy is None:
            # init model params
            policy_X = np.random.multivariate_normal(np.full(self.state_dim, mu), sigma * np.identity(self.state_dim),
                                                     size=(self.n_features,))
            policy_y = target_noise * np.random.randn(self.n_features, self.n_actions)

            # augmented states would be initalized with .7, but we already have sin and cos given
            length_scales = np.ones(self.state_dim)

            self.policy = RBFController(n_actions=self.n_actions, rollout=self.rollout, length_scales=length_scales)
            self.policy.fit(policy_X, policy_y)

        self.policy.optimize()

        # # init model params
        # policy_X = np.random.multivariate_normal(np.full(self.state_dim, mu), sigma * np.identity(self.state_dim),
        #                                          size=(self.n_features,))
        # policy_y = target_noise * np.random.randn((self.n_features, self.n_actions))
        #
        # # augmented states would be initalized with .7
        # length_scales = np.ones((self.n_actions, self.state_dim))
        #
        # policy = RBFController(n_actions=self.n_actions)
        # policy.set_hyper_params(policy_X, policy_y, length_scales)
        # return policy

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
