import gym
import numpy as np
import quanser_robots

from PILCO.MGPR import MGPR
from PILCO.RBFController import RBFController


class PILCO(object):

    def __init__(self, env_name, seed, n_features):
        # general
        self.env_name = env_name
        self.seed = seed
        self.n_features = n_features
        self.noise_var = None

        # env
        self.env = quanser_robots.GentlyTerminating(gym.make(self.env_name))
        self.env = gym.make(self.env_name)
        self.env.seed(self.seed)

        # dynamics model
        # TODO learn length scale by evidence maximization
        self.mgp = MGPR(dim=self.env.observation_space.shape[0])

        self.states = []
        self.actions = []

    def run(self, n_features, n_init):
        # sample dataset with random actions
        X = []
        y = []
        rewards = []

        self.noise_var = np.random.normal(0, np.identity(
            self.env.observation_space[0]))  # TODO learn noise variance by evidence maximization
        # self.noise_var = np.diag(np.std(X, axis=1))  # TODO Figure this out

        i = 0
        while i < n_init:
            state_prev = self.env.reset()
            done = False

            while not done or i < n_init:
                action = self.env.sample()
                state, reward, _, _ = self.env.step(action)

                # state-action pair as input
                X.append(np.append(state_prev, action))

                # delta with following state as output plus some noise
                epsilon = np.random.normal(0, self.noise_var)
                y.append(state - state_prev + epsilon)

                rewards.append(reward)

                state_prev = state
                i += 1

        # convert to numpy
        X = np.array(X)
        y = np.array(y)
        rewards = np.array(rewards)

        # init model params

        # dimension of state vector
        D = self.env.observation_space[0]
        # dimension of action vector
        F = self.env.action_space

        W = np.random.normal(0, np.identity(n_features), size=(D, n_features))
        sigma = np.random.normal(0, np.identity(X.shape[1]))
        mu = np.random.normal(0, 1, n_features)

        # create controller/policy with those params
        policy = RBFController(W, sigma, mu)

        while True:
            convergence = False
            self.learn_dynamics_model(X, y)
            # TODO model based policy search

            while True:
                self.analytic_approximate_policy_evaluation()
                self.policy_improvement()
                W, sigma, mu = self.update_params(W, sigma, mu)

                if convergence:
                    break

            policy.update_params(W, sigma, mu)
            X_test, y_test, reward_test = self.execute_test_run(policy)
            np.append(X, X_test)
            np.append(y, y_test)
            np.append(rewards, reward_test)

    def learn_dynamics_model(self, X, y):
        self.mgp.fit(X, y)

    def analytic_approximate_policy_evaluation(self):
        # TODO
        # Compute mean and covar of policy/control dist
        # Compute cross covar[x-1, u-1]
        # approx state control dist p(\tildex-1) = p(x-1, u-1) = N(x\tilde-1|mu-1,sigma-1)
        #

        raise NotImplementedError

    def policy_improvement(self):
        raise NotImplementedError

    def update_params(self, W, sigma, mu):
        # use CG or L-BFGS for updates
        # TODO
        return W, sigma, mu

    def execute_test_run(self, policy):
        X = []
        y = []
        rewards = []

        state_prev = self.env.reset()
        done = False

        while not done:
            # TODO
            action = policy.predict(state_prev)
            state, reward, done, _ = self.env.step(action)

            # create history and create new training instance
            X.append(np.append(state_prev, action))
            epsilon = np.random.normal(0, self.noise_var)
            y.append(state - state_prev + epsilon)

            state_prev = state

        return X, y, reward

    # def compute_deltas(self, x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    #     """
    #     Returns the deltas between the states plus some noise
    #     :param x: sequence of recorded states
    #     :param sigma: diagonal matrix containing the std for each state index
    #     """
    #     deltas = []
    #
    #     for i, x1 in enumerate(x):
    #         epsilon = np.random.normal(0, sigma)
    #         if i == 0:
    #             deltas.append(x1 + epsilon)
    #         deltas.append(x1 - x[i - 1] + epsilon)
    #
    #     return deltas
