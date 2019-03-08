import datetime
import logging
import os

import autograd.numpy as np
import matplotlib.pyplot as plt
import quanser_robots
from autograd import value_and_grad
from scipy.optimize import minimize

from pilco.controller.controller import Controller
from pilco.controller.linear_controller import LinearController
from pilco.controller.rbf_controller import RBFController
from pilco.cost_function.loss import Loss
from pilco.gaussian_process.gaussian_process import GaussianProcess
from pilco.gaussian_process.multivariate_gp import MultivariateGP
from pilco.gaussian_process.sparse_multivariate_gp import SparseMultivariateGP
from pilco.util.util import load_model, get_env, get_joint_dist

# define the plotting style
plt.style.use('seaborn-whitegrid')
# avoid import optim issues
quanser_robots


class PILCO(object):

    def __init__(self, args, loss: Loss):
        """
        :param args: Cmd-line parameters, see pilco_runner.py for more details
        :param loss: loss object which defines the cost for the given environment.
                              This function is used for policy optimization.
        """

        # -----------------------------------------------------
        # cmd line parameters for later access
        self.args = args

        # -----------------------------------------------------
        # env setup
        # check if the requested environment is a real robot env
        self.env = get_env(args.env_name)

        self.max_samples_test_run = args.max_samples_test_run

        self.env.seed(args.seed)
        np.random.seed(args.seed)

        if args.env_name == "Pendulum-v0":
            self.state_names = ["cos($\\theta$)", "sin($\\theta$)", "$\\dot{\\theta}$"]
        elif "Cartpole" in args.env_name:
            self.state_names = ["x", "sin($\\theta$)", "cos($\\theta$)", "$\\dot{x}$", "$\\dot{\\theta}$"]
        elif args.env_name == "Qube-v0":
            self.state_names = ["sin($\\theta$)", "cos($\\theta$)", "sin($\\alpha$)", "cos($\\alpha$)",
                                "$\\dot{\\theta}$", "$\\dot{\\alpha}$"]

        # get the number of available action from the environment
        self.state_dim = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        # -----------------------------------------------------
        # dynamics and policy model
        self.dynamics_model = None
        self.policy = None

        # -----------------------------------------------------
        # rollout variables
        self.loss = loss
        self.start_mean = args.start_state
        self.start_cov = args.start_cov

        # -----------------------------------------------------
        # Container for collected experience
        self.state_action_pairs = None
        self.state_delta = None

        # -----------------------------------------------------
        # Run parameters
        # defines if the state state-action- and state-deltas-values have been loaded
        # training is consequently continued
        self.data_loaded = False

        # -----------------------------------------------------
        # Plotting options
        # is a counter variable which is increment for each plot
        self.plot_id = 0

        # -----------------------------------------------------
        # test rendering
        self.test = args.test

    def run(self) -> None:
        """
        start pilco training run
        :return: None
        """

        if not self.data_loaded:
            # if training is not continued get initial data set from random policy
            self.sample_inital_data_set(n_init=self.args.initial_samples)

        elif "RR" in self.args.env_name and self.policy:
            # continue training policy on the real system by creating some samples first
            x_test, y_test = self.execute_test_run()

            # add test history to training data set
            self.state_action_pairs = np.append(self.state_action_pairs, x_test, axis=0)
            self.state_delta = np.append(self.state_delta, y_test, axis=0)

        for _ in range(self.args.steps):
            self.learn_dynamics_model()
            self.learn_policy()

            x_test, y_test = self.execute_test_run()

            # add test history to training data set
            self.state_action_pairs = np.append(self.state_action_pairs, x_test, axis=0)
            self.state_delta = np.append(self.state_delta, y_test, axis=0)

    def sample_inital_data_set(self, n_init: int) -> None:
        """
        sample dataset with random actions
        :param n_init: amount of samples to be generated
        :return: None
        """

        state_action_pairs = []
        state_delta = []

        i = 0
        state_prev = self.env.reset()
        done = False

        if self.args.env_name == "Pendulum-v0":
            theta = (np.arctan2(self.start_mean[1], self.start_mean[0]) + np.random.normal(0, .1, 1))[0]
            self.env.env.state = [theta, 0]
            state_prev = np.array([np.cos(theta), np.sin(theta), 0.])

        # sample more than init until current episode is over, select randomly at the end.
        while not done or i < n_init:

            # take initial random action
            if self.args.max_action:
                action = np.random.uniform(-self.args.max_action, self.args.max_action, 1)
            else:
                action = self.env.action_space.sample()
            state, reward, done, _ = self.env.step(action)

            # safe state-action pair as input for dynamics GP
            state_action_pairs.append(np.concatenate([state_prev, action]))

            # include some noise to reduce data correlations and non semi definite matrices during optimization
            noise = np.random.multivariate_normal(np.zeros(state.shape), 1e-6 * np.identity(state.shape[0]))
            state_delta.append(state - state_prev + noise)

            # reset env if terminal state was reached before max samples were generated
            if done:
                state = self.env.reset()
                if self.args.env_name == "Pendulum-v0":
                    theta = (np.arctan2(self.start_mean[1], self.start_mean[0]) + np.random.normal(0, .1, 1))[0]
                    self.env.env.state = [theta, 0]
                    state = np.array([np.cos(theta), np.sin(theta), 0.])

            state_prev = state
            i += 1

        # sample some random training samples
        idx = np.random.choice(range(0, len(state_action_pairs)), n_init, replace=False)

        self.state_action_pairs = np.array(state_action_pairs)[idx]
        self.state_delta = np.array(state_delta)[idx]

    def learn_dynamics_model(self) -> None:
        """
        Learn the dynamics model for the given environment
        :return: None
        """

        if self.dynamics_model is None:
            length_scales, sigma_f, sigma_eps = self.get_init_hyperparams()
            if self.args.inducing_points:
                self.dynamics_model = SparseMultivariateGP(x=self.state_action_pairs, y=self.state_delta,
                                                           n_targets=self.state_dim, length_scales=length_scales,
                                                           sigma_f=sigma_f, sigma_eps=sigma_eps,
                                                           n_inducing_points=self.args.inducing_points)
            else:
                self.dynamics_model = MultivariateGP(x=self.state_action_pairs, y=self.state_delta,
                                                     n_targets=self.state_dim, container=GaussianProcess,
                                                     length_scales=length_scales, sigma_f=sigma_f, sigma_eps=sigma_eps)

        else:
            self.dynamics_model.fit(self.state_action_pairs, self.state_delta)

        self.dynamics_model.optimize()

    def learn_policy(self, target_noise: float = 0.1) -> None:
        """
        learn the policy based by trajectory rollouts
        :param target_noise: noise for sampling pseudo targets
        :return: None
        """

        # initialize policy if we do not already have one
        if self.policy is None:

            # rbf policy
            if self.args.policy == "rbf":
                # init model params
                x = np.random.multivariate_normal(self.start_mean, self.start_cov, size=(self.args.features,))
                y = target_noise * np.random.randn(self.args.features, self.n_actions)

                # augmented states would be initialized with .7, but we already have sin and cos given
                # and do not need to compute this with gaussian_trig
                length_scales = np.repeat(np.ones(self.state_dim).reshape(1, -1), self.n_actions, axis=0)

                self.policy = RBFController(x, y, n_actions=self.n_actions, length_scales=length_scales)

            # linear policy
            elif self.args.policy == "linear":
                self.policy = LinearController(self.state_dim, n_actions=self.n_actions)

            else:
                raise ValueError(f"Unsupported policy {self.args.policy} found.")

        self.optimize_policy()

    def compute_trajectory_cost(self, policy: Controller, print_trajectory: bool = False) -> float:
        """
        Compute predicted cost of on trajectory rollout using current policy and dynamics.
        This is used to optimize the policy
        :param policy: policy, which decides on actions
        :param print_trajectory: print the resulting trajectory
        :return: cost of trajectory
        """

        state_mean = self.start_mean
        state_cov = self.start_cov

        cost = 0

        if print_trajectory:
            # container required plotting later on
            state_means_container = []
            state_covs_container = []
            action_means_container = []
            action_covs_container = []

            state_means_container.append(state_mean)
            state_covs_container.append(state_cov)

        for t in range(self.args.horizon):
            state_next_mean, state_next_cov, action_mean, action_cov = self.rollout(policy, state_mean, state_cov)

            # compute value of current state prediction
            l = self.loss.compute_loss(state_next_mean, state_next_cov)
            cost = cost + self.args.discount ** t * l.flatten()

            if print_trajectory:
                state_means_container.append(state_next_mean)
                state_covs_container.append(state_next_cov)
                action_means_container.append(action_mean)
                action_covs_container.append(action_cov)

            state_mean = state_next_mean
            state_cov = state_next_cov

        if print_trajectory:
            self.print_trajectory(np.array(state_means_container), np.array(state_covs_container),
                                  np.array(action_means_container), np.array(action_covs_container))

        return cost

    def rollout(self, policy, state_mean, state_cov) -> tuple:
        """
        compute a single rollout given a state mean and covariance
        :param policy: policy object which decides on action
        :param state_mean: current mean to start rollout from
        :param state_cov: current covariance to start rollout from
        :return: state_next_mean, state_next_cov, action_mean, action_cov
        """

        # ------------------------------------------------
        # get mean and covar of next action, optionally with squashing and scaling towards an action bound
        action_mean, action_cov, action_input_output_cov = policy.choose_action(state_mean, state_cov,
                                                                                bound=self.args.max_action)

        # ------------------------------------------------
        # get joint dist p(x,u)
        state_action_mean, state_action_cov, state_action_input_output_cov = get_joint_dist(state_mean, state_cov,
                                                                                            action_mean,
                                                                                            action_cov,
                                                                                            action_input_output_cov)

        # ------------------------------------------------
        # compute delta and build next state dist
        delta_mean, delta_cov, delta_input_output_cov = self.dynamics_model.predict_from_dist(state_action_mean,
                                                                                              state_action_cov)

        # cross cov is times inv(s), see matlab code
        delta_input_output_cov = state_action_input_output_cov @ delta_input_output_cov

        # ------------------------------------------------
        # compute distribution over next state
        state_next_mean = delta_mean + state_mean
        state_next_cov = delta_cov + state_cov + delta_input_output_cov + delta_input_output_cov.T

        return state_next_mean, state_next_cov, action_mean, action_cov

    def optimize_policy(self) -> None:
        """
        optimize policy with respect to pseudo inputs and targets
        :return: None
        """
        params = self.policy.get_params()
        options = {'maxiter': 150, 'disp': True}

        try:
            logging.info("Starting to optimize policy with L-BFGS-B.")
            res = minimize(fun=value_and_grad(self._optimize_hyperparams), x0=params, method='L-BFGS-B', jac=True,
                           options=options)
        except Exception:
            logging.info("Starting to optimize policy with CG.")
            res = minimize(fun=value_and_grad(self._optimize_hyperparams), x0=params, method='CG', jac=True,
                           options=options)

        self.policy.set_params(res.x)

        # Make one more run for plots
        cost = self.compute_trajectory_cost(policy=self.policy, print_trajectory=True)

        # increase trajectory length if below threshold cost
        if cost < self.args.cost_threshold:
            self.args.horizon += int(self.args.horizon * self.args.horizon_increase)
            logging.info(f"Rollout horizon was increased to {self.args.horizon}.")

    def _optimize_hyperparams(self, params):
        """
        function handle to use for scipy optimizer
        :param params: flat array of all parameters [
        :return: cost of trajectory
        """

        self.policy.set_params(params)

        # cost of trajectory
        return self.compute_trajectory_cost(self.policy, print_trajectory=False)

    def execute_test_run(self) -> tuple:
        """
        execute test run for max episode steps and return new training samples
        :return: states, state_deltas, rewards
        """

        logging.info("Starting test run.")

        x = []
        y = []
        rewards = 0

        state_prev = self.env.reset()
        # [1,3] is returned and is reduced to 1D
        state_prev = state_prev

        if self.args.env_name == "Pendulum-v0":
            theta = (np.arctan2(self.start_mean[1], self.start_mean[0]) + np.random.normal(0, .1, 1))[0]
            state_prev = np.array([np.cos(theta), np.sin(theta), 0])
            self.env.env.state = [theta, 0]

        done = False
        t = 0
        while not done:
            if not self.args.no_render:
                self.env.render()
            t += 1

            # no uncertainty during testing required
            action, _, _ = self.policy.choose_action(state_prev, 0 * np.identity(len(state_prev)),
                                                     bound=self.args.max_action)
            action = action.flatten()

            state, reward, done, _ = self.env.step(action)
            state = state

            # create history and new training instance
            x.append(np.append(state_prev, action))

            noise = np.random.multivariate_normal(np.zeros(state.shape), 1e-6 * np.identity(state.shape[0]))
            y.append(state - state_prev + noise)

            rewards += reward
            state_prev = state

        logging.info(f"reward={rewards}, episode_len={t}")

        self.save(rewards)

        if len(x) < self.max_samples_test_run:
            x = np.array(x)
            y = np.array(y)
        else:
            idx = np.random.choice(range(0, len(x)), self.max_samples_test_run, replace=False)

            x = np.array(x)[idx]
            y = np.array(y)[idx]

        return x, y

    def get_init_hyperparams(self) -> tuple:
        """
        Compute initial hyperparameters for dynamics GP
        :return: [length scales, signal variance, noise variance]
        """
        length_scales = np.repeat(np.log(np.std(self.state_action_pairs, axis=0)).reshape(1, -1), self.state_dim,
                                  axis=0)
        sigma_f = np.log(np.std(self.state_delta, axis=0))
        sigma_eps = np.log(np.std(self.state_delta, axis=0) / 10)

        return length_scales, sigma_f, sigma_eps

    def print_trajectory(self, state_means, state_covs, action_means, action_covs) -> None:

        """
        Create plot for a given trajectory
        :param state_means: means of state trajectory
        :param state_covs: covariance of state trajectory
        :param action_means: means of action trajectory
        :param action_covs: covariance of action trajectory
        :return: None
        """

        # plot state trajectory
        for i in range(self.state_dim):
            m = state_means[:, i]
            s = state_covs[:, i, i]

            x = np.arange(0, len(m))
            plt.errorbar(x, m, yerr=s, fmt='-o')
            plt.xlabel("rollout steps")
            plt.title("Trajectory prediction for {}".format(self.state_names[i]))

            if self.args.export_plots:
                from matplotlib2tikz import save as tikz_save
                tikz_save("./experiments/plots/state_trajectory" + str(self.plot_id) + str(i) + ".tex")

            plt.show()

        # plot action trajectory
        x = np.arange(0, len(action_means))
        plt.errorbar(x, action_means, yerr=action_covs, fmt='-o')
        plt.xlabel("rollout steps")
        plt.title("Trajectory prediction for actions")
        if self.args.export_plots:
            from matplotlib2tikz import save as tikz_save
            tikz_save("./experiments/plots/action_trajectory" + str(self.plot_id) + ".tex")

        plt.show()

        self.plot_id += 1

    def _load_policy(self, path):
        """
        load existing policy
        :param path: path to file
        :return: None
        """
        self.policy = load_model(path)

    def _load_dynamics(self, path):
        """
        load existing dynamics model
        :param path: path to file
        :return: None
        """
        self.dynamics_model = load_model(path)

    def _save_data(self, directory):
        """
        Saves the the stat-action pairs and targets
        :param directory: Directory where the state-actions and targets will be saved
        :return:
        """
        np.save(open(f"{directory}state-delta.npy", "wb"), self.state_delta)
        np.save(open(f"{directory}state-action.npy", "wb"), self.state_action_pairs)

    def _load_data(self, path_state_action, path_delta):
        """
        Loads the state-action and delta values
        :param path_state_action: Path where the state-actions are stored
        :param path_delta: Path where the delta values are stored
        :return:
        """
        self.state_action_pairs = np.load(open(f"{path_state_action}", "rb"))
        self.state_delta = np.load(open(f"{path_delta}", "rb"))
        self.data_loaded = True

    def save(self, rewards):
        """
        save policy, dynamics and data points to ./experiments/checkpoints
        :param rewards: rewards for naming purpose
        :return: None
        """
        # don't save the models when only testing
        if not self.test:
            timestamp = datetime.datetime.now().strftime('%Y%m%d')
            save_dir = f"./experiments/checkpoints/{timestamp}-reward-{rewards:.5f}-{self.args.env_name}/"
            try:
                # create a directory where all models and data will be saved
                os.mkdir(save_dir)
            except OSError:
                print(f"Creation of the directory {save_dir} failed")

            self.policy.save(save_dir)
            self.dynamics_model.save(save_dir)
            self._save_data(save_dir)

    def load(self, directory):
        """
        load existing policy, dynamics and data points to continue training
        :param directory: directory containing the four files:
        "policy.p", "dynamics.p", "state-action.npy", "state-delta.npy"
        :return: None
        """
        self._load_policy(f"{directory}policy.p")
        self._load_dynamics(f"{directory}dynamics.p")
        self._load_data(f"{directory}state-action.npy", f"{directory}state-delta.npy")
