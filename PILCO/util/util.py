import argparse
import logging
import time
from typing import Tuple

import autograd.numpy as np
import dill as pickle
import gym
import quanser_robots
from gym.wrappers import Monitor

from PILCO.Controller.Controller import Controller


def load_model(path):
    return pickle.load(open(path, "rb"))


def get_env(env_name, monitor=False):
    if 'RR' in env_name:
        env = quanser_robots.GentlyTerminating(gym.make(env_name))
    else:
        if monitor:
            env = Monitor(gym.make(env_name), '100_test_runs', video_callable=lambda count: count % 100 == 0,
                          force=True)
        else:
            # use the official gym env as default
            env = gym.make(env_name)
    return env


def evaluate_policy(policy: Controller, env: gym.Env, n_runs: int = 100, max_action: np.array = np.array([1]),
                    no_render: bool = False) -> None:
    """
    execute test run for given env and PILCO policy

    :return: None
    """

    rewards = np.zeros(n_runs)
    lengths = np.zeros(n_runs)

    for i in range(n_runs):
        state_prev = env.reset().flatten()

        if env.spec.id == "Pendulum-v0":
            # some noise to avoid starting upright
            theta = 0 + np.random.normal(0, .1, 1)[0]
            state_prev = np.array([np.cos(theta), np.sin(theta), 0])
            env.env.state = [theta, 0]

        done = False
        sleep = True

        while not done:
            if not no_render and "RR" not in env.spec.id and i == 0:
                env.render()
                if sleep:  # add a small delay to do a screen capture of the test run if needed
                    time.sleep(1)
                    sleep = False

            lengths[i] += 1

            # no uncertainty during testing required
            action, _, _ = policy.choose_action(state_prev, 0 * np.identity(len(state_prev)), bound=max_action)
            action = action.flatten()

            state, reward, done, _ = env.step(action)
            state = state.flatten()

            rewards[i] += reward
            state_prev = state

        logging.info(f"episode reward={rewards[i]}, episode length={lengths[i]}")
    logging.info(f"mean over {n_runs} runs: reward={rewards.mean()} +/- {rewards.std()}, length={lengths.mean()}"
                 f" +/- {lengths.std()}")


def squash_action_dist(mu: np.ndarray, sigma: np.ndarray, input_output_cov: np.ndarray, bound: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rescales and squashes the distribution x with sin(x)
    See Deisenroth(2010) Appendix A.1 for mu of sin(x), where x~N(mu, sigma)
    :param bound: max action to take
    :param mu: mean of action distribution
    :param sigma: covariance of actions distribution
    :param input_output_cov: state action input out covariance
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


def parse_args(args: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='PILCO')
    parser.add_argument('--env-name', default='CartpoleStabShort-v0',
                        help='Name of the gym environment to use.'
                             'Currently supported gym environments are:'
                             '"Pendulum-v0", "CartpoleStabShort-v0", "CartpoleStabRR-v0", "CartpoleSwingShort-v0", '
                             '"CartpoleSwingRR-v0", "Qube-v0", "QubeRR-v0"'
                             'details. If you want to use a different gym environment you need to specify '
                             '"--start-state" as well as "--target-state" (default: CartpoleStabShort-v0)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--horizon', type=int, default=100,
                        help='Number of steps for trajectory rollout (default: 100)')
    parser.add_argument('--discount', type=float, default=1.0,
                        help='Discount factor for rewards (default: 1.0)')
    parser.add_argument('--policy', type=str, default="rbf",
                        help='Type of policy to use, supported: "rbf", "linear" (default: rbf)')
    parser.add_argument('--features', type=int, default=50,
                        help='Number of features for RBF controller (default: 50)')
    parser.add_argument('--inducing-points', type=int, default=300,
                        help='Number of inducing points to approximate GP,'
                             'setting this to None results in using the full GP (default: 300)')
    parser.add_argument('--initial-samples', type=int, default=300,
                        help='Number of initial samples for learning the dynamics before first policy optimization'
                             ' (default: 300)')
    parser.add_argument('--max-samples-test-run', type=int, default=300,
                        help='Maximum samples taken from one test episode. This is required to avoid running out of'
                             'memory. (default: 300)')
    parser.add_argument('--weights', type=float, nargs="*", default=None,
                        help='Weighting for each state feature in the saturated loss. If None is given then an identity'
                             'matrix is used. If you want to disable certain features you can set them to 0, e.g. pass'
                             '"--weights 0 1 1 0 0" for "CartpoleStabShort-v0" to only specify the loss for using'
                             'the angle attributes. Make sure to use the same number of entries as there are state'
                             'dimensions (default: None)')
    parser.add_argument('--max-action', type=float, default=None,
                        help='Maximum allowed action to use, if None then a predefined max action value for the'
                             'environment will be used (default: None)')
    parser.add_argument('--steps', type=int, default=20,
                        help='Maximum number of learning steps until termination (default: 20)')
    parser.add_argument('--cost-threshold', type=float, default=None,
                        help='Specifies a threshold for the rollout cost. If cost is smaller than this value,'
                             ' the rollout horizon is increased by "horizon_increase". If "None" is used then'
                             '"cost-threshold" will be set to "-np.inf" (default: None)')
    parser.add_argument('--horizon-increase', type=float, default=0.25,
                        help='Specifies the rollout horizon\'s increase percentage after cost is smaller than'
                             'cost-threshold (default: 0.25)')
    parser.add_argument('--start-state', type=float, nargs="*", default=None,
                        help='Starting state which is used at the beginning of a trajectory rollout. If None is given '
                             'then a predetermined starting state for the supported environment is used.'
                             'Arguments needs to be passed as a list e.g. pass "--start-state 0 0 1 0 0" for the '
                             '"CartpoleStabShort-v0" environment. (default: None)')
    parser.add_argument('--target-state', type=float, nargs="*", default=None,
                        help='Target state which should be reached. If None is given '
                             'then a predetermined target state for the supported environment is used.'
                             'Arguments needs to be passed as a list e.g. pass "--target-state 0 0 -1 0 0" for the '
                             '"CartpoleStabShort-v0" environment. (default: None)')
    parser.add_argument('--start-cov', type=float, default=1e-2,
                        help='Covariance value of starting state, which is multiplied with the Identity matrix'
                             'for trajectory rollout (default: 1e-2)')
    parser.add_argument('--weight-dir', type=str, default=None,
                        help='Directory for the weights:'
                             ' "policy.p", "dynamics.p", "state-action.npy", "state-delta.npy"'
                             ' If only testing is enabled you only need to include "policy.p" in the directory'
                             ' (default: None)')
    parser.add_argument('--test', default=False, action='store_true',
                        help='Start run without training (default: False)')
    parser.add_argument('--no-log', default=False, action='store_true',
                        help='Disables exports to a log file into the log directory if set to True (default: True)')
    parser.add_argument('--export-plots', default=False, action='store_true',
                        help='Exports the trajectory plots as latex TikZ figures into "./Experiments/plots/".'
                             ' You need to install "matplotlib2tikz" if set to True. (default: False)')
    parser.add_argument('--no-render', default=False, action='store_true',
                        help='Disables rendering. (default: False)')
    parser.add_argument('--monitor', default=False, action='store_true',
                        help='Enables monitoring with video capturing of the test worker. (default: False)')

    args = parser.parse_args(args)

    # convert to numpy array if not "None" was given
    if args.max_action:
        args.max_action = np.array([args.max_action])
    if args.start_state:
        args.start_state = np.array(args.start_state)
    if args.target_state:
        args.target_state = np.array(args.target_state)
    if args.weights:
        args.weights = np.diag(args.weights)

    # set default value for cost threshold
    if not args.cost_threshold:
        args.cost_threshold = -np.inf

    # get target state as well as initial mu and cov for trajectory rollouts
    if args.env_name == "Pendulum-v0":
        # this acts like pendulum stabilization or swing up to work with easier 3D obs space
        theta = 0
        if not args.start_state:
            args.start_state = np.array([np.cos(theta), np.sin(theta), 0])
        if not args.max_action:
            args.max_action = np.array([2])

        if not args.target_state:
            args.target_state = np.array([1, 0, 0])

    elif args.env_name == "CartpoleStabShort-v0" or args.env_name == "CartpoleStabRR-v0":
        theta = np.pi
        if not args.start_state:
            args.start_state = np.array([0., np.sin(theta), np.cos(theta), 0., 0.])
        if not args.max_action:
            args.max_action = np.array([5])

        if not args.target_state:
            args.target_state = np.array([0, 0, -1, 0, 0])

    elif args.env_name == "CartpoleSwingShort-v0" or args.env_name == "CartpoleSwingRR-v0":
        theta = 0
        if not args.start_state:
            args.start_state = np.array([0., np.sin(theta), np.cos(theta), 0., 0.])
        if not args.max_action:
            args.max_action = np.array([10])

        if not args.target_state:
            args.target_state = np.array([0, 0, -1, 0, 0])

    elif args.env_name == "Qube-v0" or args.env_name == "QubeRR-v0":
        theta = 0
        alpha = 0
        if not args.start_state:
            args.start_state = np.array([np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha), 0., 0.])
        if not args.max_action:
            args.max_action = np.array([5])

        if not args.target_state:
            args.target_state = np.array([1., 0., -1., 0., 0., 0.])
    else:
        # a different environment was selected
        if not args.start_state or args.target_state:
            raise Exception("You need to specify a start and target state for your given unsupported environment.")

    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]

    args.start_cov = args.start_cov * np.identity(env.observation_space.shape[0])

    # parameter check
    if len(args.start_state) != state_dim:
        raise Exception("Your given --start-state {} is incompatible with the current state dimension of {}.".format(
            args.start_state, state_dim))
    if len(args.target_state) != state_dim:
        raise Exception("Your given --target-state {} is incompatible with the current state dimension of {}.".format(
            args.target_state, state_dim))

    if "RR" in args.env_name:
        args.no_render = True

    return args
