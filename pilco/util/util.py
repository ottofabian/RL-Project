import argparse
import logging
import time
from typing import Tuple

import autograd.numpy as np
import dill as pickle
import gym
import quanser_robots
from gym.wrappers import Monitor

from pilco.controller.controller import Controller


def load_model(path):
    return pickle.load(open(path, "rb"))


def get_env(env_name, monitor=False):
    if 'RR' in env_name:
        env = quanser_robots.GentlyTerminating(gym.make(env_name))
    else:
        if monitor:
            env = Monitor(gym.make(env_name), 'experiments/100_test_runs',
                          video_callable=lambda count: count % 100 == 0, force=True)
        else:
            # use the official gym env as default
            env = gym.make(env_name)
    return env


def evaluate_policy(policy: Controller, env: gym.Env, n_runs: int = 100, max_action: np.array = np.array([1]),
                    no_render: bool = False) -> None:
    """
    execute test run for given env and pilco policy

    :return: None
    """

    rewards = np.zeros(n_runs)
    lengths = np.zeros(n_runs)

    for i in range(n_runs):
        state_prev = env.reset().flatten()

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
    logging.info(f"best run: reward={rewards.max()}, length={lengths[rewards.argmax()]}")
    logging.info(f"worst run: reward={rewards.min()}, length={lengths[rewards.argmin()]}")


def squash_action_dist(mean: np.ndarray, cov: np.ndarray, input_output_cov: np.ndarray, bound: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rescales and squashes the distribution x with sin(x)
    See Deisenroth(2010) Appendix A.1 for mu of sin(x), where x~N(mu, sigma)
    :param bound: max action to take
    :param mean: mean of action distribution
    :param cov: covariance of actions distribution
    :param input_output_cov: state action input out covariance
    :return: mean_squashed, cov_squashed, input_output_cov_squashed
    """

    # p(u)' is squashed distribution over p(u) scaled by action space values,
    # see Deisenroth (2010), page 46, 2a)+b) and Section 2.3.2

    mean = np.atleast_2d(mean)
    cov_diag = np.atleast_2d(np.diag(cov))
    bound = np.atleast_2d(bound)

    # compute mean of squashed dist
    mean_squashed = bound * np.exp(-cov_diag / 2) * np.sin(mean)
    mean_squashed = mean_squashed.flatten()

    # covar: E[sin(x)^2] - E[sin(x)]^2
    sigma2 = -(cov_diag.T + cov_diag) / 2
    sigma2_exp = np.exp(sigma2)
    cov_squashed = ((np.exp(sigma2 + cov) - sigma2_exp) * np.cos(mean.T - mean) -
                    (np.exp(sigma2 - cov) - sigma2_exp) * np.cos(mean.T + mean))
    cov_squashed = np.dot(bound.T, bound) * cov_squashed / 2

    # compute input-output-covariance and squash through sin(x)
    input_output_cov_squashed = np.diag((bound * np.exp(-cov_diag / 2) * np.cos(mean)).flatten())
    input_output_cov_squashed = input_output_cov @ input_output_cov_squashed

    return mean_squashed, cov_squashed, input_output_cov_squashed


def get_joint_dist(state_mean, state_cov, action_mean, action_cov, input_output_cov) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    returns the joint gaussian distributions of state and action distributions
    :param state_mean: mean of state distribution
    :param state_cov: covariance of state distribution
    :param action_mean: mean of action distribution
    :param action_cov: covariance of action distribution
    :param input_output_cov: input output covariance of state-action
    :return: joint_mean, joint_cov, joint_input_output_cov
    """

    # compute joint Gaussian
    joint_mean = np.concatenate([state_mean, action_mean])

    # covariance has shape
    # [[state mean, input_output_cov]
    # [input_output_cov.T, action_cov]]
    top = np.hstack((state_cov, input_output_cov))
    bottom = np.hstack((input_output_cov.T, action_cov))
    joint_cov = np.vstack((top, bottom))

    return joint_mean, joint_cov, top


def parse_args(args: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='pilco')
    parser.add_argument('--env-name', default='CartpoleStabShort-v0',
                        help='Name of the gym environment to use. '
                             'Currently supported gym environments are: '
                             '"CartpoleStabShort-v0", "CartpoleStabRR-v0", "CartpoleSwingShort-v0", '
                             '"CartpoleSwingRR-v0", "Qube-v0", "QubeRR-v0". '
                             'If you want to use a different gym environment you need to specify '
                             '"--start-state" as well as "--target-state". (default: CartpoleStabShort-v0)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed. (default: 1)')
    parser.add_argument('--horizon', type=int, default=100,
                        help='Number of steps for trajectory rollout. (default: 100)')
    parser.add_argument('--discount', type=float, default=1.0,
                        help='Discount factor for rewards. (default: 1.0)')
    parser.add_argument('--policy', type=str, default="rbf",
                        help='Type of policy to use, supported: ["rbf", "linear"]. (default: rbf)')
    parser.add_argument('--features', type=int, default=50,
                        help='Number of features for RBF controller. (default: 50)')
    parser.add_argument('--inducing-points', type=int, default=300,
                        help='Number of inducing points to approximate GP, '
                             'setting this to 0 results in using the full GP. (default: 300)')
    parser.add_argument('--initial-samples', type=int, default=300,
                        help='Number of initial samples for learning the dynamics before first policy optimization. '
                             '(default: 300)')
    parser.add_argument('--max-samples-test-run', type=int, default=300,
                        help='Maximum samples taken from one test episode. This is required to avoid running out of '
                             'memory. (default: 300)')
    parser.add_argument('--weights', type=float, nargs="*", default=None,
                        help='Weighting for each state feature in the saturated loss. If None is given then an '
                             'identity matrix is used. If you want to disable certain features you can set them to 0, '
                             'e.g. pass "--weights 0 1 1 0 0" for "CartpoleStabShort-v0" to only specify the loss for '
                             'using the angle attributes. Make sure to use the same number of entries as there are '
                             'state dimensions. (default: None)')
    parser.add_argument('--max-action', type=float, default=None,
                        help='Maximum allowed action to use, if None then a predefined max action value for the '
                             'environment will be used. (default: None)')
    parser.add_argument('--steps', type=int, default=20,
                        help='Maximum number of learning steps until termination. (default: 20)')
    parser.add_argument('--cost-threshold', type=float, default=None,
                        help='Specifies a threshold for the rollout cost. If cost is smaller than this value, '
                             'the rollout horizon is increased by "horizon_increase". If "None" is used then '
                             '"cost-threshold" will be set to "-np.inf". (default: None)')
    parser.add_argument('--horizon-increase', type=float, default=0.25,
                        help='Specifies the rollout horizon\'s increase percentage after cost is smaller than'
                             'cost-threshold. (default: 0.25)')
    parser.add_argument('--start-state', type=float, nargs="*", default=None,
                        help='Starting state which is used at the beginning of a trajectory rollout. If None is given '
                             'then a predetermined starting state for the supported environment is used. '
                             'Arguments needs to be passed as a list e.g. pass "--start-state 0 0 1 0 0" for the '
                             '"CartpoleStabShort-v0" environment. (default: None)')
    parser.add_argument('--target-state', type=float, nargs="*", default=None,
                        help='Target state which should be reached. If None is given '
                             'then a predetermined target state for the supported environment is used. '
                             'Arguments needs to be passed as a list e.g. pass "--target-state 0 0 -1 0 0" for the '
                             '"CartpoleStabShort-v0" environment. (default: None)')
    parser.add_argument('--start-cov', type=float, default=1e-2,
                        help='Covariance value of starting state, which is multiplied with the Identity matrix '
                             'for trajectory rollout. (default: 1e-2)')
    parser.add_argument('--weight-dir', type=str, default=None,
                        help='Directory for the weights: '
                             '"policy.p", "dynamics.p", "state-action.npy", "state-delta.npy". '
                             'If only testing is enabled you only need to include "policy.p" in the directory. '
                             '(default: None)')
    parser.add_argument('--test', default=False, action='store_true',
                        help='Start run without training and evaluate for number of --test-runs (default: False)')
    parser.add_argument('--test-runs', type=int, default=10,
                        help='Number of test evaluation runs during training or in test mode (default: 10)')
    parser.add_argument('--no-log', default=False, action='store_true',
                        help='Disables exports to a log file into the log directory if set to True. (default: True)')
    parser.add_argument('--export-plots', default=False, action='store_true',
                        help='Exports the trajectory plots as latex TikZ figures into "./experiments/plots/". '
                             'You need to install "matplotlib2tikz" if set to True. (default: False)')
    parser.add_argument('--no-render', default=False, action='store_true',
                        help='Disables rendering. (default: False)')
    parser.add_argument('--monitor', default=False, action='store_true',
                        help='Enables monitoring with video capturing of the test worker. (default: False)')

    args = parser.parse_args(args)

    # create dummy_env for parameter check
    dummy_env = gym.make(args.env_name)

    # convert to numpy array if not "None" was given
    if args.max_action:
        args.max_action = np.array([args.max_action])
    else:
        # define default values for missing parameters
        args.max_action = dummy_env.action_space.high

    if args.start_state:
        if len(args.start_state) != len(dummy_env.observation_space.high):
            raise Exception(f"Your defined start_state vector of length {len(args.start_state)} is inconsistent "
                            f"with the environment state shape {len(dummy_env.observation_space.high)}")
        args.start_state = np.array(args.start_state)
    if args.target_state:
        if len(args.target_state) != len(dummy_env.observation_space.high):
            raise Exception(f"Your defined target_state vector of length {len(args.target_state)} is inconsistent "
                            f"with the environment state shape {len(dummy_env.observation_space.high)}")
        args.target_state = np.array(args.target_state)
    if args.weights:
        # check for env weight vector consistency
        if len(args.weights) != len(dummy_env.observation_space.high):
            raise Exception(f"Your defined weights vector of length {len(args.weights)} is inconsistent "
                            f"with the environment state shape {len(dummy_env.observation_space.high)}")
        args.weights = np.diag(args.weights)

    # set default value for cost threshold
    if not args.cost_threshold:
        args.cost_threshold = -np.inf

    if args.env_name == "CartpoleStabShort-v0" or args.env_name == "CartpoleStabRR-v0":
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
    elif args.env_name == "Qube-v0" or args.env_name == "QubeRR-v0" or args.env_name == "Qube-v1":
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

    if not args.start_state:
        raise Exception("No --start-state defined!")
    if not args.target_state:
        raise Exception("No --target-state state defined!")

    # parameter check
    if len(args.start_state) != state_dim:
        raise Exception("Your given --start-state {} is incompatible with the current state dimension of {}.".format(
            args.start_state, state_dim))
    if len(args.target_state) != state_dim:
        raise Exception("Your given --target-state {} is incompatible with the current state dimension of {}.".format(
            args.target_state, state_dim))

    if "RR" in args.env_name:
        args.no_render = True

    # always close gym environments if they aren't used anymore
    dummy_env.close()

    return args
