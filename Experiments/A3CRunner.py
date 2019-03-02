import logging
import time
import quanser_robots

from A3C.A3C import A3C
from Experiments.util.ColorLogger import enable_color_logging
import argparse
import multiprocessing

from Experiments.util.logger_util import show_cmd_args

quanser_robots
#
# CartpoleStabShort-v0
# ---------------------
# * installable using https://git.ias.informatik.tu-darmstadt.de/quanser/clients

seed = 1
env_name = "Pendulum-v0"
# env_name = "CartpoleStabShort-v0"
# env_name = "CartpoleSwingShort-v0"
# env_name = "CartpoleSwingRR-v0"

# env_name = "CartpoleStabRR-v0"
# env_name = "Qube-v0"

# --env-name Pendulum-v0 --gamma 0.9 --t-max 10 --worker 6 --beta 0.0001 --max-action 2

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate, in paper sampled between 1e-4 to 1e-2 (default: 1e-4)')
parser.add_argument('--lr-critic', type=float, default=1e-3,
                    help='separate critic learning rate, if no shared network is used (default: 1e-3)')
parser.add_argument('--shared-model', default=False, action='store_true',
                    help='use shared network for actor and critic (default: False)')
parser.add_argument('--value-loss-weight', type=float, default=0.5,
                    help='value loss coefficient ,'
                         'constants which defines the weighting between value and policy loss (default: 0.5)')
parser.add_argument('--discount', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--gae', default=True, action='store_true',
                    help='use general advantage estimation (default: True)')
parser.add_argument('--tau', type=float, default=0.99,
                    help='parameter for GAE (default: 0.99)')
parser.add_argument('--entropy-loss-weight', type=float, default=1e-4,
                    help='entropy term coefficient (default: 1e-4)')
parser.add_argument('--max-grad-norm', type=float, default=1,
                    help='maximum gradient norm (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--worker', type=int, default=1,
                    help='how many training workers/threads to use. If set to 1, A2C is used and as many workers'
                         ' are launched as the number of environments (default: 1)')
parser.add_argument('--rollout-steps', type=int, default=50,
                    help='number of forward steps until gradient computation (default: 50)')
parser.add_argument('--max-episode-length', type=int, default=5000,
                    help='maximum length of an episode (default: 5000)')
parser.add_argument('--env-name', default='CartpoleStabShort-v0',
                    help='name of the gym environment to use.'
                         ' All available gym environments are supported as well as'
                         'additional gym environments: https://git.ias.informatik.tu-darmstadt.de/quanser/clients.'
                         '(default: CartpoleStabShort-v0)')
parser.add_argument('--shared-optimizer', default=True, action='store_true',
                    help='use an shared optimizer. (default: True)')
parser.add_argument('--optimizer', type=str, default="adam",
                    help='used optimizer (default: adam')
parser.add_argument('--lr-scheduler', type=str, default=None,
                    help='learning rate scheduler to use, by default no scheduler. '
                         'Options [None, ExponentialLR] (default: None)')
parser.add_argument('--test', default=False, action='store_true',
                    help='do testing only (default: False)')
parser.add_argument('--path', type=str, default=None,
                    help='weight location for the models to load (default: None)')
parser.add_argument('--log-dir', type=str, default=None,
                    help='default directory for logging the environment stats(default: None)')
parser.add_argument('--max-action', type=float, default=5,
                    help='maximum allowed action to use, if None the full available action range is used (default: 5)')
parser.add_argument('--normalizer', type=str, default=None,
                    help='which normalizer to use. Available [None, "MeanStd", "MinMax"] (default: None)')
parser.add_argument('--n-envs', type=int, default=5,
                    help='amount of envs for A2C (worker=1) in order to reduce correlation in batches (default: 5)')
parser.add_argument('--save-log', default=True, action='store_true',
                    help='exports a log file into the log directory if set to True (default: True)')
parser.add_argument('--log-frequency', default=100,
                    help='defines how often a sample is logged to tensorboard to avoid unnecessary bloating.'
                         'If set to X every X metric sample will be logged. (default: 100)')
parser.add_argument('--edge-fear', default=False, action='store_true',
                    help='gives negative rewards if the cart reaches goes near the border to avoid sucidal policies.'
                         'This is meant for evaluation and not part of the original environment. (default: False)')
parser.add_argument('--squared-reward', default=False, action='store_true',
                    help='manipulates the reward by squaring it. reward = reward * reward. (default: False)')
parser.add_argument('--no-render', default=False, action='store_true',
                    help='disables rendering. (default: False)')
parser.add_argument('--monitor', default=False, action='store_true',
                    help='enables monitoring of the environment. (default: False)')

if __name__ == '__main__':

    args = parser.parse_args()
    enable_color_logging(logging_lvl=logging.DEBUG, save_log=args.save_log,
                         logfile_prefix="A3C_" + args.env_name + "_")

    logging.info(
        f'Start experiment for {args.env_name} at {time.strftime("%m/%d/%Y, %Hh:%Mm:%Ss", time.gmtime(time.time()))}')

    show_cmd_args(args)

    a3c = A3C(args)
    a3c.run()
