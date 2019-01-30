import logging
import time
import quanser_robots

from A3C.A3C import A3C
from Experiments.util.ColorLogger import enable_color_logging
import argparse
import multiprocessing

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
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate, in paper sampled between 1e-4 to 1e-2 (default: 0.0001)')
parser.add_argument('--lr-critic', type=float, default=None,
                    help='separate critic learning rate, if no shared network is used (default: None)')
parser.add_argument('--shared-model', default=False, action='store_true',
                    help='use shared network for actor and critic (default: False)')
parser.add_argument('--value-loss-weight', type=float, default=0.5,
                    help='value loss coefficient ,'
                         'constants which defines the weighting between value and policy loss (default: 0.5)')
parser.add_argument('--discount', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--gae', default=True, action='store_true',
                    help='use general advantage estimation (default: True)')
parser.add_argument('--tau', type=float, default=1.0,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-loss-weight', type=float, default=1e-4,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--max-grad-norm', type=float, default=40,
                    help='maximum gradient norm (default: 40)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--worker', type=int, default=max(multiprocessing.cpu_count() - 2, 1),
                    help='how many training workers/threads to use (default: %d)' %
                         max(multiprocessing.cpu_count() - 2, 1))
parser.add_argument('--rollout-steps', type=int, default=20,
                    help='number of forward steps until gradient computation (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=10000,
                    help='maximum length of an episode (default: 10000)')
parser.add_argument('--env-name', default='CartpoleSwingShort-v0',
                    help='name of the gym environment to use.'
                         ' All available gym environments are supported as well as'
                         'additional gym environments: https://git.ias.informatik.tu-darmstadt.de/quanser/clients.'
                         '(default: CartpoleStabShort-v0)')
parser.add_argument('--shared-optimizer', default=True, action='store_true',
                    help='use an shared optimizer. (default: True)')
parser.add_argument('--optimizer', type=str, default="rmsprop",
                    help='used optimizer (default: rmsprop')
parser.add_argument('--lr-scheduler', type=str, default=None,
                    help='learning rate scheduler to use, by default no scheduler. '
                         'Options [None, ExponentialLR] (default: None)')
parser.add_argument('--train', default=True, action='store_true',
                    help='decides if to do training (default: True)')
parser.add_argument('--path', type=str, default=None,
                    help='weight location for the models to load (default: None)')
parser.add_argument('--log-dir', type=str, default=None,
                    help='default directory for logging the environment stats(default: None)')
parser.add_argument('--max-action', type=float, default=5,
                    help='maximum allowed action to use, if None the full available action range is used (default: 5)')
parser.add_argument('--normalizer', type=str, default=None,
                    help='which normalizer to use (default: None)')
parser.add_argument('--n-envs', type=int, default=1,
                    help='amount of envs for A2C (worker=1) in order to reduce correlation in batches (default: 1)')

if __name__ == '__main__':
    enable_color_logging(debug_lvl=logging.DEBUG)

    args = parser.parse_args()
    logging.info(
        f'Start Experiment for {args.env_name} at {time.strftime("%m/%d/%Y, %Hh:%Mm:%Ss", time.gmtime(time.time()))}')
    print(args)

    a3c = A3C(args)
    a3c.run()
