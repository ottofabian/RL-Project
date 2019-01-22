import logging

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
# env_name = "Pendulum-v0"
# env_name = "CartpoleStabShort-v0"
env_name = "CartpoleSwingShort-v0"
# env_name = "CartpoleSwingRR-v0"

# env_name = "CartpoleStabRR-v0"
# env_name = "Qube-v0"

# --env-name Pendulum-v0 --gamma 0.9 --t-max 10 --worker 6 --beta 0.0001 --max-action 2

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr-actor', type=float, default=0.0001,
                    help='learning rat, in paper sampled between 1e-4 to 1e-2 (default: 0.0001)')
parser.add_argument('--lr-critic', type=float, default=0.001,
                    help='learning rate, in paper sampled between 1e-4 to 1e-2 (default: 0.001)')
# TODO: Add this to combined network
parser.add_argument('--lr-actor-critic', type=float, default=0.0001,
                    help='learning rate for combined actor critic model,'
                         ' in paper sampled between 1e-4 to 1e-2 (default: 0.001)')
parser.add_argument('--value_loss_coef', type=float, default=0.5,
                    help='value loss coefficient ,'
                         'constants which defines the weighting between value and policy loss (default: 0.5)')
parser.add_argument('--n-hidden', type=int, default=200,
                    help='amount of hidden nodes (default: 256')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--gae', type=bool, default=True,
                    help='use general advantage estimation (default: True)')
parser.add_argument('--tau', type=float, default=1.0,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--beta', type=float, default=1e-4,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--max-grad-norm', type=float, default=40,
                    help='maximum gradient norm (default: 40)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--worker', type=int, default=max(multiprocessing.cpu_count() - 2, 1),
                    help='how many training workers/threads to use (default: %d)' %
                         max(multiprocessing.cpu_count() - 2, 1))
parser.add_argument('--t-max', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=5000,
                    help='maximum length of an episode (default: 5000)')
parser.add_argument('--env-name', default='CartpoleSwingShort-v0',
                    help='name of the gym environment to use.'
                         ' All available gym environments are supported as well as'
                         'additional gym environments: https://git.ias.informatik.tu-darmstadt.de/quanser/clients.'
                         '(default: CartpoleStabShort-v0)')
parser.add_argument('--no-shared', default=False,
                    help='use an non shared optimizer.')
parser.add_argument('--optimizer', default="rmsprop",
                    help='used optimizer (default: rmsprop')
parser.add_argument('--lr_scheduler', default='None',
                    help='learning rate scheduler to use, by default no scheduler. Options [None, ExponentialLR]')
parser.add_argument('--train', default=True,
                    help='decides if to do training (default: True)')
parser.add_argument('--path_actor', default=None,
                    help='weight location for the actor (default: None)')
parser.add_argument('--path_critic', default=None,
                    help='weight location for the critic (default: False)')
parser.add_argument('--max-action', type=float, default=5,
                    help='maximum allowed action to use, if None the full available action range is used (default: 5)')
parser.add_argument('--normalize', default=True,
                    help='if True the state feature space is linearly scaled to the range of [0,1],'
                         'Note: Normalizing is only supported for the environments '
                         '["CartpoleStabShort-v0", "CartpoleSwingRR-v0", "CartpoleStabShort-v0", "CartpoleStabRR-v0"]'
                         ' (default: True)')

# a3c.run_debug(
#     path_actor="./best_models/SwingUp/works but not stable/actor_T-135307781_global-3005.6435968448095.pth.tar",
#     path_critic="./best_models/SwingUp/works but not stable/critic_T-135307786_global-3005.6435968448095.pth.tar")
#
# a3c.run_debug(
#     path_actor="./best_models/Stabilization/Reduced action range/actor_T-6719059_global-9984.922698507235.pth.tar",
#     path_critic="./best_models/Stabilization/Reduced action range/critic_T-6719074_global-9984.922698507235.pth.tar")
#
# a3c.run_debug(
#     path_actor="./best_models/Stabilization/Full action range/actor_finetuned_T-7285824_global-1266.9597491827692.pth.tar",
#     path_critic="./best_models/Stabilization/Full action range/critic_finetuned_T-7285824_global-1266.9597491827692.pth.tar")

if __name__ == '__main__':

    enable_color_logging(debug_lvl=logging.DEBUG)
    logging.info('Start Experiment')

    args = parser.parse_args()
    a3c = A3C(args)

    a3c.run_debug()
