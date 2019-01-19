import logging

import quanser_robots

from A3C.A3C import A3C
from Experiments.util.ColorLogger import enable_color_logging
import argparse

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

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
#parser.add_argument('--n-hidden', type=int, default=200,
#                    help='amount of hidden nodes (default: 256')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--gae', type=bool, default=True,
                    help='use general advantage estimation (default: True)')
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--beta', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--max-grad-norm', type=float, default=40,
                    help='value loss coefficient (default: 40)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--worker', type=int, default=6,
                    help='how many training workers to use (default: 2)')
parser.add_argument('--t-max', type=int, default=1000,
                    help='number of forward steps in A3C (default: 10)')
parser.add_argument('--max-episode-length', type=int, default=5000,
                    help='maximum length of an episode (default: 100000)')
parser.add_argument('--env-name', default='CartpoleSwingShort-v0',
                    help='environment to train on (default: CartpoleStabShort-v0)')
parser.add_argument('--no-shared', default=False,
                    help='use an non shared optimizer.')
parser.add_argument('--optimizer', default="rmsprop",
                    help='used optimizer (default: rmsprop')
parser.add_argument('--lr_scheduler', default='None',
                    help='learning rate scheduler to use, by default no scheduler. Options [None, ExponentialLR]')
parser.add_argument('--train', default=True,
                    help='decides if to do training (default: True)')
parser.add_argument('--discrete', default=False,
                    help='continiouse action space (False), discrete action space (True) (default: False)')
parser.add_argument('--path_actor', default=None,
                    help='weight location for the actor (default: None)')
parser.add_argument('--path_critic', default=None,
                    help='weight location for the critic (default: False)')
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
    a3c = A3C(n_worker=args.worker, env_name=args.env_name, is_discrete=args.discrete, seed=args.seed, optimizer_name=args.optimizer)

    a3c.run_debug(path_actor=args.path_actor, path_critic=args.path_critic, max_episodes=args.max_episode_length,
                  t_max=args.t_max, gamma=args.gamma, tau=args.tau, beta=args.beta, use_gae=args.gae)

