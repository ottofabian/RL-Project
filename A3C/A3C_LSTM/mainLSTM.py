import argparse
import os

import gym
import quanser_robots
import torch
import torch.multiprocessing as mp

from A3C.A3C_LSTM.ActorCriticNetworkLSTM import ActorCriticNetworkLSTM
from A3C.A3C_LSTM.test import test
from A3C.A3C_LSTM.train import train
from A3C.Optimizers.SharedAdam import SharedAdam
from A3C.Optimizers.SharedRMSProp import SharedRMSProp

quanser_robots

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')

parser.add_argument('--n-hidden', type=int, default=128, #256,
                    help='amount of hidden nodes (default: 256')
parser.add_argument('--gamma', type=float, default=0.995,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--gae', type=bool, default=True,
                    help='use general advantage estimation (default: True)')
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--beta', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=40,
                    help='value loss coefficient (default: 40)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=1,
                    help='how many training processes to use (default: 2)')
parser.add_argument('--t-max', type=int, default=128,
                    help='number of forward steps in A3C (default: 10)')
<<<<<<< HEAD
parser.add_argument('--max-episode-length', type=int, default=5000,
                    help='maximum length of an episode (default: 5000)')
parser.add_argument('--env-name', default='CartpoleSwingShort-v0',
                    help='environment to train on (default: CartpoleStabShort-v0)')
parser.add_argument('--no-shared', default=False,
                    help='use an non shared optimizer.')
parser.add_argument('--optimizer', default="rmsprop",
                    help='used optimizer (default: rmsprop')

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    env = gym.make(args.env_name)
    shared_model = ActorCriticNetworkLSTM(n_inputs=env.observation_space.shape[0], action_space=env.action_space,
                                          n_hidden=args.n_hidden)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    elif args.optimizer == "rmsprop":
        optimizer = SharedRMSProp(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()
    elif args.optimizer == "adam":
        optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    processes = []

    T = mp.Value('i', 0)
    global_reward = mp.Value("d", 0)

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, T, global_reward))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, T, global_reward, optimizer))
        p.start()
        processes.append(p)

    print('before join')
    for p in processes:
        p.join()
