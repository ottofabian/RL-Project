import time

import gym
import numpy as np
import quanser_robots
import torch
from torch.multiprocessing import Value, Process

from a3c.train_test import train, test
from a3c.util.util import get_model, get_shared_optimizer


class A3C(object):

    def __init__(self, args) -> None:
        """

        :param n_worker: Number of workers/threads to spawn which conduct the a3c algorithm.
        :param env_name: Name of the gym environment to use. All available gym environments are supported as well as
                         additional gym environments: https://git.ias.informatik.tu-darmstadt.de/quanser/clients.
        :param lr: Constant learning rate for all workers.
        :param is_discrete: Boolean, indicating if the target variable is discrete or continuous.
                            This setting has effect on the network architecture as well as the loss function used.
                            For more detail see: p.12 - Asynchronous Methods for Deep Reinforcement Learning.pdf
        :param optimizer_name: Optimizer used for shared weight updates. Possible arguments are 'rmsprop', 'adam'.
        :param is_train: If true enable training, use false if you only deploy the policy for testing
        """
        self.args = args

        # global counter
        self.T = Value('i', 0)
        self.global_reward = Value('d', -np.inf)

        # worker handling
        self.worker_pool = []

        # validity check for input parameter
        if args.optimizer not in ['rmsprop', 'adam']:
            raise Exception('Your given optimizer %s is currently not supported. Choose either "rmsprop" or "adam"',
                            args.optimizer)

    def run(self):

        torch.manual_seed(self.args.seed)

        if "RR" in self.args.env_name:
            env = quanser_robots.GentlyTerminating(gym.make(self.args.env_name))
        else:
            env = gym.make(self.args.env_name)

        optimizer = None
        critic_optimizer = None
        model_critic = None

        if self.args.shared_model:
            model = get_model(env=env, shared=self.args.shared_model, path=self.args.path, T=self.T,
                              global_reward=self.global_reward)
            if self.args.shared_optimizer:
                optimizer = get_shared_optimizer(model=model, optimizer_name=self.args.optimizer, lr=self.args.lr,
                                                 path=self.args.path)
        else:
            model, model_critic = get_model(env=env, shared=self.args.shared_model, path=self.args.path, T=self.T,
                                            global_reward=self.global_reward)
            if self.args.shared_optimizer:
                optimizer, critic_optimizer = get_shared_optimizer(model=model, optimizer_name=self.args.optimizer,
                                                                   lr=self.args.lr, path=self.args.path,
                                                                   model_critic=model_critic,
                                                                   optimizer_name_critic=self.args.optimizer,
                                                                   lr_critic=self.args.lr_critic)

        lr_scheduler = None
        lr_scheduler_critic = None

        if self.args.lr_scheduler == "exponential":
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            if critic_optimizer:
                lr_scheduler_critic = torch.optim.lr_scheduler.ExponentialLR(critic_optimizer, gamma=0.99)

            # raise NotImplementedError()

        p = Process(target=test, args=(
            self.args, self.args.worker, model, self.T, self.global_reward, optimizer, model_critic, critic_optimizer))
        p.start()
        self.worker_pool.append(p)

        if not self.args.test:
            for wid in range(0, self.args.worker):
                p = Process(target=train, args=(
                    self.args, wid, model, self.T, self.global_reward, optimizer, model_critic,
                    critic_optimizer, lr_scheduler, lr_scheduler_critic))
                p.start()
                self.worker_pool.append(p)
                time.sleep(1)

            for p in self.worker_pool:
                p.join()

    def stop(self):
        self.worker_pool = []
        self.T = Value('i', 0)
        self.global_reward = Value('d', 0)
