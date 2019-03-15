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
        Constructor
        :param args: Cmd-line arguments
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
        """
        Start A3C worker and test thread
        :return:
        """
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
            if not self.args.no_shared_optimizer:
                optimizer = get_shared_optimizer(model=model, optimizer_name=self.args.optimizer, lr=self.args.lr,
                                                 path=self.args.path)
        else:
            model, model_critic = get_model(env=env, shared=self.args.shared_model, path=self.args.path, T=self.T,
                                            global_reward=self.global_reward)
            if not self.args.no_shared_optimizer:
                optimizer, critic_optimizer = get_shared_optimizer(model=model, optimizer_name=self.args.optimizer,
                                                                   lr=self.args.lr, path=self.args.path,
                                                                   model_critic=model_critic,
                                                                   optimizer_name_critic=self.args.optimizer,
                                                                   lr_critic=self.args.lr_critic)

        lr_scheduler = None
        lr_scheduler_critic = None

        if not self.args.no_shared_optimizer and self.args.lr_scheduler == "exponential":
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            if critic_optimizer:
                lr_scheduler_critic = torch.optim.lr_scheduler.ExponentialLR(critic_optimizer, gamma=0.99)

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
