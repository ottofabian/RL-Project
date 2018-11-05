import logging
from multiprocessing import Value, Array, Lock, Pool

import gym
import torch

from A3C.ActorCritic import ActorCritic
from A3C.SharedRMSProp import SharedRMSProp
from A3C.Worker import Worker


class A3C(object):

    def __init__(self, n_worker, env_name):
        self.seed = 123
        self.env_name = env_name
        self.lr = 1e-4  # Paper sampled between 1e-4 to 1e-2

        # parameters
        self.theta = Array()  # TODO
        self.theta_v = Array()  # TODO
        self.T = Value('i', 0)  # global counter
        self.lock = Lock()

        self.n_worker = n_worker
        self.worker_pool = []

        self.logger = logging.getLogger(__name__)

    def create_worker(self, theta, theta_v, T, env_name):
        """
        Create new worker instance
        :return:
        """
        worker = Worker(theta=theta, theta_v=theta_v, T=T, env_name=env_name)
        self.worker_pool.append(worker)

    def run(self):
        torch.manual_seed(self.seed)
        env = gym.make(self.env_name)
        shared_model = ActorCritic(env.observation_space.shape[0], env.action_space)
        shared_model.share_memory()

        optimizer = SharedRMSProp(shared_model.parameters(), lr=self.lr)
        optimizer.share_memory()

        # TODO
        # worker
        Pool()