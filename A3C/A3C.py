import logging
from multiprocessing import Value

import gym
import quanser_robots
import quanser_robots.cartpole
import quanser_robots.cartpole.cartpole
import torch

from A3C.ActorCriticNetwork import ActorCriticNetwork
from A3C.SharedRMSProp import SharedRMSProp
from A3C.Worker import Worker


class A3C(object):

    def __init__(self, n_worker: int, env_name: str, lr: float = 1e-4, is_discrete: bool = False) -> None:
        self.seed = 123
        self.env_name = env_name
        self.lr = lr  # Paper sampled between 1e-4 to 1e-2
        self.is_discrete = is_discrete

        # global counter
        self.T = Value('i', 0)

        # worker handling
        self.n_worker = n_worker
        self.worker_pool = []

        self.logger = logging.getLogger(__name__)

    def run(self):
        torch.manual_seed(self.seed)
        env = quanser_robots.GentlyTerminating(gym.make(self.env_name))
        # env = gym.make(self.env_name)
        global_model = ActorCriticNetwork(env.observation_space.shape[0], env.action_space)
        global_model.share_memory()

        # TODO
        optimizer = SharedRMSProp(global_model.parameters(), lr=self.lr)
        optimizer.share_memory()

        w = Worker(env_name=self.env_name, worker_id=self.n_worker, global_model=global_model, T=self.T, seed=self.seed,
                   lr=self.lr, t_max=200, optimizer=None, is_train=False, is_discrete=self.is_discrete)
        w.start()
        self.worker_pool.append(w)

        for wid in range(0, self.n_worker):
            self.logger.info("Worker {} created".format(wid))
            w = Worker(env_name=self.env_name, worker_id=wid, global_model=global_model, T=self.T,
                       seed=self.seed, lr=self.lr, n_steps=20, t_max=1000, gamma=.99, tau=1, beta=.01,
                       value_loss_coef=.5, optimizer=None, is_train=True, is_discrete=self.is_discrete)
            w.start()
            self.worker_pool.append(w)

        for w in self.worker_pool:
            w.join()

    def stop(self):
        self.worker_pool = []
        self.T = Value('i', 0)
