import logging

import gym
import torch
from torch.multiprocessing import Value, Lock, Process

from A3C.ActorCriticNetwork import ActorCriticNetwork
from A3C.Optimizers.SharedAdam import SharedAdam
from A3C.Optimizers.SharedRMSProp import SharedRMSProp
from A3C.Test import test, train


class A3C(object):

    def __init__(self, n_worker: int, env_name: str, lr: float = 1e-4, is_discrete: bool = False,
                 seed: int = 123, optimizer_name='rmsprop') -> None:
        """

        :param n_worker: Number of workers/threads to spawn which conduct the A3C algorithm.
        :param env_name: Name of the gym environment to use. All available gym environments are supported as well as
                         additional gym environments: https://git.ias.informatik.tu-darmstadt.de/quanser/clients.
        :param lr: Constant learning rate for all workers.
        :param is_discrete: Boolean, indicating if the target variable is discrete or continuous.
                            This setting has effect on the network architecture as well as the loss function used.
                            For more detail see: p.12 - Asynchronous Methods for Deep Reinforcement Learning.pdf
        :param optimizer_name: Optimizer used for shared weight updates. Possible arguments are 'rmsprop', 'adam'.
        """
        self.seed = seed
        self.env_name = env_name
        self.lr = lr  # Paper sampled between 1e-4 to 1e-2
        self.is_discrete = is_discrete

        # global counter
        self.T = Value('i', 0)
        self.global_reward = Value('d', 0)

        # worker handling
        self.n_worker = n_worker
        self.worker_pool = []
        self.lock = Lock()

        self.logger = logging.getLogger(__name__)

        # validity check for input parameter
        if optimizer_name not in ['rmsprop', 'adam']:
            raise Exception('Your given optimizer %s is currently not supported. Choose either "rmsprop" or "adam"',
                            optimizer_name)
        self.optimizer_name = optimizer_name

    def run(self):
        torch.manual_seed(self.seed)
        # env = quanser_robots.GentlyTerminating(gym.make(self.env_name))
        env = gym.make(self.env_name)
        shared_model = ActorCriticNetwork(env.observation_space.shape[0], env.action_space, self.is_discrete)
        shared_model.share_memory()

        if self.optimizer_name == 'rmsprop':
            optimizer = SharedRMSProp(shared_model.parameters(), lr=self.lr)
        elif self.optimizer_name == 'adam':
            optimizer = SharedAdam(shared_model.parameters(), lr=self.lr)
        else:
            raise Exception('Unexpected optimizer_name: %s' % self.optimizer_name)

        optimizer.share_memory()

        # start the test worker which is visualized to see how the current progress is
        # w = Worker(env_name=self.env_name, worker_id=self.n_worker, shared_model=shared_model, T=self.T,
        #            seed=self.seed, lr=0, n_steps=0, t_max=100000, gamma=0, tau=0, beta=0,
        #            value_loss_coef=0, optimizer=None, is_train=False, use_gae=True, is_discrete=self.is_discrete,
        #            lock=self.lock, global_reward=self.global_reward)
        # w.start()
        # self.worker_pool.append(w)
        #
        # # start all training workers which update the model parameters
        # for wid in range(0, self.n_worker):
        #     self.logger.info("Worker {} created".format(wid))
        #     w = Worker(env_name=self.env_name, worker_id=wid, shared_model=shared_model, T=self.T,
        #                seed=self.seed, lr=self.lr, n_steps=5, t_max=100000, gamma=.9, tau=1, beta=.005,
        #                value_loss_coef=1, optimizer=None, is_train=True, use_gae=False,
        #                is_discrete=self.is_discrete, lock=self.lock, global_reward=self.global_reward)
        #     w.start()
        #     self.worker_pool.append(w)
        #
        # for w in self.worker_pool:
        #     w.join()

        p = Process(target=test, args=(self.env_name, self.n_worker, shared_model, self.seed, self.T, 10000,
                                       self.is_discrete, self.global_reward))
        p.start()
        self.worker_pool.append(p)

        for rank in range(0, self.n_worker):
            p = Process(target=train, args=(self.env_name, self.n_worker, shared_model, self.seed, self.T, self.lr,
                                            10, 10000, .99, 1, .01, .5, optimizer, True, self.is_discrete,
                                            self.global_reward))
            p.start()
            self.worker_pool.append(p)

        for p in self.worker_pool:
            p.join()

    def stop(self):
        self.worker_pool = []
        self.T = Value('i', 0)
        self.global_reward = Value('d', 0)
