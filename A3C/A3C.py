import logging
from multiprocessing import Value, Lock

import gym
import torch

from A3C.ActorCriticNetwork import ActorCriticNetwork
from A3C.Optimizers.SharedAdam import SharedAdam
from A3C.Optimizers.SharedRMSProp import SharedRMSProp
from A3C.Worker import Worker


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
        global_model = ActorCriticNetwork(env.observation_space.shape[0], env.action_space, self.is_discrete)
        global_model.share_memory()

        if self.optimizer_name == 'rmsprop':
            optimizer = SharedRMSProp(global_model.parameters(), lr=self.lr)
        elif self.optimizer_name == 'adam':
            optimizer = SharedAdam(global_model.parameters(), lr=self.lr)
        else:
            raise Exception('Unexpected optimizer_name: %s' % self.optimizer_name)

        optimizer.share_memory()

        # start the test worker which is visualized to see how the current progress is
        w = Worker(env_name=self.env_name, worker_id=self.n_worker, global_model=global_model, T=self.T,
                   seed=self.seed, lr=0, n_steps=0, t_max=100000, gamma=0, tau=0, beta=0,
                   value_loss_coef=0, optimizer=None, is_train=False, use_gae=False, is_discrete=self.is_discrete,
                   lock=self.lock)
        w.start()
        self.worker_pool.append(w)

        # start all training workers which update the model parameters
        for wid in range(0, self.n_worker):
            self.logger.info("Worker {} created".format(wid))
            w = Worker(env_name=self.env_name, worker_id=wid, global_model=global_model, T=self.T,
                       seed=self.seed, lr=self.lr, n_steps=5, t_max=100000, gamma=.5, tau=.75, beta=.01,
                       value_loss_coef=.5, optimizer=optimizer, is_train=True, use_gae=False,
                       is_discrete=self.is_discrete, lock=self.lock)
            w.start()
            self.worker_pool.append(w)

        # for w in self.worker_pool:
        #     w.join()

    def stop(self):
        self.worker_pool = []
        self.T = Value('i', 0)
