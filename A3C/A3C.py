import logging

import gym
import quanser_robots
from torch.multiprocessing import Value, Lock, Process

from A3C.Models.ActorCriticNetwork import ActorCriticNetwork
from A3C.Models.ActorNetwork import ActorNetwork
from A3C.Models.CriticNetwork import CriticNetwork
from A3C.Optimizers.SharedAdam import SharedAdam
from A3C.Optimizers.SharedRMSProp import SharedRMSProp
from A3C.Worker import Worker
from A3C.split_network_debug import test, train
from Experiments.util.model_save import load_saved_model


class A3C(object):

    def __init__(self, n_worker: int, env_name: str, is_discrete: bool = False,
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
        self.lr = 0.0001  # Paper sampled between 1e-4 to 1e-2
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
        # torch.manual_seed(self.seed)
        if "RR" in self.env_name:
            env = quanser_robots.GentlyTerminating(gym.make(self.env_name))
        else:
            env = gym.make(self.env_name)

        shared_model = ActorCriticNetwork(n_inputs=env.observation_space.shape[0],
                                          action_space=env.action_space,
                                          n_hidden=512)

        if self.optimizer_name == 'rmsprop':
            optimizer = SharedRMSProp(shared_model.parameters(), lr=self.lr)
            optimizer.share_memory()
        elif self.optimizer_name == 'adam':
            optimizer = SharedAdam(shared_model.parameters(), lr=self.lr)
            optimizer.share_memory()
        else:
            optimizer = None
            # raise Exception('Unexpected optimizer_name: %s' % self.optimizer_name)

        # start the test worker which is visualized to see how the current progress is
        w = Worker(env_name=self.env_name, worker_id=self.n_worker, shared_model=shared_model,
                   T=self.T, seed=self.seed, lr=0, n_steps=0, t_max=5000, gamma=0, tau=0,
                   beta=0, value_loss_coef=0, optimizer=optimizer, is_train=False, use_gae=True,
                   is_discrete=self.is_discrete, global_reward=self.global_reward)
        w.start()
        self.worker_pool.append(w)

        # start all training workers which update the model parameters
        for wid in range(0, self.n_worker):
            self.logger.info("Worker {} created".format(wid))
            w = Worker(env_name=self.env_name, worker_id=wid, shared_model=shared_model, T=self.T,
                       seed=self.seed, lr=0.0001, n_steps=10, t_max=5000, gamma=.99, tau=1,
                       beta=.01, value_loss_coef=.5, optimizer=optimizer, is_train=True,
                       use_gae=False, is_discrete=self.is_discrete,
                       global_reward=self.global_reward)
            w.start()
            self.worker_pool.append(w)

        for w in self.worker_pool:
            w.join()

    def run_debug(self, path_actor=None, path_critic=None):

        if "RR" in self.env_name:
            env = quanser_robots.GentlyTerminating(gym.make(self.env_name))
        else:
            env = gym.make(self.env_name)

        shared_model_critic = CriticNetwork(env.observation_space.shape[0],
                                            env.action_space, self.is_discrete)
        shared_model_actor = ActorNetwork(env.observation_space.shape[0],
                                          env.action_space, self.is_discrete)

        shared_model_critic.share_memory()
        shared_model_actor.share_memory()

        if self.optimizer_name == 'rmsprop':
            optimizer_actor = SharedRMSProp(shared_model_actor.parameters(), lr=0.0001)
            optimizer_critic = SharedRMSProp(shared_model_critic.parameters(), lr=0.001)
            optimizer_actor.share_memory()
            optimizer_critic.share_memory()
        elif self.optimizer_name == 'adam':
            optimizer_actor = SharedAdam(shared_model_actor.parameters(), lr=0.0001)
            optimizer_critic = SharedAdam(shared_model_critic.parameters(), lr=0.001)
            optimizer_actor.share_memory()
            optimizer_critic.share_memory()
        else:
            optimizer_actor = None
            optimizer_critic = None

        if path_actor is not None:
            if optimizer_actor is not None:

                load_saved_model(shared_model_actor, path_actor, optimizer_actor)
            else:
                load_saved_model(shared_model_actor, path_actor)

        if path_critic is not None:
            if optimizer_critic is not None:
                load_saved_model(shared_model_critic, path_critic, optimizer_critic)
            else:
                load_saved_model(shared_model_critic, path_critic)

        p = Process(target=test, args=(
            self.env_name, self.n_worker, shared_model_actor, shared_model_critic,
            self.seed, self.T, 5000, optimizer_actor, optimizer_critic, self.is_discrete, self.global_reward))
        p.start()
        self.worker_pool.append(p)

        for rank in range(0, self.n_worker):
            p = Process(target=train, args=(
                self.env_name, self.n_worker, shared_model_actor, shared_model_critic, self.seed,
                self.T, 10, 5000, .9, 1, .01, optimizer_actor, optimizer_critic, True, self.is_discrete,
                self.global_reward))
            p.start()
            self.worker_pool.append(p)

        for p in self.worker_pool:
            p.join()

    def stop(self):
        self.worker_pool = []
        self.T = Value('i', 0)
        self.global_reward = Value('d', 0)
