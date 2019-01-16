import logging
import torch

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
from Experiments.util.model_save import load_saved_model, save_checkpoint


class A3C(object):

    def __init__(self, n_worker: int, env_name: str, is_discrete: bool = False,
                 seed: int = 123, optimizer_name='rmsprop', is_train: bool = True) -> None:
        """

        :param n_worker: Number of workers/threads to spawn which conduct the A3C algorithm.
        :param env_name: Name of the gym environment to use. All available gym environments are supported as well as
                         additional gym environments: https://git.ias.informatik.tu-darmstadt.de/quanser/clients.
        :param lr: Constant learning rate for all workers.
        :param is_discrete: Boolean, indicating if the target variable is discrete or continuous.
                            This setting has effect on the network architecture as well as the loss function used.
                            For more detail see: p.12 - Asynchronous Methods for Deep Reinforcement Learning.pdf
        :param optimizer_name: Optimizer used for shared weight updates. Possible arguments are 'rmsprop', 'adam'.
        :param is_train: If true enable training, use false if you only deploy the policy for testing
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

        self.is_train = is_train

        self.logger = logging.getLogger(__name__)

        # validity check for input parameter
        if optimizer_name not in ['rmsprop', 'adam']:
            raise Exception('Your given optimizer %s is currently not supported. Choose either "rmsprop" or "adam"',
                            optimizer_name)

        self.optimizer_name = optimizer_name

    def run(self, path=None):
        if "RR" in self.env_name:
            env = quanser_robots.GentlyTerminating(gym.make(self.env_name))
        else:
            env = gym.make(self.env_name)

        max_action = 10

        shared_model = ActorCriticNetwork(n_inputs=env.observation_space.shape[0],
                                          action_space=env.action_space,
                                          n_hidden=64,
                                          max_action=max_action)

        if self.optimizer_name == 'rmsprop':
            optimizer = SharedRMSProp(shared_model.parameters(), lr=0.001)
            optimizer.share_memory()
        elif self.optimizer_name == 'adam':
            optimizer = SharedAdam(shared_model.parameters(), lr=0.00001)
            optimizer.share_memory()
        else:
            optimizer = None
            # raise Exception('Unexpected optimizer_name: %s' % self.optimizer_name)

        if path is not None:
            if optimizer is not None:
                load_saved_model(shared_model, path, self.T, self.global_reward, optimizer)
            else:
                load_saved_model(shared_model, path, self.T, self.global_reward)

        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        scheduler = None

        # start the test worker which is visualized to see how the current progress is
        w = Worker(env_name=self.env_name, worker_id=self.n_worker, shared_model=shared_model,
                   T=self.T, seed=self.seed, lr=0, max_episodes=5000, t_max=0, gamma=0, tau=0,
                   beta=0, value_loss_coef=0, optimizer=optimizer, scheduler=scheduler, is_train=False, use_gae=True,
                   is_discrete=self.is_discrete, global_reward=self.global_reward, max_action=max_action)
        w.start()
        self.worker_pool.append(w)

        # start all training workers which update the model parameters
        for wid in range(0, self.n_worker):
            self.logger.info("Worker {} created".format(wid))
            w = Worker(env_name=self.env_name, worker_id=wid, shared_model=shared_model, T=self.T,
                       seed=self.seed, lr=None, max_episodes=5000, t_max=32, gamma=.995, tau=1,
                       beta=.1, value_loss_coef=.5, optimizer=optimizer, scheduler=scheduler, is_train=True,
                       use_gae=False, is_discrete=self.is_discrete,
                       global_reward=self.global_reward, max_action=max_action)
            w.start()
            self.worker_pool.append(w)

        for w in self.worker_pool:
            w.join()

    def run_debug(self, path_actor=None, path_critic=None):

        torch.manual_seed(self.seed)

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

        # okish swing up:
        # optimizer_actor = SharedRMSProp(shared_model_actor.parameters(), lr=0.0001)
        # optimizer_critic = SharedRMSProp(shared_model_critic.parameters(), lr=0.0005)
        # p = Process(target=train, args=(
        #     self.env_name, rank, shared_model_actor, shared_model_critic, self.seed,
        #     self.T, 5000, 128, .9975, 1, .05, optimizer_actor, optimizer_critic, scheduler_actor,
        #     scheduler_critic, True, self.is_discrete,

        if self.optimizer_name == 'rmsprop':
            optimizer_actor = SharedRMSProp(shared_model_actor.parameters(), lr=0.0001)
            optimizer_critic = SharedRMSProp(shared_model_critic.parameters(), lr=0.0005)
            optimizer_actor.share_memory()
            optimizer_critic.share_memory()
        elif self.optimizer_name == 'adam':
            optimizer_actor = SharedAdam(shared_model_actor.parameters(), lr=0.00001)
            optimizer_critic = SharedAdam(shared_model_critic.parameters(), lr=0.00001)
            optimizer_actor.share_memory()
            optimizer_critic.share_memory()
        else:
            optimizer_actor = None
            optimizer_critic = None

        # scheduler_critic = torch.optim.lr_scheduler.ExponentialLR(optimizer_critic, gamma=0.95)
        # scheduler_actor = torch.optim.lr_scheduler.ExponentialLR(optimizer_actor, gamma=0.95)
        scheduler_actor = None
        scheduler_critic = None

        if path_actor is not None:
            if optimizer_actor is not None:

                load_saved_model(shared_model_actor, path_actor, self.T, self.global_reward, optimizer_actor)
            else:
                load_saved_model(shared_model_actor, path_actor, self.T, self.global_reward)

        if path_critic is not None:
            if optimizer_critic is not None:
                load_saved_model(shared_model_critic, path_critic, self.T, self.global_reward, optimizer_critic)
            else:
                load_saved_model(shared_model_critic, path_critic, self.T, self.global_reward)

        p = Process(target=test, args=(
            self.env_name, self.n_worker, shared_model_actor, shared_model_critic,
            self.seed, self.T, 5000, optimizer_actor, optimizer_critic, self.is_discrete, self.global_reward))
        p.start()
        self.worker_pool.append(p)

        if self.is_train:
            if "RR" not in self.env_name:
                for rank in range(0, self.n_worker):
                    p = Process(target=train, args=(
                        self.env_name, rank, shared_model_actor, shared_model_critic, self.seed,
                        self.T, 5000, 128, .995, 1, .1, optimizer_actor, optimizer_critic, scheduler_actor,
                        scheduler_critic, True, self.is_discrete, self.global_reward))
                    p.start()
                    self.worker_pool.append(p)

                for p in self.worker_pool:
                    p.join()

    def stop(self):
        self.worker_pool = []
        self.T = Value('i', 0)
        self.global_reward = Value('d', 0)
