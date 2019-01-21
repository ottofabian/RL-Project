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

    def __init__(self, args) -> None:
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
        self.args = args

        # global counter
        self.T = Value('i', 0)
        self.global_reward = Value('d', 0)

        # worker handling
        self.worker_pool = []
        self.lock = Lock()

        self.logger = logging.getLogger(__name__)

        # validity check for input parameter
        if args.optimizer not in ['rmsprop', 'adam']:
            raise Exception('Your given optimizer %s is currently not supported. Choose either "rmsprop" or "adam"',
                            args.optimizer)

    def run(self):
        if "RR" in self.args.env_name:
            env = quanser_robots.GentlyTerminating(gym.make(self.args.env_name))
        else:
            env = gym.make(self.args.env_name)

        shared_model = ActorCriticNetwork(n_inputs=env.observation_space.shape[0],
                                          action_space=env.action_space,
                                          n_hidden=64,
                                          max_action=self.args.max_action)

        if self.args.optimizer == 'rmsprop':
            optimizer = SharedRMSProp(shared_model.parameters(), lr=self.args.lr_combined_actor_critic)
            optimizer.share_memory()
        elif self.args.optimizer == 'adam':
            optimizer = SharedAdam(shared_model.parameters(), lr=self.args.lr_actor_critic)
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
        w = Worker(args=self.args, worker_id=self.args.worker, shared_model=shared_model, T=self.T,
                   optimizer=optimizer, scheduler=scheduler, is_train=False,
                   global_reward=self.global_reward)
        w.start()
        self.worker_pool.append(w)

        # start all training workers which update the model parameters
        for wid in range(0, self.args.worker):
            self.logger.info("Worker {} created".format(wid))
            w = Worker(self.args, worker_id=wid, shared_model=shared_model, T=self.T,
                       optimizer=optimizer, scheduler=scheduler, is_train=True,
                       global_reward=self.global_reward)
            w.start()
            self.worker_pool.append(w)

        for w in self.worker_pool:
            w.join()

    def run_debug(self):

        torch.manual_seed(self.args.seed)

        if "RR" in self.args.env_name:
            env = quanser_robots.GentlyTerminating(gym.make(self.args.env_name))
        else:
            env = gym.make(self.args.env_name)

        shared_model_critic = CriticNetwork(env.observation_space.shape[0],
                                            env.action_space, self.args.n_hidden)
        shared_model_actor = ActorNetwork(env.observation_space.shape[0],
                                          env.action_space, self.args.n_hidden, self.args.max_action)

        shared_model_critic.share_memory()
        shared_model_actor.share_memory()

        # okish swing up:
        # optimizer_actor = SharedRMSProp(shared_model_actor.parameters(), lr=0.0001)
        # optimizer_critic = SharedRMSProp(shared_model_critic.parameters(), lr=0.0005)
        # p = Process(target=train, args=(
        #     self.env_name, rank, shared_model_actor, shared_model_critic, self.seed,
        #     self.T, 5000, 128, .9975, 1, .05, optimizer_actor, optimizer_critic, scheduler_actor,
        #     scheduler_critic, True, self.is_discrete,

        if self.args.optimizer == 'rmsprop':
            optimizer_actor = SharedRMSProp(shared_model_actor.parameters(), lr=self.args.lr_actor)
            optimizer_critic = SharedRMSProp(shared_model_critic.parameters(), lr=self.args.lr_critic)
            optimizer_actor.share_memory()
            optimizer_critic.share_memory()
        elif self.args.optimizer == 'adam':
            optimizer_actor = SharedAdam(shared_model_actor.parameters(), lr=self.args.lr_actor)
            optimizer_critic = SharedAdam(shared_model_critic.parameters(), lr=self.args.lr_critic)
            optimizer_actor.share_memory()
            optimizer_critic.share_memory()
        else:
            optimizer_actor = None
            optimizer_critic = None

        # scheduler_critic = torch.optim.lr_scheduler.ExponentialLR(optimizer_critic, gamma=0.95)
        # scheduler_actor = torch.optim.lr_scheduler.ExponentialLR(optimizer_actor, gamma=0.95)
        scheduler_actor = None
        scheduler_critic = None

        if self.args.path_actor is not None:
            if optimizer_actor is not None:

                load_saved_model(shared_model_actor, self.args.path_actor, self.T, self.global_reward, optimizer_actor)
            else:
                load_saved_model(shared_model_actor, self.args.path_actor, self.T, self.global_reward)

        if self.args.path_critic is not None:
            if optimizer_critic is not None:
                load_saved_model(shared_model_critic, self.args.path_critic, self.T, self.global_reward, optimizer_critic)
            else:
                load_saved_model(shared_model_critic, self.args.path_critic, self.T, self.global_reward)

        p = Process(target=test, args=(self.args,
            self.args.worker, shared_model_actor, shared_model_critic,
            self.T, optimizer_actor, optimizer_critic, self.global_reward))
        p.start()
        self.worker_pool.append(p)

        if self.args.train:
            if "RR" not in self.args.env_name:
                for wid in range(0, self.args.worker):
                    p = Process(target=train, args=(
                        self.args, wid, shared_model_actor, shared_model_critic,
                        self.T, optimizer_actor, optimizer_critic, scheduler_actor,
                        scheduler_critic, self.global_reward))
                    p.start()
                    self.worker_pool.append(p)

                for p in self.worker_pool:
                    p.join()

    def stop(self):
        self.worker_pool = []
        self.T = Value('i', 0)
        self.global_reward = Value('d', 0)
