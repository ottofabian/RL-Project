import logging
import time

import torch

import gym
import quanser_robots
from torch.multiprocessing import Value, Lock, Process

from A3C.Models.ActorCriticNetwork import ActorCriticNetwork
from A3C.Optimizers.SharedAdam import SharedAdam
from A3C.Optimizers.SharedRMSProp import SharedRMSProp
from A3C.Worker import Worker
from A3C.train_test import test, train
from A3C.util.save_and_load.model_save import load_saved_model, get_model, get_optimizer


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

    # def run(self):
    #
    #     shared_model = ActorCriticNetwork(n_inputs=env.observation_space.shape[0],
    #                                       action_space=env.action_space,
    #                                       max_action=self.args.max_action)
    #
    #     if "RR" in self.args.env_name:
    #         env = quanser_robots.GentlyTerminating(gym.make(self.args.env_name))
    #     else:
    #         env = gym.make(self.args.env_name)
    #
    #     if self.args.optimizer == 'rmsprop':
    #         optimizer = SharedRMSProp(shared_model.parameters(), lr=self.args.lr_combined_actor_critic)
    #         optimizer.share_memory()
    #     elif self.args.optimizer == 'adam':
    #         optimizer = SharedAdam(shared_model.parameters(), lr=self.args.lr_actor_critic)
    #         optimizer.share_memory()
    #     else:
    #         optimizer = None
    #         # raise Exception('Unexpected optimizer_name: %s' % self.optimizer_name)
    #
    #     if path is not None:
    #         if optimizer is not None:
    #             load_saved_model(shared_model, path, self.T, self.global_reward, optimizer)
    #         else:
    #             load_saved_model(shared_model, path, self.T, self.global_reward)
    #
    #     # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    #     scheduler = None
    #
    #     # start the test worker which is visualized to see how the current progress is
    #     w = Worker(args=self.args, worker_id=self.args.worker, shared_model=shared_model, T=self.T,
    #                optimizer=optimizer, scheduler=scheduler, is_train=False,
    #                global_reward=self.global_reward)
    #     w.start()
    #     self.worker_pool.append(w)
    #
    #     # start all training workers which update the model parameters
    #     for wid in range(0, self.args.worker):
    #         self.logger.info("Worker {} created".format(wid))
    #         w = Worker(self.args, worker_id=wid, shared_model=shared_model, T=self.T,
    #                    optimizer=optimizer, scheduler=scheduler, is_train=True,
    #                    global_reward=self.global_reward)
    #         w.start()
    #         self.worker_pool.append(w)
    #
    #     for w in self.worker_pool:
    #         w.join()

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
            model = get_model(env=env, shared=self.args.shared_model, path=self.args.path, T=self.T, global_reward=self.global_reward)
            if self.args.shared_optimizer:
                optimizer = get_optimizer(model=model, optimizer_name=self.args.optimizer, lr=self.args.lr,
                                          path=self.args.path)
        else:
            model, model_critic = get_model(env=env, shared=self.args.shared_model, path=self.args.path, T=self.T,
                                            global_reward=self.global_reward)
            optimizer, critic_optimizer = get_optimizer(model=model, optimizer_name=self.args.optimizer,
                                                        lr=self.args.lr, path=self.args.path, model_critic=model_critic,
                                                        optimizer_name_critic=self.args.optimizer,
                                                        lr_critic=self.args.lr_critic)

        # okish swing up:
        # optimizer_actor = SharedRMSProp(shared_model_actor.parameters(), lr=0.0001)
        # optimizer_critic = SharedRMSProp(shared_model_critic.parameters(), lr=0.0005)
        # p = Process(target=train, args=(
        #     self.env_name, rank, shared_model_actor, shared_model_critic, self.seed,
        #     self.T, 5000, 128, .9975, 1, .05, optimizer_actor, optimizer_critic, lr_scheduler,
        #     lr_scheduler_critic, True, self.is_discrete,

        lr_scheduler = None
        lr_scheduler_critic = None

        if self.args.lr_scheduler:
            # TODO
            # lr_scheduler_critic = torch.optim.lr_scheduler.ExponentialLR(optimizer_critic, gamma=0.95)
            # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_actor, gamma=0.95)
            raise NotImplementedError()

        p = Process(target=test, args=(
            self.args, self.args.worker, model, self.T, self.global_reward, optimizer, model_critic, critic_optimizer))
        p.start()
        self.worker_pool.append(p)

        if self.args.train:
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
