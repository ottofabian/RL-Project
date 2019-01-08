import logging
import math
import shutil
import time

import gym
import numpy as np
import quanser_robots
import quanser_robots.cartpole
import quanser_robots.cartpole.cartpole
import torch
from tensorboardX import SummaryWriter
from torch.autograd import Variable
# from multiprocessing import Value, Process
from torch.multiprocessing import Value, Process
from torch.optim import Optimizer

from A3C.Models.ActorCriticNetwork import ActorCriticNetwork
from Experiments.util.model_save import save_checkpoint

pi = Variable(torch.FloatTensor([math.pi]))


def sync_grads(model: ActorCriticNetwork, shared_model: ActorCriticNetwork) -> None:
    """
    This method synchronizes the grads of the local network with the global network.
    :return:
    :param model: local worker model
    :param shared_model: shared global model
    :return:
    """
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def normal(x, mu, sigma_sq):
    a = (-1 * (Variable(x) - mu).pow(2) / (2 * sigma_sq)).exp()
    b = 1 / (2 * sigma_sq * pi.expand_as(sigma_sq)).sqrt()
    return a * b


class Worker(Process):
    # DEBUG list to store the taken actions by the worker threads
    actions_taken_training = []
    values_training = []

    def __init__(self, env_name: str, worker_id: int, shared_model: ActorCriticNetwork, seed: int, T: Value,
                 lr: float = 1e-4, max_episodes: int = 10, t_max: int = 100000, gamma: float = .99,
                 tau: float = 1, beta: float = .01, value_loss_coef: float = .5,
                 optimizer: Optimizer = None, scheduler: torch.optim.lr_scheduler = None, is_train: bool = True,
                 use_gae: bool = True, is_discrete: bool = False, global_reward=None, max_action=None):
        """
        Initialize Worker thread for A3C algorithm
        :param max_episodes: maximum episodes for training
        :param global_reward: global shared running reward
        :param use_gae: use Generalize Advantage Estimate
        :param t_max: max steps before making an update
        :param env_name: gym environment name
        :param worker_id: number of workers
        :param T: global shared counter
        :param optimizer: torch optimizer instance, either shared Optimizer or None for individual
        :param beta: entropy weight factor
        :param tau: bias variance trade-off factor for GAE
        :param gamma: discount factor
        :param shared_model: shared global model to get the parameters from
        :param seed: seed to ensure reproducibility
        :param lr: learning rate for the workers NN
        :param value_loss_coef: factor for scaling the value loss
        """
        super(Worker, self).__init__()

        self.is_discrete = is_discrete

        # separate env for each worker
        self.env_name = env_name

        # check if the requested environment is a quanser robot env
        if 'RR' in self.env_name:
            self.env = quanser_robots.GentlyTerminating(gym.make(self.env_name))
        else:
            # use the official gym env as default
            self.env = gym.make(self.env_name)

        self.env.seed(seed + worker_id)

        # DEBUG information about the environment
        # logging.debug('Environment {}'.format(self.env_name))
        # logging.debug('Observation Space: {}'.format(self.env.observation_space))
        # logging.debug('Action Space: {}'.format(self.env.action_space))
        # logging.debug('Action Range: [{}, {}]'.format(self.env.action_space.low, self.env.action_space.high))

        # training params
        self.max_episodes = max_episodes
        self.tau = tau
        self.discount = gamma
        self.beta = beta
        self.value_loss_coef = value_loss_coef
        self.use_gae = use_gae
        self.scheduler = scheduler

        # training and testing params
        self.seed = seed
        self.lr = lr
        self.t_max = t_max
        self.is_train = is_train
        self.max_action = max_action if max_action is not None else np.asscalar(self.env.action_space.high)

        # shared params
        self.optimizer = optimizer
        self.global_model = shared_model
        self.worker_id = worker_id
        self.T = T
        self.global_reward = global_reward

        # logging instance
        self.logger = logging.getLogger(__name__)

    def run(self):
        if self.is_train:
            self._train()
            logging.debug('._train')
        else:
            self._test()

    def _train(self):
        """
        Start worker in training mode, i.e. training the shared model with backprop
        loosely based on https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py
        :return: self
        """

        torch.manual_seed(self.seed + self.worker_id)

        # init local NN instance for worker thread
        model = ActorCriticNetwork(n_inputs=self.env.observation_space.shape[0],
                                   action_space=self.env.action_space,
                                   n_hidden=self.global_model.n_hidden,
                                   max_action=self.max_action)
        # max_action=self.action_space.high)

        # if no shared optimizer is provided use individual one
        if self.optimizer is None:
            self.optimizer = torch.optim.RMSprop(self.global_model.parameters(), lr=0.0001)
            # optimizer = torch.optim.Adam(global_model.parameters(), lr=lr)

        model.train()

        writer = SummaryWriter()

        state = torch.from_numpy(np.array(self.env.reset()))

        t = 0
        iter_ = 0
        episode_reward = 0

        while True:
            # Get state of the global model
            model.load_state_dict(self.global_model.state_dict())

            # containers for computing loss
            values = []
            log_probs = []
            rewards = []
            entropies = []

            # reward_sum = 0
            for step in range(self.t_max):
                t += 1

                value, mu, sigma = model(Variable(state))

                dist = torch.distributions.Normal(mu, sigma)

                # ------------------------------------------
                # # select action
                # eps = Variable(torch.randn(mu.size()))
                # action = (mu + sigma.sqrt() * eps).data
                action = dist.rsample().detach()

                # ------------------------------------------
                # Compute statistics for loss
                # prob = normal(action, mu, sigma)

                # entropy = -0.5 * (sigma + 2 * pi.expand_as(sigma)).log() + .5
                entropy = dist.entropy()
                entropies.append(entropy)

                # log_prob = prob.log()
                log_prob = dist.log_prob(action)

                # make selected move
                state, reward, done, _ = self.env.step(action.numpy())
                # TODO: check if this is better
                # reward -= np.array(state)[0]
                episode_reward += reward

                # reward = min(max(-1, reward), 1)

                done = done or t >= self.max_episodes

                with self.T.get_lock():
                    self.T.value += 1

                if self.scheduler is not None and self.worker_id == 0 and self.T.value % 500000 and iter_ != 0:
                    # TODO improve the call frequency
                    self.scheduler.step(self.T.value / 500000)

                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)

                # reset env to beginning if done
                if done:
                    t = 0
                    state = self.env.reset()

                    # keep track of the avg overall global reward
                    with self.global_reward.get_lock():
                        if self.global_reward.value == 0:
                            self.global_reward.value = episode_reward
                        else:
                            self.global_reward.value = .99 * self.global_reward.value + 0.01 * episode_reward
                    if self.worker_id == 0:
                        writer.add_scalar("global_reward", self.global_reward.value, iter_)
                    episode_reward = 0

                state = torch.from_numpy(np.array(state))

                # end if terminal state or max episodes are reached
                if done:
                    break

            R = torch.zeros(1, 1)

            # if non terminal state is present set R to be value of current state
            if not done:
                v, _, _ = model(state)
                R = v.detach()

            # R = Variable(R)
            values.append(R)
            # compute loss and backprop
            actor_loss = 0
            critic_loss = 0
            gae = torch.zeros(1, 1)

            # iterate over rewards from most recent to the starting one
            for i in reversed(range(len(rewards))):
                R = rewards[i] + R * self.discount
                advantage = R - values[i]
                critic_loss = critic_loss + 0.5 * advantage.pow(2)
                if self.use_gae:
                    # Generalized Advantage Estimation
                    delta_t = rewards[i] + self.discount * values[i + 1] - values[i]
                    gae = gae * self.discount * self.tau + delta_t
                    actor_loss = actor_loss - log_probs[i] * gae.detach() - self.beta * entropies[i]
                else:
                    actor_loss = actor_loss - log_probs[i] * advantage.data - self.beta * entropies[i]

            # zero grads to avoid computation issues in the next step
            self.optimizer.zero_grad()

            # compute combined loss of actor_loss and critic_loss
            # avoid overfitting on value loss by scaling it down
            (actor_loss + self.value_loss_coef * critic_loss).mean().backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

            sync_grads(model, self.global_model)
            self.optimizer.step()

            iter_ += 1

            if self.worker_id == 0:
                grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                        for p in model.parameters()
                                        if p.grad is not None])

                valuelist = [v.data for v in values]
                writer.add_scalar("mean_values", np.mean(valuelist), iter_)
                writer.add_scalar("min_values", np.min(valuelist), iter_)
                writer.add_scalar("max_values", np.max(valuelist), iter_)
                writer.add_scalar("batch_rewards", np.mean(np.array(rewards)), iter_)
                writer.add_scalar("loss_policy", actor_loss, iter_)
                writer.add_scalar("loss_value", critic_loss, iter_)
                writer.add_scalar("grad_l2", np.sqrt(np.mean(np.square(grads))), iter_)
                writer.add_scalar("grad_max", np.max(np.abs(grads)), iter_)
                writer.add_scalar("grad_var", np.var(grads), iter_)
                for param_group in self.optimizer.param_groups:
                    writer.add_scalar("lr", param_group['lr'], iter_)

    def _test(self):
        """
        Start worker in _test mode, i.e. no training is done, only testing is used to validate current performance
        loosely based on https://github.com/ikostrikov/pytorch-a3c/blob/master/_test.py
        :return:
        """

        torch.manual_seed(self.seed + self.worker_id)
        self.env.seed(self.seed + self.worker_id)

        # get an instance of the current global model state
        model = ActorCriticNetwork(n_inputs=self.env.observation_space.shape[0],
                                   action_space=self.env.action_space,
                                   n_hidden=self.global_model.n_hidden,
                                   max_action=self.max_action)
        model.eval()

        state = torch.from_numpy(np.array(self.env.reset()))
        reward_sum = 0

        t = 0
        done = False

        while True:

            # Get params from shared global model
            model.load_state_dict(self.global_model.state_dict())

            rewards = np.zeros(10)
            eps_len = np.zeros(10)

            # make 10 runs to get current avg performance
            for i in range(10):
                while not done:
                    t += 1

                    if i == 0:
                        self.env.render()

                    with torch.no_grad():
                        # select mean of normal dist as action --> Expectation
                        _, mu, _ = model(Variable(state))
                        action = mu.detach()

                    # make selected move
                    state, reward, done, _ = self.env.step(action.numpy())
                    done = done or t >= self.max_episodes
                    reward_sum += reward

                    if done:
                        # reset current cumulated reward and episode counter as well as env
                        rewards[i] = reward_sum
                        reward_sum = 0

                        eps_len[i] = t
                        t = 0

                        state = self.env.reset()

                    state = torch.from_numpy(np.array(state))
                done = False

            print("T={} -- mean reward={} -- mean episode length={} -- global reward={}".format(self.T.value,
                                                                                                rewards.mean(),
                                                                                                eps_len.mean(),
                                                                                                self.global_reward.value))
            save_checkpoint({
                'epoch': self.T.value,
                'state_dict': model.state_dict(),
                'global_reward': self.global_reward.value,
                'optimizer': self.optimizer.state_dict() if self.optimizer is not None else None,
            }, filename="./checkpoints/critic_T-{}_global-{}.pth.tar".format(self.T.value, self.global_reward.value))

            # delay _test run for 10s to give the network some time to train
            time.sleep(10)
