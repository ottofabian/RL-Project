import logging
import time
from multiprocessing import Value
from threading import Thread

import gym
import numpy as np
import quanser_robots
import quanser_robots.cartpole
import quanser_robots.cartpole.cartpole
import torch
import torch.nn.functional as F
from gym.spaces import Discrete, Box
from torch.optim import Optimizer

from A3C.ActorCriticNetwork import ActorCriticNetwork


def sync_grads(model: ActorCriticNetwork, shared_model: ActorCriticNetwork) -> None:
    """
    This method synchronizes the grads of the local network with the global network.
    :return:
    :param model: local worker model
    :param shared_model: shared global model
    :return:
    """
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


class Worker(Thread):

    def __init__(self, env_name: str, worker_id: int, global_model: ActorCriticNetwork, seed: int, T: Value,
                 lr: float = 1e-4, n_steps: int = 0, t_max: int = 100000, gamma: float = .99,
                 tau: float = 1, beta: float = .01, value_loss_coef: float = .5,
                 optimizer: Optimizer = None, is_train: bool = True, use_gae: bool = True,
                 is_discrete: bool = False, lock=None) -> None:
        """
        Initialize Worker thread for A3C algorithm
        :param use_gae: use Generalize Advantage Estimate
        :param t_max: maximum episodes for training
        :param env_name: gym environment name
        :param worker_id: number of workers
        :param T: global shared counter
        :param optimizer: torch optimizer instance, either shared Optimizer or None for individual
        :param beta: entropy weight factor
        :param tau: TODO hyperparam for GAE
        :param gamma: discount factor
        :param global_model: shared global model to get the parameters from
        :param seed: seed to ensure reproducibility
        :param lr: learning rate for the workers NN
        :param n_steps: amount of steps for training
        :param value_loss_coef: factor for scaling the value loss
        """
        super(Worker, self).__init__()

        self.is_discrete = is_discrete

        # separate env for each worker
        self.env_name = env_name

        # check if the requested environment is a quanser robot env
        if self.env_name in ['CartpoleStabShort-v0']:
            self.env = quanser_robots.GentlyTerminating(gym.make(self.env_name))
        else:
            # use the official gym env as default
            self.env = gym.make(self.env_name)

        # training params
        self.n_steps = n_steps
        self.tau = tau
        self.gamma = gamma
        self.beta = beta
        self.value_loss_coef = value_loss_coef
        self.use_gae = use_gae

        # training and testing params
        self.seed = seed
        self.lr = lr
        self.t_max = t_max
        self.is_train = is_train

        # shared params
        self.optimizer = optimizer
        self.global_model = global_model
        self.worker_id = worker_id
        self.T = T
        self.lock = lock

        # logging instance
        self.logger = logging.getLogger(__name__)

    def run(self):
        if self.is_train:
            self._train()
        else:
            self._test()

    def _train(self):
        """
        Start worker in training mode, i.e.training the shared model with backprop
        loosely based on https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py
        :return: None
        """
        # ensure different seed for each worker thread
        torch.manual_seed(self.seed + self.worker_id)
        self.env.seed(self.seed + self.worker_id)

        # init local NN instance for worker thread
        model = ActorCriticNetwork(self.env.observation_space.shape[0], self.env.action_space, self.is_discrete)

        # if no shared optimizer is provided use individual one
        if self.optimizer is None:
            self.optimizer = torch.optim.RMSprop(self.global_model.parameters(), lr=self.lr)

        model.train()

        state = self.env.reset()
        state = torch.Tensor(state)
        done = True

        t = 0
        while True:
            # Get state of the global model
            model.load_state_dict(self.global_model.state_dict())

            # containers for computing loss
            values = []
            log_probs = []
            rewards = []
            entropies = []

            for step in range(self.n_steps):
                t += 1

                if self.is_discrete:
                    # forward pass
                    value, logit = model(state.unsqueeze(0))

                    # prop dist over actions
                    prob = F.softmax(logit, dim=-1)
                    log_prob = F.log_softmax(logit, dim=-1)

                    # compute entropy for loss regularization
                    entropy = -(log_prob * prob).sum(1, keepdim=True)

                    # choose action based on prob dist
                    action = prob.multinomial(num_samples=1).detach()
                    log_prob = log_prob.gather(1, action)

                else:
                    # forward pass
                    value, mu, sigma = model(state.unsqueeze(0))
                    # print("Training worker {} -- mu: {} -- sigma: {}".format(self.worker_id, mu, sigma))

                    # assuming action space is in -high/high
                    high = np.asscalar(self.env.action_space.high)
                    low = np.asscalar(self.env.action_space.low)
                    mu = high * mu

                    # prop dist over actions
                    prob = torch.distributions.Normal(mu.view(-1, ).detach(), sigma.view(-1, ).detach())

                    # entropy for regularization
                    entropy = prob.entropy()

                    # sample during training for exploration
                    action = prob.sample(self.env.action_space.shape)

                    # avoid sampling outside the allowed range of action_space
                    action = np.clip(action, low, high)
                    # print("Training worker {} action: {}".format(self.worker_id, action))
                    log_prob = prob.log_prob(action)

                entropies.append(entropy)

                # make selected move
                if isinstance(self.env.action_space, Discrete):
                    action = action.numpy()[0, 0]
                elif isinstance(self.env.action_space, Box):
                    action = action.numpy()[0]

                state, reward, done, _ = self.env.step(action)
                done = done or t >= self.t_max

                with self.T.get_lock():
                    self.T.value += 1

                # reset env to ensure to get latest state
                if done:
                    t = 0
                    state = self.env.reset()

                state = torch.Tensor(state)
                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)

                # end if terminal state or max episodes are reached
                if done:
                    break

            R = torch.zeros(1, 1)

            # if non terminal state is present set R to be value of current state
            if not done:
                value = model(state.unsqueeze(0))[0]
                R = value.detach()

            values.append(R)

            # compute loss and backprop
            self._compute_loss(R, rewards, values, log_probs, entropies)

            self.lock.acquire()
            sync_grads(model, self.global_model)
            self.lock.release()

            self.optimizer.step()

    def _compute_loss(self, R: torch.Tensor, rewards: list, values: list, log_probs: list, entropies: list) -> None:
        policy_loss = 0
        value_loss = 0

        gae = torch.zeros(1, 1)

        # iterate over rewards from most recent to the starting one
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R
            advantage = R - values[i]
            value_loss += advantage.pow(2)

            if self.use_gae:
                # Generalized Advantage Estimation
                delta_t = rewards[i] + self.gamma * values[i + 1].detach() - values[i].detach()
                gae = gae * self.gamma * self.tau + delta_t
                policy_loss -= log_probs[i] * gae.detach() - self.beta * entropies[i]
            else:
                policy_loss -= log_probs[i] * advantage - self.beta * entropies[i]

        # zero grads to avoid computation issues in the next step
        self.optimizer.zero_grad()

        # compute combined loss of policy_loss and value_loss
        # avoid overfitting on value loss by scaling it down
        combined_loss = policy_loss + self.value_loss_coef * value_loss
        combined_loss.mean().backward()

    def _test(self):
        """
        Start worker in _test mode, i.e. no training is done, only testing is used to validate current performance
        loosely based on https://github.com/ikostrikov/pytorch-a3c/blob/master/_test.py
        :return:
        """

        torch.manual_seed(self.seed + self.worker_id)
        self.env.seed(self.seed + self.worker_id)

        # get an instance of the current global model state
        model = ActorCriticNetwork(self.env.observation_space.shape[0], self.env.action_space, self.is_discrete)
        model.eval()

        state = self.env.reset()
        state = torch.Tensor(state)
        reward_sum = 0
        done = True

        t = 0
        while True:
            self.env.render()
            t += 1

            # Get params from shared global model
            if done:
                model.load_state_dict(self.global_model.state_dict())

            # forward pass
            with torch.no_grad():
                if self.is_discrete:
                    _, logit = model(state.unsqueeze(0))

                    # prob dist of action space, select best action
                    prob = F.softmax(logit, dim=-1)
                    action = prob.max(1, keepdim=True)[1]

                else:
                    # select mean of normal dist as action --> Expectation
                    _, mu, _ = model(state.unsqueeze(0))
                    # print(mu)
                    action = mu

            if isinstance(self.env.action_space, Discrete):
                action = action.numpy()[0, 0]
            elif isinstance(self.env.action_space, Box):
                action = action.numpy()[0]

            state, reward, done, _ = self.env.step(action)
            done = done or t >= self.t_max
            reward_sum += reward

            # print current performance if terminal state or max episodes was reached
            if done:
                print("T={}, reward={}, episode_len={}".format(self.T.value, reward_sum, t))
                # reset current cumulated reward and episode counter as well as env
                reward_sum = 0
                t = 0
                state = self.env.reset()
                # delay _test run for 10s to give the network some time to train
                time.sleep(10)

            state = torch.Tensor(state)
