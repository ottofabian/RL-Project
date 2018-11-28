import logging
import math
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
from torch.autograd import Variable
from torch.optim import Optimizer

from A3C.ActorCriticNetwork import ActorCriticNetwork

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


class Worker(Thread):
    # DEBUG list to store the taken actions by the worker threads
    actions_taken_training = []
    values_training = []

    def __init__(self, env_name: str, worker_id: int, global_model: ActorCriticNetwork, seed: int, T: Value,
                 lr: float = 1e-4, n_steps: int = 0, t_max: int = 100000, gamma: float = .99,
                 tau: float = 1, beta: float = .01, value_loss_coef: float = .5,
                 optimizer: Optimizer = None, is_train: bool = True, use_gae: bool = True,
                 is_discrete: bool = False, lock=None, global_reward=None) -> None:
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

        # DEBUG information about the environment
        logging.debug('Chosen Environment %s' % self.env_name)
        logging.debug('Observation Space: %s' % self.env.observation_space)
        logging.debug('Action Space: %s' % self.env.action_space)
        if self.is_discrete is False:
            logging.debug('Action Range: [%.3f, %.3f]' % (self.env.action_space.low, self.env.action_space.high))
        # print(self.env.action_space)

        # training params
        self.n_steps = n_steps
        self.tau = tau
        self.discount = gamma
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
        self.global_reward = global_reward

        # logging instance
        self.logger = logging.getLogger(__name__)

        self.pi = Variable(torch.Tensor([math.pi]))

    def run(self):
        if self.is_train:
            self._train()
            logging.debug('._train')
        else:
            self._test()

    def _normal(self, x, mu, sigma_sq):
        a = (-1 * (x - mu).pow(2) / (2 * sigma_sq)).exp()
        b = 1 / (2 * sigma_sq * self.pi.expand_as(sigma_sq)).sqrt()
        return (a * b).log()

    def _train(self):
        """
        Start worker in training mode, i.e.training the shared model with backprop
        loosely based on https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py
        :return: self
        """

        # ensure different seed for each worker thread
        torch.manual_seed(self.seed + self.worker_id)
        self.env.seed(self.seed + self.worker_id)

        # init local NN instance for worker thread
        model = ActorCriticNetwork(self.env.observation_space.shape[0], self.env.action_space, self.is_discrete)
        model.train()

        # if no shared optimizer is provided use individual one
        if self.optimizer is None:
            self.optimizer = torch.optim.RMSprop(self.global_model.parameters(), lr=self.lr)

        state = torch.from_numpy(np.array(self.env.reset()))

        t = 0
        while True:
            # Get state of the global model
            model.load_state_dict(self.global_model.state_dict())

            # containers for computing loss
            values = []
            log_probs = []
            rewards = []
            entropies = []

            reward_sum = 0

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

                    value, mu, sigma = model(state.float())
                    # print("Training worker {} -- mu: {} -- sigma: {}".format(self.worker_id, mu, sigma))

                    dist = model.distribution(mu, sigma)

                    # ------------------------------------------
                    # select action
                    action = dist.sample()

                    # assuming action space is in -high/high
                    # env is currently clipping internally
                    # high = np.asscalar(self.env.action_space.high)
                    # low = np.asscalar(self.env.action_space.low)
                    # action = np.clip(action, low, high)

                    # ------------------------------------------
                    # Compute statistics for loss
                    log_prob = dist.log_prob(action)
                    entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(dist.scale)

                    # -----------------------------------------------------------------
                    # eps = Variable(torch.randn(mu.size()))
                    # action = (mu + sigma.sqrt() * eps).data.numpy()
                    #
                    # # assuming action space is in -high/high
                    # high = np.asscalar(self.env.action_space.high)
                    # low = np.asscalar(self.env.action_space.low)
                    # action = np.clip(action, low, high)
                    #
                    # action = Variable(torch.from_numpy(action))
                    #
                    # entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(dist.scale)  # exploration

                    # Worker.actions_taken_training.append(action)
                    # Worker.values.append(value.detach().numpy())

                    # print("Training worker {} action: {}".format(self.worker_id, action))

                # make selected move
                if isinstance(self.env.action_space, Discrete):
                    action = action.numpy()[0, 0]
                elif isinstance(self.env.action_space, Box):
                    action = action.numpy()

                state, reward, done, _ = self.env.step(np.array(action))
                reward_sum = reward_sum + reward
                done = done or t >= self.t_max

                with self.T.get_lock():
                    self.T.value += 1

                reward = min(max(-1, reward), 1)

                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)

                # reset env to ensure to get latest state
                if done:
                    t = 0
                    state = self.env.reset()

                state = torch.from_numpy(np.array(state))

                # end if terminal state or max episodes are reached
                if done:
                    break

            with self.global_reward.get_lock():
                if self.global_reward.value == 0:
                    self.global_reward.value = reward_sum
                else:
                    self.global_reward.value = self.global_reward.value * 0.99 + reward_sum * 0.01

            R = torch.zeros(1, 1)

            # if non terminal state is present set R to be value of current state
            if not done:
                R = model(state.float())[0].detach()

            values.append(R)

            # compute loss and backprop

            actor_loss = []
            critic_loss = []
            gae = torch.zeros(1, 1)

            # iterate over rewards from most recent to the starting one
            for i in reversed(range(len(rewards))):
                R = rewards[i] + R * self.discount
                advantage = R - values[i]
                critic_loss.append(0.5 * advantage.pow(2))
                if self.use_gae:
                    # Generalized Advantage Estimation
                    delta_t = rewards[i] + self.discount * values[i + 1].data - values[i].detach()
                    gae = gae * self.discount * self.tau + delta_t
                    actor_loss.append(-log_probs[i] * Variable(gae) - self.beta * entropies[i])
                else:
                    actor_loss.append(-log_probs[i] * advantage.detach() - self.beta * entropies[i])

            # zero grads to avoid computation issues in the next step
            self.optimizer.zero_grad()

            # compute combined loss of actor_loss and critic_loss
            # avoid overfitting on value loss by scaling it down
            combined_loss = torch.stack(actor_loss).sum() + self.value_loss_coef * torch.stack(critic_loss).sum()
            # combined_loss.mean().backward()
            combined_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
            sync_grads(model, self.global_model)

            self.optimizer.step()

    def _compute_loss(self, R: torch.Tensor, rewards: list, values: list, log_probs: list, entropies: list,
                      model: ActorCriticNetwork) -> None:
        actor_loss = 0
        critic_loss = 0

        gae = torch.zeros(1, 1)

        # iterate over rewards from most recent to the starting one
        for i in reversed(range(len(rewards))):
            R = rewards[i] + R * self.discount
            advantage = R - values[i]
            critic_loss = critic_loss + 0.5 * advantage.pow(2)
            if self.use_gae:
                # Generalized Advantage Estimataion
                delta_t = rewards[i] + self.discount * values[i + 1].data - values[i]
                gae = gae * self.discount * self.tau + delta_t
                actor_loss = actor_loss - log_probs[i] * gae.detach() - self.beta * entropies[i]
            else:
                actor_loss = actor_loss - log_probs[i] * advantage - self.beta * entropies[i]

        # zero grads to avoid computation issues in the next step
        model.zero_grad()
        self.optimizer.zero_grad()

        # compute combined loss of actor_loss and critic_loss
        # avoid overfitting on value loss by scaling it down
        combined_loss = actor_loss + self.value_loss_coef * critic_loss
        combined_loss.mean().backward()
        # combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50)

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
        # model.load_state_dict(self.global_model.state_dict())

        state = torch.Tensor(self.env.reset())
        reward_sum = 0

        t = 0

        taken_actions_test = []

        iteration = 0
        done = False

        while True:

            # Get params from shared global model
            model.load_state_dict(self.global_model.state_dict())

            rewards = np.zeros(10)
            eps_len = np.zeros(10)

            for i in range(10):
                while not done:
                    if i == 0:
                        self.env.render()
                    t += 1
                    # forward pass
                    with torch.no_grad():
                        if self.is_discrete:
                            _, logit = model(state.float())

                            # prob dist of action space, select best action
                            prob = F.softmax(logit, dim=-1)
                            action = prob.max(1, keepdim=True)[1]

                        else:
                            # select mean of normal dist as action --> Expectation
                            _, mu, _ = model(state.float())
                            # print(value, mu, sigma)
                            action = mu
                            taken_actions_test.append(action)

                    # make selected move
                    if isinstance(self.env.action_space, Discrete):
                        action = action.numpy()[0, 0]
                    elif isinstance(self.env.action_space, Box):
                        action = action.numpy()  # [0]

                    state, reward, done, _ = self.env.step(action)
                    done = done or t >= self.t_max
                    reward_sum = reward_sum + reward

                    # print current performance if terminal state or max episodes was reached
                    if done:
                        # print("T={}, reward={}, episode_len={}".format(self.T.value, reward_sum, t))

                        # reset current cumulated reward and episode counter as well as env
                        rewards[i] = reward_sum
                        reward_sum = 0

                        eps_len[i] = t
                        t = 0

                        state = self.env.reset()

                        # if iteration == 3 and self.is_discrete is False:
                        #     plt.hist(Worker.actions_taken_training)
                        #     plt.title('Taken Action during Training')
                        #     plt.show()
                        #
                        #     plt.hist(Worker.actions_taken_training)
                        #     plt.title('Taken Action during Test-Phase')
                        #     plt.show()

                        # delay _test run for 10s to give the network some time to train
                        # time.sleep(10)
                        # plt.figure()
                        # iteration += 1

                    state = torch.from_numpy(np.array(state))
                done = False

            print("T={} -- mean reward={} -- mean episode length={} -- global reward={}".format(self.T.value,
                                                                                                rewards.mean(),
                                                                                                eps_len.mean(),
                                                                                                self.global_reward.value))
            # delay _test run for 10s to give the network some time to train
            time.sleep(10)

            # plt.figure()
            # plt.hist(Worker.values)
            # plt.title('Predicted Values during Training')
            # plt.show()
