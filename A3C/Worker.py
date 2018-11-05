import logging
import time
from multiprocessing import Value
from threading import Thread

import gym
import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from A3C.ActorCriticNetwork import ActorCriticNetwork


class Worker(Thread):

    def __init__(self, env_name: str, worker_id: int, global_model: ActorCriticNetwork, seed: int, T: Value,
                 lr: float = 1e-4, n_steps: int = 0, t_max: int = 100000, gamma: float = .99,
                 tau: float = 1, beta: float = .01, value_loss_coef: float = .5,
                 optimizer: Optimizer = None, is_train: bool = True) -> None:
        """

        :param t_max: maximum episodes for training
        :param env_name: gym environment name
        :param worker_id: number of workers
        :param T: global shared counter
        :param optimizer: torch optimizer instance, either shared Optimizer or None for individual
        :param beta: Entropy weight factor
        :param tau:
        :param gamma: discount factor
        :param global_model:
        :param seed:
        :param lr: learning rate for the workers NN
        :param n_steps: amount of steps for training
        :param value_loss_coef:
        """
        super().__init__()

        # separate env for each worker
        self.env_name = env_name
        self.env = gym.make(self.env_name)

        # training params
        self.n_steps = n_steps
        self.tau = tau
        self.gamma = gamma
        self.beta = beta
        self.value_loss_coef = value_loss_coef

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

        # logging instance
        self.logger = logging.getLogger(__name__)

    def train(self):
        # code from https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py

        torch.manual_seed(self.seed + self.worker_id)

        self.env.seed(self.seed + self.worker_id)

        model = ActorCriticNetwork(self.env.observation_space.shape[0], self.env.action_space)

        if self.optimizer is None:
            self.optimizer = torch.optim.RMSprop(self.global_model.parameters(), lr=self.lr)

        model.train()

        state = self.env.reset()
        state = torch.from_numpy(state)
        done = True

        t = 0
        while True:
            # Get state of the global model
            model.load_state_dict(self.global_model.state_dict())

            values = []
            log_probs = []
            rewards = []
            entropies = []

            # print(self.n_steps, self.is_train)

            for step in range(self.n_steps):
                t += 1
                print(self.worker_id, self.T)
                value, logit, = model(state.unsqueeze(0))
                prob = F.softmax(logit, dim=-1)
                log_prob = F.log_softmax(logit, dim=-1)

                # compute entropy for loss
                entropy = -(log_prob * prob).sum(1, keepdim=True)
                entropies.append(entropy)

                action = prob.multinomial(num_samples=1).detach()
                log_prob = log_prob.gather(1, action)

                state, reward, done, _ = self.env.step(action.numpy()[0, 0])

                done = done or t >= self.t_max
                # reward = max(min(reward, 1), -1)

                with self.T.get_lock():
                    self.T.value += 1

                if done:
                    t = 0
                    state = self.env.reset()

                state = torch.from_numpy(state)
                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)

                # end if terminal state or max episodes are reached
                if done:
                    break

            R = torch.zeros(1, 1)

            # if non terminal state is present
            if not done:
                value, _ = model(state.unsqueeze(0))
                R = value.detach()

            values.append(R)

            self.compute_loss(R, rewards, values, log_probs, entropies)

            # ensure model grads are shared
            for param, shared_param in zip(model.parameters(), self.global_model.parameters()):
                if shared_param.grad is not None:
                    return
                shared_param._grad = param.grad

            self.optimizer.step()
            print("end", self.worker_id)

    def compute_loss(self, R, rewards, values, log_probs, entropies):
        policy_loss = 0
        value_loss = 0

        gae = torch.zeros(1, 1)

        # iterate over rewards from most recent to the starting one
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            # TODO not sure about this part
            delta_t = rewards[i] + self.gamma * values[i + 1] - values[i]
            gae = gae * self.gamma * self.tau + delta_t

            policy_loss = policy_loss - log_probs[i] * gae.detach() - self.beta * entropies[i]

        self.optimizer.zero_grad()

        (policy_loss + self.value_loss_coef * value_loss).backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

    def test(self):

        torch.manual_seed(self.seed + self.worker_id)

        env = gym.make(self.env_name)
        env.seed(self.seed + self.worker_id)

        model = ActorCriticNetwork(env.observation_space.shape[0], env.action_space)
        model.eval()

        state = env.reset()
        env.render()
        state = torch.from_numpy(state)
        reward_sum = 0
        done = True

        start_time = time.time()

        # a quick hack to prevent the agent from stucking
        # actions = deque(maxlen=100)
        t = 0
        while True:
            t += 1
            # Get params from global model
            if done:
                model.load_state_dict(self.global_model.state_dict())

            with torch.no_grad():
                value, logit = model(state.unsqueeze(0))
            prob = F.softmax(logit, dim=-1)
            action = prob.max(1, keepdim=True)[1].numpy()[0, 0]

            state, reward, done, _ = env.step(action)
            done = done or t >= self.t_max
            reward_sum += reward

            # a quick hack to prevent the agent from stucking
            # actions.append(action)
            # if actions.count(actions[0]) == actions.maxlen:
            #     done = True

            if done:
                print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), self.T.value,
                    self.T.value / (time.time() - start_time), reward_sum, t))
                reward_sum = 0
                t = 0
                # actions.clear()
                state = env.reset()
                time.sleep(60)

            state = torch.from_numpy(state)

    def run(self):
        if self.is_train:
            self.train()
        else:
            self.test()
