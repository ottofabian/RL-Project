import logging

import gym
import torch
import torch.nn.functional as F

from A3C.ActorCritic import ActorCritic


class Worker(object):

    def __init__(self, env_name, n_worker, global_model, T, seed, lr, n_steps, t_max, gamma, tau, beta, value_loss_coef,
                 optimizer=None):
        """

        :param t_max: maximum episodes for training
        :param env_name: gym environment name
        :param n_worker: number of workers
        :param T: global shared counter
        :param optimizer:
        :param beta: Entropy weight factor
        :param tau: object
        :param gamma: discount factor
        :param global_model:
        :param seed:
        :param lr:
        :param n_steps:
        :param value_loss_coef:
        """
        # separate env for each worker
        self.env_name = env_name
        self.env = gym.make(self.env_name)

        # training params
        self.seed = seed
        self.lr = lr
        self.n_steps = n_steps
        self.t_max = t_max
        self.tau = tau
        self.gamma = gamma
        self.beta = beta
        self.value_loss_coef = value_loss_coef

        # shared params
        self.optimizer = optimizer
        self.global_model = global_model
        self.n_worker = n_worker
        self.T = T

        # logging instance
        self.logger = logging.getLogger(__name__)

    def run(self):
        # code from https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py

        torch.manual_seed(self.seed + self.n_worker)

        self.env.seed(self.seed + self.n_worker)

        model = ActorCritic(self.env.observation_space.shape[0], self.env.action_space)

        if self.optimizer is None:
            optimizer = torch.optim.RMSprop(self.global_model.parameters(), lr=self.lr)

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

            for step in range(self.n_steps):
                t += 1
                value, logit, = model(state.unsqueeze(0))
                prob = F.softmax(logit, dim=-1)
                log_prob = F.log_softmax(logit, dim=-1)

                # compute entropy for loss
                entropy = -(log_prob * prob).sum(1, keepdim=True)
                entropies.append(entropy)

                action = prob.multinomial(num_samples=1).detach()
                log_prob = log_prob.gather(1, action)

                state, reward, done, _ = self.env.step(action.numpy())

                done = done or t >= self.t_max
                reward = max(min(reward, 1), -1)

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
                value, _, _ = model(state.unsqueeze(0))
                R = value.detach()

            values.append(R)

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

                policy_loss -= log_probs[i] * gae.detach() - self.beta * entropies[i]

            optimizer.zero_grad()

            (policy_loss + self.value_loss_coef * value_loss).backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

            # ensure model grads are shared
            for param, shared_param in zip(model.parameters(), self.global_model.parameters()):
                if shared_param.grad is not None:
                    return
                shared_param._grad = param.grad

            optimizer.step()
