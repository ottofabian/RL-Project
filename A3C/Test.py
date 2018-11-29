import math
import time

import gym
import numpy as np
import torch
import torch.nn.functional as F
from gym.spaces import Discrete, Box
from torch.autograd import Variable
from torch.multiprocessing import Value
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
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def test(env_name: str, worker_id: int, shared_model: ActorCriticNetwork, seed: int, T: Value, t_max: int = 100000,
         is_discrete: bool = False, global_reward: Value = None):
    """
    Start worker in _test mode, i.e. no training is done, only testing is used to validate current performance
    loosely based on https://github.com/ikostrikov/pytorch-a3c/blob/master/_test.py
    :return:
    """
    torch.manual_seed(seed + worker_id)

    env = gym.make(env_name)
    env.seed(seed + worker_id)

    # get an instance of the current global model state
    model = ActorCriticNetwork(env.observation_space.shape[0], env.action_space, is_discrete)
    model.eval()
    # model.load_state_dict(global_model.state_dict())

    state = torch.from_numpy(np.array(env.reset()))
    reward_sum = 0

    t = 0
    taken_actions_test = []
    iteration = 0
    done = True

    while True:

        # Get params from shared global model
        model.load_state_dict(shared_model.state_dict())

        rewards = np.zeros(10)
        eps_len = np.zeros(10)

        # make 10 runs to get current avg performance
        for i in range(10):
            while not done:
                t += 1

                if i == 0:
                    env.render()

                with torch.no_grad():

                    if is_discrete:
                        _, logit = model(state.float())

                        # prob dist of action space, select best action
                        prob = F.softmax(logit, dim=-1)
                        action = prob.max(1, keepdim=True)[1]

                    else:
                        # select mean of normal dist as action --> Expectation
                        _, mu, _ = model(Variable(state))
                        action = mu.data
                        taken_actions_test.append(action)

                # make selected move
                if isinstance(env.action_space, Discrete):
                    action = action.numpy()[0, 0]
                elif isinstance(env.action_space, Box):
                    action = action.numpy()

                state, reward, done, _ = env.step(action)
                done = done or t >= t_max
                reward_sum += reward

                if done:
                    # reset current cumulated reward and episode counter as well as env
                    rewards[i] = reward_sum
                    reward_sum = 0

                    eps_len[i] = t
                    t = 0

                    state = env.reset()

                    # if iteration == 3 and is_discrete is False:
                    #     plt.hist(Worker.actions_taken_training)
                    #     plt.title('Taken Action during Training')
                    #     plt.show()
                    #
                    #     plt.hist(Worker.actions_taken_training)
                    #     plt.title('Taken Action during Test-Phase')
                    #     plt.show()

                    # plt.figure()
                    iteration += 1

                state = torch.from_numpy(np.array(state))
            done = False

        print("T={} -- mean reward={} -- mean episode length={} -- global reward={}".format(T.value,
                                                                                            rewards.mean(),
                                                                                            eps_len.mean(),
                                                                                            global_reward.value))
        # delay _test run for 10s to give the network some time to train
        time.sleep(10)

        # plt.figure()
        # plt.hist(Worker.values)
        # plt.title('Predicted Values during Training')
        # plt.show()


def train(env_name: str, worker_id: int, shared_model: ActorCriticNetwork, seed: int, T: Value, lr: float = 1e-4,
          n_steps: int = 0, t_max: int = 100000, gamma: float = .99, tau: float = 1, beta: float = .01,
          value_loss_coef: float = .5, optimizer: Optimizer = None, use_gae: bool = True, is_discrete: bool = False,
          global_reward=None):
    """
    Start worker in training mode, i.e. training the shared model with backprop
    loosely based on https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py
    :return: self
    """
    torch.manual_seed(seed + worker_id)

    env = gym.make(env_name)
    # ensure different seed for each worker thread
    env.seed(seed + worker_id)

    # init local NN instance for worker thread
    model = ActorCriticNetwork(env.observation_space.shape[0], env.action_space, is_discrete)

    # if no shared optimizer is provided use individual one
    if optimizer is None:
        optimizer = torch.optim.RMSprop(shared_model.parameters(), lr=lr)
        # optimizer = torch.optim.Adam(global_model.parameters(), lr=lr)

    model.train()

    state = torch.from_numpy(np.array(env.reset()))

    t = 0
    while True:
        # Get state of the global model
        model.load_state_dict(shared_model.state_dict())

        # containers for computing loss
        values = []
        log_probs = []
        rewards = []
        entropies = []

        # reward_sum = 0

        for step in range(n_steps):
            t += 1

            if is_discrete:
                # forward pass
                value, logit = model(state)

                # prop dist over actions
                prob = F.softmax(logit, dim=-1)
                log_prob = F.log_softmax(logit, dim=-1)

                # compute entropy for loss regularization
                entropy = -(log_prob * prob).sum(1, keepdim=True)

                # choose action based on prob dist
                action = prob.multinomial(num_samples=1).detach()
                log_prob = log_prob.gather(1, action)

            else:

                value, mu, sigma = model(Variable(state))
                # print("Training worker {} -- mu: {} -- sigma: {}".format(worker_id, mu, sigma))

                # dist = torch.distributions.Normal(mu, sigma)

                # ------------------------------------------
                # select action
                eps_v = Variable(torch.randn(mu.size()))
                action = (mu + sigma.sqrt() * eps_v).data
                # action = dist.sample().detach()

                # assuming action space is in -high/high
                # env is currently clipping internally
                # high = np.asscalar(env.action_space.high)
                # low = np.asscalar(env.action_space.low)
                # action = np.clip(action, low, high)

                # ------------------------------------------
                # Compute statistics for loss
                exp_ = (-1 * (Variable(action) - mu).pow(2) / (2 * sigma)).exp()
                scale = 1 / (2 * sigma * math.pi).sqrt()
                log_prob = (exp_ * scale).log()
                # log_prob = dist.log_prob(action)

                entropy = -0.5 * ((sigma + 2 * math.pi).log() + 1.)
                # entropy = dist.entropy()

                # -----------------------------------------------------------------
                # eps = Variable(torch.randn(mu.size()))
                # action = (mu + sigma.sqrt() * eps).data.numpy()
                #
                # # assuming action space is in -high/high
                # high = np.asscalar(env.action_space.high)
                # low = np.asscalar(env.action_space.low)
                # action = np.clip(action, low, high)
                #
                # action = Variable(torch.from_numpy(action))
                #
                # entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(dist.scale)  # exploration

                # Worker.actions_taken_training.append(action)
                # Worker.values.append(value.detach().numpy())

                # print("Training worker {} action: {}".format(worker_id, action))

            # make selected move
            if isinstance(env.action_space, Discrete):
                action = action.numpy()[0, 0]
            elif isinstance(env.action_space, Box):
                action = action.numpy()

            state, reward, done, _ = env.step(action)

            done = done or t >= t_max

            with T.get_lock():
                T.value += 1

            reward = min(max(-1, reward), 1)

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)

            # reset env to beginning if done
            if done:
                t = 0
                state = env.reset()

            state = torch.from_numpy(np.array(state))

            # end if terminal state or max episodes are reached
            if done:
                break

        # TODO: Implement this correctly
        with global_reward.get_lock():
            if global_reward.value == 0:
                global_reward.value = sum(rewards)
            else:
                global_reward.value = global_reward.value * 0.99 + sum(rewards) * 0.01

        R = torch.zeros(1, 1)

        # if non terminal state is present set R to be value of current state
        if not done:
            R = model(Variable(state))[0].data

        values.append(Variable(R))
        # compute loss and backprop
        actor_loss = 0
        critic_loss = 0
        gae = torch.zeros(1, 1)
        R = Variable(R)

        # iterate over rewards from most recent to the starting one
        for i in reversed(range(len(rewards))):
            R = rewards[i] + R * gamma
            advantage = R - values[i]
            critic_loss = critic_loss + 0.5 * advantage.pow(2)
            if use_gae:
                # Generalized Advantage Estimation
                delta_t = rewards[i] + gamma * values[i + 1].data - values[i].data
                gae = gae * gamma * tau + delta_t
                actor_loss = actor_loss - log_probs[i] * Variable(gae) - beta * entropies[i]
            else:
                actor_loss = actor_loss - log_probs[i] * advantage.data - beta * entropies[i]

        # zero grads to avoid computation issues in the next step
        optimizer.zero_grad()

        # compute combined loss of actor_loss and critic_loss
        # avoid overfitting on value loss by scaling it down
        (actor_loss + value_loss_coef * critic_loss).backward()
        # combined_loss.mean().backward()
        # combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

        sync_grads(model, shared_model)
        optimizer.step()
