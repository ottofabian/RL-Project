import math
import time

import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.multiprocessing import Value
from torch.optim import Optimizer

from A3C.Models.ActorCriticNetwork import ActorCriticNetwork
from A3C.Models.ActorNetwork import ActorNetwork
from A3C.Models.CriticNetwork import CriticNetwork


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
        shared_param._grad = param.grad  #


pi = Variable(torch.Tensor([math.pi]))


def normal(x, mu, sigma_sq):
    a = (-1 * (Variable(x) - mu).pow(2) / (2 * sigma_sq)).exp()
    b = 1 / (2 * sigma_sq * pi.expand_as(sigma_sq)).sqrt()
    return a * b


def test(env_name: str, worker_id: int, shared_model_actor: ActorNetwork, shared_model_critic: CriticNetwork, seed: int,
         T: Value, t_max: int = 100000, is_discrete: bool = False, global_reward: Value = None):
    """
    Start worker in _test mode, i.e. no training is done, only testing is used to validate current performance
    loosely based on https://github.com/ikostrikov/pytorch-a3c/blob/master/_test.py
    :return:
    """
    torch.manual_seed(seed + worker_id)

    env = gym.make(env_name)
    env.seed(seed + worker_id)

    # get an instance of the current global model state
    model_actor = ActorNetwork(env.observation_space.shape[0], env.action_space, is_discrete)
    model_critic = CriticNetwork(env.observation_space.shape[0], env.action_space, is_discrete)
    model_actor.eval()
    model_critic.eval()
    # model.load_state_dict(global_model.state_dict())

    state = torch.from_numpy(np.array(env.reset()))
    reward_sum = 0

    t = 0
    done = False

    while True:

        # Get params from shared global model
        model_critic.load_state_dict(shared_model_critic.state_dict())
        model_actor.load_state_dict(shared_model_actor.state_dict())

        rewards = np.zeros(10)
        eps_len = np.zeros(10)

        # make 10 runs to get current avg performance
        for i in range(10):
            while not done:
                t += 1

                if i == 0:
                    env.render()

                with torch.no_grad():

                    # select mean of normal dist as action --> Expectation
                    mu, _ = model_actor(Variable(state))
                    action = mu.data

                state, reward, done, _ = env.step(action.numpy())
                done = done or t >= t_max
                reward_sum += reward

                if done:
                    # reset current cumulated reward and episode counter as well as env
                    rewards[i] = reward_sum
                    reward_sum = 0

                    eps_len[i] = t
                    t = 0

                    state = env.reset()

                state = torch.from_numpy(np.array(state))
            done = False

        print("T={} -- mean reward={} -- mean episode length={} -- global reward={}".format(T.value,
                                                                                            rewards.mean(),
                                                                                            eps_len.mean(),
                                                                                            global_reward.value))
        # delay _test run for 10s to give the network some time to train
        time.sleep(10)


def train(env_name: str, worker_id: int, shared_model_actor: ActorNetwork, shared_model_critic: CriticNetwork,
          seed: int, T: Value, lr: float = 1e-4, n_steps: int = 0, t_max: int = 100000, gamma: float = .99,
          tau: float = 1, beta: float = .01, value_loss_coef: float = .5, optimizer: Optimizer = None,
          use_gae: bool = True, is_discrete: bool = False, global_reward=None):
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
    model_actor = ActorNetwork(env.observation_space.shape[0], env.action_space, is_discrete)
    model_critic = CriticNetwork(env.observation_space.shape[0], env.action_space, is_discrete)

    # if no shared optimizer is provided use individual one
    if optimizer is None:
        optimizer_critic = torch.optim.RMSprop(shared_model_critic.parameters(), lr=0.001)
        optimizer_actor = torch.optim.RMSprop(shared_model_actor.parameters(), lr=0.0001)
        # optimizer = torch.optim.Adam(global_model.parameters(), lr=lr)

    model_actor.train()
    model_critic.train()

    writer = SummaryWriter()

    state = torch.from_numpy(np.array(env.reset()))

    t = 0
    iter_ = 0

    while True:
        # Get state of the global model
        model_critic.load_state_dict(shared_model_critic.state_dict())
        model_actor.load_state_dict(shared_model_actor.state_dict())

        # containers for computing loss
        values = []
        log_probs = []
        rewards = []
        entropies = []

        states = []

        # reward_sum = 0
        for step in range(n_steps):
            t += 1

            value = model_critic(Variable(state))
            mu, sigma = model_actor(Variable(state))

            states.append(state)

            # ------------------------------------------
            # # select action
            eps = Variable(torch.randn(mu.size()))
            action = (mu + sigma.sqrt() * eps).data

            # ------------------------------------------
            # Compute statistics for loss
            prob = normal(action, mu, sigma)

            entropy = -0.5 * (sigma + 2 * pi.expand_as(sigma)).log() + .5
            entropies.append(entropy)

            log_prob = prob.log()

            # make selected move
            state, reward, done, _ = env.step(action.numpy())

            reward = min(max(-1, reward), 1)

            done = done or t >= t_max

            with T.get_lock():
                T.value += 1

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            # reset env to beginning if done
            if done:
                t = 0
                state = env.reset()

            state = torch.from_numpy(np.array(state))

            # end if terminal state or max episodes are reached
            if done:
                break

        # # TODO: Implement this correctly
        with global_reward.get_lock():
            if global_reward.value == 0:
                global_reward.value = sum(rewards)
            else:
                global_reward.value = global_reward.value * 0.99 + sum(rewards) * 0.01

        R = torch.zeros(1, 1)

        # if non terminal state is present set R to be value of current state
        if not done:
            R = model_critic(Variable(state)).data

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
        optimizer_critic.zero_grad()
        optimizer_actor.zero_grad()

        critic_loss.backward()
        actor_loss.backward()

        # compute combined loss of actor_loss and critic_loss
        # avoid overfitting on value loss by scaling it down
        # (actor_loss + value_loss_coef * critic_loss).backward()
        # combined_loss.mean().backward()
        # combined_loss.backward()

        torch.nn.utils.clip_grad_norm_(model_critic.parameters(), 40)
        torch.nn.utils.clip_grad_norm_(model_actor.parameters(), 40)

        sync_grads(model_critic, shared_model_critic)
        sync_grads(model_actor, shared_model_actor)
        optimizer_critic.step()
        optimizer_actor.step()

        iter_ += 1

        grads_critic = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                       for p in model_critic.parameters()
                                       if p.grad is not None])

        grads_actor = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                      for p in model_actor.parameters()
                                      if p.grad is not None])

        writer.add_scalar("mean_values", np.mean([v.data for v in values]), iter_)
        writer.add_scalar("batch_rewards", np.mean(np.array(rewards)), iter_)
        writer.add_scalar("loss_policy", actor_loss, iter_)
        writer.add_scalar("loss_value", critic_loss, iter_)
        writer.add_scalar("grad_l2_actor", np.sqrt(np.mean(np.square(grads_actor))), iter_)
        writer.add_scalar("grad_max_actor", np.max(np.abs(grads_actor)), iter_)
        writer.add_scalar("grad_var_actor", np.var(grads_actor), iter_)
        writer.add_scalar("grad_l2_critic", np.sqrt(np.mean(np.square(grads_critic))), iter_)
        writer.add_scalar("grad_max_critic", np.max(np.abs(grads_critic)), iter_)
        writer.add_scalar("grad_var_critic", np.var(grads_critic), iter_)

        # if iter % 50 == 0:
        #     loss_mean_actor.append(np.mean(np.array(loss_container_actor[-50])))
        #     loss_mean_critic.append(np.mean(np.array(loss_container_critic[-50])))
        #
        #     plt.figure(0)
        #     plt.plot(loss_container_critic)
        #     plt.title("Critic Loss")
        #
        #     # plt.plot(np.arange(50, len(loss_container_critic)+1, 50), loss_mean_critic)
        #     # plt.title("Critic Loss Mean over 50")
        #
        #     plt.figure(1)
        #     plt.plot(loss_container_critic)
        #     plt.title("Actor Loss")
        #
        #     # plt.plot(np.arange(50, len(loss_container_actor)+1, 50), loss_mean_actor)
        #     # plt.title(, "Actor Loss Mean over 50")
        #
        #     plt.show()

        # model_critic.load_state_dict(shared_model_critic.state_dict())
        #
        # print("Value before training:")
        # print([v.data for v in values])
        # print("Value after training:")
        # print(model_critic(torch.stack(states)))
