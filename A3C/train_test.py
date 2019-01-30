import copy
import math
import time

import gym
import numpy as np
import quanser_robots
import torch
from torch.autograd import Variable
from torch.multiprocessing import Value
from torch.optim import Optimizer

from A3C.Models.ActorCriticNetwork import ActorCriticNetwork
from A3C.Models.CriticNetwork import CriticNetwork
from A3C.Worker import save_checkpoint
from tensorboardX import SummaryWriter

from A3C.util.normalizer.base_normalizer import BaseNormalizer
from A3C.util.normalizer.mean_std_normalizer import MeanStdNormalizer
from A3C.util.save_and_load.model_save import get_normalizer


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


def test(args, worker_id: int, shared_model: torch.nn.Module, T: Value, global_reward: Value = None,
         optimizer: Optimizer = None, shared_model_critic: CriticNetwork = None, optimizer_critic: Optimizer = None):
    """
    Start worker in _test mode, i.e. no training is done, only testing is used to validate current performance
    loosely based on https://github.com/ikostrikov/pytorch-a3c/blob/master/_test.py
    :param args:
    :param worker_id:
    :param shared_model:
    :param shared_model_critic:
    :param T:
    :param optimizer:
    :param optimizer_critic:
    :param global_reward:
    :return:
    """
    torch.manual_seed(args.seed + worker_id)

    if "RR" in args.env_name:
        env = quanser_robots.GentlyTerminating(gym.make(args.env_name))
    else:
        env = gym.make(args.env_name)

    env.seed(args.seed + worker_id)

    normalizer = get_normalizer(args.normalizer)

    # get an instance of the current global model state
    model = copy.deepcopy(shared_model)
    model.eval()

    if shared_model_critic:
        model_critic = copy.deepcopy(shared_model_critic)
        model_critic.eval()

    state = torch.from_numpy(env.reset())

    writer = SummaryWriter(comment='_test')
    start_time = time.time()

    t = 0
    episode_reward = 0

    done = False
    iter_ = 0
    best_global_reward = None

    while True:

        # Get params from shared global model
        model.load_state_dict(shared_model.state_dict())
        if not args.shared_model:
            model_critic.load_state_dict(shared_model_critic.state_dict())

        rewards = []
        eps_len = []

        # make 10 runs to get current avg performance
        for i in range(10):
            while not done:
                t += 1

                if i == 0 and t % 1 == 0:
                    env.render()

                # apply min/max scaling on the environment

                with torch.no_grad():

                    # select mean of normal dist as action --> Expectation
                    if args.shared_model:
                        _, mu, _ = model(normalizer(state))
                    else:
                        mu, _ = model(normalizer(state))

                    action = mu.detach()

                state, reward, done, _ = env.step(np.clip(action.numpy(), -args.max_action, args.max_action))

                done = done or t >= args.max_episode_length
                episode_reward += reward

                if done:
                    # reset current cumulated reward and episode counter as well as env
                    rewards.append(episode_reward)
                    episode_reward = 0

                    eps_len.append(t)
                    t = 0

                    state = env.reset()

                state = torch.from_numpy(state)

            # necessary to make more than one run
            done = False

        time_print = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))

        rewards = np.mean(rewards)

        print(f"Time: {time_print}, T={T.value} -- mean reward={rewards}"
              f"-- mean episode length={np.mean(eps_len)} -- global reward={global_reward.value}")

        writer.add_scalar("mean_test_reward", rewards, int(T.value))

        if best_global_reward is None or global_reward.value > best_global_reward:
            best_global_reward = global_reward.value
            model_type = 'shared' if args.shared_model else 'split'

            save_checkpoint({
                'epoch': T.value,
                'model': shared_model.state_dict(),
                'model_critic': shared_model_critic.state_dict() if shared_model_critic else None,
                'global_reward': global_reward.value,
                # only save optimizers if shared ones are used
                'optimizer': optimizer.state_dict() if optimizer else None,
                'optimizer_critic': optimizer_critic.state_dict() if optimizer_critic else None,
            }, filename=f"./checkpoints/model_{model_type}_T-{T.value}_global-{global_reward.value}.pth.tar")

        # delay _test run for 10s to give the network some time to train
        # time.sleep(10)
        iter_ += 1


def train(args, worker_id: int, shared_model: torch.nn.Module, T: Value, global_reward: Value,
          optimizer: Optimizer = None, shared_model_critic: CriticNetwork = None, optimizer_critic: Optimizer = None,
          lr_scheduler: torch.optim.lr_scheduler = None, lr_scheduler_critic: torch.optim.lr_scheduler = None):
    """
    Start worker in training mode, i.e. training the shared model with backprop
    loosely based on https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py
    :return: None
    """
    torch.manual_seed(args.seed + worker_id)
    print(f"Training Worker {worker_id} started")

    env = gym.make(args.env_name)
    # ensure different seed for each worker thread
    env.seed(args.seed + worker_id)

    normalizer = get_normalizer(args.normalizer)

    # init local NN instance for worker thread
    model = copy.deepcopy(shared_model)
    model.train()

    if shared_model_critic:
        model_critic = copy.deepcopy(shared_model_critic)
        model_critic.train()

    # if no shared optimizer is provided use individual one
    if not optimizer:
        if args.optimizer == "rmsprop":
            optimizer = torch.optim.RMSprop(shared_model.parameters(), lr=args.lr)
        elif args.optimizer == "adam":
            optimizer = torch.optim.Adam(shared_model.parameters(), lr=args.lr)

    if shared_model_critic and not optimizer_critic:
        if args.optimizer == "rmsprop":
            optimizer_critic = torch.optim.RMSprop(shared_model_critic.parameters(), lr=args.lr_critic)
        elif args.optimizer == "adam":
            optimizer_critic = torch.optim.Adam(shared_model_critic.parameters(), lr=args.lr_critic)

    state = torch.from_numpy(env.reset())

    t = 0
    iter_ = 0
    episode_reward = 0

    if worker_id == 0:
        writer = SummaryWriter()

    while True:
        # Get state of the global model
        model.load_state_dict(shared_model.state_dict())
        if not args.shared_model:
            model_critic.load_state_dict(shared_model_critic.state_dict())

        # containers for computing loss
        values = []
        log_probs = []
        rewards = []
        entropies = []

        # reward_sum = 0
        for step in range(args.t_max):
            t += 1

            if args.shared_model:
                value, mu, std = model(normalizer(state))
            else:
                mu, std = model(normalizer(state))
                value = model_critic(normalizer(state))

            dist = torch.distributions.Normal(mu, std)

            # ------------------------------------------
            # # select action
            action = dist.sample()
            # ------------------------------------------
            # Compute statistics for loss
            entropy = dist.entropy()
            log_prob = dist.log_prob(action)

            # make selected move
            state, reward, done, _ = env.step(np.clip(action.detach().numpy(), -args.max_action, args.max_action))
            # TODO: check if this is better
            # reward += (-state[2]+1) * np.exp(- np.abs(state[0]) ** 2)
            episode_reward += reward

            # reward = min(max(-1, reward), 1)

            done = done or t >= args.max_episode_length  # probably don't set terminal state if max_episode length

            with T.get_lock():
                T.value += 1

            if worker_id == 0 and T.value % 500000 and iter_ != 0:
                if lr_scheduler:
                    # TODO improve the call frequency
                    lr_scheduler.step(T.value / 500000)
                if lr_scheduler_critic:
                    # TODO improve the call frequency
                    lr_scheduler_critic.step(T.value / 500000)

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)

            # reset env to beginning if done
            if done:
                t = 0
                state = env.reset()

                # keep track of the avg overall global reward
                with global_reward.get_lock():
                    if global_reward.value == 0:
                        global_reward.value = episode_reward
                    else:
                        global_reward.value = .99 * global_reward.value + 0.01 * episode_reward
                if worker_id == 0:
                    writer.add_scalar("reward/global", global_reward.value, T.value)
                episode_reward = 0

            state = torch.from_numpy(state)

            # end if terminal state or max episodes are reached
            if done:
                break

        G = torch.zeros(1, 1)

        # if non terminal state is present set G to be value of current state
        if not done:
            if args.shared_model:
                v, _, _ = model(normalizer(state))
                G = v.detach()
            else:
                G = model_critic(normalizer(state)).detach()

        values.append(G)
        # compute loss and backprop
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0
        advantages = torch.zeros(1, 1)

        rewards = torch.Tensor(rewards)

        # iterate over rewards from most recent to the starting one
        for i in reversed(range(len(rewards))):
            G = rewards[i] + G * args.gamma
            adv = G - values[i]
            value_loss = value_loss + .5 * adv.pow(2)
            entropy_loss = entropy_loss + entropies[i]
            if args.gae:
                # Generalized Advantage Estimation
                td_error = rewards[i] + args.gamma * values[i + 1] - values[i]
                advantages = advantages * args.gamma * args.tau + td_error
                policy_loss = policy_loss - log_probs[i] * advantages.detach()
            else:
                policy_loss = policy_loss - log_probs[i] * adv.detach()

        # zero grads to reset the gradients
        optimizer.zero_grad()

        if args.shared_model:
            # combined loss for shared architecture
            (policy_loss + args.value_loss_coef * value_loss - args.beta * entropy_loss).mean().backward()
        else:
            optimizer_critic.zero_grad()

            value_loss.mean().backward()
            (policy_loss - args.beta * entropy_loss).mean().backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        sync_grads(model, shared_model)
        optimizer.step()

        if not args.shared_model:
            torch.nn.utils.clip_grad_norm_(model_critic.parameters(), args.max_grad_norm)
            sync_grads(model_critic, shared_model_critic)
            optimizer_critic.step()

        iter_ += 1

        if worker_id == 0:
            temp_T = T.value

            if args.shared_model:
                grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                        for p in model.parameters()
                                        if p.grad is not None])

                writer.add_scalar("grad/mean", np.mean(grads), temp_T)
                writer.add_scalar("grad/l2", np.sqrt(np.mean(np.square(grads))), temp_T)
                writer.add_scalar("grad/max", np.max(np.abs(grads)), temp_T)
                writer.add_scalar("grad/var", np.var(grads), temp_T)
                for param_group in optimizer.param_groups:
                    writer.add_scalar("lr", param_group['lr'], temp_T)
            else:
                grads_critic = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                               for p in model_critic.parameters()
                                               if p.grad is not None])

                grads_actor = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                              for p in model.parameters()
                                              if p.grad is not None])

                writer.add_scalar("grad/actor/mean", np.mean(grads_actor), temp_T)
                writer.add_scalar("grad/actor/l2", np.sqrt(np.mean(np.square(grads_actor))), temp_T)
                writer.add_scalar("grad/actor/max", np.max(np.abs(grads_actor)), temp_T)
                writer.add_scalar("grad/actor/var", np.var(grads_actor), temp_T)

                writer.add_scalar("grad/critic/mean", np.mean(grads_critic), temp_T)
                writer.add_scalar("grad/critic/l2", np.sqrt(np.mean(np.square(grads_critic))), temp_T)
                writer.add_scalar("grad/critic/max", np.max(np.abs(grads_critic)), temp_T)
                writer.add_scalar("grad/critic/var", np.var(grads_critic), temp_T)
                for param_group in optimizer.param_groups:
                    writer.add_scalar("lr/actor", param_group['lr'], temp_T)
                for param_group in optimizer_critic.param_groups:
                    writer.add_scalar("lr/critic", param_group['lr'], temp_T)

            valuelist = [v.data for v in values]

            writer.add_scalar("values/mean", np.mean(valuelist), temp_T)
            writer.add_scalar("values/min", np.min(valuelist), temp_T)
            writer.add_scalar("values/max", np.max(valuelist), temp_T)
            writer.add_scalar("reward/batch", np.mean(np.array(rewards)), temp_T)
            writer.add_scalar("loss/policy", policy_loss, temp_T)
            writer.add_scalar("loss/value", value_loss, temp_T)
            writer.add_scalar("loss/entropy", entropy_loss, temp_T)
