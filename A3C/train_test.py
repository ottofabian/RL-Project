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
from A3C.Models.ActorNetwork import ActorNetwork
from A3C.Models.CriticNetwork import CriticNetwork
from A3C.Worker import save_checkpoint
from tensorboardX import SummaryWriter
from Experiments.util.MinMaxScaler import MinMaxScaler


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


def normal(x, mu, variance):
    pi = Variable(torch.Tensor([math.pi])).float()

    a = (-1 * (x - mu).pow(2) / (2 * variance)).exp()
    b = 1 / (2 * variance * pi.expand_as(variance)).sqrt()
    return a * b


def test(args, worker_id: int, shared_model_actor: ActorNetwork, shared_model_critic: CriticNetwork,
         T: Value, optimizer_actor: Optimizer = None,
         optimizer_critic: Optimizer = None, global_reward: Value = None, min_max_scaler: MinMaxScaler = None):
    """
    Start worker in _test mode, i.e. no training is done, only testing is used to validate current performance
    loosely based on https://github.com/ikostrikov/pytorch-a3c/blob/master/_test.py
    :param args:
    :param worker_id:
    :param shared_model_actor:
    :param shared_model_critic:
    :param T:
    :param optimizer_actor:
    :param optimizer_critic:
    :param global_reward:
    :param min_max_scaler:
    :return:
    """
    torch.manual_seed(args.seed + worker_id)

    if "RR" in args.env_name:
        env = quanser_robots.GentlyTerminating(gym.make(args.env_name))
    else:
        env = gym.make(args.env_name)

    env.seed(args.seed + worker_id)

    # get an instance of the current global model state
    model_actor = ActorNetwork(n_inputs=env.observation_space.shape[0], n_outputs=env.action_space.shape[0],
                               max_action=args.max_action)
    model_critic = CriticNetwork(n_inputs=env.observation_space.shape[0])

    model_actor.eval()
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
        model_critic.load_state_dict(shared_model_critic.state_dict())
        model_actor.load_state_dict(shared_model_actor.state_dict())

        rewards = []
        eps_len = []

        # make 10 runs to get current avg performance
        for i in range(10):
            while not done:
                t += 1

                if i == 0 and t % 1 == 0:
                    env.render()

                # apply min/max scaling on the environment
                if min_max_scaler:
                    state_normalized = min_max_scaler.normalize_state(state)
                else:
                    state_normalized = state

                with torch.no_grad():

                    # select mean of normal dist as action --> Expectation
                    mu, _ = model_actor(state_normalized)
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
        writer.add_scalar("mean_test_reward", rewards, int(T.value))

        if best_global_reward is None or global_reward.value > best_global_reward:
            best_global_reward = global_reward.value
            save_checkpoint({
                'epoch': T.value,
                'state_dict': shared_model_actor.state_dict(),
                'global_reward': global_reward.value,
                'optimizer': optimizer_actor.state_dict() if optimizer_actor is not None else None,
            }, filename=f"./checkpoints/actor_T-{T.value}_global-{global_reward.value}.pth.tar")

            save_checkpoint({
                'epoch': T.value,
                'state_dict': shared_model_critic.state_dict(),
                'global_reward': global_reward.value,
                'optimizer': optimizer_critic.state_dict() if optimizer_critic is not None else None,
            }, filename=f"./checkpoints/critic_T-{T.value}_global-{global_reward.value}.pth.tar")

        # delay _test run for 10s to give the network some time to train
        time.sleep(10)
        iter_ += 1


def train(args, worker_id: int, shared_model_actor: ActorNetwork, shared_model_critic: CriticNetwork,
          T: Value,
          optimizer_actor: Optimizer = None,
          optimizer_critic: Optimizer = None, scheduler_actor: torch.optim.lr_scheduler = None,
          scheduler_critic: torch.optim.lr_scheduler = None,
          global_reward=None, min_max_scaler: MinMaxScaler = None):
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

    # init local NN instance for worker thread
    model_actor = ActorNetwork(n_inputs=env.observation_space.shape[0], n_outputs=env.action_space.shape[0],
                               max_action=args.max_action)
    model_critic = CriticNetwork(n_inputs=env.observation_space.shape[0])

    # if no shared optimizer is provided use individual one
    if optimizer_actor is None:
        if args.optimizer == "rmsprop":
            optimizer_actor = torch.optim.RMSprop(shared_model_actor.parameters(), lr=args.lr_actor)
        elif args.optimizer == "adam":
            optimizer_actor = torch.optim.Adam(shared_model_actor.parameters(), lr=args.lr_actor)

    if optimizer_critic is None:
        if args.optimizer == "rmsprop":
            optimizer_critic = torch.optim.RMSprop(shared_model_critic.parameters(), lr=args.lr_critic)
        elif args.optimizer == "adam":
            optimizer_critic = torch.optim.Adam(shared_model_critic.parameters(), lr=args.lr_critic)

    model_actor.train()
    model_critic.train()

    state = torch.from_numpy(env.reset())

    t = 0
    iter_ = 0
    episode_reward = 0

    if worker_id == 0:
        writer = SummaryWriter()

    while True:
        # Get state of the global model
        model_critic.load_state_dict(shared_model_critic.state_dict())
        model_actor.load_state_dict(shared_model_actor.state_dict())

        # containers for computing loss
        values = []
        log_probs = []
        rewards = []
        entropies = []

        # reward_sum = 0
        for step in range(args.t_max):
            t += 1

            # apply min/max scaling on the environment
            if min_max_scaler:
                state_normalized = min_max_scaler.normalize_state(state)
            else:
                state_normalized = state

            value = model_critic(state_normalized)
            mu, variance = model_actor(state_normalized)

            dist = torch.distributions.Normal(mu, variance)

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
                if scheduler_actor is not None:
                    # TODO improve the call frequency
                    scheduler_actor.step(T.value / 500000)
                if scheduler_critic is not None:
                    # TODO improve the call frequency
                    scheduler_critic.step(T.value / 500000)

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
                    writer.add_scalar("global_reward", global_reward.value, iter_)
                episode_reward = 0

            state = torch.from_numpy(state)

            # end if terminal state or max episodes are reached
            if done:
                break

        G = torch.zeros(1, 1)

        # if non terminal state is present set G to be value of current state
        if not done:
            G = model_critic(state).detach()

        values.append(G)
        # compute loss and backprop
        policy_loss = 0
        value_loss = 0
        advantages = torch.zeros(1, 1)

        rewards = torch.Tensor(rewards)

        # iterate over rewards from most recent to the starting one
        for i in reversed(range(len(rewards))):
            G = rewards[i] + G * args.gamma
            adv = G - values[i]
            value_loss = value_loss + .5 * adv.pow(2)
            entropy_loss = args.beta * entropies[i]
            if args.gae:
                # Generalized Advantage Estimation
                td_error = rewards[i] + args.gamma * values[i + 1] - values[i]
                advantages = advantages * args.gamma * args.tau + td_error
                policy_loss = policy_loss - log_probs[i] * advantages.detach()
            else:
                policy_loss = policy_loss - log_probs[i] * adv.detach()

            policy_loss = policy_loss - entropy_loss

        # zero grads to avoid computation issues in the next step
        optimizer_critic.zero_grad()
        optimizer_actor.zero_grad()

        value_loss.mean().backward()
        policy_loss.mean().backward()

        # compute combined loss of policy_loss and value_loss
        # avoid overfitting on value loss by scaling it down
        # (policy_loss + value_loss_coef * value_loss).mean().backward()

        torch.nn.utils.clip_grad_norm_(model_critic.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(model_actor.parameters(), args.max_grad_norm)

        sync_grads(model_critic, shared_model_critic)
        sync_grads(model_actor, shared_model_actor)

        optimizer_critic.step()
        optimizer_actor.step()

        iter_ += 1

        if worker_id == 0:
            grads_critic = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                           for p in model_critic.parameters()
                                           if p.grad is not None])

            grads_actor = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                          for p in model_actor.parameters()
                                          if p.grad is not None])

            valuelist = [v.data for v in values]

            writer.add_scalar("mean_values", np.mean(valuelist), iter_)
            writer.add_scalar("min_values", np.min(valuelist), iter_)
            writer.add_scalar("max_values", np.max(valuelist), iter_)
            writer.add_scalar("batch_rewards", np.mean(np.array(rewards)), iter_)
            writer.add_scalar("loss_policy", policy_loss, iter_)
            writer.add_scalar("loss_value", value_loss, iter_)
            writer.add_scalar("grad_l2_actor", np.sqrt(np.mean(np.square(grads_actor))), iter_)
            writer.add_scalar("grad_max_actor", np.max(np.abs(grads_actor)), iter_)
            writer.add_scalar("grad_var_actor", np.var(grads_actor), iter_)
            writer.add_scalar("grad_l2_critic", np.sqrt(np.mean(np.square(grads_critic))), iter_)
            writer.add_scalar("grad_max_critic", np.max(np.abs(grads_critic)), iter_)
            writer.add_scalar("grad_var_critic", np.var(grads_critic), iter_)
            for param_group in optimizer_actor.param_groups:
                writer.add_scalar("lr_actor", param_group['lr'], iter_)
            for param_group in optimizer_critic.param_groups:
                writer.add_scalar("lr_critic", param_group['lr'], iter_)
