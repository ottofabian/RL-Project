import math

import gym
import numpy as np
import quanser_robots
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from A3C.A3C_LSTM.ActorCriticNetworkLSTM import ActorCriticNetworkLSTM

quanser_robots


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


pi = Variable(torch.Tensor([math.pi])).float()


def normal(x, mu, variance):
    a = (-1 * (x - mu).pow(2) / (2 * variance)).exp()
    b = 1 / (2 * variance * pi.expand_as(variance)).sqrt()
    return a * b


def train(rank, args, shared_model, T, global_reward, optimizer=None):
    torch.manual_seed(args.seed + rank)

    env = gym.make(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCriticNetworkLSTM(n_inputs=env.observation_space.shape[0], action_space=env.action_space,
                                   n_hidden=args.n_hidden)
    if optimizer is None:
        if args.optimizer == "rmsprop":
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        elif args.optimizer == "adam":
            optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    writer = SummaryWriter()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    iter_ = 0
    episode_length = 0
    episode_reward = 0
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())

        if done:
            cx = torch.zeros(1, args.n_hidden)
            hx = torch.zeros(1, args.n_hidden)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.t_max):
            episode_length += 1
            value, mu, variance, (hx, cx) = model((state, (hx, cx)))
            mu = torch.clamp(mu, -5., 5.)

            # dist = torch.distributions.Normal(mu, variance)

            # reparametrization trick through rsample
            eps = Variable(torch.randn(mu.size()))
            action = (mu + variance.sqrt() * eps).detach()
            # action = dist.rsample().detach()

            # ------------------------------------------
            # Compute statistics for loss

            entropy = .5 * ((variance * 2 * pi.expand_as(variance)).log() + 1)
            # print("entropy:", entropy, dist.entropy())
            entropies.append(entropy)

            prob = normal(Variable(action), mu, variance)
            log_prob = (prob + 1e-6).log()
            entropies.append(entropy)

            state, reward, done, _ = env.step(action.detach().numpy().flatten())
            done = done or episode_length >= args.max_episode_length
            episode_reward += reward

            reward = max(min(float(reward), 1.), -1.)
            # reward = float(reward)

            with T.get_lock():
                T.value += 1

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                episode_length = 0
                state = env.reset()

                # keep track of the avg overall global reward
                with global_reward.get_lock():
                    if global_reward.value == 0:
                        global_reward.value = episode_reward
                    else:
                        global_reward.value = .99 * global_reward.value + 0.01 * episode_reward
                    writer.add_scalar("global_reward", global_reward.value, iter_)
                episode_reward = 0

            state = torch.from_numpy(state).view(1, env.observation_space.shape[0])

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _, _ = model((state, (hx, cx)))
            R = value.detach()

        R = Variable(R)
        values.append(R)
        policy_loss = 0
        critic_loss = 0
        gae = torch.zeros(1, 1)
        # iterate over rewards from most recent to the starting one
        for i in reversed(range(len(rewards))):
            R = rewards[i] + R * args.gamma
            advantage = R - values[i]
            critic_loss = critic_loss + 0.5 * advantage.pow(2)
            if args.gae:
                # Generalized Advantage Estimation
                delta_t = rewards[i] + args.gamma * values[i + 1].data - values[i].data
                gae = gae * args.gamma * args.tau + delta_t
                policy_loss = policy_loss - log_probs[i] * Variable(gae) - args.beta * entropies[i]
            else:
                policy_loss = policy_loss - log_probs[i] * advantage.data - args.beta * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * critic_loss).mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

        iter_ += 1

        grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                for p in model.parameters()
                                if p.grad is not None])

        writer.add_scalar("loss_policy", policy_loss, iter_)
        writer.add_scalar("loss_value", critic_loss, iter_)
        writer.add_scalar("values", np.mean([v.data for v in values]), iter_)
        writer.add_scalar("batch_rewards", np.mean(np.array(rewards)), iter_)
        writer.add_scalar("grad_l2", np.sqrt(np.mean(np.square(grads))), iter_)
        writer.add_scalar("grad_max", np.max(np.abs(grads)), iter_)
        writer.add_scalar("grad_variance", np.var(grads), iter_)
