import gym
import numpy as np
import quanser_robots
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from A3C.A3C_LSTM.ActorCriticNetworkLSTM import ActorCriticNetworkLSTM

quanser_robots


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


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
    state = torch.from_numpy(np.array(state))
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

        for step in range(args.num_steps):
            episode_length += 1
            value, mu, sigma, (hx, cx) = model((state, (hx, cx)))

            dist = torch.distributions.Normal(mu, sigma)
            # reparametrization trick through rsample
            action = dist.rsample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            entropies.append(entropy)

            state, reward, done, _ = env.step(action.detach().numpy()[0])
            done = done or episode_length >= args.t_max
            episode_reward += reward

            # reward = max(min(float(reward), 1.), -1.)
            reward = float(reward)

            with T.get_lock():
                T.value += 1

            values.append(value)
            log_probs.append(log_prob)
            # rewards.append((reward + 8) / 8)
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

            state = torch.from_numpy(np.array(state).flatten())

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _, _ = model((state, (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            if args.gae:
                delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
                gae = gae * args.gamma * args.tau + delta_t
                policy_loss = policy_loss - log_probs[i] * gae.detach() - args.beta * entropies[i]
            else:
                policy_loss = policy_loss - log_probs[i] * advantage.detach() - args.beta * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

        iter_ += 1

        grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                for p in model.parameters()
                                if p.grad is not None])

        writer.add_scalar("loss_policy", policy_loss, iter_)
        writer.add_scalar("loss_value", value_loss, iter_)
        writer.add_scalar("values", np.mean([v.data for v in values]), iter_)
        writer.add_scalar("batch_rewards", np.mean(np.array(rewards)), iter_)
        writer.add_scalar("grad_l2", np.sqrt(np.mean(np.square(grads))), iter_)
        writer.add_scalar("grad_max", np.max(np.abs(grads)), iter_)
        writer.add_scalar("grad_variance", np.var(grads), iter_)