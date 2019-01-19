import time

import gym
import numpy as np
import quanser_robots
import torch
from torch.autograd import Variable

from A3C.A3C_LSTM.ActorCriticNetworkLSTM import ActorCriticNetworkLSTM

quanser_robots


def test(rank, args, shared_model, T, global_reward):
    torch.manual_seed(args.seed + rank)

    env = gym.make(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCriticNetworkLSTM(n_inputs=env.observation_space.shape[0], action_space=env.action_space,
                                   n_hidden=args.n_hidden)
    model.eval()

    start_time = time.time()

    while True:

        reward_sum = np.zeros(10)
        episode_length = np.zeros(10)

        # make 10 test runs
        for i in range(10):

            state = env.reset()
            state = torch.from_numpy(state)
            done = False

            # Sync with the shared model
            model.load_state_dict(shared_model.state_dict())

            cx = torch.zeros(1, args.n_hidden)
            hx = torch.zeros(1, args.n_hidden)

            # run until termination
            while not done:

                # only render first run
                if i == 0:
                    env.render()

                episode_length[i] += 1

                cx = cx.detach()
                hx = hx.detach()

                with torch.no_grad():
                    _, mu, _, (hx, cx) = model((Variable(state), (hx, cx)))

                action = mu

                state, reward, done, _ = env.step(action.numpy().flatten())
                done = done or episode_length[i] >= args.max_episode_length
                reward_sum[i] += reward

                state = torch.from_numpy(state)

        print(
            "Total training time: {} -- T={}, FPS={:.0f} -- "
            "mean episode reward: {:4.5f} -- mean episode length: {:4.0f} -- global reward: {:4.5f}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), T.value,
                T.value / (time.time() - start_time), np.mean(reward_sum), np.mean(episode_length),
                global_reward.value))
        time.sleep(10)
