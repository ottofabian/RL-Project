"""
@file: A2C.py
@project: RL-Project

Please describe what the content of this file is about
# Adapted from https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On
# & https://github.com/colinskow/move37

"""
import gym
import numpy as np
import ptan
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim
from tensorboardX import SummaryWriter

from A3C.A3C import A3C
from A3C.Models.ActorCriticNetwork import ActorCriticNetwork

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
NUM_ENVS = 50

BELLMAN_STEPS = 4
CLIP_GRAD = 0.1


class A2C(A3C):

    def __init__(self, env_name: str, lr: float = 1e-4, is_discrete: bool = False,
                 seed: int = 123, optimizer_name='rmsprop') -> None:
        """

        :param env_name: Name of the gym environment to use. All available gym environments are supported as well as
                         additional gym environments: https://git.ias.informatik.tu-darmstadt.de/quanser/clients.
        :param lr: Constant learning rate for all workers.
        :param is_discrete: Boolean, indicating if the target variable is discrete or continuous.
                            This setting has effect on the network architecture as well as the loss function used.
                            For more detail see: p.12 - Asynchronous Methods for Deep Reinforcement Learning.pdf
        :param optimizer_name: Optimizer used for shared weight updates. Possible arguments are 'rmsprop', 'adam'.
        """

        super(A2C, self).__init__(1, env_name, lr, is_discrete, seed, optimizer_name)

        self.env = gym.make(self.env_name)
        self.model = ActorCriticNetwork(self.env.observation_space.shape[0], self.env.action_space, self.is_discrete)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-3)
        self.n_steps = BATCH_SIZE #128
        self.max_steps = 100000
        self.discount = GAMMA
        print(self.model)
        self.use_gae = False

        self.value_loss_coef = 1.
        self.tau = 1
        self.beta = ENTROPY_BETA

    def train(self):

        # the tensorboard log dir will be created at ./Experiments/runs
        writer = SummaryWriter() #log_dir='./logs', comment="-a3c-data_" + self.env_name) # + "_" + args.name)

        step_idx = 0

        state = self.env.reset()

        with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:

            while step_idx < self.max_steps:

                states_v = []
                q_vals_v = []
                actions_t = []
                # log_prob_v = []

                # containers for computing loss
                values = []
                log_probs = []
                rewards = []
                entropies = []

                reward_sum = 0

                done = False
                step = 0
                reward_sum = 0

                while step < self.n_steps and done is False:
                    step_idx += 1
                    step += 1

                    if self.is_discrete:
                        # forward pass
                        value, logit = self.model(torch.Tensor(state).unsqueeze(0))

                        # prop dist over actions
                        prob = F.softmax(logit, dim=-1)
                        log_prob = F.log_softmax(logit, dim=-1)

                        # compute entropy for loss regularization
                        entropy = -(log_prob * prob).sum(1, keepdim=True)

                        # choose action based on prob dist
                        action = prob.multinomial(num_samples=1)
                        #action = np.random.choice(range(len(prob.flatten())), p=prob.flatten())
                        log_prob = log_prob.gather(1, action)
                        state, reward, done, _ = self.env.step(int(action.detach().numpy()))

                        #log_prob_v.append(prob[0, action])
                        if step_idx % 1000 == 0:
                            done = False
                            state = self.env.reset()
                            while done is False:

                                # forward pass
                                with torch.no_grad():
                                    if self.is_discrete:
                                        _, logit = self.model(torch.Tensor(state).unsqueeze(0))

                                        # prob dist of action space, select best action
                                        prob = F.softmax(logit, dim=-1)
                                        action = prob.max(1, keepdim=True)[1]

                                action = action.numpy()[0, 0]
                                state, reward, done, _ = self.env.step(action)
                                reward_sum += reward
                                self.env.render()

                        q_val = float(reward)

                        if done is True:
                            q_val = 0
                            state = self.env.reset()
                        #states_v, actions_t, q_vals_v = common.unpack_batch(batch, net, last_val_gamma=GAMMA**BELLMAN_STEPS, device=device)

                        states_v.append(state)
                        q_vals_v.append([q_val])
                        actions_t.append(action)
                        rewards.append(reward)
                        log_probs.append(log_prob)
                        entropies.append(entropy)
                        values.append(value)

                    else:
                        # forward pass
                        value, mu, sigma = self.model(state)
                        print("Training -- mu: {} -- sigma: {}".format(self.worker_id, mu, sigma))

                        # assuming action space is in -high/high
                        high = self.env.action_space.high
                        low = self.env.action_space.low

                        # ------------------------------------------
                        # select action

                        # prob dist over actions
                        prob = torch.distributions.Normal(mu, sigma)
                        # sample during training for exploration
                        action = prob.sample()

                        # avoid sampling outside the allowed range of action_space
                        # action = np.clip(action, low, high)

                        print("Training action: {}".format(action))

                        # ------------------------------------------
                        # Compute statistics for loss

                        # entropy for regularization
                        entropy = prob.entropy()
                        # log prob of action
                        log_prob = prob.log_prob(action)

                R = torch.zeros(1, 1)

                if done is False:
                    R = values[-1]

                states_v = torch.Tensor(np.array(states_v))
                q_vals_v = torch.Tensor(np.array(q_vals_v))
                #actions_t = torch.tensor((np.array(actions_t, dtype=np.long)), dtype=torch.int32)
                #log_prob_v = torch.Tensor(np.array(log_prob_v))

                q_vals_v = q_vals_v + GAMMA * R

                # clear the old gradients
                self.optimizer.zero_grad()

                value_v, logits_v = self.model(torch.Tensor(states_v))
                #logging.debug('value_v: %s' % value_v)
                #logging.debug('q_vals_v: %s' % q_vals_v)
                loss_value_v = F.mse_loss(value_v, q_vals_v)

                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = q_vals_v - value_v.detach()
                log_prob_actions_v = adv_v * log_prob_v[:, actions_t]
                loss_policy_v = -log_prob_actions_v.mean()

                prob_v = F.softmax(logits_v, dim=1)
                entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

                # calculate policy gradients only
                loss_policy_v.backward(retain_graph=True)
                grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                        for p in self.model.parameters()
                                        if p.grad is not None])

                # apply entropy and value gradients
                loss_v = entropy_loss_v + loss_value_v
                loss_v.backward()
                nn_utils.clip_grad_norm_(self.model.parameters(), CLIP_GRAD)
                self.optimizer.step()
                # get full loss
                loss_v += loss_policy_v

                # compute loss and backprop
                #value_loss, policy_loss, combined_loss = self._compute_loss(R, rewards, values, log_probs, entropies)

                #logging.info('Reward: %d' % q_vals_v.sum())
                #logging.debug('loss_value_v: %.3f' % loss_value_v)
                #logging.debug('loss_policy_v: %.3f' % loss_policy_v)

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", q_vals_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_value_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)
                tb_tracker.track("grad_l2", np.sqrt(np.mean(np.square(grads))), step_idx)
                tb_tracker.track("grad_max", np.max(np.abs(grads)), step_idx)
                tb_tracker.track("grad_var", np.var(grads), step_idx)

                #writer.add_scalar("advantage", adv_v[0], step_idx)
                #writer.add_scalar("values", value_v[0], step_idx)
                #writer.add_scalar("batch_rewards", q_vals_v[0], step_idx)
                #writer.add_scalar("loss_entropy", entropy_loss_v, step_idx)

                #writer.add_scalar("value_loss", loss_value_v, step_idx)
                #writer.add_scalar("policy_loss", loss_policy_v, step_idx)
                #writer.add_scalar("combined_loss", loss_v, step_idx)

            writer.close()

    def _compute_loss(self, R: torch.Tensor, rewards: list, values: list, log_probs: list, entropies: list):

        policy_loss = 0
        value_loss = 0

        gae = torch.zeros(1, 1)

        # iterate over rewards from most recent to the starting one
        for i in reversed(range(len(rewards))):
            # clear the old gradients
            self.optimizer.zero_grad()

            R = rewards[i] + R * self.discount
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            #value_loss.backward()

            if self.use_gae:
                # Generalized Advantage Estimation
                delta_t = rewards[i] + self.discount * values[i + 1].data - values[i]
                gae = gae * self.discount * self.tau + delta_t
                policy_loss = policy_loss - log_probs[i] * gae.detach() - self.beta * entropies[i]
            else:
                policy_loss = policy_loss + log_probs[i] * advantage - self.beta * entropies[i]
                #policy_loss.backward()

        # compute combined loss of policy_loss and value_loss
        # avoid overfitting on value loss by scaling it down
        combined_loss = policy_loss + self.value_loss_coef * value_loss
        combined_loss.backward() #.mean()
        # combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP_GRAD)

        self.optimizer.step()

        return value_loss, policy_loss, combined_loss
