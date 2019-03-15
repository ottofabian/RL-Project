import argparse
import os

from baselines import bench
from tensorboardX import SummaryWriter
from torch.multiprocessing import Value

import numpy as np
import gym
import torch

from typing import Tuple, Union, List

from torch.nn import Module

from a3c.models.actor_critic_network import ActorCriticNetwork
from a3c.models.actor_network import ActorNetwork
from a3c.models.critic_network import CriticNetwork
from a3c.optimizers.shared_adam import SharedAdam
from a3c.optimizers.shared_rmsprop import SharedRMSProp
from a3c.util.normalizer.base_normalizer import BaseNormalizer
from a3c.util.normalizer.mean_std_normalizer import MeanStdNormalizer


def sync_grads(model: ActorCriticNetwork, global_model: ActorCriticNetwork) -> None:
    """
    This method synchronizes the grads of the local network with the global network.
    :return:
    :param model: local worker model
    :param global_model: shared global model
    :return:
    """
    for param, global_param in zip(model.parameters(), global_model.parameters()):
        if global_param.grad is not None:
            return
        global_param._grad = param.grad  #


def save_checkpoint(state: dict, path='./experiments/checkpoint.pth.tar') -> None:
    """
    save model checkpoint.
    Example of checkpoint dict:
    {
        'epoch': T.value,
        'model': model.state_dict(),
        'model_critic': model_critic.state_dict(),
        'global_reward': global_reward.value,
        'optimizer': optimizer.state_dict(),
        'optimizer_critic': optimizer_critic.state_dict()
    }
    :param state: dict of checkpoint info
    :param path: path to save the file to
    :return: None
    """
    torch.save(state, path)


def load_saved_optimizer(optimizer: torch.optim.Optimizer, path: str, optimizer_critic: torch.optim.Optimizer = None):
    """
    load optimizer statistics
    :param optimizer: model to load params for
    :param path: path to load parameters from
    :param optimizer_critic: possible separate critic optimizer to load if non shared network is used
    :return:
    """
    if os.path.isfile(path):
        print(f"=> loading optimizer checkpoint '{path}'")
        checkpoint = torch.load(path)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if optimizer_critic:
            optimizer_critic.load_state_dict(checkpoint['optimizer_critic'])
        print(
            f"=> loaded optimizer checkpoint '{path}' (T: {checkpoint['epoch']} "
            f"-- global reward: {checkpoint['global_reward']})")
    else:
        print(f"=> no optimizer checkpoint found at '{path}'")


def load_saved_model(model: Module, path: str, T: Value, global_reward: Value, model_critic: Module = None) -> None:
    """
    load saved model from file
    :param model: model to load params for
    :param path: path to load parameters from
    :param T: global steps counter, to continue training
    :param model_critic: possible separate critic model to load if non shared network is used
    :return: None
    """
    if os.path.isfile(path):
        print(f"=> loading model checkpoint '{path}'")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        T.value = checkpoint['epoch']
        global_reward.value = checkpoint['global_reward']
        if model_critic:
            model_critic.load_state_dict(checkpoint['model_critic'])
        print(
            f"=> loaded model checkpoint '{path}' (T: {checkpoint['epoch']} "
            f"-- global reward: {checkpoint['global_reward']})")
    else:
        print(f"=> no model checkpoint found at '{path}'")


def get_model(env: gym.Env, shared: bool = False, path: str = None, T: Value = None,
              global_reward: Value = None) -> Union[Tuple[Module, Module], Module]:
    """
    return either one shared model or two separate networks
    :param env: gym environment to determine in- and outputs
    :param shared: use shared model or two separate actor and critic models
    :param path: path to load model from or List of paths for non shared models [actor_path, value_path]
    :param T: global steps counter, to continue training
    :param global_reward: global reward counter to continue training
    :return: pytorch modules for either one shared model or two separate actor and critic models
    """
    if shared:
        shared_model = ActorCriticNetwork(n_inputs=env.observation_space.shape[0], n_actions=env.action_space.shape[0])
        shared_model.share_memory()

        # load model if path specified
        if path is not None:
            load_saved_model(shared_model, path, T, global_reward)

        return shared_model
    else:
        shared_model_critic = CriticNetwork(n_inputs=env.observation_space.shape[0])
        shared_model_actor = ActorNetwork(n_inputs=env.observation_space.shape[0], n_actions=env.action_space.shape[0])

        shared_model_critic.share_memory()
        shared_model_actor.share_memory()

        # load model if path specified
        if path is not None:
            load_saved_model(shared_model_actor, path, T, global_reward, model_critic=shared_model_critic)

        return shared_model_actor, shared_model_critic


def get_optimizer(optimizer_name: str, model: torch.nn.Module, lr: float, model_critic: torch.nn.Module = None,
                  lr_critic: float = None) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """
    return optimizer for given model and optional second critic model without shared statistics
    :param model: model to create optimizer for
    :param optimizer_name: which optimizer to use
    :param lr: learning rate for optimizer
    :param model_critic: possible separate critic model to load if non shared network is used
    :param lr_critic: learning rate for separate critic model
    :return: Optimizer instance or tuple of two optimizers, 2nd instance is None or critic_optimizer
    """
    if optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("Selected optimizer is not supported as shared optimizer.")

    if model_critic:
        if optimizer_name == "rmsprop":
            optimizer_critic = torch.optim.RMSprop(model_critic.parameters(), lr=lr_critic)
        elif optimizer_name == "adam":
            optimizer_critic = torch.optim.Adam(model_critic.parameters(), lr=lr_critic)

        return optimizer, optimizer_critic
    return optimizer, None


def get_shared_optimizer(model: Module, optimizer_name: str, lr: float, path=None, model_critic: Module = None,
                         optimizer_name_critic: str = None, lr_critic: float = None) \
        -> Union[torch.optim.Optimizer, Tuple[torch.optim.Optimizer, torch.optim.Optimizer]]:
    """
    get optimizer for given model and optional second critic model with shared statistics
    :param model: model to create optimizer for
    :param optimizer_name: which optimizer to use
    :param lr: learning rate for optimizer
    :param path: load saved optimizer for previous run
    :param model_critic: possible separate critic model to load if non shared network is used
    :param optimizer_name_critic: optimizer for separate critic model
    :param lr_critic: learning rate for separate critic model
    :return: Optimizer instance or tuple of two optimizers
    """

    if optimizer_name == 'rmsprop':
        optimizer = SharedRMSProp(model.parameters(), lr=lr)

    elif optimizer_name == 'adam':
        optimizer = SharedAdam(model.parameters(), lr=lr)

    else:
        raise ValueError("Selected optimizer is not supported as shared optimizer.")

    optimizer.share_memory()

    optimizer_critic = None

    if model_critic:
        if optimizer_name_critic == 'rmsprop':
            optimizer_critic = SharedRMSProp(model_critic.parameters(), lr=lr_critic)

        elif optimizer_name_critic == 'adam':
            optimizer_critic = SharedAdam(model_critic.parameters(), lr=lr_critic)

        else:
            raise ValueError("Selected optimizer is not supported as shared optimizer.")

    if path is not None:
        load_saved_optimizer(optimizer, path, optimizer_critic)

    if model_critic:
        return optimizer, optimizer_critic

    return optimizer


def get_normalizer(normalizer_type: str) -> BaseNormalizer:
    """
    returns normalizer instance based on specified type
    :param normalizer_type: string of normalizer instance
    :return: BaseNormalizer instance
    """
    if normalizer_type == "MinMax":
        normalizer = MinMaxNormalizer()

    elif normalizer_type == "MeanStd":
        normalizer = MeanStdNormalizer()

    else:
        # this does nothing
        normalizer = BaseNormalizer(read_only=True)

    return normalizer


def make_env(env_id: str, seed: int, rank: int, log_dir=None) -> callable:
    """
    returns callable to create gym environment or monitor
    from https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/component/envs.py
    :param env_id: gym id
    :param seed: seed for env
    :param rank: rank if multiple env are used
    :param log_dir: default log for env, results in returning a monitor instance
    :return: callable
    """

    def _get_env():
        env = gym.make(env_id)
        env.seed(seed + rank)

        if log_dir is not None:
            env = bench.Monitor(env=env, filename=os.path.join(log_dir, str(rank)), allow_early_resets=True)

        return env

    return _get_env


def log_to_tensorboard(writer: SummaryWriter, model: Module, optimizer: torch.optim.Optimizer,
                       rewards: List[torch.Tensor], values: List[torch.Tensor], loss: float, policy_loss: float,
                       value_loss: float, entropy_loss: float, iteration: int, model_critic: Module = None,
                       optimizer_critic: torch.optim.Optimizer = None) -> None:
    """
    log training info to tensorboard
    :param writer: tensorboard writer
    :param model: current global model/ for split model: actor model
    :param optimizer: current optimizer/ for split model: actor optimizer
    :param rewards: list of tensor rewards of last test run
    :param values: list of tensor values of last test run
    :param loss: combined loss value
    :param policy_loss: policy loss value
    :param value_loss: value loss value
    :param entropy_loss: entropy loss value
    :param iteration: current iteration to log for (best use global counter T)
    :param model_critic: optional critic model for split architecture
    :param optimizer_critic: optional critic optimizer for split architecture
    :return: None
    """
    if not model_critic:
        grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                for p in model.parameters()
                                if p.grad is not None])

        writer.add_scalar("grad/mean", np.mean(grads), iteration)
        writer.add_scalar("grad/l2", np.sqrt(np.mean(np.square(grads))), iteration)
        writer.add_scalar("grad/max", np.max(np.abs(grads)), iteration)
        writer.add_scalar("grad/var", np.var(grads), iteration)
        for param_group in optimizer.param_groups:
            writer.add_scalar("lr", param_group['lr'], iteration)
    else:
        grads_critic = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                       for p in model_critic.parameters()
                                       if p.grad is not None])

        grads_actor = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                      for p in model.parameters()
                                      if p.grad is not None])

        writer.add_scalar("grad/actor/mean", np.mean(grads_actor), iteration)
        writer.add_scalar("grad/actor/l2", np.sqrt(np.mean(np.square(grads_actor))), iteration)
        writer.add_scalar("grad/actor/max", np.max(np.abs(grads_actor)), iteration)
        writer.add_scalar("grad/actor/var", np.var(grads_actor), iteration)

        writer.add_scalar("grad/critic/mean", np.mean(grads_critic), iteration)
        writer.add_scalar("grad/critic/l2", np.sqrt(np.mean(np.square(grads_critic))), iteration)
        writer.add_scalar("grad/critic/max", np.max(np.abs(grads_critic)), iteration)
        writer.add_scalar("grad/critic/var", np.var(grads_critic), iteration)
        for param_group in optimizer.param_groups:
            writer.add_scalar("lr/actor", param_group['lr'], iteration)
        for param_group in optimizer_critic.param_groups:
            writer.add_scalar("lr/critic", param_group['lr'], iteration)

    valuelist = [v.detach().numpy() for v in values]

    writer.add_scalar("values/mean", np.mean(valuelist), iteration)
    writer.add_scalar("values/min", np.min(valuelist), iteration)
    writer.add_scalar("values/max", np.max(valuelist), iteration)
    writer.add_scalar("reward/batch", np.mean(np.array([r.detach().numpy() for r in rewards])), iteration)
    writer.add_scalar("loss", loss, iteration)
    writer.add_scalar("loss/policy", policy_loss, iteration)
    writer.add_scalar("loss/value", value_loss, iteration)
    writer.add_scalar("loss/entropy", entropy_loss, iteration)


def shape_reward(args, reward: np.ndarray):
    """
    Optional reward shaping

    :param args: Cmd-line parameter
    :param reward: Reward vector
    :return:
    """

    if args.squared_reward:
        # use quadratic of reward
        # for idx, r in enumerate(reward):
        #     reward[idx] *= reward[idx]
        reward = reward ** 2

    if args.scale_reward:
        reward *= args.scale_reward

    return reward


def parse_args(args: list) -> argparse.Namespace:
    """
    parse console arguments
    :param args: console arguments
    :return: parsed consoled arguments
    """
    parser = argparse.ArgumentParser(description='a3c')
    parser.add_argument('--env-name', default='CartpoleStabShort-v0',
                        help='Name of the gym environment to use. '
                             'All environments based on OpenAI\'s gym are supported. '
                             '(default: CartpoleStabShort-v0)')
    parser.add_argument('--rollout-steps', type=int, default=50,
                        help='Number of forward steps for n-step return. (default: 50)')
    parser.add_argument('--max-action', type=float, default=None,
                        help='Maximum allowed action to use,'
                             'if None the full available action range is used. (default: None)')
    parser.add_argument('--shared-model', default=False, action='store_true',
                        help='Use shared network for actor and critic. (default: False)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate for shared model or actor network. (default: 1e-4)')
    parser.add_argument('--lr-critic', type=float, default=1e-3,
                        help='Separate critic learning rate, if no shared network is used. (default: 1e-3)')
    parser.add_argument('--value-loss-weight', type=float, default=0.5,
                        help='Value loss coefficient, which defines the weighting between value and policy loss '
                             'for shared model. (default: 0.5)')
    parser.add_argument('--discount', type=float, default=0.99,
                        help='Discount factor for rewards. (default: 0.99)')
    parser.add_argument('--no-gae', default=False, action='store_true',
                        help='Disable general advantage estimation. (default: False)')
    parser.add_argument('--tau', type=float, default=0.99,
                        help='Adjusts the bias-variance tradeoff for GAE. (default: 0.99)')
    parser.add_argument('--entropy-loss-weight', type=float, default=1e-4,
                        help='Entropy term coefficient. (default: 1e-4)')
    parser.add_argument('--max-grad-norm', type=float, default=1,
                        help='Maximum gradient norm. (default: 1)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed. (default: 1)')
    parser.add_argument('--worker', type=int, default=1,
                        help='Number of training workers/threads to use. If set to 1, A2C is used. '
                             'For >1 A3C is used with the specified number of workers. (default: 1)')
    parser.add_argument('--n-envs', type=int, default=5,
                        help='Number of environment for A2C (--worker 1) in one batch. (default: 5)')
    parser.add_argument('--max-episode-length', type=int, default=5000,
                        help='Maximum length of an episode. (default: 5000)')
    parser.add_argument('--no-shared-optimizer', default=False, action='store_true',
                        help='Use optimizer with shared statistics. (default: False)')
    parser.add_argument('--optimizer', type=str, default="adam",
                        help='Type of optimizer, supported: [rmsprop, adam]. (default: adam)')
    parser.add_argument('--lr-scheduler', type=str, default=None,
                        help='Type of learning rate scheduler to use, supported: [None, exponential]. (default: None)')
    parser.add_argument('--lr-scheduler-step', type=int, default=50000,
                        help='Number of steps before lr decay with gamma .99, (default: 50000)')
    parser.add_argument('--normalizer', type=str, default=None,
                        help='Type of normalizer, supported: [None, MeanStd, MinMax]. (default: None)')
    parser.add_argument('--test', default=False, action='store_true',
                        help='Start run without training and evaluate for number of --test-runs (default: False)')
    parser.add_argument('--test-runs', type=int, default=10,
                        help='Number of test evaluation runs during training or in test mode (default: 10)')
    parser.add_argument('--path', type=str, default=None,
                        help='Weight location for the models to load. (default: None)')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory for monitor logging of each environment. (default: None)')
    parser.add_argument('--no-log', default=False, action='store_true',
                        help='Avoid exporting a log file to the log directory. (default: False)')
    parser.add_argument('--log-frequency', type=int, default=100,
                        help='Defines how often a sample is logged to tensorboard to avoid unnecessary bloating. '
                             'If set to X every X metric sample will be logged. (default: 100)')
    parser.add_argument('--squared-reward', default=False, action='store_true',
                        help='Manipulates the reward by squaring it.'
                             'This reward shaping is meant for evaluation and not part of the original environment. '
                             '(default: False)')
    parser.add_argument('--scale-reward', type=float, default=None,
                        help='Multiply reward by specified factor. '
                             'This reward shaping is meant for evaluation and not part of the original environment. '
                             '(default: None)')
    parser.add_argument('--no-render', default=False, action='store_true',
                        help='Disables rendering. (default: False)')
    parser.add_argument('--monitor', default=False, action='store_true',
                        help='Enables monitoring with video capturing of the test worker. (default: False)')

    args = parser.parse_args(args)

    # create dummy_env for parameter check
    dummy_env = gym.make(args.env_name)

    if not args.max_action:
        # define default values for missing parameters
        args.max_action = dummy_env.action_space.high[0]

    # always close gym environments if they aren't used anymore
    dummy_env.close()

    return args
