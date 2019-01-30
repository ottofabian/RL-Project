import os

from baselines import bench
from torch.multiprocessing import Value

import numpy as np
import gym
import torch

from typing import Tuple, Union

from torch.nn import Module

from A3C.Models.ActorCriticNetwork import ActorCriticNetwork
from A3C.Models.ActorNetwork import ActorNetwork
from A3C.Models.CriticNetwork import CriticNetwork
from A3C.Optimizers.SharedAdam import SharedAdam
from A3C.Optimizers.SharedRMSProp import SharedRMSProp
from A3C.util.normalizer.base_normalizer import BaseNormalizer
from A3C.util.normalizer.mean_std_normalizer import MeanStdNormalizer


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


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


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


def get_optimizer(optimizer_name: str, shared_model: torch.nn.Module, lr: float,
                  shared_model_critic: torch.nn.Module = None, lr_critic: float = None):
    """

    :param optimizer_name:
    :param shared_model:
    :param lr:
    :param shared_model_critic:
    :param lr_critic:
    :return:
    """
    if optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(shared_model.parameters(), lr=lr)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(shared_model.parameters(), lr=lr)
    else:
        raise ValueError("Selected optimizer is not supported as shared optimizer.")

    if shared_model_critic:
        if optimizer_name == "rmsprop":
            optimizer_critic = torch.optim.RMSprop(shared_model_critic.parameters(), lr=lr_critic)
        elif optimizer_name == "adam":
            optimizer_critic = torch.optim.Adam(shared_model_critic.parameters(), lr=lr_critic)

        return optimizer, optimizer_critic
    return optimizer, None


def get_shared_optimizer(model: Module, optimizer_name: str, lr: float, path=None, model_critic: Module = None,
                         optimizer_name_critic: str = None, lr_critic: float = None) -> Union[
    torch.optim.Optimizer, Tuple[torch.optim.Optimizer, torch.optim.Optimizer]]:
    """
    return optimizer for given model and optional second critic model
    :param model: model to create optimizer for
    :param optimizer_name: which optimizer to use
    :param lr: learning rate for optimizer
    :param path: load saved optimizer for previous run
    :param model_critic: possible separate critic model to load if non shared network is used
    :param optimizer_name_critic: optimizer for separate critic model
    :param lr_critic: learning rate for separate critic model
    :return:
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


def get_normalizer(normalizer_type: str):
    if normalizer_type == "MinMax":
        # TODO fix this to api
        raise NotImplementedError()

    elif normalizer_type == "MeanStd":
        normalizer = MeanStdNormalizer()

    else:
        # this does nothing
        normalizer = BaseNormalizer(read_only=True)

    return normalizer


def make_env(env_id, seed, rank, log_dir=None):
    """
    from https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/component/envs.py
    :param env_id: gym id
    :param seed: seed for env
    :param rank: rank if multiple env are used
    :param log_dir: default log for env
    :return:
    """

    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)

        if log_dir is not None:
            env = bench.Monitor(env=env, filename=os.path.join(log_dir, str(rank)), allow_early_resets=True)

        return env

    return _thunk


def log_to_tensorboard(writer, model, optimizer, rewards, values, loss, policy_loss, value_loss, entropy_loss, iteration,
                       model_critic=None, optimizer_critic=None):
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
