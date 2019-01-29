import os
from torch.multiprocessing import Value

import gym
import torch

from typing import Tuple, Union, List

from torch.nn import Module

from A3C.Models.ActorCriticNetwork import ActorCriticNetwork
from A3C.Models.ActorNetwork import ActorNetwork
from A3C.Models.CriticNetwork import CriticNetwork
from A3C.Optimizers.SharedAdam import SharedAdam
from A3C.Optimizers.SharedRMSProp import SharedRMSProp
from A3C.util.normalizer.base_normalizer import BaseNormalizer
from A3C.util.normalizer.mean_std_normalizer import MeanStdNormalizer


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


def get_optimizer(model: Module, optimizer_name: str, lr: float, path=None, model_critic: Module = None,
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
