import os

import torch


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def load_saved_model(model, path, T, global_reward, optimizer=None):
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        T.value = checkpoint['epoch']
        global_reward.value = checkpoint['global_reward']
        print("=> loaded checkpoint '{}' (T: {} -- global reward: {})"
              .format(path, checkpoint['epoch'], checkpoint['global_reward']))
    else:
        print("=> no checkpoint found at '{}'".format(path))
