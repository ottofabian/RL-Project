import os

import torch


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def load_saved_model(model, path, optimizer=None):
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(path))
