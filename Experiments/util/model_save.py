import torch


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
