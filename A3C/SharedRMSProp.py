import torch.optim as optim


# code from https://github.com/ikostrikov/pytorch-a3c/blob/master/my_optim.py

class SharedRMSProp(optim.RMSprop):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedRMSProp, self).__init__(params, lr, betas, eps, weight_decay)

    def share_memory(self):
        raise NotImplementedError
