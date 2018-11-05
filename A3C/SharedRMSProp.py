import torch.optim as optim


# code from https://github.com/ikostrikov/pytorch-a3c/blob/master/my_optim.py

class SharedRMSProp(optim.RMSprop):

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        super(SharedRMSProp, self).__init__(params, lr, alpha, eps, weight_decay, momentum, centered)

    def share_memory(self):
        raise NotImplementedError
