"""
@file: SharedAdam.py

Definition of the Adam optimizer for shared usage.
One example for shared usage is multiprocessing for example.
The class inherits the functionality from the default pytorch Adam implementation and calls .share_memory_()
"""

import torch


class SharedAdam(torch.optim.Adam):

        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                     weight_decay=0, amsgrad=False):

            # call the parent class with the Adam parameter settings
            super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                             amsgrad=amsgrad)

            # initialize the states
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                    # call share memory for all state paramters
                    state['exp_avg'].share_memory_()
                    state['exp_avg_sq'].share_memory_()
