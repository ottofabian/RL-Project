"""
@file: Kernel_C
Created on 10.11.18
@project: RL-Project
@author: queensgambit

Please describe what the content of this file is about
"""
from sklearn.gaussian_process.kernels import Kernel
import numpy as np


class Kernel_C(Kernel):
    """
    White noise.
    """

    def init(self):
        super().init()
        self.num_hyp = lambda x: 1

    def call(self, loghyp, x, z=None):
        loghyp = np.atleast_2d(loghyp)
        n, D = x.shape
        s2 = np.exp(2 * loghyp)  # [E, 1]
        s2 = s2.reshape(-1, 1, 1)

        if z is None:
            K = s2 * np.expand_dims(np.eye(n), 0)
        else:
            K = 0
        return K
