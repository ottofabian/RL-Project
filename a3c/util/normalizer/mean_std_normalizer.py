#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
from baselines.common.running_mean_std import RunningMeanStd

from a3c.util.normalizer.base_normalizer import BaseNormalizer
import torch


class MeanStdNormalizer(BaseNormalizer):
    def __init__(self, read_only=False, clip=10.0, epsilon=1e-8):

        super().__init__(read_only)
        self.rms = None
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, x):
        x = np.asarray(x)
        if self.rms is None:
            self.rms = RunningMeanStd(shape=(1,) + x.shape[1:])
        if not self.read_only:
            self.rms.update(x)
        return torch.from_numpy(np.clip((x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon),
                                        -self.clip, self.clip))
