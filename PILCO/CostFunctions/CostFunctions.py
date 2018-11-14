"""
@file: CostFunctions

Defines different costfunctions for several environments.
These functions are used for the policy optimization in PILCO.
"""

import numpy as np


def cartpolebase_costfunc(x, a):
    """
    Returns the reward taken an action from a specific state

    Extracted CartpoleBase.from CartpoleBase._rmd()
    https://git.ias.informatik.tu-darmstadt.de/quanser/clients/blob/master/quanser_robots/cartpole/base.py
    :param x: Environment state
    :param a: Taken action
    :return: Reward (float)
    """
    _, sin_th, cos_th, _, _ = x
    # th = .5 * (np.arcsin(sin_th) + np.arccos(cos_th))
    # rwd = -np.cos(th)
    rwd = -cos_th

    return np.float32(rwd)
