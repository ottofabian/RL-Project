"""
@file: CostFunctions

Defines different costfunctions for several environments.
These functions are used for the policy optimization in PILCO.
"""

import numpy as np


def cartpolebase_rewardfunc(x, a):
    """
    Returns the reward taken an action from a specific state

    Extracted CartpoleBase.from CartpoleBase._rmd()
    https://git.ias.informatik.tu-darmstadt.de/quanser/clients/blob/master/quanser_robots/cartpole/base.py
    :param x: Environment state
    :param a: Taken action
    :return: Reward (float)
    """
    _, _, cos_th, _, _ = x
    rwd = -cos_th

    return np.float32(rwd)


def cartpolebase_costfunc(x, a):
    """
    Returns the reward taken an action from a specific state

    Extracted CartpoleBase.from CartpoleBase._rmd()
    https://git.ias.informatik.tu-darmstadt.de/quanser/clients/blob/master/quanser_robots/cartpole/base.py
    :param x: Environment state
    :param a: Taken action
    :return: Reward (float)
    """
    _, _, cos_th, _, _ = x
    # rwd = -np.cos(th)
    cost = cos_th

    return np.float32(cost)
