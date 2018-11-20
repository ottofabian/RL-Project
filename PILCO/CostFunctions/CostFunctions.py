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
    Returns the cost taken an action from a specific state

    Extracted CartpoleBase.from CartpoleBase._rmd()
    https://git.ias.informatik.tu-darmstadt.de/quanser/clients/blob/master/quanser_robots/cartpole/base.py
    :param x: Environment state
    :param a: Taken action
    :return: Reward (float)
    """
    _, _, cos_th, _, _ = x
    # rwd = -np.cos(th)

    return np.float32(cos_th)


def cartpolebase_costfunc_dist(mu, sigma):
    """
    returns the cost and takes into account the current certainty of this action.
    See Deisenroth(2010) A.1 for mean computation of cos
    :param x: Environment state
    :param a: Taken action
    :return: Reward (float)
    """

    # TODO: Maybe better Deisenroth(2010) page 54
    _, _, cos_th, _, _ = mu

    return np.float32(cos_th)

