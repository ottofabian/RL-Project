import os

import numpy as np
import oct2py

from PILCO.CostFunctions.SaturatedLoss import SaturatedLoss

octave = oct2py.Oct2Py()
dir_path = os.path.dirname(os.path.realpath("__file__")) + "/Matlab_Code"
octave.addpath(dir_path)


def test_cost():
    np.random.seed(1)
    state_dim = 2
    mu = np.random.rand(1, state_dim)
    sigma = np.random.rand(state_dim, state_dim)
    sigma = sigma.dot(sigma.T)

    target_state = np.random.rand(state_dim)
    T_inv = np.random.randn(state_dim, state_dim)

    # TODO PILCO object with saturation
    loss = SaturatedLoss(state_dim=state_dim, target_state=target_state, T_inv=T_inv)

    M, S, C = loss.compute_cost(mu, sigma)
    C = np.linalg.solve(sigma, np.eye(state_dim)) @ C

    cost = oct2py.io.Struct()
    cost.z = loss.target_state.T
    cost.W = loss.T_inv

    M_mat, S_mat, C_mat = octave.lossSat(cost, mu.T, sigma, nout=3)

    np.testing.assert_allclose(M, M_mat, rtol=1e-9)
    np.testing.assert_allclose(S, S_mat, rtol=1e-9)
    np.testing.assert_allclose(C, C_mat, rtol=1e-9)


if __name__ == '__main__':
    test_cost()
