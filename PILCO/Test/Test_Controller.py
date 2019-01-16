import os

import numpy as np
import oct2py

from PILCO.Controller.RBFController import RBFController, squash_action_dist

# https://github.com/nrontsis/PILCO/edit/master/tests/test_controllers.py
octave = oct2py.Oct2Py()
dir_path = os.path.dirname(os.path.realpath("__file__")) + "/Matlab_Code"
print(dir_path)
octave.addpath(dir_path)


def compute_action_wrapper(controller, m, s, squash=False):
    return controller.choose_action(m, s, squash)


def test_rbf():
    np.random.seed(0)

    state_dim = 5
    n_actions = 1
    n_features = 10

    # Training Dataset
    X0 = np.random.rand(n_features, state_dim)
    A = np.random.rand(state_dim, n_actions)
    Y0 = np.sin(X0).dot(A) + 1e-3 * (np.random.rand(n_features, n_actions) - 0.5)
    length_scales = np.random.rand(state_dim)

    rbf = RBFController(n_actions=n_actions, n_features=n_features, compute_cost=None, length_scales=length_scales)
    rbf.fit(X0, Y0)

    # Generate input
    mu = np.random.rand(1, state_dim)
    sigma = np.random.rand(state_dim, state_dim)
    sigma = sigma.dot(sigma.T)  # Make sigma positive semidefinite

    M, S, V = rbf.choose_action(mu, sigma, None)
    # V is already multiplied with S, have to revert that to run positive test
    V = np.linalg.solve(sigma, np.eye(sigma.shape[0])) @ V

    # convert data to the struct expected by the MATLAB implementation
    length_scales = length_scales.reshape(n_actions, state_dim)
    sigma_f = rbf.get_sigma_fs()
    sigma_eps = rbf.get_sigma_eps()

    hyp = np.hstack(
        (length_scales,
         sigma_f,
         sigma_eps)
    ).T

    gpmodel = oct2py.io.Struct()
    gpmodel.hyp = hyp
    gpmodel.inputs = X0
    gpmodel.targets = Y0

    M_mat, S_mat, V_mat = octave.gp2(gpmodel, mu.T, sigma, nout=3)
    M_mat = np.asarray([M_mat])
    S_mat = np.atleast_2d(S_mat)
    V_mat = np.asarray(V_mat)

    assert M.shape == M_mat.T.shape
    assert S.shape == S_mat.shape
    assert V.shape == V_mat.shape

    print(M)
    print(M_mat.T)

    np.testing.assert_allclose(M, M_mat.T, rtol=1e-5)
    np.testing.assert_allclose(S, S_mat, rtol=1e-5)
    np.testing.assert_allclose(V, V_mat, rtol=1e-5)


def test_squash():
    np.random.seed(0)
    n_actions = 1

    mu = np.random.rand(1, n_actions)  # But MATLAB defines it as m'
    sigma = np.random.rand(n_actions, n_actions)
    sigma = sigma.dot(sigma.T)
    i_o_cov = np.ones(mu.T.shape)
    e = np.array([7.0])

    M, S, V = squash_action_dist(mu, sigma, i_o_cov, e)

    M_mat, S_mat, V_mat = octave.gSin(mu.T, sigma, e, nout=3)
    M_mat = np.atleast_2d([M_mat])
    S_mat = np.atleast_2d(S_mat)
    V_mat = np.atleast_2d(V_mat)

    assert M.shape == M_mat.T.shape
    assert S.shape == S_mat.shape
    assert V.shape == V_mat.shape

    np.testing.assert_allclose(M, M_mat.T, rtol=1e-5)
    np.testing.assert_allclose(S, S_mat, rtol=1e-5)
    np.testing.assert_allclose(V, V_mat, rtol=1e-5)


if __name__ == '__main__':
    test_rbf()
    test_squash()
