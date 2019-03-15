import os

import numpy as np
import oct2py

from pilco.controller.linear_controller import LinearController
from pilco.controller.rbf_controller import RBFController
from pilco.util.util import squash_action_dist

octave = oct2py.Oct2Py()
dir_path = "pilco/test/matlab_code"
print(dir_path)
octave.addpath(dir_path)


def test_rbf():
    np.random.seed(0)

    state_dim = 5
    n_actions = 2
    n_features = 10

    # Training Dataset
    X0 = np.random.rand(n_features, state_dim)
    A = np.random.rand(state_dim, n_actions)
    Y0 = np.sin(X0).dot(A) + 1e-3 * (np.random.rand(n_features, n_actions) - 0.5)
    length_scales = np.random.rand(n_actions, state_dim)

    rbf = RBFController(X0, Y0, n_actions=n_actions, length_scales=length_scales)

    # Generate input
    mu = np.random.rand(1, state_dim)
    sigma = np.random.rand(state_dim, state_dim)
    sigma = sigma.dot(sigma.T)  # Make sigma positive semidefinite

    M, S, V = rbf.choose_action(mu, sigma, None)

    # V is already multiplied with S, have to revert that to run positive test
    V = np.linalg.solve(sigma, np.eye(sigma.shape[0])) @ V

    # convert data to the struct expected by the MATLAB implementation
    length_scales = length_scales.reshape(n_actions, state_dim)
    sigma_f = rbf.sigma_f()
    sigma_eps = rbf.sigma_eps()

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
    M_mat = np.asarray(M_mat)[:, 0]
    S_mat = np.atleast_2d(S_mat)
    V_mat = np.asarray(V_mat)

    assert M.shape == M_mat.T.shape
    assert S.shape == S_mat.shape
    assert V.shape == V_mat.shape

    np.testing.assert_allclose(M, M_mat.T, rtol=1e-5)
    np.testing.assert_allclose(S, S_mat, rtol=1e-5)
    np.testing.assert_allclose(V, V_mat, rtol=1e-5)


def test_set_params_rbf():
    np.random.seed(0)

    state_dim = 5
    n_actions = 2
    n_features = 10

    X0 = np.random.rand(n_features, state_dim)
    A = np.random.rand(state_dim, n_actions)
    Y0 = np.sin(X0).dot(A) + 1e-3 * (np.random.rand(n_features, n_actions) - 0.5)
    length_scales = np.random.rand(n_actions, state_dim)

    rbf = RBFController(X0, Y0, n_actions=n_actions, length_scales=length_scales)
    sigma_eps = rbf.sigma_eps()
    sigma_f = rbf.sigma_f()

    rbf.set_params(rbf.get_params())
    assert np.all(X0 == rbf.x)
    assert np.all(Y0 == rbf.y)
    assert np.all(length_scales == rbf.length_scales())
    assert np.all(sigma_eps == rbf.sigma_eps())
    assert np.all(sigma_f == rbf.sigma_f())


def test_set_params_linear():
    np.random.seed(0)

    state_dim = 5
    n_actions = 2

    linear = LinearController(n_actions=n_actions, state_dim=state_dim)
    W = linear.weights
    b = linear.bias

    linear.set_params(linear.get_params())
    assert np.all(W == linear.weights)
    assert np.all(b == linear.bias)


def test_linear():
    np.random.seed(0)
    state_dim = 3
    n_actions = 2
    # Generate input
    m = np.random.rand(1, state_dim)
    s = np.random.rand(state_dim, state_dim)
    s = s.dot(s.T)

    bound = None

    W = np.random.rand(state_dim, n_actions)
    b = np.random.rand(1, n_actions)

    linear = LinearController(state_dim=state_dim, n_actions=n_actions)
    linear.weights = W
    linear.bias = b

    M, S, V = linear.choose_action(m, s, bound=bound)

    # convert data to the struct expected by the MATLAB implementation
    policy = oct2py.io.Struct()
    policy.p = oct2py.io.Struct()
    policy.p.w = W.T
    policy.p.b = b.T

    # Call function in octave
    M_mat, S_mat, V_mat = octave.conlin(policy, m.T, s, nout=3)

    assert M.shape == M_mat.T.shape
    assert S.shape == S_mat.shape
    assert V.shape == V_mat.shape

    np.testing.assert_allclose(M, M_mat.T, rtol=1e-4)
    np.testing.assert_allclose(S, S_mat, rtol=1e-4)
    np.testing.assert_allclose(V, V_mat, rtol=1e-4)


def test_squash():
    np.random.seed(0)
    n_actions = 2

    mu = np.random.rand(n_actions)
    sigma = np.random.rand(n_actions, n_actions)
    sigma = sigma.dot(sigma.T)
    i_o_cov = np.ones(mu.T.shape)
    bound = np.array([7.])

    M, S, V = squash_action_dist(mu, sigma, i_o_cov, bound)

    bound = bound.reshape(1, -1)
    M_mat, S_mat, V_mat = octave.gSin(mu, sigma, bound, nout=3)
    M_mat = np.asarray(M_mat)[:, 0]
    # S_mat = np.atleast_2d(S_mat)
    # V_mat = np.atleast_2d(V_mat)

    V_mat = V_mat @ i_o_cov

    assert M.shape == M_mat.T.shape
    assert S.shape == S_mat.shape
    assert V.shape == V_mat.shape

    np.testing.assert_allclose(M, M_mat.T, rtol=1e-5)
    np.testing.assert_allclose(S, S_mat, rtol=1e-5)
    np.testing.assert_allclose(V, V_mat, rtol=1e-5)


if __name__ == '__main__':
    test_rbf()
    test_linear()
    test_squash()
    test_set_params_rbf()
