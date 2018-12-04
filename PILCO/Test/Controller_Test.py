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
    state_dim = 5  # Input dimension
    n_actions = 1  # Number of outputs
    n_features = 1  # basis functions

    # Training Dataset
    X0 = np.random.rand(100, state_dim)
    A = np.random.rand(state_dim, n_actions)
    Y0 = np.sin(X0).dot(A) + 1e-3 * (np.random.rand(100, n_actions) - 0.5)  # Just something smooth
    length_scales = np.ones((state_dim,))

    rbf = RBFController(n_actions=n_actions, n_features=n_features, rollout=None, length_scales=length_scales)
    rbf.fit(X0, Y0)

    # Generate input
    m = np.random.rand(1, state_dim)  # But MATLAB defines it as m'
    s = np.random.rand(state_dim, state_dim)
    s = s.dot(s.T)  # Make s positive semidefinite

    M, S, V = compute_action_wrapper(rbf, m, s)

    # convert data to the struct expected by the MATLAB implementation
    length_scales = rbf.get_length_scales()
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

    # Call gp0 in octave
    M_mat, S_mat, V_mat = octave.gp2(gpmodel, m.T, s, nout=3)
    M_mat = np.asarray([M_mat])
    S_mat = np.atleast_2d(S_mat)
    V_mat = np.asarray(V_mat)

    assert M.shape == M_mat.T.shape
    assert S.shape == S_mat.shape
    assert V.shape == V_mat.shape
    np.testing.assert_allclose(M, M_mat.T, rtol=1e-4)
    np.testing.assert_allclose(S, S_mat, rtol=1e-4)
    np.testing.assert_allclose(V, V_mat, rtol=1e-4)


def test_squash():
    np.random.seed(0)
    d = 1  # Control dimensions

    mu = np.random.rand(1, d)  # But MATLAB defines it as m'
    sigma = np.random.rand(d, d)
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

    np.testing.assert_allclose(M, M_mat.T, rtol=1e-9)
    np.testing.assert_allclose(S, S_mat, rtol=1e-9)
    np.testing.assert_allclose(V, V_mat, rtol=1e-9)


if __name__ == '__main__':
    test_rbf()
    test_squash()
