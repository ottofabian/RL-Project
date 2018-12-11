import os

import numpy as np
import oct2py

from PILCO.GaussianProcess.GaussianProcess import GaussianProcess
from PILCO.GaussianProcess.MultivariateGP import MultivariateGP

octave = oct2py.Oct2Py()
dir_path = os.path.dirname(os.path.realpath("__file__")) + "/Matlab_Code"
octave.addpath(dir_path)


def test_predictions():
    np.random.seed(0)
    state_dim = 3
    n_targets = 2

    # Training Dataset
    X0 = np.random.rand(100, state_dim)
    A = np.random.rand(state_dim, n_targets)
    Y0 = np.sin(X0).dot(A) + 1e-3 * (np.random.rand(100, n_targets) - 0.5)  # Just something smooth
    length_scales = np.random.rand(state_dim)

    mgpr = MultivariateGP(length_scales=length_scales, n_targets=n_targets, container=GaussianProcess)
    mgpr.fit(X0, Y0)

    mgpr.optimize()

    # Generate input
    m = np.random.rand(1, state_dim)  # But MATLAB defines it as m'
    s = np.random.rand(state_dim, state_dim)
    s = s.dot(s.T)  # Make s positive semidefinite

    _ = mgpr.predict_from_dist(m, s)

    # Change the dataset to avoid any caching issues for K and beta
    X0 = 5 * np.random.rand(100, state_dim)
    mgpr.fit(X0, Y0)

    M, S, V = mgpr.predict_from_dist(m, s)
    # V = np.linalg.solve(s, np.eye(s.shape[0])) @ V

    # convert data to the struct expected by the MATLAB implementation
    length_scales = mgpr.get_length_scales()
    sigma_f = mgpr.get_sigma_fs()
    sigma_eps = mgpr.get_sigma_eps()

    hyp = np.hstack(
        (length_scales,
         sigma_f,
         sigma_eps)
    ).T

    gpmodel = oct2py.io.Struct()
    gpmodel.hyp = hyp
    gpmodel.inputs = X0
    gpmodel.targets = Y0

    # Call function in octave
    M_mat, S_mat, V_mat = octave.gp0(gpmodel, m.T, s, nout=3)
    M_mat = np.asarray(M_mat).flatten()
    S_mat = np.atleast_2d(S_mat)
    V_mat = np.asarray(V_mat)

    assert M.shape == M_mat.T.shape
    assert S.shape == S_mat.shape
    assert V.shape == V_mat.shape
    np.testing.assert_allclose(M, M_mat.T, rtol=1e-5)
    np.testing.assert_allclose(S, S_mat, rtol=1e-5)
    np.testing.assert_allclose(V, V_mat, rtol=1e-5)


if __name__ == '__main__':
    test_predictions()
