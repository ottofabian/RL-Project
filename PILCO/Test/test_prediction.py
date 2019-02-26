import os

import numpy as np
import oct2py

from PILCO.GaussianProcess.GaussianProcess import GaussianProcess
from PILCO.GaussianProcess.MultivariateGP import MultivariateGP
from PILCO.GaussianProcess.SparseMultivariateGP import SparseMultivariateGP

octave = oct2py.Oct2Py()
dir_path = os.path.dirname(os.path.realpath("__file__")) + "/Matlab_Code"
octave.addpath(dir_path)


def test_mgpr():
    np.random.seed(1)

    state_dim = 3
    n_targets = 2

    n_samples = 100

    # Training Dataset
    X0 = np.random.rand(n_samples, state_dim)
    A = np.random.rand(state_dim, n_targets)
    Y0 = np.sin(X0).dot(A) + 1e-3 * (np.random.rand(n_samples, n_targets) - 0.5)
    length_scales = np.random.rand(n_targets, state_dim)
    sigma_f = np.ones(state_dim)
    sigma_eps = np.ones(state_dim)

    mgpr = MultivariateGP(container=GaussianProcess, length_scales=length_scales, n_targets=n_targets, sigma_f=sigma_f,
                          sigma_eps=sigma_eps)
    mgpr.fit(X0, Y0)

    mgpr.optimize()

    # Generate input
    mu = np.random.rand(1, state_dim)
    sigma = np.random.rand(state_dim, state_dim)
    sigma = sigma.dot(sigma.T)

    _ = mgpr.predict_from_dist(mu, sigma)

    # Change the dataset to avoid any caching issues for K and beta
    X0 = 5 * np.random.rand(100, state_dim)
    mgpr.fit(X0, Y0)

    M, S, V = mgpr.predict_from_dist(mu, sigma)

    # convert data to the struct expected by the MATLAB implementation
    length_scales = mgpr.length_scales()
    sigma_f = mgpr.sigma_fs()
    sigma_eps = mgpr.sigma_eps()

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
    M_mat, S_mat, V_mat = octave.gp0(gpmodel, mu.T, sigma, nout=3)
    M_mat = np.asarray(M_mat).flatten()
    S_mat = np.atleast_2d(S_mat)
    V_mat = np.asarray(V_mat)

    assert M.shape == M_mat.T.shape
    assert S.shape == S_mat.shape
    assert V.shape == V_mat.shape
    np.testing.assert_allclose(M, M_mat.T, rtol=1e-5)
    np.testing.assert_allclose(S, S_mat, rtol=1e-5)
    np.testing.assert_allclose(V, V_mat, rtol=1e-5)


def test_smgpr():
    np.random.seed(1)

    state_dim = 3
    n_targets = 1

    n_samples = 5
    n_inducing_points = 2

    # Training Dataset
    X0 = np.random.rand(n_samples, state_dim)
    A = np.random.rand(state_dim, n_targets)
    Y0 = np.sin(X0).dot(A) + 1e-3 * (np.random.rand(n_samples, n_targets) - 0.5)
    length_scales = np.random.rand(n_targets, state_dim)
    sigma_eps = np.ones(state_dim)
    sigma_f = np.ones(state_dim)

    smgpr = SparseMultivariateGP(length_scales=length_scales, n_targets=n_targets, X=X0, y=Y0,
                                 n_inducing_points=n_inducing_points, container=GaussianProcess, sigma_eps=sigma_eps,
                                 sigma_f=sigma_f)
    smgpr.optimize()

    # Generate input
    mu = np.random.rand(1, state_dim)
    sigma = np.random.rand(state_dim, state_dim)
    sigma = sigma.dot(sigma.T)

    _ = smgpr.predict_from_dist(mu, sigma)

    # Change the dataset to avoid any caching issues for K and beta
    X0 = 5 * np.random.rand(n_samples, state_dim)
    smgpr.fit(X0, Y0)

    M, S, V = smgpr.predict_from_dist(mu, sigma)

    # convert data to the struct expected by the MATLAB implementation
    length_scales = smgpr.length_scales()
    sigma_f = smgpr.sigma_fs()
    sigma_eps = smgpr.sigma_eps()

    hyp = np.hstack(
        (length_scales,
         sigma_f,
         sigma_eps)
    ).T

    gpmodel = oct2py.io.Struct()
    gpmodel.hyp = hyp
    gpmodel.inputs = X0
    gpmodel.targets = Y0
    gpmodel.induce = np.stack([gp.Z.T for gp in smgpr.gp_container]).T
    # gpmodel.induce = smgpr.gp_container[0].Z

    # Call function in octave
    M_mat, S_mat, V_mat = octave.gp1(gpmodel, mu.T, sigma, nout=3)
    M_mat = np.asarray(M_mat).flatten()
    S_mat = np.atleast_2d(S_mat)
    V_mat = np.asarray(V_mat)

    assert M.shape == M_mat.T.shape
    assert S.shape == S_mat.shape
    assert V.shape == V_mat.shape
    np.testing.assert_allclose(M, M_mat.T, rtol=1e-3)
    np.testing.assert_allclose(S, S_mat, rtol=1e-3)
    np.testing.assert_allclose(V, V_mat, rtol=1e-3)


if __name__ == '__main__':
    test_mgpr()
    test_smgpr()
