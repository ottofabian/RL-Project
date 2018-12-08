import os

import numpy as np
import oct2py

from PILCO.GaussianProcess.MultivariateGP import MultivariateGP

octave = oct2py.Oct2Py()
dir_path = os.path.dirname(os.path.realpath("__file__")) + "/tests/Matlab Code"
octave.addpath(dir_path)


def test_predictions():
    np.random.seed(0)
    d = 3  # Input dimension
    k = 2  # Number of outputs

    # Training Dataset
    X0 = np.random.rand(100, d)
    A = np.random.rand(d, k)
    Y0 = np.sin(X0).dot(A) + 1e-3 * (np.random.rand(100, k) - 0.5)  # Just something smooth
    mgpr = MultivariateGP(X0, Y0)

    mgpr.optimize()

    # Generate input
    m = np.random.rand(1, d)  # But MATLAB defines it as m'
    s = np.random.rand(d, d)
    s = s.dot(s.T)  # Make s positive semidefinite

    M, S, V = mgpr.predict_from_dist(m, s)

    # Change the dataset and predict again. Just to make sure that we don't cache something we shouldn't.
    X0 = 5 * np.random.rand(100, d)
    mgpr.set_XY(X0, Y0)

    M, S, V = mgpr.predict_from_dist(m, s)

    # convert data to the struct expected by the MATLAB implementation
    lengthscales = np.stack([model.kern.lengthscales.value for model in mgpr.models])
    variance = np.stack([model.kern.variance.value for model in mgpr.models])
    noise = np.stack([model.likelihood.variance.value for model in mgpr.models])

    hyp = np.log(np.hstack(
        (lengthscales,
         np.sqrt(variance[:, None]),
         np.sqrt(noise[:, None]))
    )).T

    gpmodel = oct2py.io.Struct()
    gpmodel.hyp = hyp
    gpmodel.inputs = X0
    gpmodel.targets = Y0

    # Call function in octave
    M_mat, S_mat, V_mat = octave.gp0(gpmodel, m.T, s, nout=3)

    assert M.shape == M_mat.T.shape
    assert S.shape == S_mat.shape
    assert V.shape == V_mat.shape
    np.testing.assert_allclose(M, M_mat.T, rtol=1e-4)
    np.testing.assert_allclose(S, S_mat, rtol=1e-4)
    np.testing.assert_allclose(V, V_mat, rtol=1e-4)


if __name__ == '__main__':
    test_predictions()