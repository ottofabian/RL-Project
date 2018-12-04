import logging
import os

import numpy as np
import oct2py

from PILCO.Controller.RBFController import RBFController
from PILCO.PILCO import PILCO

octave = oct2py.Oct2Py(logger=oct2py.get_log())
octave.logger = oct2py.get_log('new_log')
octave.logger.setLevel(logging.INFO)
dir_path = os.path.dirname(os.path.realpath("__file__")) + "/Matlab Code"
octave.addpath(dir_path)


def test_cascade():
    # TODO make this test work for my code
    np.random.seed(0)
    d = 2  # State dimenstion
    k = 1  # Controller's output dimension
    horizon = 10
    e = np.array([[10.0]])  # Max control input. Set too low can lead to Cholesky failures.

    # Training Dataset
    X0 = np.random.rand(100, d + k)
    A = np.random.rand(d + k, d)
    Y0 = np.sin(X0).dot(A) + 1e-3 * (np.random.rand(100, d) - 0.5)  # Just something smooth
    pilco = PILCO(env_name=None, seed=1, n_features=None, Horizon=horizon, cost_function=None, squash=False)
    length_scales = np.ones((d,))

    rbf = RBFController(n_actions=k, n_features=n_features, rollout=None, length_scales=length_scales)
    rbf.fit(X0, Y0)

    pilco.state_action_pairs = X0
    pilco.state_delta = Y0
    # pilco.controller.max_action = e
    # pilco.optimize()

    # Generate input
    m = np.random.rand(1, d)  # But MATLAB defines it as m'
    s = np.random.rand(d, d)
    s = s.dot(s.T)  # Make s positive semidefinite

    M, S, reward = pilco.rollout(policy=rbf)

    # convert data to the struct expected by the MATLAB implementation
    policy = oct2py.io.Struct()
    policy.p = oct2py.io.Struct()
    policy.p.w = pilco.controller.W.value
    policy.p.b = pilco.controller.b.value.T
    policy.maxU = e

    # convert data to the struct expected by the MATLAB implementation
    length_scales = rbf.get_length_scales()
    sigma_f = rbf.get_sigma_fs()
    sigma_eps = rbf.get_sigma_eps()

    hyp = np.hstack(
        (length_scales,
         sigma_f,
         sigma_eps)
    ).T

    dynmodel = oct2py.io.Struct()
    dynmodel.hyp = hyp
    dynmodel.inputs = X0
    dynmodel.targets = Y0

    plant = oct2py.io.Struct()
    plant.angi = np.zeros(0)
    plant.angi = np.zeros(0)
    plant.poli = np.arange(d) + 1
    plant.dyni = np.arange(d) + 1
    plant.difi = np.arange(d) + 1

    # Call function in octave
    M_mat, S_mat = octave.pred(rbf, plant, dynmodel, m.T, s, horizon, nout=2, verbose=True)
    # Extract only last element of the horizon
    M_mat = M_mat[:, -1]
    S_mat = S_mat[:, :, -1]

    np.testing.assert_allclose(M[0], M_mat.T, rtol=1e-4)
    np.testing.assert_allclose(S, S_mat, rtol=1e-4)


if __name__ == '__main__':
    test_cascade()
