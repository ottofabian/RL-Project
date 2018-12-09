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


def test_rollout():
    np.random.seed(0)
    state_dim = 2
    n_actions = 1
    horizon = 10
    e = np.array([[10.0]])  # Max control input. Set too low can lead to Cholesky failures.
    n_features = 10

    # Training Dataset
    X0 = np.random.rand(100, state_dim + n_actions)
    A = np.random.rand(state_dim + n_actions, state_dim)
    Y0 = np.sin(X0).dot(A) + 1e-3 * (np.random.rand(100, state_dim) - 0.5)  # Just something smooth
    pilco = PILCO(env_name="Pendulum-v0", seed=1, n_features=n_features, Horizon=horizon, loss=None, squash=True)

    pilco.state_action_pairs = X0
    pilco.state_delta = Y0

    pilco.learn_dynamics_model()
    length_scales_rbf = np.ones((state_dim,))

    rbf = RBFController(n_actions=n_actions, n_features=n_features, length_scales=length_scales_rbf, rollout=None)
    rbf.fit(X0, Y0)

    # Generate input
    m = np.random.rand(1, state_dim)  # But MATLAB defines it as m'
    s = np.random.rand(state_dim, state_dim)
    s = s.dot(s.T)  # Make s positive semidefinite

    M, S = pilco.rollout_debug(policy=rbf, state_mu=m, state_cov=s, bound=e)

    # convert data to the struct expected by the MATLAB implementation
    length_scales_rbf = rbf.get_length_scales()
    sigma_f_rbf = rbf.get_sigma_fs()
    sigma_eps_rbf = rbf.get_sigma_eps()

    hyp = np.hstack(
        (length_scales_rbf,
         sigma_f_rbf,
         sigma_eps_rbf)
    ).T

    policy = oct2py.io.Struct()
    policy.p = oct2py.io.Struct()
    policy.hyp = hyp
    policy.inputs = X0
    policy.targets = Y0
    policy.maxU = e

    # generate
    length_scales = pilco.dynamics_model.get_length_scales()
    sigma_f = pilco.dynamics_model.get_sigma_fs()
    sigma_eps = pilco.dynamics_model.get_sigma_eps()

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
    plant.poli = np.arange(state_dim) + 1
    plant.dyni = np.arange(state_dim) + 1
    plant.difi = np.arange(state_dim) + 1

    # Call function in octave
    M_mat, S_mat = octave.pred(m.T, s, plant, dynmodel, policy, nout=2, verbose=True)
    # Extract only last element of the horizon
    M_mat = M_mat[:, -1]
    S_mat = S_mat[:, :, -1]

    np.testing.assert_allclose(M[-1], M_mat.T, rtol=1e-5)
    np.testing.assert_allclose(S[-1], S_mat, rtol=1e-5)


if __name__ == '__main__':
    test_rollout()
