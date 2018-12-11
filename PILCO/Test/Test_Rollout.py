import logging
import os

import numpy as np
import oct2py

from PILCO.Controller.RBFController import RBFController
from PILCO.PILCO import PILCO

octave = oct2py.Oct2Py(logger=oct2py.get_log())
octave.logger = oct2py.get_log('new_log')
octave.logger.setLevel(logging.INFO)
dir_path = os.path.dirname(os.path.realpath("__file__")) + "/Matlab_Code"
octave.addpath(dir_path)


def test_rollout():
    np.random.seed(0)
    state_dim = 2
    n_actions = 1
    n_targets = 2

    n_features_rbf = 100
    e = np.array([10.0])

    horizon = 10

    # ---------------------------------------------------------------------------------------

    # setup policy
    X0_rbf = np.random.rand(n_features_rbf, state_dim)
    A_rbf = np.random.rand(state_dim, n_actions)
    Y0_rbf = np.sin(X0_rbf).dot(A_rbf) + 1e-3 * (np.random.rand(n_features_rbf, n_actions) - 0.5)
    length_scales_rbf = np.random.rand(state_dim)

    rbf = RBFController(n_actions=n_actions, n_features=n_features_rbf, rollout=None, length_scales=length_scales_rbf)
    rbf.fit(X0_rbf, Y0_rbf)

    # ---------------------------------------------------------------------------------------

    # Pilco setup

    # take any env, to avoid issues with gym.make
    # matlab is specified with squashing
    pilco = PILCO(env_name="Pendulum-v0", seed=1, n_features=n_features_rbf, Horizon=horizon, loss=None, squash=True)

    # Training Dataset for dynamics model
    X0_dyn = np.random.rand(100, state_dim + n_actions)
    A_dyn = np.random.rand(state_dim + n_actions, n_targets)
    Y0_dyn = np.sin(X0_dyn).dot(A_dyn) + 1e-3 * (np.random.rand(100, n_targets) - 0.5)  # Just something smooth

    pilco.state_action_pairs = X0_dyn
    pilco.state_delta = Y0_dyn
    pilco.state_dim = state_dim
    pilco.n_actions = n_actions
    pilco.bound = e
    pilco.T = horizon

    pilco.learn_dynamics_model()

    # ---------------------------------------------------------------------------------------

    # Generate input
    m = np.random.rand(1, state_dim)
    s = np.random.rand(state_dim, state_dim)
    s = s.dot(s.T)

    M = []
    S = []

    M.append(m)
    S.append(s)

    mm = m
    ss = s

    for i in range(horizon):
        mm, ss, _, _ = pilco.rollout(policy=rbf, state_mu=mm.flatten(), state_cov=ss)
        M.append(mm)
        S.append(ss)

    M = np.array(M)
    S = np.array(S)
    # ---------------------------------------------------------------------------------------

    # generate dynmodel for matlab
    length_scales_dyn = pilco.dynamics_model.get_length_scales()
    sigma_f_dyn = pilco.dynamics_model.get_sigma_fs()
    sigma_eps_dyn = pilco.dynamics_model.get_sigma_eps()

    hyp_dyn = np.hstack(
        (length_scales_dyn,
         sigma_f_dyn,
         sigma_eps_dyn)
    ).T

    dynmodel = oct2py.io.Struct()
    dynmodel.hyp = hyp_dyn
    dynmodel.inputs = X0_dyn
    dynmodel.targets = Y0_dyn

    # ---------------------------------------------------------------------------------------

    # generate rbf policy for matlab
    length_scales_rbf = rbf.get_length_scales()
    sigma_f_rbf = rbf.get_sigma_fs()
    sigma_eps_rbf = rbf.get_sigma_eps()

    hyp_rbf = np.hstack(
        (length_scales_rbf,
         sigma_f_rbf,
         sigma_eps_rbf)
    ).T

    policy = oct2py.io.Struct()
    policy.p = oct2py.io.Struct()
    policy.p.hyp = hyp_rbf
    policy.p.inputs = X0_rbf
    policy.p.targets = Y0_rbf
    policy.maxU = e

    # ---------------------------------------------------------------------------------------

    # set fake environment
    plant = oct2py.io.Struct()
    plant.angi = np.zeros(0)
    plant.poli = np.arange(state_dim) + 1
    plant.dyni = np.arange(state_dim) + 1
    plant.difi = np.arange(state_dim) + 1

    # ---------------------------------------------------------------------------------------

    M_mat, S_mat = octave.pred(policy, plant, dynmodel, m.T, s, horizon, nout=2, verbose=True)

    # check after first iteration
    np.testing.assert_allclose(M[1], M_mat[:, 1].T, rtol=1e-5)
    np.testing.assert_allclose(S[1], S_mat[:, :, 1], rtol=1e-5)

    # check after last iteration
    np.testing.assert_allclose(M[-1], M_mat[:, -1].T, rtol=1e-5)
    np.testing.assert_allclose(S[-1], S_mat[:, :, -1], rtol=1e-5)


if __name__ == '__main__':
    test_rollout()
