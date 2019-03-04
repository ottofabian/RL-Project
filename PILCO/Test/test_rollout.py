import logging
import os

import numpy as np
import oct2py

from PILCO.Controller.RBFController import RBFController
from PILCO.PILCO import PILCO
from PILCO.util.util import parse_args

octave = oct2py.Oct2Py(logger=oct2py.get_log())
octave.logger = oct2py.get_log('new_log')
octave.logger.setLevel(logging.INFO)
dir_path = os.path.dirname(os.path.realpath("__file__")) + "/Matlab_Code"
octave.addpath(dir_path)


def test_rollout():
    np.random.seed(0)

    # specifiy state and actions space
    state_dim = 2
    n_actions = 1
    n_targets = 2

    # how many samples for dynamics
    n_samples = 100

    # policy features and actions range
    n_features_rbf = 20
    bound = np.array([10.0])

    # trajectory length
    horizon = 10

    # ---------------------------------------------------------------------------------------

    # setup policy
    X0_rbf = np.random.rand(n_features_rbf, state_dim)
    A_rbf = np.random.rand(state_dim, n_actions)
    Y0_rbf = np.sin(X0_rbf).dot(A_rbf) + 1e-3 * (np.random.rand(n_features_rbf, n_actions) - 0.5)
    length_scales_rbf = np.random.rand(n_actions, state_dim)

    rbf = RBFController(X0_rbf, Y0_rbf, n_actions=n_actions, length_scales=length_scales_rbf)
    # ---------------------------------------------------------------------------------------

    # Pilco setup

    # take any env, to avoid issues with gym.make
    # matlab is specified with squashing, so we assume bound bound
    args = parse_args([])
    args.max_action = bound
    args.env_name = "MountainCarContinuous-v0"
    args.features = None
    args.horizon = horizon
    args.inducing_points = None

    pilco = PILCO(args, loss=None)

    # Training Dataset for dynamics model
    X0_dyn = np.random.rand(n_samples, state_dim + n_actions)
    A_dyn = np.random.rand(state_dim + n_actions, n_targets)
    Y0_dyn = np.sin(X0_dyn).dot(A_dyn) + 1e-3 * (np.random.rand(n_samples, n_targets) - 0.5)

    # set observed data set manually
    pilco.state_action_pairs = X0_dyn
    pilco.state_delta = Y0_dyn
    pilco.state_dim = state_dim
    pilco.n_actions = n_actions
    # pilco.T = horizon
    # pilco.bound = bound

    pilco.learn_dynamics_model()

    # ---------------------------------------------------------------------------------------

    # Generate input
    mean = np.random.rand(1, state_dim)
    sigma = np.random.rand(state_dim, state_dim)
    sigma = sigma.dot(sigma.T)

    M = []
    S = []

    M.append(mean.flatten())
    S.append(sigma)

    mm = mean
    ss = sigma

    for i in range(horizon):
        mm, ss, _, _ = pilco.rollout(policy=rbf, state_mu=mm.flatten(), state_cov=ss)
        M.append(mm)
        S.append(ss)

    M = np.array(M)
    S = np.array(S)
    # ---------------------------------------------------------------------------------------

    # generate dynmodel for matlab
    length_scales_dyn = pilco.dynamics_model.length_scales()
    sigma_f_dyn = pilco.dynamics_model.sigma_f()
    sigma_eps_dyn = pilco.dynamics_model.sigma_eps()

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
    length_scales_rbf = rbf.length_scales()
    sigma_f_rbf = rbf.sigma_f()
    sigma_eps_rbf = rbf.sigma_eps()

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
    policy.maxU = bound

    # ---------------------------------------------------------------------------------------

    # set fake environment
    plant = oct2py.io.Struct()
    plant.angi = np.zeros(0)
    plant.poli = np.arange(state_dim) + 1
    plant.dyni = np.arange(state_dim) + 1
    plant.difi = np.arange(state_dim) + 1

    # ---------------------------------------------------------------------------------------

    M_mat, S_mat = octave.pred(policy, plant, dynmodel, mean.T, sigma, horizon, nout=2, verbose=True)

    # check initial values before rollout, avoids stupid mistakes
    np.testing.assert_allclose(M[0], M_mat[:, 0].T, rtol=1e-5)
    np.testing.assert_allclose(S[0], S_mat[:, :, 0], rtol=1e-5)

    # check after first iteration
    np.testing.assert_allclose(M[1], M_mat[:, 1].T, rtol=1e-5)
    np.testing.assert_allclose(S[1], S_mat[:, :, 1], rtol=1e-5)

    # check after last iteration
    np.testing.assert_allclose(M[-1], M_mat[:, -1].T, rtol=1e-5)
    np.testing.assert_allclose(S[-1], S_mat[:, :, -1], rtol=1e-5)


if __name__ == '__main__':
    test_rollout()
