import os

import numpy as np
import oct2py

from PILCO.Controller.RBFController import RBFController
from PILCO.CostFunctions.SaturatedLoss import SaturatedLoss
from PILCO.PILCO import PILCO

octave = oct2py.Oct2Py()
dir_path = os.path.dirname(os.path.realpath("__file__")) + "/Matlab_Code"
octave.addpath(dir_path)


def test_cost():
    np.random.seed(0)

    state_dim = 2
    mu = np.random.rand(1, state_dim)
    sigma = np.random.rand(state_dim, state_dim)
    sigma = sigma.dot(sigma.T)

    target_state = np.random.rand(state_dim)
    T_inv = np.random.randn(state_dim, state_dim)

    loss = SaturatedLoss(state_dim=state_dim, target_state=target_state, T_inv=T_inv)

    M, S, C = loss.compute_cost(mu, sigma)
    C = np.linalg.solve(sigma, np.eye(state_dim)) @ C

    cost = oct2py.io.Struct()
    cost.z = loss.target_state.T
    cost.W = loss.T_inv

    M_mat, _, _, S_mat, _, _, C_mat, _, _ = octave.lossSat(cost, mu.T, sigma, nout=9)

    np.testing.assert_allclose(M, M_mat, rtol=1e-9)
    np.testing.assert_allclose(S, S_mat, rtol=1e-9)
    np.testing.assert_allclose(C, C_mat, rtol=1e-9)


def test_trajectory_cost():
    np.random.seed(0)

    state_dim = 2
    n_actions = 1
    n_targets = 2

    n_features_rbf = 100
    e = np.array([10.0])

    horizon = 10

    # Default initial distribution for computing trajectory cost
    mu = np.random.randn(1, state_dim)
    sigma = np.random.randn(state_dim, state_dim)
    sigma = sigma.dot(sigma.T)

    # some random target state to reach
    target_state = np.random.rand(state_dim)

    # ---------------------------------------------------------------------------------------
    # Policy setup

    X0_rbf = np.random.rand(n_features_rbf, state_dim)
    A_rbf = np.random.rand(state_dim, n_actions)
    Y0_rbf = np.sin(X0_rbf).dot(A_rbf) + 1e-3 * (np.random.rand(n_features_rbf, n_actions) - 0.5)
    length_scales_rbf = np.random.rand(state_dim)

    rbf = RBFController(n_actions=n_actions, n_features=n_features_rbf, compute_cost=None,
                        length_scales=length_scales_rbf)
    rbf.fit(X0_rbf, Y0_rbf)

    # ---------------------------------------------------------------------------------------
    # Pilco setup

    # setup loss
    T_inv = np.random.randn(state_dim, state_dim)
    loss = SaturatedLoss(state_dim=state_dim, target_state=target_state, T_inv=T_inv)

    # take any env, to avoid issues with gym.make
    # matlab is specified with squashing
    pilco = PILCO(env_name="MountainCarContinuous-v0", seed=1, n_features=n_features_rbf, Horizon=horizon, loss=loss,
                  bound=e, gamma=1, start_mu=mu.flatten(), start_cov=sigma)

    # Training Dataset for dynamics model
    X0_dyn = np.random.rand(100, state_dim + n_actions)
    A_dyn = np.random.rand(state_dim + n_actions, n_targets)
    Y0_dyn = np.sin(X0_dyn).dot(A_dyn) + 1e-3 * (np.random.rand(100, n_targets) - 0.5)

    # set observed data set manually
    pilco.state_action_pairs = X0_dyn
    pilco.state_delta = Y0_dyn
    pilco.state_dim = state_dim
    pilco.n_actions = n_actions

    pilco.learn_dynamics_model()

    # ---------------------------------------------------------------------------------------

    M = pilco.compute_trajectory_cost(policy=rbf)

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

    # generate loss struct for matlab
    cost = oct2py.io.Struct()
    cost.z = loss.target_state.T
    cost.W = loss.T_inv

    # ---------------------------------------------------------------------------------------

    # set fake environment
    plant = oct2py.io.Struct()
    plant.angi = np.zeros(0)
    plant.poli = np.arange(state_dim) + 1
    plant.dyni = np.arange(state_dim) + 1
    plant.difi = np.arange(state_dim) + 1

    # ---------------------------------------------------------------------------------------

    M_mat, S_mat = octave.predcost(mu.T, sigma, dynmodel, plant, policy, cost, horizon, nout=2, verbose=True)

    # compare sum of cost
    np.testing.assert_allclose(M, np.sum(M_mat), rtol=1e-5)
    # np.testing.assert_allclose(S, S_mat, rtol=1e-5)


if __name__ == '__main__':
    test_cost()
    test_trajectory_cost()
