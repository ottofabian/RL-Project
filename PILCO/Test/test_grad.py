from autograd import grad, jacobian
from autograd.test_util import check_grads
from scipy.optimize import check_grad

import numpy as np

from PILCO.Controller.RBFController import RBFController
from PILCO.CostFunctions.SaturatedLoss import SaturatedLoss
from PILCO.PILCO import PILCO


def test_grad_mgpr():
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
    # Pilco setup

    # setup loss
    T_inv = np.random.randn(state_dim, state_dim)
    loss = SaturatedLoss(state_dim=state_dim, target_state=target_state, T_inv=T_inv)

    # take any env, to avoid issues with gym.make
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
    # Policy setup

    X0_rbf = np.random.rand(n_features_rbf, state_dim)
    A_rbf = np.random.rand(state_dim, n_actions)
    Y0_rbf = np.sin(X0_rbf).dot(A_rbf) + 1e-3 * (np.random.rand(n_features_rbf, n_actions) - 0.5)
    length_scales_rbf = np.random.rand(n_actions, state_dim)

    rbf = RBFController(n_actions=n_actions, n_features=n_features_rbf, compute_cost=None,
                        length_scales=length_scales_rbf)
    rbf.compute_cost = pilco.compute_trajectory_cost
    rbf.fit(X0_rbf, Y0_rbf)

    # ---------------------------------------------------------------------------------------

    params = np.array([gp.wrap_policy_hyperparams() for gp in rbf.gp_container]).flatten()
    np.allclose(grad(rbf._optimize_hyperparams)(params), jacobian(rbf._optimize_hyperparams)(params))


def test_grad_smgpr():
    np.random.seed(0)

    state_dim = 2
    n_actions = 1
    n_targets = 2
    n_inducing_points = 10
    n_samples = 100

    n_features_rbf = 50
    e = np.array([10.0])

    horizon = 10

    # Default initial distribution for computing trajectory cost
    mu = np.random.randn(1, state_dim)
    sigma = np.random.randn(state_dim, state_dim)
    sigma = sigma.dot(sigma.T)

    # some random target state to reach
    target_state = np.random.rand(state_dim)

    # ---------------------------------------------------------------------------------------
    # Pilco setup

    # setup loss
    T_inv = np.random.randn(state_dim, state_dim)
    loss = SaturatedLoss(state_dim=state_dim, target_state=target_state, T_inv=T_inv)

    # take any env, to avoid issues with gym.make
    pilco = PILCO(env_name="MountainCarContinuous-v0", seed=1, n_features=n_features_rbf, Horizon=horizon, loss=loss,
                  bound=e, gamma=1, start_mu=mu.flatten(), start_cov=sigma, n_inducing_points=n_inducing_points)

    # Training Dataset for dynamics model
    X0_dyn = np.random.rand(n_samples, state_dim + n_actions)
    A_dyn = np.random.rand(state_dim + n_actions, n_targets)
    Y0_dyn = np.sin(X0_dyn).dot(A_dyn) + 1e-3 * (np.random.rand(n_samples, n_targets) - 0.5)

    # set observed data set manually
    pilco.state_action_pairs = X0_dyn
    pilco.state_delta = Y0_dyn
    pilco.state_dim = state_dim
    pilco.n_actions = n_actions

    pilco.learn_dynamics_model()

    # ---------------------------------------------------------------------------------------
    # Policy setup

    X0_rbf = np.random.rand(n_features_rbf, state_dim)
    A_rbf = np.random.rand(state_dim, n_actions)
    Y0_rbf = np.sin(X0_rbf).dot(A_rbf) + 1e-3 * (np.random.rand(n_features_rbf, n_actions) - 0.5)
    length_scales_rbf = np.random.rand(n_actions, state_dim)

    rbf = RBFController(n_actions=n_actions, n_features=n_features_rbf, compute_cost=None,
                        length_scales=length_scales_rbf)
    rbf.compute_cost = pilco.compute_trajectory_cost
    rbf.fit(X0_rbf, Y0_rbf)

    # ---------------------------------------------------------------------------------------

    params = np.array([gp.wrap_policy_hyperparams() for gp in rbf.gp_container]).flatten()
    np.allclose(grad(rbf._optimize_hyperparams)(params), jacobian(rbf._optimize_hyperparams)(params))


if __name__ == '__main__':
    test_grad_mgpr()
    test_grad_smgpr()
