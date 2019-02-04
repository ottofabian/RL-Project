from functools import partial

from autograd import grad, jacobian
from autograd.test_util import check_grads
from scipy.optimize import check_grad

from autograd import numpy as np

from PILCO.Controller.RBFController import RBFController, squash_action_dist
from PILCO.CostFunctions.SaturatedLoss import SaturatedLoss
from PILCO.PILCO import PILCO

check_grads = partial(check_grads, modes=['rev'])


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
    rbf.fit(X0_rbf, Y0_rbf)
    pilco.policy = rbf

    # ---------------------------------------------------------------------------------------

    params = np.array([gp.wrap_policy_hyperparams() for gp in rbf.gp_container]).flatten()
    np.allclose(grad(pilco._optimize_hyperparams)(params), jacobian(pilco._optimize_hyperparams)(params))

    # def check_it(x):
    #     x1 = x[:state_dim]
    #     x2 = x[state_dim:].reshape(state_dim, state_dim)
    #     return rbf._optimize_hyperparams()

    # grad_error = check_grad(func=rbf._optimize_hyperparams, grad=grad(rbf._optimize_hyperparams), x0=params)
    # print(grad_error)
    # np.testing.assert_almost_equal(grad_error, 0, decimal=7)

    check_grads(pilco._optimize_hyperparams)(params)


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
    rbf.fit(X0_rbf, Y0_rbf)

    pilco.policy = rbf
    # ---------------------------------------------------------------------------------------

    params = np.array([gp.wrap_policy_hyperparams() for gp in rbf.gp_container]).flatten()
    np.allclose(grad(pilco._optimize_hyperparams)(params), jacobian(pilco._optimize_hyperparams)(params))

    def helper(x):
        return pilco._optimize_hyperparams(x)

    check_grads(helper)(params)


def test_grad_loss():
    np.random.seed(0)

    state_dim = 2
    mu = np.random.rand(1, state_dim)
    sigma = np.random.rand(state_dim, state_dim)
    sigma = sigma.dot(sigma.T)

    target_state = np.random.rand(state_dim)
    T_inv = np.random.randn(state_dim, state_dim)

    loss = SaturatedLoss(state_dim=state_dim, target_state=target_state, T_inv=T_inv)

    # grad_error = check_grad(func=loss.compute_loss, grad=grad(loss.compute_loss), x0=params)
    check_grads(loss.compute_loss)(mu, sigma)


def test_grad_squash():
    np.random.seed(0)

    n_actions = 1
    mu = np.random.rand(1, n_actions)
    sigma = np.random.rand(n_actions, n_actions)
    sigma = sigma.dot(sigma.T)
    i_o_cov = np.eye(n_actions)
    e = np.array([7.0])

    def helper1(x):
        m2, s2, io2 = squash_action_dist(x, sigma, i_o_cov, e)
        return m2

    def helper2(x):
        x = x.reshape(n_actions, n_actions)
        m2, s2, io2 = squash_action_dist(mu, x, i_o_cov, e)
        return s2

    def helper3(x):
        x = x.reshape(n_actions, n_actions)
        m2, s2, io2 = squash_action_dist(mu, sigma, x, e)
        return io2

    check_grads(helper1)(mu.flatten())
    check_grads(helper2)(sigma.flatten())
    check_grads(helper3)(i_o_cov.flatten())


def test_grad_rollout():
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

    # ---------------------------------------------------------------------------------------

    # setup policy
    X0_rbf = np.random.rand(n_features_rbf, state_dim)
    A_rbf = np.random.rand(state_dim, n_actions)
    Y0_rbf = np.sin(X0_rbf).dot(A_rbf) + 1e-3 * (np.random.rand(n_features_rbf, n_actions) - 0.5)
    length_scales_rbf = np.random.rand(n_actions, state_dim)

    rbf = RBFController(n_actions=n_actions, n_features=n_features_rbf, compute_cost=None,
                        length_scales=length_scales_rbf)
    rbf.fit(X0_rbf, Y0_rbf)

    # ---------------------------------------------------------------------------------------

    # Pilco setup

    # take any env, to avoid issues with gym.make
    # matlab is specified with squashing, so we assume bound bound
    pilco = PILCO(env_name="MountainCarContinuous-v0", seed=1, n_features=None, Horizon=None, loss=None,
                  bound=bound, n_inducing_points=50)

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

    target_state = np.random.rand(state_dim)
    T_inv = np.random.randn(state_dim, state_dim)

    loss = SaturatedLoss(state_dim=state_dim, target_state=target_state, T_inv=T_inv)

    def helper1(x):
        m = x[0:state_dim]
        s = x[state_dim:state_dim + state_dim * state_dim].reshape(state_dim, state_dim)
        state_next_mu, state_next_cov, action_mu, action_cov = pilco.rollout(rbf, m, s)
        return loss.compute_loss(state_next_mu, state_next_cov)

    def helper2(x):
        m = x[0:state_dim]
        s = x[state_dim:state_dim + state_dim * state_dim].reshape(state_dim, state_dim)
        state_next_mu, state_next_cov, action_mu, action_cov = pilco.rollout(rbf, m, s)
        return state_next_mu

    def helper3(x):
        m = x[0:state_dim]
        s = x[state_dim:state_dim + state_dim * state_dim].reshape(state_dim, state_dim)
        state_next_mu, state_next_cov, action_mu, action_cov = pilco.rollout(rbf, m, s)
        return state_next_cov

    check_grads(helper1)(np.concatenate([mean.flatten(), sigma.flatten()]))
    check_grads(helper2)(np.concatenate([mean.flatten(), sigma.flatten()]))
    check_grads(helper3)(np.concatenate([mean.flatten(), sigma.flatten()]))

    # z = check_grad(helper1, jacobian(helper1), np.concatenate([mean.flatten(), sigma.flatten()]))
    # print(z)


if __name__ == '__main__':
    test_grad_mgpr()
    test_grad_smgpr()
    test_grad_loss()
    test_grad_rollout()
    test_grad_squash()
