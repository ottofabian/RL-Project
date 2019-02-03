import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import quanser_robots

np.random.seed(0)

env = gym.make('CartpoleStabShort-v0')


def rollout(policy, timesteps):
    X = []
    Y = []
    env.reset()
    x, _, _, _ = env.step(np.array([0]))
    for timestep in range(timesteps):
        env.render()
        u = policy(x)
        x_new, _, done, _ = env.step(u)
        if done:
            break
        X.append(np.hstack((x, u)))
        noise = np.random.multivariate_normal(np.zeros(x.shape), 1e-6 * np.identity(x.shape[0]))
        Y.append(x_new - x + noise)
        x = x_new
    return np.stack(X), np.stack(Y)


def random_policy(x):
    return env.action_space.sample()


def pilco_policy(x):
    return pilco.compute_action(x[None, :])[0, :]


# Initial random rollouts to generate a dataset
X, Y = rollout(policy=random_policy, timesteps=40)
for i in range(1, 5):
    X_, Y_ = rollout(policy=random_policy, timesteps=40)
    X = np.vstack((X, X_))
    Y = np.vstack((Y, Y_))

state_dim = Y.shape[1]
control_dim = X.shape[1] - state_dim
controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=25)
# controller = LinearController(state_dim=state_dim, control_dim=control_dim)

# pilco = PILCO(X, Y, controller=controller, horizon=40)
# Example of user provided reward function, setting a custom target state
R = ExponentialReward(state_dim=state_dim, t=np.array([0., 0., -1., 0., 0.]))
pilco = PILCO(X, Y, controller=controller, horizon=40, reward=R,
              m_init=np.array([0., np.sin(np.pi), np.cos(np.pi), 0., 0.]).reshape(1,-1))

# Example of fixing a parameter, optional, for a linear controller only
# pilco.controller.b = np.array([[0.0]])
# pilco.controller.b.trainable = False

for rollouts in range(50):
    pilco.optimize()
    # import pdb

    # pdb.set_trace()
    X_new, Y_new = rollout(policy=pilco_policy, timesteps=200)
    # Update dataset
    X = np.vstack((X, X_new))
    Y = np.vstack((Y, Y_new))
    pilco.mgpr.set_XY(X, Y)
