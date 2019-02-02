import logging
import gym
from autograd import numpy as np

from Experiments.util.ColorLogger import enable_color_logging
from PILCO.CostFunctions.SaturatedLoss import SaturatedLoss
from PILCO.PILCO import PILCO


def main():
    enable_color_logging(logging_lvl=logging.DEBUG)
    logging.info('Start Experiment')

    seed = 1
    # env_name = "Pendulum-v0"

    env_name = "CartpoleStabShort-v0"
    # env_name = "CartpoleStabRR-v0"
    # env_name = "CartpoleSwingShort-v0"
    # env_name = "CartpoleSwingRR-v0"
    # env_name = "Qube-v0"
    # env_name = "QubeRR-v0"

    env = gym.make(env_name)

    n_inital_samples = 300
    max_samples_per_test_run = 300
    n_inducing_points = 300
    n_features = 10
    horizon = 100

    # get target state value for computing loss
    if "Cartpole" in env_name:
        target_state = np.array([0, 0, -1, 0, 0])
    elif "Pendulum" in env_name:
        target_state = np.array([1, 0, 0])

    # get initial mu and cov for trajectory rollouts
    if env_name == "Pendulum-v0":
        # this acts like pendulum stabilization or swing up to work with easier 3D obs space
        theta = 0
        start_mu = np.array([np.cos(theta), np.sin(theta), 0])
        bound = np.array([2])
    elif env_name == "CartpoleStabShort-v0":
        theta = np.pi
        start_mu = np.array([0., np.sin(theta), np.cos(theta), 0., 0.])
        bound = np.array([5])
    elif env_name == "CartpoleSwingShort-v0":
        theta = 0
        start_mu = np.array([0., np.sin(theta), np.cos(theta), 0., 0.])
        bound = np.array([5])

    start_cov = 1e-2 * np.identity(env.observation_space.shape[0])
    # --------------------------------------------------------
    # Alternatives:

    # state_cov = X[:, :self.state_dim].std(axis=0)
    # state_cov = np.cov(X[:, :self.state_dim], rowvar=False
    # --------------------------------------------------------

    loss = SaturatedLoss(state_dim=env.observation_space.shape[0], target_state=target_state)
    pilco = PILCO(env_name=env_name, seed=seed, n_features=n_features, Horizon=horizon, loss=loss,
                  max_samples_per_test_run=max_samples_per_test_run, gamma=1, start_mu=start_mu, start_cov=start_cov, bound=bound,
                  n_inducing_points=n_inducing_points)
    pilco.run(n_samples=n_inital_samples)


if __name__ == '__main__':
    main()
