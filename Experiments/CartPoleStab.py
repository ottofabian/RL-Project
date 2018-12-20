import logging

import gym
import numpy as np
import quanser_robots

from A3C.A3C import A3C
from Experiments.util.ColorLogger import enable_color_logging
from PILCO.CostFunctions.SaturatedLoss import SaturatedLoss
from PILCO.PILCO import PILCO

quanser_robots
#
# CartpoleStabShort-v0
# ---------------------
# * installable using https://git.ias.informatik.tu-darmstadt.de/quanser/clients

seed = 1
# env_name = "Pendulum-v0"
# env_name = "CartpoleStabShort-v0"
env_name = "CartpoleSwingShort-v0"
env_name = "CartpoleStabRR-v0"
# env_name = "Qube-v0"

enable_color_logging(debug_lvl=logging.DEBUG)
logging.info('Start Experiment')
a3c = A3C(n_worker=1, env_name=env_name, is_discrete=False, seed=seed, optimizer_name='rmsprop')
# a3c.run()
# a3c.run_debug(path_actor="./best_models/Stabilization/actor_T-6722137_global-6995.186679900222.pth.tar",
#               path_critic="./best_models/Stabilization/critic_T-6722140_global-6995.186679900222.pth.tar")
a3c.run_debug()

# env = gym.make(env_name)
# max_episode_steps = 200
#
# if "Cartpole" in env_name:
#     target_state = np.array([0, 0, -1, 0, 0])
# elif "Pendulum" in env_name:
#     target_state = np.array([1, 0, 0])
#
# loss = SaturatedLoss(state_dim=env.observation_space.shape[0], target_state=target_state)
# pilco = PILCO(env_name=env_name, seed=seed, n_features=100, Horizon=40, loss=loss, max_episode_steps=max_episode_steps,
#               gamma=.99)
# pilco.run(n_samples=max_episode_steps)
