import logging

import quanser_robots

from Experiments.util.ColorLogger import enable_color_logging
from PILCO.CostFunctions.CostFunctions import cartpolebase_costfunc
from PILCO.PILCO import PILCO

quanser_robots

# currently tested on the following environments:
# CartPole-v0
# ------------
# SETTINGS
# * discrete=True

# RESULTS
# T=968, reward=21.0, episode_len=21
# T=14525, reward=63.0, episode_len=63
# T=27720, reward=49.0, episode_len=49
# T=43597, reward=180.0, episode_len=180
# T=60367, reward=200.0, episode_len=200
# T=76268, reward=200.0, episode_len=200
#
# CartpoleStabShort-v0
# ---------------------
# * installable using https://git.ias.informatik.tu-darmstadt.de/quanser/clients
# SETTINGS
# * discrete=False
#
# RESULTS
# T=3736, reward=174.13807806372643, episode_len=200
# T=19431, reward=190.78671061992645, episode_len=200
# T=35006, reward=199.9990211725235, episode_len=200
# T=50697, reward=199.99907058477402, episode_len=200

seed = 1

enable_color_logging(debug_lvl=logging.DEBUG)
logging.info('Start Experiment')
# a3c = A3C(n_worker=4, env_name='CartPole-v0', lr=1e-4, is_discrete=True, seed = seed, optimizer_name='adam')
# a2c = A2C(env_name='CartpoleStabShort-v0', lr=1e-4, is_discrete=False, seed=seed, optimizer_name='adam')
# a2c.train()
# a3c = A3C(n_worker=3, env_name='CartpoleStabShort-v0', lr=0.0001, is_discrete=False, seed=seed,
#           optimizer_name='rmsprop')
# a3c = A3C(n_worker=1, env_name='Pendulum-v0', lr=0.0001, is_discrete=False, seed=seed, optimizer_name='rmsprop')
# a3c.run()

# n_features in paper was 100
# pilco = PILCO(env_name='CartpoleStabShort-v0', seed=seed, n_features=100, Horizon=20,
#               cost_function=cartpolebase_costfunc, target_state=[0, 0, -1, 0, 0])
pilco = PILCO(env_name='CartpoleSwingShort-v0', seed=seed, n_features=2, Horizon=20,
              cost_function=cartpolebase_costfunc, target_state=[0, 0, -1, 0, 0])
pilco.run(200)
