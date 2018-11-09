from A3C.A3C import A3C

from PILCO.PILCO import PILCO

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

# a3c = A3C(n_worker=4, env_name='CartPole-v0', lr=1e-3, is_discrete=True)

a3c = A3C(n_worker=4, env_name='CartpoleStabShort-v0', lr=1e-3, is_discrete=False)
a3c.run()


pilco = PILCO(env_name='CartPole-v0', seed=42, n_features=4)
pilco.run(50)
