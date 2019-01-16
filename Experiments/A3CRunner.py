import logging

import quanser_robots

from A3C.A3C import A3C
from Experiments.util.ColorLogger import enable_color_logging

quanser_robots
#
# CartpoleStabShort-v0
# ---------------------
# * installable using https://git.ias.informatik.tu-darmstadt.de/quanser/clients

seed = 1
# env_name = "Pendulum-v0"
# env_name = "CartpoleStabShort-v0"
env_name = "CartpoleSwingShort-v0"
# env_name = "CartpoleSwingRR-v0"

# env_name = "CartpoleStabRR-v0"
# env_name = "Qube-v0"

enable_color_logging(debug_lvl=logging.DEBUG)
logging.info('Start Experiment')
a3c = A3C(n_worker=1, env_name=env_name, is_discrete=False, seed=seed, optimizer_name='rmsprop')
# a3c.run_debug(
#     path_actor="./best_models/SwingUp/works but not stable/actor_T-135307781_global-3005.6435968448095.pth.tar",
#     path_critic="./best_models/SwingUp/works but not stable/critic_T-135307786_global-3005.6435968448095.pth.tar")
#
# a3c.run_debug(
#     path_actor="./best_models/Stabilization/Reduced action range/actor_T-6719059_global-9984.922698507235.pth.tar",
#     path_critic="./best_models/Stabilization/Reduced action range/critic_T-6719074_global-9984.922698507235.pth.tar")
#
# a3c.run_debug(
#     path_actor="./best_models/Stabilization/Full action range/actor_finetuned_T-7285824_global-1266.9597491827692.pth.tar",
#     path_critic="./best_models/Stabilization/Full action range/critic_finetuned_T-7285824_global-1266.9597491827692.pth.tar")

# a3c.run_debug()
# a3c.run()

