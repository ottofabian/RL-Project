# from A3C.A3C import A3C
# from PILCO.PILCO import PILCO

# algorithm = PILCO()
# algorithm = A3C()
from A3C.A3C import A3C

a3c = A3C(n_worker=1, env_name='CartPole-v0')
a3c.run()
