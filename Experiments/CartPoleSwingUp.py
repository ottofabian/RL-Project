from A3C.A3C import A3C

a3c = A3C(n_worker=4, env_name='CartPoleSwing-v0')
a3c.run()
