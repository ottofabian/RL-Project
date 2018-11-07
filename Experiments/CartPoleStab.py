from A3C.A3C import A3C

a3c = A3C(n_worker=4, env_name='Pendulum-v0', lr=1e-4)
a3c.run()
