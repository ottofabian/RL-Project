from A3C.A3C import A3C

a3c = A3C(n_worker=4, env_name='CartpoleStabShort-v0', lr=1e-4, is_discrete=False)
a3c.run()
