## Overview of Experiments on the [Quanser-Robot](https://git.ias.informatik.tu-darmstadt.de/quanser/clients) environments



# A3C (Asynchronous Advantage Actor-Critic) / A2C 

## CartpoleStabShort-v0

|lr-actor|lr-critic|shared-model|value-loss-weight|discount| tau | entropy-loss-weight | max-grad-norm | seed | worker | rollout steps|max-episode-length|shared-optimizer|optimizer|lr-scheduler|max-action|normalizer|n-envs| test reward (10 episodes) | steps        | training time |
| ------ | ------- | -----------| ----------------|------- | --- | --------------------| ------------- | -----| ------ | ------------ | ---------------- | -------------- | ------- | ---------- | -------- | -------- | ---- | ------------------------- | ------------ | ------------- |
|0.0001  | 0.001   | FALSE      | 0.5             | 0.99   | 1.0 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | 8228.67 - 9999.97         | 938750 - 53420290 | 06m 27s  |

* [x] solved


## CartpoleSwingShort-v0

* [x] solved

## Qube-v0 / a.k.a Furuta-Pendulum

* [x] solved


# PILCO 

## Pendulum-v0

* [x] solved


## CartpoleStabShort-v0

* [ ] solved


## CartpoleSwingShort-v0

* [ ] solved

## Qube-v0 / a.k.a Furuta-Pendulum

* [ ] solved