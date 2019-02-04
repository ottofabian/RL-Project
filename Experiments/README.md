# Overview of Experiments on the [Quanser-Robot](https://git.ias.informatik.tu-darmstadt.de/quanser/clients) environments

---

# A3C (Asynchronous Advantage Actor-Critic) / A2C 

## CartpoleStabShort-v0

| done | comment                          | solved env | type |lr-actor|lr-critic|shared-model|value-loss-weight|discount| tau | entropy-loss-weight | max-grad-norm | seed | worker | rollout steps|max-episode-length|shared-optimizer|optimizer|lr-scheduler|max-action|normalizer|n-envs| test reward (10 episodes) | steps                | training time          |
| ---- | -------------------------------- | ---------- | ---  |------- | ------- | -----------| ----------------|------- | --- | --------------------| ------------- | -----| ------ | ------------ | ---------------- | -------------- | ------- | ---------- | -------- | -------- | ---- | ------------------------- | -------------------- | ---------------------- |
| [x]  | baseline                         | [x]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 1.0 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | 8228.67 - 9999.97         | 938,750 - 53,420,290 | 06m 27s - 06h 00m 11s  |
| [ ]  | baseline, type=A3C (QG)          | [?]        | A3C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 1.0 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | XXXXXXX - XXXXXXX         | XXXXXXX - XXXXXXXXXX | XXXXXXX - XXXXXXXXXXX  |
| [ ]  | baseline, tau=0.99 (BD)          | [?]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   |0.99 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | XXXXXXX - XXXXXXX         | XXXXXXX - XXXXXXXXXX | XXXXXXX - XXXXXXXXXXX  |
| [ ]  | baseline, shared-model=True (QG) | [?]        | A2C  | 0.0001 | 0.001   | TRUE       | 0.5             | 0.99   | 1.0 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | XXXXXXX - XXXXXXX         | XXXXXXX - XXXXXXXXXX | XXXXXXX - XXXXXXXXXXX  |
| [ ]  | baseline, lr-actor=0.001 (BD)    | [?]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 1.0 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | XXXXXXX - XXXXXXX         | XXXXXXX - XXXXXXXXXX | XXXXXXX - XXXXXXXXXXX  |
| [ ]  | baseline, n-envs=10 (QG)         | [?]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 1.0 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 10   | XXXXXXX - XXXXXXX         | XXXXXXX - XXXXXXXXXX | XXXXXXX - XXXXXXXXXXX  |
| [ ]  | baseline, optimize=rmsprop (BD)  | [?]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 1.0 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | XXXXXXX - XXXXXXX         | XXXXXXX - XXXXXXXXXX | XXXXXXX - XXXXXXXXXXX  |

* [x] solved in simulation 
* [x] solved in Q-Lab (real world) [2018.12.20]

 _Our learnt policy was able to stabilize the pole on the first trial.
 We set our maximum action to +/- 10. This resulted in a rather rough policy where the cartpole robot needed to correct
 its position in order to keep the pole balanced._


## CartpoleSwingShort-v0

| done | solved env | type |lr-actor|lr-critic|shared-model|value-loss-weight|discount| tau | entropy-loss-weight | max-grad-norm | seed | worker | rollout steps|max-episode-length|shared-optimizer|optimizer|lr-scheduler|max-action|normalizer|n-envs| test reward (10 episodes) | steps                    | training time          |
| ---- | ---------- | ---  |------- | ------- | -----------| ----------------|------- | --- | --------------------| ------------- | -----| ------ | ------------ | ---------------- | -------------- | ------- | ---------- | -------- | -------- | ---- | ------------------------- | ------------------------ | ---------------------- |
| [x]  | [x]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 1.0 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | 2.85 - 3.44               | XXXXXXX - XXXXXXX        | XXXXXXX - XXXXXXXXXXX  |

* [x] solved in simulation
* [ ] solved for Q-Lab

## Qube-v0 / a.k.a Furuta-Pendulum

| done | solved env | type |lr-actor|lr-critic|shared-model|value-loss-weight|discount| tau | entropy-loss-weight | max-grad-norm | seed | worker | rollout steps|max-episode-length|shared-optimizer|optimizer|lr-scheduler|max-action|normalizer|n-envs| test reward (10 episodes) | steps                    | training time              |
| ---- | ---------- | ---  |------- | ------- | -----------| ----------------|------- | --- | --------------------| ------------- | -----| ------ | ------------ | ---------------- | -------------- | ------- | ---------- | -------- | -------- | ---- | ------------------------- | ------------------------ | -------------------------- |
| [x]  | [x]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 1.0 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | 2.85 - 3.44               | 15,7983,845 - 72,839,490 | 10h 00m 00s - 15h 00m 00s  |

* [x] solved in simulation 
* [ ] solved in Q-Lab (real world)


# PILCO 

## Pendulum-v0

* [x] solved in simulation
* [ ] solved in Q-Lab (real world) _-> we don't have a robot experiment for the pendulum_


## CartpoleStabShort-v0

* [ ] solved in simulation
* [ ] solved in Q-Lab (real world)


## CartpoleSwingShort-v0

* [ ] solved in simulation
* [ ] solved in Q-Lab (real world)

## Qube-v0 / a.k.a Furuta-Pendulum

* [ ] solved in simulation
* [ ] solved in Q-Lab (real world)
