# Overview of Experiments on the [Quanser-Robot](https://git.ias.informatik.tu-darmstadt.de/quanser/clients) environments

---

# A3C (Asynchronous Advantage Actor-Critic) / A2C 

## CartpoleStabShort-v0

| done | comment                          | solved env | type |lr-actor|lr-critic|shared-model|value-loss-weight|discount| tau | entropy-loss-weight | max-grad-norm | seed | worker | rollout steps|max-episode-length|shared-optimizer|optimizer|lr-scheduler|max-action|normalizer|n-envs| test reward (10 episodes) | steps                | training time          |
| ---- | -------------------------------- | ---------- | ---  |------- | ------- | -----------| ----------------|------- | --- | --------------------| ------------- | -----| ------ | ------------ | ---------------- | -------------- | ------- | ---------- | -------- | -------- | ---- | ------------------------- | -------------------- | ---------------------- |
| [x]  | baseline (QG)                    | [x]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 1.0 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | 8228.67 - 9999.97         | 938,750 - 53,420,290 | 06m 27s - 06h 00m 11s  |
| [ ]  | baseline, type=A3C (BD)          | [?]        | A3C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 1.0 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | XXXXXXX - XXXXXXX         | XXXXXXX - XXXXXXXXXX | XXXXXXX - XXXXXXXXXXX  |
| [ ]  | baseline, tau=0.99 (BD)          | [?]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   |0.99 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | XXXXXXX - XXXXXXX         | XXXXXXX - XXXXXXXXXX | XXXXXXX - XXXXXXXXXXX  |
| [ ]  | baseline, tau=0.95 (BD)          | [?]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   |0.95 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | XXXXXXX - XXXXXXX         | XXXXXXX - XXXXXXXXXX | XXXXXXX - XXXXXXXXXXX  |
| [ ]  | baseline, tau=0.90 (BD)          | [?]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   |0.90 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | XXXXXXX - XXXXXXX         | XXXXXXX - XXXXXXXXXX | XXXXXXX - XXXXXXXXXXX  |
| [ ]  | baseline, shared-model=True (BD) | [?]        | A2C  | 0.0001 | 0.001   | TRUE       | 0.5             | 0.99   | 1.0 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | XXXXXXX - XXXXXXX         | XXXXXXX - XXXXXXXXXX | XXXXXXX - XXXXXXXXXXX  |
| [ ]  | baseline, lr-actor=0.001 (BD)    | [?]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 1.0 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | XXXXXXX - XXXXXXX         | XXXXXXX - XXXXXXXXXX | XXXXXXX - XXXXXXXXXXX  |
| [ ]  | baseline, n-envs=10 (BD)         | [?]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 1.0 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 10   | XXXXXXX - XXXXXXX         | XXXXXXX - XXXXXXXXXX | XXXXXXX - XXXXXXXXXXX  |
| [ ]  | baseline, optimize=rmsprop (BD)  | [?]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 1.0 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | rmsprop | None       | +/-5     | None     | 5    | XXXXXXX - XXXXXXX         | XXXXXXX - XXXXXXXXXX | XXXXXXX - XXXXXXXXXXX  |
| [ ]  | baseline, rollout=100 (BD)       | [?]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 1.0 | 0.0001              | 1.0           | 1    |  1     | 100          | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | XXXXXXX - XXXXXXX         | XXXXXXX - XXXXXXXXXX | XXXXXXX - XXXXXXXXXXX  |
| [x]  | baseline, rollout=20 (QG,L)      | [?]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 1.0 | 0.0001              | 1.0           | 1    |  1     | 20           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | XXXXXXX - XXXXXXX         | 403,100 - 58.181,185 | 00h 06m 11s - 07h 29m 59s  |

* [x] solved in simulation 
* [x] solved in Q-Lab (real world) [2018.12.20]

 _Our learnt policy was able to stabilize the pole on the first trial.
 We set our maximum action to +/- 10. This resulted in a rather rough policy where the cartpole robot needed to correct
 its position in order to keep the pole balanced._


## CartpoleSwingShort-v0

| done | comment                          | solved env | type |lr-actor|lr-critic|shared-model|value-loss-weight|discount| tau | entropy-loss-weight | max-grad-norm | seed | worker | rollout steps|max-episode-length|shared-optimizer|optimizer|lr-scheduler|max-action|normalizer|n-envs| test reward (10 episodes) | steps                    | training time          |
| ---- | -------------------------------- | ---------- | ---  |------- | ------- | -----------| ----------------|------- | --- | --------------------| ------------- | -----| ------ | ------------ | ---------------- | -------------- | ------- | ---------- | -------- | -------- | ---- | ------------------------- | ------------------------ | ---------------------- |
| [x]  | baseline (BD)                    | [x]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 1.0 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-10    | None     | 5    | 9518                      | 13,630,085               | 08h 08m 47s            |
| [x]  | baseline, action_space=+/-5 (QG) | [x]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 1.0 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | 4992.49                   | 9,245,565                | 01h 05m 17s            |

* [x] solved in simulation
* [ ] solved for Q-Lab

## Qube-v0 / a.k.a Furuta-Pendulum

| done | solved env | type |lr-actor|lr-critic|shared-model|value-loss-weight|discount| tau | entropy-loss-weight | max-grad-norm | seed | worker | rollout steps|max-episode-length|shared-optimizer|optimizer|lr-scheduler|max-action|normalizer|n-envs| test reward (10 episodes) | steps                    | training time              |
| ---- | ---------- | ---  |------- | ------- | -----------| ----------------|------- | --- | --------------------| ------------- | -----| ------ | ------------ | ---------------- | -------------- | ------- | ---------- | -------- | -------- | ---- | ------------------------- | ------------------------ | -------------------------- |
| [x]  | [x] (QG)   | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 1.0 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | 2.85 - 3.44               | 15,7983,845 - 72,839,490 | 10h 00m 00s - 15h 00m 00s  |

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
