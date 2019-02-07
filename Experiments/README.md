# Overview of Experiments on the [Quanser-Robot](https://git.ias.informatik.tu-darmstadt.de/quanser/clients) environments

---

# A3C (Asynchronous Advantage Actor-Critic) / A2C 

We changed the default environment episode length from 10,000 to 5,000 in order to avoid computational overhead.
For stabilization there was no advantage of training with longer episode length in the final policy.
With the Swing-up we fine-tuned our final policy with 5,000 episode length using 10,000 length in order to stay within the 
environment boundaries.

## CartpoleStabShort-v0

| done | contributor | comment                                     | solved env | type |lr-actor|lr-critic|shared-model|value-loss-weight|discount| tau | entropy-loss-weight | max-grad-norm | seed | worker | rollout steps|max-episode-length|shared-optimizer|optimizer|lr-scheduler|max-action|normalizer|n-envs| test reward (10 episodes) | steps                | training time          |
| ---- | ----------- | ------------------------------------------- | ---------- | ---  |------- | ------- | -----------| ----------------|------- | --- | --------------------| ------------- | -----| ------ | ------------ | ---------------- | -------------- | ------- | ---------- | -------- | -------- | ---- | ------------------------- | -------------------- | ---------------------- |
| [x]  | BD          | baseline                                    | [x]        | A2C  | 1e-4   | 1e-3    | FALSE      | 0.5             | 0.99   |0.99 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    |           9999.92         |            2,736,500 |           00h 48m 29s  |
| [x]  | QG          | baseline, tau=1.0                           | [x]        | A2C  | 1e-4   | 1e-3    | FALSE      | 0.5             | 0.99   | 1.0 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    |9999.33 - 9999.97          | 1,941,500 - 53,420,290 | 00h 13m 22s - 06h 00m 11s  |
| [x]  | BD          | baseline, tau=0.95                          | [x]        | A2C  | 1e-4   | 1e-3    | FALSE      | 0.5             | 0.99   |0.95 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    |           9999.39         |            1,726,550 |           00h 31m 33s  |
| [x]  | BD          | baseline, tau=0.90                          | [x]        | A2C  | 1e-4   | 1e-3    | FALSE      | 0.5             | 0.99   |0.90 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    |           9999.85         |            2,275,295 |           00h 39m 51s  |
| [x]  | BD          | baseline, type=A3C                          | [x]        | A3C  | 1e-4   | 1e-3    | FALSE      | 0.5             | 0.99   |0.99 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    |           9996.78         |            9,611,694 |           08h 09m 04s  |
| [x]  | BD          | baseline, shared-model=True                 | [x]        | A2C  | 1e-4   | 1e-3    | TRUE       | 0.5             | 0.99   |0.99 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    |           9996.70         |            6,637,935 |           01h 25m 00s  |
| [x]  | BD          | baseline, lr-actor=0.001                    | [x]        | A2C  | 1e-3   | 1e-3    | FALSE      | 0.5             | 0.99   |0.99 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    |           9999.86         |            5,239,500 |           01h 34m 04s  |
| [x]  | BD          | baseline, lr-critic=0.01                    | [x]        | A2C  | 1e-4   | 1e-2    | FALSE      | 0.5             | 0.99   |0.99 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    |           9999.20         |              781,630 |           00h 14m 13s  |
| [ ]  | BD          | baseline, lr-critic=0.01, lr-actor=0.001    | [?]        | A2C  | 1e-3   | 1e-2    | FALSE      | 0.5             | 0.99   |0.99 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    |            999.34         |              629,435 |           00h 11m 41s  |
| [ ]  | QG          | baseline, n-envs=10                         | [?]        | A2C  | 1e-4   | 1e-3    | FALSE      | 0.5             | 0.99   |0.99 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 10   | XXXXXXX - XXXXXXX         | XXXXXXX - XXXXXXXXXX | XXXXXXX - XXXXXXXXXXX  |
| [ ]  | BD          | baseline, optimize=rmsprop                  | [?]        | A2C  | 1e-4   | 1e-3    | FALSE      | 0.5             | 0.99   |0.99 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | rmsprop | None       | +/-5     | None     | 5    | XXXXXXX - XXXXXXX         | XXXXXXX - XXXXXXXXXX | XXXXXXX - XXXXXXXXXXX  |
| [ ]  | BD          | baseline, rollout=100                       | [?]        | A2C  | 1e-4   | 1e-3    | FALSE      | 0.5             | 0.99   |0.99 | 0.0001              | 1.0           | 1    |  1     | 100          | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | XXXXXXX - XXXXXXX         | XXXXXXX - XXXXXXXXXX | XXXXXXX - XXXXXXXXXXX  |
| [ ]  | BD          | baseline, rollout=20                        | [?]        | A2C  | 1e-4   | 1e-3    | FALSE      | 0.5             | 0.99   |0.99 | 0.0001              | 1.0           | 1    |  1     | 20           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | XXXXXXX - XXXXXXX         | XXXXXXX - XXXXXXXXXX | XXXXXXX - XXXXXXXXXX   |
| [ ]  | BD          | baseline, discount=0.999                    | [?]        | A2C  | 1e-4   | 1e-3    | FALSE      | 0.5             | 0.999  |0.99 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | XXXXXXX - XXXXXXX         | XXXXXXX - XXXXXXXXXX | XXXXXXX - XXXXXXXXXX   |
| [ ]  | BD          | baseline, discount=0.95                     | [?]        | A2C  | 1e-4   | 1e-3    | FALSE      | 0.5             | 0.95   |0.99 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | XXXXXXX - XXXXXXX         | XXXXXXX - XXXXXXXXXX | XXXXXXX - XXXXXXXXXX   |
| [ ]  | BD          | baseline, discount=0.90                     | [?]        | A2C  | 1e-4   | 1e-3    | FALSE      | 0.5             | 0.90   |0.99 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | XXXXXXX - XXXXXXX         | XXXXXXX - XXXXXXXXXX | XXXXXXX - XXXXXXXXXX   |
| [ ]  | QG,L        | baseline, lr-scheduler=Step                 | [?]        | A2C  | 1e-4   | 1e-3    | FALSE      | 0.5             | 0.99   |0.99 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | XXXXXXX - XXXXXXX         | XXXXXXX - XXXXXXXXXX | XXXXXXX - XXXXXXXXXX   |
| [ ]  | QG,L        | baseline, normalizer=True                   | [?]        | A2C  | 1e-4   | 1e-3    | FALSE      | 0.5             | 0.99   |0.99 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | XXXXXXX - XXXXXXX         | XXXXXXX - XXXXXXXXXX | XXXXXXX - XXXXXXXXXX   |

* [x] solved in simulation 
* [x] solved in Q-Lab (real world) [2018.12.20]

 _Our learnt policy was able to stabilize the pole on the first trial.
 We set our maximum action to +/- 10. This resulted in a rather rough policy where the cartpole robot needed to correct
 its position in order to keep the pole balanced._


## CartpoleSwingShort-v0

| done | contributor | comment                              | solved env | type |lr-actor|lr-critic|shared-model|value-loss-weight|discount| tau | entropy-loss-weight | max-grad-norm | seed | worker | rollout steps|max-episode-length|shared-optimizer|optimizer|lr-scheduler|max-action|normalizer|n-envs| test reward (10 episodes) | steps                    | training time          |
| ---- | ----------- | ------------------------------------ | ---------- | ---  |------- | ------- | -----------| ----------------|------- | --- | --------------------| ------------- | -----| ------ | ------------ | ---------------- | -------------- | ------- | ---------- | -------- | -------- | ---- | ------------------------- | ------------------------ | ---------------------- |
| [x]  | QG          | baseline                             | [x]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 0.99| 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-10    | None     | 5    | 9523                      | 15,060,000               | 01h 45m 04s            |
| [ ]  | QG          | baseline, action_space=+/-5          | [ ]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 0.99| 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | 4992.49                   | 9,245,565                | 01h 05m 17s            |
| [x]  | BD          | baseline, tau=1.0                    | [x]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 1.0 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-10    | None     | 5    | 9518                      | 13,630,085               | 08h 08m 47s            |
| [ ]  | QG          | baseline, tau=0.95                   | [ ]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 0.95| 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-10    | None     | 5    | 9518                      | 13,630,085               | 08h 08m 47s            |
| [ ]  | QG          | baseline, tau=0.90                   | [x]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 0.9 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-10    | None     | 5    | 9518                      | 13,630,085               | 08h 08m 47s            |
| [x]  | QG          | baseline, tau=1.0, action_space=+/-5 | [-]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 0.99| 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | 4992.49                   | 9,245,565                | 01h 05m 17s            |
| [ ]  | QG          | baseline, type=A3C                   | [ ]        | A3C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 0.99| 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-10    | None     | 5    | 9518                      | 13,630,085               | 08h 08m 47s            |
| [ ]  | BD          | baseline, rollout=20                 | [ ]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 0.99| 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-10    | None     | 5    | 9518                      | 13,630,085               | 08h 08m 47s            |
| [ ]  | QG          | baseline, rollout=100                | [ ]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 0.99| 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-10    | None     | 5    | 9518                      | 13,630,085               | 08h 08m 47s            |
| [ ]  | QG          | baseline, discount=0.95              | [ ]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.95   | 0.99| 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-10    | None     | 5    | 9518                      | 13,630,085               | 08h 08m 47s            |
| [ ]  | QG          | baseline, n_envs=10                  | [ ]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.95   | 0.99| 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-10    | None     | 10   | 9518                      | 13,630,085               | 08h 08m 47s            |

* [x] solved in simulation
* [ ] solved for Q-Lab

## Qube-v0 / a.k.a Furuta-Pendulum

| done | contributor | comment                               | solved env | type |lr-actor|lr-critic|shared-model|value-loss-weight|discount| tau | entropy-loss-weight | max-grad-norm | seed | worker | rollout steps|max-episode-length|shared-optimizer|optimizer|lr-scheduler|max-action|normalizer|n-envs| test reward (10 episodes) | steps                    | training time              |
| ---- | ------------| ------------------------------------- | ---------- | ---- |------- | ------- | -----------| ----------------|------- | --- | --------------------| ------------- | -----| ------ | ------------ | ---------------- | -------------- | ------- | ---------- | -------- | -------- | ---- | ------------------------- | ------------------------ | -------------------------- |
| [ ]  | QG          | baseline                              | [?]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 0.99| 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | 2.85 - 3.44               | 15,7983,845 - 72,839,490 | 10h 00m 00s - 15h 00m 00s  |
| [x]  | QG          | baseline, tau=1.0                     | [x]        | A2C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 1.0 | 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | 2.85 - 3.44               | 15,7983,845 - 72,839,490 | 10h 00m 00s - 15h 00m 00s  |
| [ ]  | QG          | baseline, A3C                         | [?]        | A3C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 0.99| 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | 2.85 - 3.44               | 15,7983,845 - 72,839,490 | 10h 00m 00s - 15h 00m 00s  |
| [ ]  | QG          | baseline, scale-reward(100)           | [?]        | A3C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 0.99| 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | 2.85 - 3.44               | 15,7983,845 - 72,839,490 | 10h 00m 00s - 15h 00m 00s  |
| [ ]  | QG          | baseline, no-exp-reward               | [?]        | A3C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 0.99| 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | 2.85 - 3.44               | 15,7983,845 - 72,839,490 | 10h 00m 00s - 15h 00m 00s  |
| [ ]  | QG          | baseline, entropy-loss-weight 1e-7    | [?]        | A3C  | 0.0001 | 0.001   | FALSE      | 0.5             | 0.99   | 0.99| 0.0001              | 1.0           | 1    |  1     | 50           | 5000             | TRUE           | adam    | None       | +/-5     | None     | 5    | 2.85 - 3.44               | 15,7983,845 - 72,839,490 | 10h 00m 00s - 15h 00m 00s  |

* change rewards definition in clients.py

* [x] solved in simulation 
* [ ] solved in Q-Lab (real world)


# PILCO

## Pendulum-v0

* [ ] solved in simulation
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