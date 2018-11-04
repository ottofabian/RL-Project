import gym

from A3C.A3C import A3C
from PILCO.PILCO import PILCO

algorithm = PILCO()
algorithm = A3C()

env = gym.make('CartPole-v0')
env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())  # take a random action
