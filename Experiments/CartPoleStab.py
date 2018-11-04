import gym

# from A3C.A3C import A3C
# from PILCO.PILCO import PILCO

# algorithm = PILCO()
# algorithm = A3C()


env = gym.make('CartPole-v0')
print(env.action_space)
print(env.observation_space)

env.reset()

# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t + 1))
#             break
