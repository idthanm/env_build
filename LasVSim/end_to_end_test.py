from LasVSim.endtoend import EndtoendEnv, RewardWrapper, ObservationWrapper
# import gym
import matplotlib.pyplot as plt
import numpy as np
import os

def action_fn(obs):
    # grid_list, supplement_vector = obs
    return np.array([1, 0])


# class Testt:
#     def __init__(self):
#         pass
#
#     def testtt(self, b):
#         self.a = 1
#         b = b
#         def fn():
#             return b
#         return fn
#
#     def change_a(self, x):
#         fn = self.testtt(x)
#         self.a = fn()
#         return self.a


if __name__ == '__main__':
    env = EndtoendEnv('./Scenario/Highway_endtoend/')
    env = RewardWrapper(env)
    env = ObservationWrapper(env)
    done = 0
    episode_num = 10
    for i in range(episode_num):  # run episode_num episodes
        done = 0
        obs = env.reset()
        ret = 0
        while not done:
            action = action_fn(obs)
            obs, (position_bias, velocity_bias, heading_bias, rew), done, info = env.step(action)
            ret += rew

            print('reward: ', rew, 'other: ', position_bias, velocity_bias, heading_bias)

        print('return: ', ret)


