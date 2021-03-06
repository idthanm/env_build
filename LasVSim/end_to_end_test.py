from LasVSim.endtoend import CrossroadEnd2end
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
    env = CrossroadEnd2end('./Scenario/Intersection_endtoend/')
    done = 0
    episode_num = 10
    for i in range(episode_num):  # run episode_num episodes
        done = 0
        kwarg = dict(init_state=[3.75/2, -18-2.5, 0, 90],
                     task=0)
        obs = env.reset(**kwarg)
        ret = 0
        ite = 0
        while not done:
            ite += 1
            action = action_fn(obs)
            obs, rew, done, info = env.step(action)
            env.render()
            ret += rew
            print('reward: ', rew)

        print('return: ', ret)


