# from LasVSim.endtoend import EndtoendEnv
# # import gym
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# def action_fn(obs):
#     detected_objects, all_objects, ego_dynamics, ego_info = obs
#     return np.array([1, 0])
#
# env = EndtoendEnv('./Scenario/Highway_endtoend/simulation_setting_file.xml')
# done = 0
# episode_num = 10
# for i in range(episode_num):  # run episode_num episodes
#     obs = env.reset()
#     while not done:
#         action = action_fn(obs)
#         obs, rew, done, info = env.step(action)
class Testt:
    def __init__(self):
        pass

    def testtt(self, b):
        self.a = 1
        b = b
        def fn():
            return b
        return fn

    def change_a(self, x):
        fn = self.testtt(x)
        self.a = fn()
        return self.a


if __name__ == '__main__':
    testt = Testt()
    a = testt.change_a(1)
    print(a)
    a = testt.change_a(2)
    print(a)



