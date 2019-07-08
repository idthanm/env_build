from LasVSim.endtoend import EndtoendEnv
import gym
import matplotlib.pyplot as plt
import numpy as np
import os

env = EndtoendEnv('./Library/default_simulation_setting.xml')
# plt.figure(1)
# x = np.linspace(0, 2, 200)  # 在0到2pi之间，均匀产生200点的数组
#
# y = 0.75 * x**2 - 0.25 * x**3  # 半径
# plt.plot(x, y)
# plt.show()
abp = os.path.abspath('.')
print(abp)
print('aaaaa')
# with open('Library/vehicle_model_library.csv') as f:
#     a = f.readline()
#     print(a)

