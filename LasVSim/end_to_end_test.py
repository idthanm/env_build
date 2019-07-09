from LasVSim.endtoend import EndtoendEnv
import gym
import matplotlib.pyplot as plt
import numpy as np
import os

def action_fn(obs):
    detected_objects, all_objects, ego_dynamics, ego_info = obs
    return np.array([1, 0])

env = EndtoendEnv('./Scenario/Highway_endtoend/simulation_setting_file.xml')
done = 0
episode_num = 10
for i in range(episode_num):  # run episode_num episodes
    obs = env.reset()
    while not done:
        action = action_fn(obs)
        obs, rew, done, info = env.step(action)



