from LasVSim.endtoend import EndtoendEnv, ObservationWrapper
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
from gym.wrappers import ObservationWrapper

# def action_fn(last_timestep_vector, horizon):
#     ego_v, ego_heading, ego_length, ego_width, dist2current_lane_center, \
#     egolane_index, dist2roadleft, dist2roadright = last_timestep_vector
#
#     def _interested_lane_index(ego_lane_index):
#         info_list = [[1, 0, None], [2, 1, 0], [3, 2, 1], [None, 3, 2]]  # left, middle, right
#         return info_list[ego_lane_index]
#
#     def _laneindex2centery(lane_index):
#         center_y_list = [-150 - 7 * 3.75 / 2, -150 - 5 * 3.75 / 2, -150 - 3 * 3.75 / 2, -150 - 1 * 3.75 / 2]
#         return center_y_list[lane_index]
#
#     acc = np.random.random() * 4 - 2
#     goal_delta_v = acc * horizon * 100/1000
#     goal_delta_x = np.random.random() * 40 + 10
#     egolane_index = int(egolane_index)
#     goal_lane_index = np.random.choice(np.array(_interested_lane_index(egolane_index)), 1)[0]
#     goal_lane_index = goal_lane_index if goal_lane_index is not None else egolane_index
#     goal_y = _laneindex2centery(goal_lane_index)
#     return goal_delta_x, goal_y, goal_delta_v
def action_fn():
    upper_action = np.random.choice([0, 1, 2])
    delta_x_norm = (np.random.random() - 0.5) * 2
    acc_norm = (np.random.random() - 0.5) * 2

    delta_x = np.clip((delta_x_norm + 1) / 2 * 50 + 10, 10, 60)
    acc = np.clip(acc_norm * 3, -3, 3)
    return upper_action, delta_x, acc


if __name__ == '__main__':
    env = gym.make('EndtoendEnv-v0', setting_path='./Scenario/Highway_endtoend/', plan_horizon=30, history_len=10)
    env = ObservationWrapper(env)
    done = 0
    episode_num = 10
    for i in range(episode_num):  # run episode_num episodes
        done = 0
        obs = env.reset(init_state=[-800, -150-3.75*5/2, 10, 0])
        last_timestep_vector = obs[-1][-8:]
        ret = 0
        while not done:
            action = action_fn()
            print('action:', action)
            obs, rew, done, info = env.step(action)
            env.render()
            last_timestep_vector = obs[-1][-8:]
            print(info, rew)
            ret += rew

        print('return: ', ret)


