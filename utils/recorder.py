#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/12/11
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: recorder.py
# =====================================
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as ticker
from matplotlib.pyplot import MultipleLocator
import math
sns.set(style="darkgrid")


class Recorder(object):
    def __init__(self):
        self.val2record = ['v_x', 'v_y', 'r', 'x', 'y', 'phi',
                           'steer', 'a_x', 'delta_y', 'delta_v', 'delta_phi',
                           'cal_time', 'ref_index', 'beta', 'path_values', 'ss_time']
        self.val2plot = ['v_x', 'r',
                         'steer', 'a_x',
                         'cal_time', 'ref_index', 'beta']
        self.key2label = dict(v_x='Velocity [m/s]',
                              r='Yaw rate [rad/s]',
                              steer='Steer angle [$\circ$]',
                              a_x='Acceleration [$\mathrm {m/s^2}$]',
                              # a_x='Acceleration [$m/s^2$]',
                              cal_time='Computing time [ms]',
                              ref_index='Selected path',
                              beta='Side slip angle[$\circ$]')
        self.ego_info_dim = 6
        self.per_tracking_info_dim = 3
        self.num_future_data = 0
        self.data_across_all_episodes = []
        self.val_list_for_an_episode = []

    def reset(self,):
        if self.val_list_for_an_episode:
            self.data_across_all_episodes.append(self.val_list_for_an_episode)
        self.val_list_for_an_episode = []

    def record(self, obs, act, cal_time, ref_index, path_values, ss_time):
        ego_info, tracking_info, _ = obs[:self.ego_info_dim], \
                                     obs[self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (
                                               self.num_future_data + 1)], \
                                     obs[self.ego_info_dim + self.per_tracking_info_dim * (
                                               self.num_future_data + 1):]
        v_x, v_y, r, x, y, phi = ego_info
        delta_y, delta_phi, delta_v = tracking_info[:3]
        steer, a_x = act[0]*0.4, act[1]*3-1.

        # transformation
        clip_random = np.random.uniform(-0.1, 0.1)
        a_x = np.clip(a_x, -3.0, 1.5 + clip_random)
        beta = 0 if v_x == 0 else np.arctan(v_y/v_x) * 180 / math.pi
        steer = steer * 180 / math.pi
        self.val_list_for_an_episode.append(np.array([v_x, v_y, r, x, y, phi, steer, a_x, delta_y,
                                        delta_phi, delta_v, cal_time, ref_index, beta, path_values, ss_time]))

    def save(self, logdir):
        np.save(logdir + '/data_across_all_episodes.npy', np.array(self.data_across_all_episodes))

    def load(self, logdir):
        self.data_across_all_episodes = np.load(logdir + '/data_across_all_episodes.npy', allow_pickle=True)

    def plot_ith_episode_curves(self, i):
        episode2plot = self.data_across_all_episodes[i]
        real_time = np.array([0.1*i for i in range(len(episode2plot))])
        all_data = [np.array([vals_in_a_timestep[index] for vals_in_a_timestep in episode2plot])
                    for index in range(len(self.val2record))]
        data_dict = dict(zip(self.val2record, all_data))
        color = ['cyan', 'indigo', 'magenta', 'coral', 'b', 'brown', 'c']
        i = 0
        for key in data_dict.keys():
            if key in self.val2plot:
                f = plt.figure(key)
                ax = f.add_axes([0.20, 0.12, 0.78, 0.86])
                if key == 'ref_index':
                    sns.lineplot(real_time, data_dict[key] + 1, linewidth=2, palette="bright", color=color[i])
                    plt.ylim([0.5, 3.5])
                    x_major_locator = MultipleLocator(10)
                    # ax.xaxis.set_major_locator(x_major_locator)
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                elif key == 'v_x':
                    sns.lineplot(real_time, data_dict[key], linewidth=2, palette="bright", color=color[i])
                    plt.ylim([-0.5, 10.])
                elif key == 'cal_time':
                    sns.lineplot(real_time, data_dict[key] * 1000, linewidth=2, palette="bright", color=color[i])
                    plt.ylim([0, 10])
                elif key == 'a_x':
                    sns.lineplot(real_time, np.clip(data_dict[key], -3.0, 1.5), linewidth=2, palette="bright", color=color[i])
                    # sns.lineplot(real_time, data_dict[key], linewidth=2, palette="bright", color=color[i])
                    plt.ylim([-4.5, 2.0])
                elif key == 'steer':
                    sns.lineplot(real_time, data_dict[key], linewidth=2, palette="bright", color=color[i])
                    plt.ylim([-25, 25])
                elif key == 'beta':
                    sns.lineplot(real_time, data_dict[key], linewidth=2, palette="bright", color=color[i])
                    plt.ylim([-15, 15])
                elif key == 'r':
                    sns.lineplot(real_time, data_dict[key], linewidth=2, palette="bright", color=color[i])
                    plt.ylim([-0.8, 0.8])
                else:
                    sns.lineplot(real_time, data_dict[key], linewidth=2, palette="bright", color=color[i])

                ax.set_ylabel(self.key2label[key], fontsize=15)
                ax.set_xlabel("Time [s]", fontsize=15)
                plt.yticks(fontsize=15)
                plt.xticks(fontsize=15)
                i += 1
        plt.show()






