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
sns.set(style="darkgrid")


class Recorder(object):
    def __init__(self):
        self.val2record = ['v_x', 'v_y', 'r', 'x', 'y', 'phi',
                           'steer', 'a_x', 'delta_y', 'delta_v', 'delta_phi',
                           'cal_time', 'ref_index']
        self.val2plot = ['v_x', 'r',
                         'steer', 'a_x',
                         'cal_time', 'ref_index']
        self.key2label = dict(v_x='Velocity [m/s]',
                              r='Yaw rate [rad/s]',
                              steer='Steer angle [rad]',
                              a_x='Expected acceleration [$m^2$/s]',
                              cal_time='Computing time [s]',
                              ref_index='Selected path')
        self.ego_info_dim = 6
        self.per_tracking_info_dim = 3
        self.num_future_data = 0
        self.data_across_all_episodes = []
        self.val_list_for_an_episode = []

    def reset(self,):
        self.data_across_all_episodes.append(self.val_list_for_an_episode)
        self.val_list_for_an_episode = []

    def record(self, obs, act, cal_time, ref_index):
        ego_info, tracking_info, _ = obs[:self.ego_info_dim], \
                                    obs[self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (
                                               self.num_future_data + 1)], \
                                    obs[self.ego_info_dim + self.per_tracking_info_dim * (
                                               self.num_future_data + 1):]
        v_x, v_y, r, x, y, phi = ego_info
        delta_y, delta_phi, delta_v = tracking_info[:3]
        steer, a_x = act[0]*0.4, act[1]*3-1.
        self.val_list_for_an_episode.append(np.array([v_x, v_y, r, x, y, phi, steer, a_x, delta_y,
                                        delta_phi, delta_v, cal_time, ref_index]))

    def save(self):
        np.save('./data_across_all_episodes.npy', np.array(self.data_across_all_episodes))

    def load(self):
        self.data_across_all_episodes = np.load('./data_across_all_episodes.npy')

    def plot_current_episode_curves(self):
        real_time = np.array([0.1 * i for i in range(len(self.val_list_for_an_episode))])
        all_data = [np.array([vals_in_a_timestep[index] for vals_in_a_timestep in self.val_list_for_an_episode])
                    for index in range(len(self.val2record))]
        data_dict = dict(zip(self.val2record, all_data))
        for key in data_dict.keys():
            if key in self.val2plot:
                f = plt.figure(key)
                ax = f.add_axes([0.20, 0.12, 0.78, 0.86])
                sns.lineplot(real_time, data_dict[key], linewidth=2, palette="bright")
                ax.set_ylabel(self.key2label[key], fontsize=15)
                ax.set_xlabel("Time [s]", fontsize=15)
                plt.yticks(fontsize=15)
                plt.xticks(fontsize=15)
        plt.show()

    def plot_ith_episode_curves(self, i):
        episode2plot = self.data_across_all_episodes[i]
        real_time = np.array([0.1*i for i in range(len(episode2plot))])
        all_data = [np.array([vals_in_a_timestep[index] for vals_in_a_timestep in episode2plot])
                    for index in range(len(self.val2record))]
        data_dict = dict(zip(self.val2record, all_data))
        for key in data_dict.keys():
            if key in self.val2plot:
                f = plt.figure(key)
                ax = f.add_axes([0.20, 0.12, 0.78, 0.86])
                sns.lineplot(real_time, data_dict[key], linewidth=2, palette="bright")
                ax.set_ylabel(self.key2label[key], fontsize=15)
                ax.set_xlabel("Time [s]", fontsize=15)
                plt.yticks(fontsize=15)
                plt.xticks(fontsize=15)
        plt.show()






