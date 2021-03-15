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
import pandas as pd
sns.set(style="darkgrid")

WINDOWSIZE = 15


class Recorder(object):
    def __init__(self):
        self.val2record = ['v_x', 'v_y', 'r', 'x', 'y', 'phi',
                           'steer', 'a_x', 'delta_y', 'delta_v', 'delta_phi',
                           'cal_time', 'ref_index', 'beta', 'path_values', 'ss_time', 'is_ss']
        self.val2plot = ['v_x', 'r',
                         'steer', 'a_x',
                         'cal_time', 'ref_index', 'beta', 'path_values', 'is_ss']
        self.key2label = dict(v_x='Velocity [m/s]',
                              r='Yaw rate [rad/s]',
                              steer='Steer angle [$\circ$]',
                              a_x='Acceleration [$\mathrm {m/s^2}$]',
                              cal_time='Computing time [ms]',
                              ref_index='Selected path',
                              beta='Side slip angle[$\circ$]',
                              path_values='Path value',
                              is_ss='Safety shield',)

        self.comp2record = ['v_x', 'v_y', 'r', 'x', 'y', 'phi', 'adp_steer', 'adp_a_x', 'mpc_steer', 'mpc_a_x',
                            'delta_y', 'delta_v', 'delta_phi', 'adp_time', 'mpc_time', 'adp_ref', 'mpc_ref', 'beta']

        self.ego_info_dim = 6
        self.per_tracking_info_dim = 3
        self.num_future_data = 0
        self.data_across_all_episodes = []
        self.val_list_for_an_episode = []
        self.comp_list_for_an_episode = []
        self.comp_data_for_all_episodes = []

    def reset(self,):
        if self.val_list_for_an_episode:
            self.data_across_all_episodes.append(self.val_list_for_an_episode)
        if self.comp_list_for_an_episode:
            self.comp_data_for_all_episodes.append(self.comp_list_for_an_episode)
        self.val_list_for_an_episode = []
        self.comp_list_for_an_episode = []

    def record(self, obs, act, cal_time, ref_index, path_values, ss_time, is_ss):
        ego_info, tracking_info, _ = obs[:self.ego_info_dim], \
                                     obs[self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (
                                               self.num_future_data + 1)], \
                                     obs[self.ego_info_dim + self.per_tracking_info_dim * (
                                               self.num_future_data + 1):]
        v_x, v_y, r, x, y, phi = ego_info
        delta_y, delta_phi, delta_v = tracking_info[:3]
        steer, a_x = act[0]*0.4, act[1]*2.25 - 0.75

        # transformation
        beta = 0 if v_x == 0 else np.arctan(v_y/v_x) * 180 / math.pi
        steer = steer * 180 / math.pi
        self.val_list_for_an_episode.append(np.array([v_x, v_y, r, x, y, phi, steer, a_x, delta_y,
                                        delta_phi, delta_v, cal_time, ref_index, beta, path_values, ss_time, is_ss]))

    # For comparison of MPC and ADP
    def record_compare(self, obs, adp_act, mpc_act, adp_time, mpc_time, adp_ref, mpc_ref, mode='ADP'):
        ego_info, tracking_info, _ = obs[:self.ego_info_dim], \
                                     obs[self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (
                                               self.num_future_data + 1)], \
                                     obs[self.ego_info_dim + self.per_tracking_info_dim * (
                                               self.num_future_data + 1):]
        v_x, v_y, r, x, y, phi = ego_info
        delta_y, delta_phi, delta_v = tracking_info[:3]
        adp_steer, adp_a_x = adp_act[0]*0.4, adp_act[1]*3-1.
        mpc_steer, mpc_a_x = mpc_act[0], mpc_act[1]

        # transformation
        beta = 0 if v_x == 0 else np.arctan(v_y/v_x) * 180 / math.pi
        adp_steer = adp_steer * 180 / math.pi
        mpc_steer = mpc_steer * 180 / math.pi
        self.comp_list_for_an_episode.append(np.array([v_x, v_y, r, x, y, phi, adp_steer, adp_a_x, mpc_steer, mpc_a_x,
                                            delta_y, delta_phi, delta_v, adp_time, mpc_time, adp_ref, mpc_ref, beta]))

    def save(self, logdir):
        np.save(logdir + '/data_across_all_episodes.npy', np.array(self.data_across_all_episodes))
        np.save(logdir + '/comp_data_for_all_episodes.npy', np.array(self.comp_data_for_all_episodes))

    def load(self, logdir):
        self.data_across_all_episodes = np.load(logdir + '/data_across_all_episodes.npy', allow_pickle=True)
        self.comp_data_for_all_episodes = np.load(logdir + '/comp_data_for_all_episodes.npy', allow_pickle=True)

    def plot_and_save_ith_episode_curves(self, i, save_dir, isshow=True):
        episode2plot = self.data_across_all_episodes[i]
        real_time = np.array([0.1*i for i in range(len(episode2plot))])
        all_data = [np.array([vals_in_a_timestep[index] for vals_in_a_timestep in episode2plot])
                    for index in range(len(self.val2record))]
        data_dict = dict(zip(self.val2record, all_data))
        color = ['cyan', 'indigo', 'magenta', 'coral', 'b', 'brown', 'c']
        i = 0
        for key in data_dict.keys():
            if key in self.val2plot:
                f = plt.figure(key, figsize=(6, 5))
                if key == 'ref_index':
                    ax = f.add_axes([0.11, 0.12, 0.88, 0.86])
                    sns.lineplot(real_time, data_dict[key] + 1, linewidth=2, palette="bright", color='indigo')
                    plt.ylim([0.5, 3.5])
                    x_major_locator = MultipleLocator(10)
                    # ax.xaxis.set_major_locator(x_major_locator)
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                elif key == 'v_x':
                    df = pd.DataFrame(dict(time=real_time, data=data_dict[key]))
                    df['data_smo'] = df['data'].rolling(WINDOWSIZE, min_periods=1).mean()
                    ax = f.add_axes([0.11, 0.12, 0.88, 0.86])
                    sns.lineplot('time', 'data_smo', linewidth=2,
                                 data=df, palette="bright", color='indigo')
                    plt.ylim([-0.5, 10.])
                elif key == 'cal_time':
                    df = pd.DataFrame(dict(time=real_time, data=data_dict[key] * 1000))
                    df['data_smo'] = df['data'].rolling(WINDOWSIZE, min_periods=1).mean()
                    ax = f.add_axes([0.11, 0.12, 0.88, 0.86])
                    sns.lineplot('time', 'data_smo', linewidth=2,
                                 data=df, palette="bright", color='indigo')
                    plt.ylim([0, 10])
                elif key == 'a_x':
                    df = pd.DataFrame(dict(time=real_time, data=data_dict[key]))
                    df['data_smo'] = df['data'].rolling(WINDOWSIZE, min_periods=1).mean()
                    ax = f.add_axes([0.14, 0.12, 0.86, 0.86])
                    sns.lineplot('time', 'data_smo', linewidth=2,
                                 data=df, palette="bright", color='indigo')
                    plt.ylim([-4.5, 2.0])
                elif key == 'steer':
                    df = pd.DataFrame(dict(time=real_time, data=data_dict[key]))
                    df['data_smo'] = df['data'].rolling(WINDOWSIZE, min_periods=1).mean()
                    ax = f.add_axes([0.15, 0.12, 0.85, 0.86])
                    sns.lineplot('time', 'data_smo', linewidth=2,
                                 data=df, palette="bright", color='indigo')
                    plt.ylim([-25, 25])
                elif key == 'beta':
                    df = pd.DataFrame(dict(time=real_time, data=data_dict[key]))
                    df['data_smo'] = df['data'].rolling(WINDOWSIZE, min_periods=1).mean()
                    ax = f.add_axes([0.15, 0.12, 0.85, 0.86])
                    sns.lineplot('time', 'data_smo', linewidth=2,
                                 data=df, palette="bright", color='indigo')
                    plt.ylim([-15, 15])
                elif key == 'r':
                    df = pd.DataFrame(dict(time=real_time, data=data_dict[key]))
                    df['data_smo'] = df['data'].rolling(WINDOWSIZE, min_periods=1).mean()
                    ax = f.add_axes([0.15, 0.12, 0.85, 0.86])
                    sns.lineplot('time', 'data_smo', linewidth=2,
                                 data=df, palette="bright", color='indigo')
                    plt.ylim([-0.8, 0.8])
                elif key == 'path_values':
                    path_values = data_dict[key]
                    df1 = pd.DataFrame(dict(time=real_time, data=-path_values[:, 0], path_index='Path 1'))
                    df2 = pd.DataFrame(dict(time=real_time, data=-path_values[:, 1], path_index='Path 2'))
                    df3 = pd.DataFrame(dict(time=real_time, data=-path_values[:, 2], path_index='Path 3'))
                    total_dataframe = df1.append([df2, df3], ignore_index=True)
                    ax = f.add_axes([0.15, 0.12, 0.85, 0.86])
                    sns.lineplot('time', 'data', linewidth=2, hue='path_index',
                                 data=total_dataframe, palette="bright", color='indigo')
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles=handles, labels=labels, loc='lower left', frameon=False)
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                elif key == 'is_ss':
                    df = pd.DataFrame(dict(time=real_time, data=data_dict[key]))
                    ax = f.add_axes([0.15, 0.12, 0.85, 0.86])
                    sns.lineplot('time', 'data', linewidth=2,
                                 data=df, palette="bright", color='indigo')
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                else:
                    ax = f.add_axes([0.11, 0.12, 0.88, 0.86])
                    sns.lineplot(real_time, data_dict[key], linewidth=2, palette="bright", color='indigo')

                ax.set_ylabel(self.key2label[key], fontsize=15)
                ax.set_xlabel("Time [s]", fontsize=15)
                plt.yticks(fontsize=15)
                plt.xticks(fontsize=15)
                plt.savefig(save_dir + '/{}.pdf'.format(key))
                if not isshow:
                    plt.close(f)
                i += 1
        if isshow:
            plt.show()

    def plot_mpc_rl(self, i):
        episode2plot = self.comp_data_for_all_episodes[i] if i is not None else self.comp_list_for_an_episode
        real_time = np.array([0.1 * i for i in range(len(episode2plot))])
        all_data = [np.array([vals_in_a_timestep[index] for vals_in_a_timestep in episode2plot])
                    for index in range(len(self.comp2record))]
        data_dict = dict(zip(self.comp2record, all_data))

        df_mpc = pd.DataFrame({'algorithms': 'MPC',
                               'iteration': real_time,
                               'steer': data_dict['mpc_steer'],
                               'acc': data_dict['mpc_a_x'],
                               'time': data_dict['mpc_time'],
                               'ref_path': data_dict['mpc_ref'] + 1
                               })
        df_rl = pd.DataFrame({'algorithms': 'ADP',
                              'iteration': real_time,
                              'steer': data_dict['adp_steer'],
                              'acc': data_dict['adp_a_x'],
                              'time': data_dict['adp_time'],
                              'ref_path': data_dict['adp_ref'] + 1
                              })

        total_df = df_mpc.append([df_rl], ignore_index=True)
        f1 = plt.figure(1)
        ax1 = f1.add_axes([0.155, 0.12, 0.82, 0.86])
        sns.lineplot(x="iteration", y="steer", hue="algorithms", data=total_df, linewidth=2, palette="bright", )
        ax1.set_ylabel('Front wheel angle [$\circ$]', fontsize=15)
        ax1.set_xlabel("Time[s]", fontsize=15)
        ax1.legend(frameon=False, fontsize=15)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)

        f2 = plt.figure(2)
        ax2 = f2.add_axes([0.155, 0.12, 0.82, 0.86])
        sns.lineplot(x="iteration", y="acc", hue="algorithms", data=total_df, linewidth=2, palette="bright", )
        ax2.set_ylabel('Acceleration [$\mathrm {m/s^2}$]', fontsize=15)
        ax2.set_xlabel('Time[s]', fontsize=15)
        ax2.legend(frameon=False, fontsize=15)
        # plt.xlim(0, 3)
        # plt.ylim(-40, 80)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)

        f3 = plt.figure(3)
        ax3 = f3.add_axes([0.155, 0.12, 0.82, 0.86])
        sns.lineplot(x="iteration", y="time", hue="algorithms", data=total_df, linewidth=2, palette="bright", )
        plt.yscale('log')
        ax3.set_ylabel('Computing time [ms]', fontsize=15)
        ax3.set_xlabel("Time[s]", fontsize=15)
        handles, labels = ax3.get_legend_handles_labels()
        # ax3.legend(handles=handles[1:], labels=labels[1:], loc='upper left', frameon=False, fontsize=15)
        ax3.legend(handles=handles[:], labels=labels[:], frameon=False, fontsize=15)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)

        f4 = plt.figure(4)
        ax4 = f4.add_axes([0.155, 0.12, 0.82, 0.86])
        sns.lineplot(x="iteration", y="ref_path", hue="algorithms", data=total_df, dashes=True, linewidth=2, palette="bright", )
        ax4.lines[1].set_linestyle("--")
        ax4.set_ylabel('Selected path', fontsize=15)
        ax4.set_xlabel("Time[s]", fontsize=15)
        ax4.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax4.legend(frameon=False, fontsize=15)
        # plt.xlim(0, 3)
        # plt.ylim(-40, 80)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.show()






