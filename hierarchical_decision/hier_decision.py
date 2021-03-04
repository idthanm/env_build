#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/12/12
# @Author  : Yang Guan; Yangang Ren (Tsinghua Univ.)
# @FileName: hier_decision.py
# =====================================

import datetime
import json
import os
from math import cos, sin, pi

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dynamics_and_models import EnvironmentModel, ReferencePath
from endtoend import CrossroadEnd2end
from endtoend_env_utils import rotate_coordination, CROSSROAD_SIZE, LANE_WIDTH, LANE_NUMBER
from hierarchical_decision.multi_path_generator import MultiPathGenerator
from utils.load_policy import LoadPolicy
from utils.misc import TimerStat
from utils.recorder import Recorder


class HierarchicalDecision(object):
    def __init__(self, task, train_exp_dir, ite, logdir=None):
        self.task = task
        self.policy = LoadPolicy('../utils/models/{}/{}'.format(task, train_exp_dir), ite)
        self.env = CrossroadEnd2end(training_task=self.task)
        self.model = EnvironmentModel(self.task, mode='selecting')
        self.recorder = Recorder()
        self.episode_counter = -1
        self.step_counter = -1
        self.obs = None
        self.stg = None
        self.step_timer = TimerStat()
        self.ss_timer = TimerStat()
        self.logdir = logdir
        if self.logdir is not None:
            config = dict(task=task, train_exp_dir=train_exp_dir, ite=ite)
            with open(self.logdir + '/config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
        self.fig = plt.figure(figsize=(8, 8))
        plt.ion()
        self.hist_posi = []
        self.reset()

    def reset(self,):
        self.obs = self.env.reset()
        self.stg = MultiPathGenerator()
        self.recorder.reset()
        self.hist_posi = []
        if self.logdir is not None:
            self.episode_counter += 1
            os.makedirs(self.logdir + '/episode{}'.format(self.episode_counter))
            self.step_counter = -1
            self.recorder.save(self.logdir)
        return self.obs

    def safe_shield(self, real_obs, traj):
        action_bound = 1.0
        action_safe_set = [[[0, -action_bound]]]
        real_obs = tf.convert_to_tensor(real_obs[np.newaxis, :])
        obs = real_obs
        self.model.add_traj(obs, traj)
        total_punishment = 0.0
        for step in range(3):
            action = self.policy.run(obs)
            _, _, _, _, veh2veh4real, _ = self.model.rollout_out(action)
            total_punishment += veh2veh4real
        if total_punishment != 0:
            print('original action will cause collision within three steps!!!')
            for safe_action in action_safe_set:
                obs = real_obs
                total_punishment = 0
                # for step in range(1):
                #     obs, veh2veh4real = self.model.safety_calculation(obs, safe_action)
                #     total_punishment += veh2veh4real
                #     if veh2veh4real != 0:   # collide
                #         break
                # if total_punishment == 0:
                #     print('found the safe action', safe_action)
                #     safe_action = np.array(safe_action)
                #     break
                # else:
                #     print('still collide')
                #     safe_action = self.policy.run(real_obs).numpy().squeeze(0)
                return np.array(safe_action).squeeze(0)
        else:
            safe_action = self.policy.run(real_obs).numpy()[0]
            return safe_action

    def step(self):
        self.step_counter += 1
        with self.step_timer:
            path_list = self.stg.generate_path(self.task)
            obs_list = []

            # select optimal path
            for path in path_list:
                self.env.set_traj(path)
                obs_list.append(self.env._get_obs())
            all_obs = tf.stack(obs_list, axis=0)
            path_values = self.policy.obj_value_batch(all_obs).numpy()
            path_index = int(np.argmax(path_values))

            self.env.set_traj(path_list[path_index])
            self.obs_real = obs_list[path_index]

            # obtain safe action
            # with self.ss_timer:
            #     safe_action = self.safe_shield(self.obs_real, path_list[path_index])
            safe_action = self.policy.run(self.obs_real).numpy()

            print('ALL TIME:', self.step_timer.mean)
        # deal with red light temporally
        # if self.env.v_light != 0 and -25 > self.env.ego_dynamics['y'] > -35 and self.env.training_task != 'right':
        #     scaled_steer = 0.
        #     if self.env.ego_dynamics['v_x'] == 0.0:
        #         scaled_a_x = 0.33
        #     else:
        #         scaled_a_x = np.random.uniform(-0.6, -0.4)
        #     safe_action = np.array([scaled_steer, scaled_a_x], dtype=np.float32)
        self.render(path_list, path_values, path_index)
        self.recorder.record(self.obs_real, safe_action, self.step_timer.mean,
                             path_index, path_values, self.ss_timer.mean)
        self.obs, r, done, info = self.env.step(safe_action)
        return done

    def render(self, traj_list, path_values, path_index):
        square_length = CROSSROAD_SIZE
        extension = 40
        lane_width = LANE_WIDTH
        light_line_width = 3
        dotted_line_style = '--'
        solid_line_style = '-'

        plt.cla()
        ax = plt.axes([-0.05, -0.05, 1.1, 1.1])
        for ax in self.fig.get_axes():
            ax.axis('off')
        ax.axis("equal")

        # ----------arrow--------------
        plt.arrow(lane_width / 2, -square_length / 2 - 10, 0, 5, color='b')
        plt.arrow(lane_width / 2, -square_length / 2 - 10 + 5, -0.5, 0, color='b', head_width=1)
        plt.arrow(lane_width * 1.5, -square_length / 2 - 10, 0, 4, color='b', head_width=1)
        plt.arrow(lane_width * 2.5, -square_length / 2 - 10, 0, 5, color='b')
        plt.arrow(lane_width * 2.5, -square_length / 2 - 10 + 5, 0.5, 0, color='b', head_width=1)

        # ----------horizon--------------

        plt.plot([-square_length / 2 - extension, -square_length / 2], [0.3, 0.3], color='orange')
        plt.plot([-square_length / 2 - extension, -square_length / 2], [-0.3, -0.3], color='orange')
        plt.plot([square_length / 2 + extension, square_length / 2], [0.3, 0.3], color='orange')
        plt.plot([square_length / 2 + extension, square_length / 2], [-0.3, -0.3], color='orange')

        #
        for i in range(1, LANE_NUMBER + 1):
            linestyle = dotted_line_style if i < LANE_NUMBER else solid_line_style
            linewidth = 1 if i < LANE_NUMBER else 2
            plt.plot([-square_length / 2 - extension, -square_length / 2], [i * lane_width, i * lane_width],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([square_length / 2 + extension, square_length / 2], [i * lane_width, i * lane_width],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([-square_length / 2 - extension, -square_length / 2], [-i * lane_width, -i * lane_width],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([square_length / 2 + extension, square_length / 2], [-i * lane_width, -i * lane_width],
                     linestyle=linestyle, color='black', linewidth=linewidth)

        # ----------vertical----------------
        plt.plot([0.3, 0.3], [-square_length / 2 - extension, -square_length / 2], color='orange')
        plt.plot([-0.3, -0.3], [-square_length / 2 - extension, -square_length / 2], color='orange')
        plt.plot([0.3, 0.3], [square_length / 2 + extension, square_length / 2], color='orange')
        plt.plot([-0.3, -0.3], [square_length / 2 + extension, square_length / 2], color='orange')

        #
        for i in range(1, LANE_NUMBER + 1):
            linestyle = dotted_line_style if i < LANE_NUMBER else solid_line_style
            linewidth = 1 if i < LANE_NUMBER else 2
            plt.plot([i * lane_width, i * lane_width], [-square_length / 2 - extension, -square_length / 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([i * lane_width, i * lane_width], [square_length / 2 + extension, square_length / 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([-i * lane_width, -i * lane_width], [-square_length / 2 - extension, -square_length / 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([-i * lane_width, -i * lane_width], [square_length / 2 + extension, square_length / 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)

        v_light = self.env.v_light
        if v_light == 0:
            v_color, h_color = 'green', 'red'
        elif v_light == 1:
            v_color, h_color = 'orange', 'red'
        elif v_light == 2:
            v_color, h_color = 'red', 'green'
        else:
            v_color, h_color = 'red', 'orange'

        plt.plot([0, (LANE_NUMBER - 1) * lane_width], [-square_length / 2, -square_length / 2],
                 color=v_color, linewidth=light_line_width)
        plt.plot([(LANE_NUMBER - 1) * lane_width, LANE_NUMBER * lane_width], [-square_length / 2, -square_length / 2],
                 color='green', linewidth=light_line_width)

        plt.plot([-LANE_NUMBER * lane_width, -(LANE_NUMBER - 1) * lane_width], [square_length / 2, square_length / 2],
                 color='green', linewidth=light_line_width)
        plt.plot([-(LANE_NUMBER - 1) * lane_width, 0], [square_length / 2, square_length / 2],
                 color=v_color, linewidth=light_line_width)

        plt.plot([-square_length / 2, -square_length / 2], [0, -(LANE_NUMBER - 1) * lane_width],
                 color=h_color, linewidth=light_line_width)
        plt.plot([-square_length / 2, -square_length / 2], [-(LANE_NUMBER - 1) * lane_width, -LANE_NUMBER * lane_width],
                 color='green', linewidth=light_line_width)

        plt.plot([square_length / 2, square_length / 2], [(LANE_NUMBER - 1) * lane_width, 0],
                 color=h_color, linewidth=light_line_width)
        plt.plot([square_length / 2, square_length / 2], [LANE_NUMBER * lane_width, (LANE_NUMBER - 1) * lane_width],
                 color='green', linewidth=light_line_width)

        # ----------Oblique--------------
        plt.plot([LANE_NUMBER * lane_width, square_length / 2], [-square_length / 2, -LANE_NUMBER * lane_width],
                 color='black', linewidth=2)
        plt.plot([LANE_NUMBER * lane_width, square_length / 2], [square_length / 2, LANE_NUMBER * lane_width],
                 color='black', linewidth=2)
        plt.plot([-LANE_NUMBER * lane_width, -square_length / 2], [-square_length / 2, -LANE_NUMBER * lane_width],
                 color='black', linewidth=2)
        plt.plot([-LANE_NUMBER * lane_width, -square_length / 2], [square_length / 2, LANE_NUMBER * lane_width],
                 color='black', linewidth=2)

        def is_in_plot_area(x, y, tolerance=5):
            if -square_length / 2 - extension + tolerance < x < square_length / 2 + extension - tolerance and \
                    -square_length / 2 - extension + tolerance < y < square_length / 2 + extension - tolerance:
                return True
            else:
                return False

        def draw_rotate_rec(x, y, a, l, w, c):
            bottom_left_x, bottom_left_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
            ax.add_patch(plt.Rectangle((x + bottom_left_x, y + bottom_left_y), w, l, edgecolor=c,
                                       facecolor='white', angle=-(90 - a), zorder=50))

        def plot_phi_line(x, y, phi, color):
            line_length = 3
            x_forw, y_forw = x + line_length * cos(phi * pi / 180.), \
                             y + line_length * sin(phi * pi / 180.)
            plt.plot([x, x_forw], [y, y_forw], color=color, linewidth=0.5)

        # plot cars
        for veh in self.env.all_vehicles:
            veh_x = veh['x']
            veh_y = veh['y']
            veh_phi = veh['phi']
            veh_l = veh['l']
            veh_w = veh['w']
            if is_in_plot_area(veh_x, veh_y):
                plot_phi_line(veh_x, veh_y, veh_phi, 'black')
                draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, 'black')

        # plot_interested vehs
        # for mode, num in self.veh_mode_dict.items():
        #     for i in range(num):
        #         veh = self.interested_vehs[mode][i]
        #         veh_x = veh['x']
        #         veh_y = veh['y']
        #         veh_phi = veh['phi']
        #         veh_l = veh['l']
        #         veh_w = veh['w']
        #         task2color = {'left': 'b', 'straight': 'c', 'right': 'm'}
        #
        #         if is_in_plot_area(veh_x, veh_y):
        #             plot_phi_line(veh_x, veh_y, veh_phi, 'black')
        #             task = MODE2TASK[mode]
        #             color = task2color[task]
        #             draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, color, linestyle=':')

        ego_v_x = self.env.ego_dynamics['v_x']
        ego_v_y = self.env.ego_dynamics['v_y']
        ego_r = self.env.ego_dynamics['r']
        ego_x = self.env.ego_dynamics['x']
        ego_y = self.env.ego_dynamics['y']
        ego_phi = self.env.ego_dynamics['phi']
        ego_l = self.env.ego_dynamics['l']
        ego_w = self.env.ego_dynamics['w']
        ego_alpha_f = self.env.ego_dynamics['alpha_f']
        ego_alpha_r = self.env.ego_dynamics['alpha_r']
        alpha_f_bound = self.env.ego_dynamics['alpha_f_bound']
        alpha_r_bound = self.env.ego_dynamics['alpha_r_bound']
        r_bound = self.env.ego_dynamics['r_bound']

        plot_phi_line(ego_x, ego_y, ego_phi, 'fuchsia')
        draw_rotate_rec(ego_x, ego_y, ego_phi, ego_l, ego_w, 'fuchsia')
        self.hist_posi.append((ego_x, ego_y))
        # plot history data
        for hist_x, hist_y in self.hist_posi:
            plt.scatter(hist_x, hist_y, color='fuchsia', alpha=0.1)
        # plot future data
        tracking_info = self.obs[
                        self.env.ego_info_dim:self.env.ego_info_dim + self.env.per_tracking_info_dim * (self.env.num_future_data + 1)]
        future_path = tracking_info[self.env.per_tracking_info_dim:]
        for i in range(self.env.num_future_data):
            delta_x, delta_y, delta_phi = future_path[i * self.env.per_tracking_info_dim:
                                                      (i + 1) * self.env.per_tracking_info_dim]
            path_x, path_y, path_phi = ego_x + delta_x, ego_y + delta_y, ego_phi - delta_phi
            plt.plot(path_x, path_y, 'g.')
            plot_phi_line(path_x, path_y, path_phi, 'g')

        delta_, _, _ = tracking_info[:3]
        indexs, points = self.env.ref_path.find_closest_point(np.array([ego_x], np.float32), np.array([ego_y], np.float32))
        path_x, path_y, path_phi = points[0][0], points[1][0], points[2][0]
        # plt.plot(path_x, path_y, 'g.')
        delta_x, delta_y, delta_phi = ego_x - path_x, ego_y - path_y, ego_phi - path_phi

        # plot real time traj
        try:
            color = ['blue', 'coral', 'darkcyan']
            for i, item in enumerate(traj_list):
                if i == path_index:
                    plt.plot(item.path[0], item.path[1], color=color[i])
                else:
                    plt.plot(item.path[0], item.path[1], color=color[i], alpha=0.3)
                indexs, points = item.find_closest_point(np.array([ego_x], np.float32), np.array([ego_y], np.float32))
                path_x, path_y, path_phi = points[0][0], points[1][0], points[2][0]
                plt.plot(path_x, path_y, color=color[i])
        except Exception:
            pass

        # text
        # text_x, text_y_start = -120, 60
        # ge = iter(range(0, 1000, 4))
        # plt.text(text_x, text_y_start - next(ge), 'ego_x: {:.2f}m'.format(ego_x))
        # plt.text(text_x, text_y_start - next(ge), 'ego_y: {:.2f}m'.format(ego_y))
        # plt.text(text_x, text_y_start - next(ge), 'path_x: {:.2f}m'.format(path_x))
        # plt.text(text_x, text_y_start - next(ge), 'path_y: {:.2f}m'.format(path_y))
        # plt.text(text_x, text_y_start - next(ge), 'delta_: {:.2f}m'.format(delta_))
        # plt.text(text_x, text_y_start - next(ge), 'delta_x: {:.2f}m'.format(delta_x))
        # plt.text(text_x, text_y_start - next(ge), 'delta_y: {:.2f}m'.format(delta_y))
        # plt.text(text_x, text_y_start - next(ge), r'ego_phi: ${:.2f}\degree$'.format(ego_phi))
        # plt.text(text_x, text_y_start - next(ge), r'path_phi: ${:.2f}\degree$'.format(path_phi))
        # plt.text(text_x, text_y_start - next(ge), r'delta_phi: ${:.2f}\degree$'.format(delta_phi))
        # plt.text(text_x, text_y_start - next(ge), 'v_x: {:.2f}m/s'.format(ego_v_x))
        # plt.text(text_x, text_y_start - next(ge), 'exp_v: {:.2f}m/s'.format(self.env.exp_v))
        # plt.text(text_x, text_y_start - next(ge), 'v_y: {:.2f}m/s'.format(ego_v_y))
        # plt.text(text_x, text_y_start - next(ge), 'yaw_rate: {:.2f}rad/s'.format(ego_r))
        # plt.text(text_x, text_y_start - next(ge), 'yaw_rate bound: [{:.2f}, {:.2f}]'.format(-r_bound, r_bound))
        #
        # plt.text(text_x, text_y_start - next(ge), r'$\alpha_f$: {:.2f} rad'.format(ego_alpha_f))
        # plt.text(text_x, text_y_start - next(ge), r'$\alpha_f$ bound: [{:.2f}, {:.2f}] '.format(-alpha_f_bound,
        #                                                                                         alpha_f_bound))
        # plt.text(text_x, text_y_start - next(ge), r'$\alpha_r$: {:.2f} rad'.format(ego_alpha_r))
        # plt.text(text_x, text_y_start - next(ge), r'$\alpha_r$ bound: [{:.2f}, {:.2f}] '.format(-alpha_r_bound,
        #                                                                                         alpha_r_bound))
        # if self.env.action is not None:
        #     steer, a_x = self.env.action[0], self.env.action[1]
        #     plt.text(text_x, text_y_start - next(ge),
        #              r'steer: {:.2f}rad (${:.2f}\degree$)'.format(steer, steer * 180 / np.pi))
        #     plt.text(text_x, text_y_start - next(ge), 'a_x: {:.2f}m/s^2'.format(a_x))
        #
        # text_x, text_y_start = 70, 60
        # ge = iter(range(0, 1000, 4))
        #
        # # done info
        # plt.text(text_x, text_y_start - next(ge), 'done info: {}'.format(self.env.done_type))
        #
        # # reward info
        # if self.env.reward_info is not None:
        #     for key, val in self.env.reward_info.items():
        #         plt.text(text_x, text_y_start - next(ge), '{}: {:.4f}'.format(key, val))
        #
        # # indicator for trajectory selection
        # text_x, text_y_start = -18, -70
        # ge = iter(range(0, 1000, 6))
        # if path_values is not None:
        #     for i, value in enumerate(path_values):
        #         if i == path_index:
        #             plt.text(text_x, text_y_start - next(ge), 'Path reward={:.4f}'.format(value[0]), fontsize=14,
        #                      color=color[i], fontstyle='italic')
        #         else:
        #             plt.text(text_x, text_y_start - next(ge), 'Path reward={:.4f}'.format(value[0]), fontsize=12,
        #                      color=color[i], fontstyle='italic')
        plt.show()
        plt.pause(0.001)
        if self.logdir is not None:
            plt.savefig(self.logdir + '/episode{}'.format(self.episode_counter) + '/step{}.pdf'.format(self.step_counter))


def plot_data(logdir, i):
    recorder = Recorder()
    recorder.load(logdir)
    recorder.plot_ith_episode_curves(i)


def main():
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = './results/{time}'.format(time=time_now)
    os.makedirs(logdir)
    hier_decision = HierarchicalDecision('left', 'experiment-2021-03-04-16-44-45', 120000, logdir)

    for i in range(300):
        done = 0
        while not done:
            done = hier_decision.step()
        hier_decision.reset()


def plot_static_path():
    square_length = CROSSROAD_SIZE
    extension = 20
    lane_width = LANE_WIDTH
    light_line_width = 3
    dotted_line_style = '--'
    solid_line_style = '-'
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes([0, 0, 1, 1])
    for ax in fig.get_axes():
        ax.axis('off')
    ax.axis("equal")

    # ----------arrow--------------
    plt.arrow(lane_width / 2, -square_length / 2 - 10, 0, 5, color='b')
    plt.arrow(lane_width / 2, -square_length / 2 - 10 + 5, -0.5, 0, color='b', head_width=1)
    plt.arrow(lane_width * 1.5, -square_length / 2 - 10, 0, 4, color='b', head_width=1)
    plt.arrow(lane_width * 2.5, -square_length / 2 - 10, 0, 5, color='b')
    plt.arrow(lane_width * 2.5, -square_length / 2 - 10 + 5, 0.5, 0, color='b', head_width=1)

    # ----------horizon--------------

    plt.plot([-square_length / 2 - extension, -square_length / 2], [0.3, 0.3], color='orange')
    plt.plot([-square_length / 2 - extension, -square_length / 2], [-0.3, -0.3], color='orange')
    plt.plot([square_length / 2 + extension, square_length / 2], [0.3, 0.3], color='orange')
    plt.plot([square_length / 2 + extension, square_length / 2], [-0.3, -0.3], color='orange')

    for i in range(1, LANE_NUMBER + 1):
        linestyle = dotted_line_style if i < LANE_NUMBER else solid_line_style
        linewidth = 1 if i < LANE_NUMBER else 2
        plt.plot([-square_length / 2 - extension, -square_length / 2], [i * lane_width, i * lane_width],
                 linestyle=linestyle, color='black', linewidth=linewidth)
        plt.plot([square_length / 2 + extension, square_length / 2], [i * lane_width, i * lane_width],
                 linestyle=linestyle, color='black', linewidth=linewidth)
        plt.plot([-square_length / 2 - extension, -square_length / 2], [-i * lane_width, -i * lane_width],
                 linestyle=linestyle, color='black', linewidth=linewidth)
        plt.plot([square_length / 2 + extension, square_length / 2], [-i * lane_width, -i * lane_width],
                 linestyle=linestyle, color='black', linewidth=linewidth)

    # ----------vertical----------------
    plt.plot([0.3, 0.3], [-square_length / 2 - extension, -square_length / 2], color='orange')
    plt.plot([-0.3, -0.3], [-square_length / 2 - extension, -square_length / 2], color='orange')
    plt.plot([0.3, 0.3], [square_length / 2 + extension, square_length / 2], color='orange')
    plt.plot([-0.3, -0.3], [square_length / 2 + extension, square_length / 2], color='orange')

    for i in range(1, LANE_NUMBER + 1):
        linestyle = dotted_line_style if i < LANE_NUMBER else solid_line_style
        linewidth = 1 if i < LANE_NUMBER else 2
        plt.plot([i * lane_width, i * lane_width], [-square_length / 2 - extension, -square_length / 2],
                 linestyle=linestyle, color='black', linewidth=linewidth)
        plt.plot([i * lane_width, i * lane_width], [square_length / 2 + extension, square_length / 2],
                 linestyle=linestyle, color='black', linewidth=linewidth)
        plt.plot([-i * lane_width, -i * lane_width], [-square_length / 2 - extension, -square_length / 2],
                 linestyle=linestyle, color='black', linewidth=linewidth)
        plt.plot([-i * lane_width, -i * lane_width], [square_length / 2 + extension, square_length / 2],
                 linestyle=linestyle, color='black', linewidth=linewidth)
    plt.plot([0, LANE_NUMBER * lane_width], [-square_length / 2, -square_length / 2],
             color='black', linewidth=light_line_width, alpha=0.3)
    plt.plot([LANE_NUMBER * lane_width, 0], [square_length / 2, square_length / 2],
             color='black', linewidth=light_line_width, alpha=0.3)
    plt.plot([-square_length / 2, -square_length / 2], [0, LANE_NUMBER * lane_width],
             color='black', linewidth=light_line_width, alpha=0.3)
    plt.plot([square_length / 2, square_length / 2], [-LANE_NUMBER * lane_width, 0],
             color='black', linewidth=light_line_width, alpha=0.3)

    # ----------Oblique--------------
    plt.plot([LANE_NUMBER * lane_width, square_length / 2], [-square_length / 2, -LANE_NUMBER * lane_width],
             color='black', linewidth=2)
    plt.plot([LANE_NUMBER * lane_width, square_length / 2], [square_length / 2, LANE_NUMBER * lane_width],
             color='black', linewidth=2)
    plt.plot([-LANE_NUMBER * lane_width, -square_length / 2], [-square_length / 2, -LANE_NUMBER * lane_width],
             color='black', linewidth=2)
    plt.plot([-LANE_NUMBER * lane_width, -square_length / 2], [square_length / 2, LANE_NUMBER * lane_width],
             color='black', linewidth=2)

    for task in ['left', 'straight', 'right']:
        path = ReferencePath(task)
        path_list = path.path_list
        control_points = path.control_points
        color = ['blue', 'coral', 'darkcyan']
        for i, (path_x, path_y, _) in enumerate(path_list):
            plt.plot(path_x[600:-600], path_y[600:-600], color=color[i])
        for i, four_points in enumerate(control_points):
            for point in four_points:
                plt.scatter(point[0], point[1], color=color[i])
            plt.plot([four_points[0][0], four_points[1][0]], [four_points[0][1], four_points[1][1]], linestyle='--', color=color[i])
            plt.plot([four_points[1][0], four_points[2][0]], [four_points[1][1], four_points[2][1]], linestyle='--', color=color[i])
            plt.plot([four_points[2][0], four_points[3][0]], [four_points[2][1], four_points[3][1]], linestyle='--', color=color[i])
    plt.savefig('./multipath_planning.pdf')
    plt.show()


if __name__ == '__main__':
    main()
    # plot_static_path()
    # plot_data('./results/2021-03-03-19-18-38', 1)


