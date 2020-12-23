#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/10/12
# @Author  : Yang Guan; Yangang Ren (Tsinghua Univ.)
# @FileName: hier_decision.py
# =====================================

import numpy as np
from endtoend import CrossroadEnd2end
from math import cos, sin, pi
from dynamics_and_models import EnvironmentModel
from hierarchical_decision.static_traj_generator import StaticTrajectoryGenerator
from endtoend_env_utils import shift_coordination, rotate_coordination, rotate_and_shift_coordination, deal_with_phi, \
    L, W, CROSSROAD_SIZE, LANE_WIDTH, LANE_NUMBER, judge_feasible, MODE2TASK, VEHICLE_MODE_DICT, VEH_NUM
from utils.load_policy import LoadPolicy
import time
import tensorflow as tf
from utils.recorder import Recorder
from utils.misc import TimerStat
import matplotlib.pyplot as plt


class HierarchicalDecision(object):
    def __init__(self, task):
        self.task = task
        if self.task == 'left':
            self.policy = LoadPolicy('G:\\env_build\\utils\\models\\left', 100000)
        elif self.task == 'right':
            self.policy = LoadPolicy('G:\\env_build\\utils\\models\\right', 145000)
        elif self.task == 'straight':
            self.policy = LoadPolicy('G:\\env_build\\utils\\models\\straight', 95000)
        self.env = CrossroadEnd2end(training_task=self.task)
        self.model = EnvironmentModel(self.task)
        self.obs = self.env.reset()
        self.stg = StaticTrajectoryGenerator(mode='static_traj')
        self.recorder = Recorder()
        self.cal_timer = TimerStat()
        self.total_time = []

    def reset(self):
        self.obs = self.env.reset()
        self.stg = StaticTrajectoryGenerator(mode='static_traj')  # mode: static_traj or dyna_traj
        self.recorder.reset()
        self.recorder.save()
        self.total_time = []
        return self.obs

    def safe_shield(self, real_obs, traj):
        action_bound = 1.0
        # action_safe_set = ([[-action_bound, -action_bound]], [[-action_bound, action_bound]],
        #                    [[action_bound, -action_bound]], [[action_bound, action_bound]])
        action_safe_set = [[[0, -action_bound]]]
        real_obs = tf.convert_to_tensor(real_obs[np.newaxis, :])
        obs = real_obs
        self.model.add_traj(obs, traj, mode='selecting')
        total_punishment = 0.0

        for step in range(3):
            action = self.policy.run(obs)
            obs, veh2veh4real = self.model.safety_calculation(obs, action)
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
        with self.cal_timer:
            start_time = time.time()
            traj_list, feature_points = self.stg.generate_traj(self.task, self.obs)
            traj_return_value = []

            for trajectory in traj_list:
                self.env.set_traj(trajectory)
                # initial state
                obs = self.env._get_obs(func='selecting')[np.newaxis, :]
                traj_value = self.policy.values(obs)
                traj_return_value.append(traj_value.numpy().squeeze().tolist())

            traj_return_value = np.array(traj_return_value, dtype=np.float32)
            path_index = np.argmax(traj_return_value[:, 0])

            self.env.set_traj(traj_list[path_index])
            self.obs_real = self.env._get_obs(func='tracking')

            # obtain safe action
            # safe_action = self.safe_shield(self.obs_real, traj_list[path_index])
            safe_action = self.policy.run(self.obs_real).numpy()
            end_time = time.time()
            print('ALL TIME:', end_time - start_time)
        if self.env.v_light != 0 and self.env.ego_dynamics['y'] < -25 and \
           self.env.ego_dynamics['y'] > -35 and self.env.training_task != 'right':
            scaled_steer = 0.
            if self.env.ego_dynamics['v_x'] == 0.0:
                scaled_a_x = 0.33
            else:
                scaled_a_x = np.random.uniform(-0.6, -0.4)
            safe_action = np.array([scaled_steer, scaled_a_x], dtype=np.float32)
        self.total_time.append(end_time - start_time)
        self.render(traj_list, traj_return_value, path_index, feature_points)
        self.recorder.record(self.obs_real, safe_action, self.cal_timer.mean, path_index)

        self.obs, r, done, info = self.env.step(safe_action)
        return done

    def render(self, traj_list, traj_return_value, path_index, feature_points):
        square_length = CROSSROAD_SIZE
        extension = 40
        lane_width = LANE_WIDTH
        light_line_width = 3
        dotted_line_style = '--'
        solid_line_style = '-'

        plt.cla()
        plt.title("Crossroad")
        ax = plt.axes(xlim=(-square_length / 2 - extension, square_length / 2 + extension),
                      ylim=(-square_length / 2 - extension, square_length / 2 + extension))
        plt.axis("equal")
        plt.axis('off')

        # ax.add_patch(plt.Rectangle((-square_length / 2, -square_length / 2),
        #                            square_length, square_length, edgecolor='black', facecolor='none'))
        ax.add_patch(plt.Rectangle((-square_length / 2 - extension, -square_length / 2 - extension),
                                   square_length + 2 * extension, square_length + 2 * extension, edgecolor='black',
                                   facecolor='none'))

        # ----------arrow--------------
        plt.arrow(lane_width / 2, -square_length / 2 - 10, 0, 5, color='b')
        plt.arrow(lane_width / 2, -square_length / 2 - 10 + 5, -0.5, 0, color='b', head_width=1)
        plt.arrow(lane_width * 1.5, -square_length / 2 - 10, 0, 5, color='b', head_width=1)
        plt.arrow(lane_width * 2.5, -square_length / 2 - 10, 0, 5, color='b')
        plt.arrow(lane_width * 2.5, -square_length / 2 - 10 + 5, 0.5, 0, color='b', head_width=1)

        # ----------horizon--------------
        plt.plot([-square_length / 2 - extension, -square_length / 2], [0, 0], color='black')
        plt.plot([square_length / 2 + extension, square_length / 2], [0, 0], color='black')

        #
        for i in range(1, LANE_NUMBER + 1):
            linestyle = dotted_line_style if i < LANE_NUMBER else solid_line_style
            plt.plot([-square_length / 2 - extension, -square_length / 2], [i * lane_width, i * lane_width],
                     linestyle=linestyle, color='black')
            plt.plot([square_length / 2 + extension, square_length / 2], [i * lane_width, i * lane_width],
                     linestyle=linestyle, color='black')
            plt.plot([-square_length / 2 - extension, -square_length / 2], [-i * lane_width, -i * lane_width],
                     linestyle=linestyle, color='black')
            plt.plot([square_length / 2 + extension, square_length / 2], [-i * lane_width, -i * lane_width],
                     linestyle=linestyle, color='black')

        # ----------vertical----------------
        plt.plot([0, 0], [-square_length / 2 - extension, -square_length / 2], color='black')
        plt.plot([0, 0], [square_length / 2 + extension, square_length / 2], color='black')

        #
        for i in range(1, LANE_NUMBER + 1):
            linestyle = dotted_line_style if i < LANE_NUMBER else solid_line_style
            plt.plot([i * lane_width, i * lane_width], [-square_length / 2 - extension, -square_length / 2],
                     linestyle=linestyle, color='black')
            plt.plot([i * lane_width, i * lane_width], [square_length / 2 + extension, square_length / 2],
                     linestyle=linestyle, color='black')
            plt.plot([-i * lane_width, -i * lane_width], [-square_length / 2 - extension, -square_length / 2],
                     linestyle=linestyle, color='black')
            plt.plot([-i * lane_width, -i * lane_width], [square_length / 2 + extension, square_length / 2],
                     linestyle=linestyle, color='black')

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
                 color='black')
        plt.plot([LANE_NUMBER * lane_width, square_length / 2], [square_length / 2, LANE_NUMBER * lane_width],
                 color='black')
        plt.plot([-LANE_NUMBER * lane_width, -square_length / 2], [-square_length / 2, -LANE_NUMBER * lane_width],
                 color='black')
        plt.plot([-LANE_NUMBER * lane_width, -square_length / 2], [square_length / 2, LANE_NUMBER * lane_width],
                 color='black')

        def is_in_plot_area(x, y, tolerance=5):
            if -square_length / 2 - extension + tolerance < x < square_length / 2 + extension - tolerance and \
                    -square_length / 2 - extension + tolerance < y < square_length / 2 + extension - tolerance:
                return True
            else:
                return False

        def draw_rotate_rec(x, y, a, l, w, color, linestyle='-'):
            RU_x, RU_y, _ = rotate_coordination(l / 2, w / 2, 0, -a)
            RD_x, RD_y, _ = rotate_coordination(l / 2, -w / 2, 0, -a)
            LU_x, LU_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
            LD_x, LD_y, _ = rotate_coordination(-l / 2, -w / 2, 0, -a)
            ax.plot([RU_x + x, RD_x + x], [RU_y + y, RD_y + y], color=color, linestyle=linestyle)
            ax.plot([RU_x + x, LU_x + x], [RU_y + y, LU_y + y], color=color, linestyle=linestyle)
            ax.plot([LD_x + x, RD_x + x], [LD_y + y, RD_y + y], color=color, linestyle=linestyle)
            ax.plot([LD_x + x, LU_x + x], [LD_y + y, LU_y + y], color=color, linestyle=linestyle)

        def plot_phi_line(x, y, phi, color):
            line_length = 5
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
            color = ['blue', 'coral', 'cyan']
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
        # for j, item_point in enumerate(feature_points):
        #     for k in range(len(item_point)):
        #         plt.scatter(item_point[k][0], item_point[k][1], c='lime')

        # plot ego dynamics
        text_x, text_y_start = -120, 60
        ge = iter(range(0, 1000, 4))
        plt.text(text_x, text_y_start - next(ge), 'ego_x: {:.2f}m'.format(ego_x))
        plt.text(text_x, text_y_start - next(ge), 'ego_y: {:.2f}m'.format(ego_y))
        plt.text(text_x, text_y_start - next(ge), 'path_x: {:.2f}m'.format(path_x))
        plt.text(text_x, text_y_start - next(ge), 'path_y: {:.2f}m'.format(path_y))
        plt.text(text_x, text_y_start - next(ge), 'delta_: {:.2f}m'.format(delta_))
        plt.text(text_x, text_y_start - next(ge), 'delta_x: {:.2f}m'.format(delta_x))
        plt.text(text_x, text_y_start - next(ge), 'delta_y: {:.2f}m'.format(delta_y))
        plt.text(text_x, text_y_start - next(ge), r'ego_phi: ${:.2f}\degree$'.format(ego_phi))
        plt.text(text_x, text_y_start - next(ge), r'path_phi: ${:.2f}\degree$'.format(path_phi))
        plt.text(text_x, text_y_start - next(ge), r'delta_phi: ${:.2f}\degree$'.format(delta_phi))
        plt.text(text_x, text_y_start - next(ge), 'v_x: {:.2f}m/s'.format(ego_v_x))
        plt.text(text_x, text_y_start - next(ge), 'exp_v: {:.2f}m/s'.format(self.env.exp_v))
        plt.text(text_x, text_y_start - next(ge), 'v_y: {:.2f}m/s'.format(ego_v_y))
        plt.text(text_x, text_y_start - next(ge), 'yaw_rate: {:.2f}rad/s'.format(ego_r))
        plt.text(text_x, text_y_start - next(ge), 'yaw_rate bound: [{:.2f}, {:.2f}]'.format(-r_bound, r_bound))

        plt.text(text_x, text_y_start - next(ge), r'$\alpha_f$: {:.2f} rad'.format(ego_alpha_f))
        plt.text(text_x, text_y_start - next(ge), r'$\alpha_f$ bound: [{:.2f}, {:.2f}] '.format(-alpha_f_bound,
                                                                                                alpha_f_bound))
        plt.text(text_x, text_y_start - next(ge), r'$\alpha_r$: {:.2f} rad'.format(ego_alpha_r))
        plt.text(text_x, text_y_start - next(ge), r'$\alpha_r$ bound: [{:.2f}, {:.2f}] '.format(-alpha_r_bound,
                                                                                                alpha_r_bound))
        if self.env.action is not None:
            steer, a_x = self.env.action[0], self.env.action[1]
            plt.text(text_x, text_y_start - next(ge),
                     r'steer: {:.2f}rad (${:.2f}\degree$)'.format(steer, steer * 180 / np.pi))
            plt.text(text_x, text_y_start - next(ge), 'a_x: {:.2f}m/s^2'.format(a_x))

        text_x, text_y_start = 70, 60
        ge = iter(range(0, 1000, 4))

        # done info
        plt.text(text_x, text_y_start - next(ge), 'done info: {}'.format(self.env.done_type))

        # reward info
        if self.env.reward_info is not None:
            for key, val in self.env.reward_info.items():
                plt.text(text_x, text_y_start - next(ge), '{}: {:.4f}'.format(key, val))

        # indicator for trajectory selection
        text_x, text_y_start = -18, -70
        ge = iter(range(0, 1000, 6))
        if traj_return_value is not None:
            for i, value in enumerate(traj_return_value):
                if i == path_index:
                    plt.text(text_x, text_y_start - next(ge), 'Path reward={:.4f}'.format(value[0]), fontsize=14,
                             color=color[i], fontstyle='italic')
                else:
                    plt.text(text_x, text_y_start - next(ge), 'Path reward={:.4f}'.format(value[0]), fontsize=12,
                             color=color[i], fontstyle='italic')
        plt.pause(0.001)


def plot_data(i):
    recorder = Recorder()
    recorder.load()
    recorder.plot_ith_episode_curves(i)


def main():
    hier_decision = HierarchicalDecision('left')
    for i in range(300):
        done = 0
        while not done:
            done = hier_decision.step()
            if done == 2:
                print('Episode {} is SUCCESS!'.format(i))
                print(np.mean(hier_decision.total_time))
                print(hier_decision.total_time)
        hier_decision.reset()


if __name__ == '__main__':
    main()
    # plot_data(3)


