#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: multi_ego.py
# =====================================

import copy
import datetime
import os
import time
from math import cos, sin, pi

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dynamics_and_models import EnvironmentModel
from endtoend import CrossroadEnd2end
from endtoend_env_utils import rotate_coordination, cal_ego_info_in_transform_coordination, \
    cal_info_in_transform_coordination, CROSSROAD_SIZE, LANE_WIDTH, LANE_NUMBER
from hierarchical_decision.multi_path_generator import MultiPathGenerator
from traffic import Traffic
from utils.load_policy import LoadPolicy

NAME2TASK = dict(DL='left', DU='straight', DR='right',
                 RD='left', RL='straight', RU='right',
                 UR='left', UD='straight', UL='right',
                 LU='left', LR='straight', LD='right')
ROTATE_ANGLE = dict(D=0, R=90, U=180, L=-90)
dirname = os.path.dirname(__file__)


class MultiEgo(object):
    def __init__(self, init_n_ego_dict):  # init_n_ego_dict is used to init traffic (mainly) and ego dynamics
        self.TASK2MODEL = dict(left=LoadPolicy('../utils/models/left', 100000),
                               straight=LoadPolicy('../utils/models/straight', 95000),
                               right=LoadPolicy('../utils/models/right', 145000),)
        self.n_ego_instance = {}
        self.n_ego_dynamics = {}
        self.n_ego_select_index = {}
        for egoID, ego_dict in init_n_ego_dict.items():
            self.n_ego_instance[egoID] = CrossroadEnd2end(training_task=NAME2TASK[egoID[:2]], display=True)

        self.mpp = MultiPathGenerator()
        self.virtual_model = dict(left=EnvironmentModel(training_task='left', mode='selecting'),
                                  straight=EnvironmentModel(training_task='straight', mode='selecting'),
                                  right=EnvironmentModel(training_task='right', mode='selecting'))

        self.reset(init_n_ego_dict)

    def reset(self, init_n_ego_dict):
        self.egoID2pop = []
        for egoID, ego_dict in init_n_ego_dict.items():
            self.n_ego_dynamics[egoID] = self.n_ego_instance[egoID]._get_ego_dynamics([ego_dict['v_x'],
                                                                                       ego_dict['v_y'],
                                                                                       ego_dict['r'],
                                                                                       ego_dict['x'],
                                                                                       ego_dict['y'],
                                                                                       ego_dict['phi']],
                                                                                      [0,
                                                                                       0,
                                                                                       self.n_ego_instance[egoID].dynamics.vehicle_params['miu'],
                                                                                       self.n_ego_instance[egoID].dynamics.vehicle_params['miu']])

    def get_next_n_ego_dynamics(self, n_ego_vehicles, v_light):
        for egoID, ego_dynamics in self.n_ego_dynamics.items():
            rotate_angle = ROTATE_ANGLE[egoID[0]]
            vehicles = n_ego_vehicles[egoID]
            vehicles_trans = cal_info_in_transform_coordination(vehicles, 0, 0, rotate_angle)
            ego_dynamics_trans = cal_ego_info_in_transform_coordination(ego_dynamics, 0, 0, rotate_angle)
            if rotate_angle == 0 or rotate_angle == 180:
                v_light_trans = v_light
            else:
                v_light_trans = 2 if v_light != 2 else 0
            self.n_ego_instance[egoID].all_vehicles = vehicles_trans
            self.n_ego_instance[egoID].ego_dynamics = ego_dynamics_trans
            self.n_ego_instance[egoID].v_light = v_light_trans

            # generate multiple paths
            task = NAME2TASK[egoID[:2]]
            path_list = self.mpp.generate_path(task)
            obs_list = []
            # evaluate each path
            for path in path_list:
                self.n_ego_instance[egoID].set_traj(path)
                obs = self.n_ego_instance[egoID]._get_obs(exit_=egoID[0])
                obs_list.append(obs)
            all_obs = tf.stack(obs_list, axis=0)
            path_values = self.TASK2MODEL[task].values(all_obs)
            # select and safety shield
            path_index = int(np.argmax(path_values[:, 0]))
            # select
            # temp = path_values[:, 0]
            # if task == 'right':
            #     temp = path_values[:, 0] + np.array([170, 0, 0])
            # path_index = np.argmax(temp)
            self.n_ego_select_index[egoID] = path_index
            # self.n_ego_instance[egoID].render(traj_list, traj_return_value, path_index, feature_points)
            self.obs_real = obs_list[path_index]

            # safe shield
            if v_light_trans == 0:
                # safe_action = self.safe_shield(self.obs_real, traj_list[path_index], egoID, task)
                safe_action = self.TASK2MODEL[task].run(self.obs_real).numpy()
            else:
                safe_action = self.TASK2MODEL[task].run(self.obs_real).numpy()
            action_trans = self.n_ego_instance[egoID]._action_transformation_for_end2end(safe_action)
            next_ego_state, next_ego_params = self.n_ego_instance[egoID]._get_next_ego_state(action_trans)
            next_ego_dynamics = self.n_ego_instance[egoID]._get_ego_dynamics(next_ego_state, next_ego_params)
            self.n_ego_dynamics[egoID] = cal_ego_info_in_transform_coordination(next_ego_dynamics, 0, 0, -rotate_angle)

        return copy.deepcopy(self.n_ego_dynamics)

    def judge_n_ego_done(self, n_ego_collision_flag):
        n_ego_done = {}
        for egoID in self.n_ego_dynamics.keys():
            ego_instance = self.n_ego_instance[egoID]
            collision_flag = n_ego_collision_flag[egoID]
            is_achieve_goal = ego_instance._is_achieve_goal()
            n_ego_done[egoID] = [collision_flag, is_achieve_goal]
        return n_ego_done

    def safe_shield(self, real_obs, traj, egoID, task=None):
        action_bound = 1.0
        action_safe_set = ([[0., -action_bound]], [[-action_bound, -action_bound]], [[-action_bound, action_bound]],
                           [[action_bound, -action_bound]], [[action_bound, action_bound]])
        # action_safe_set = [0., -action_bound]
        real_obs = real_obs[np.newaxis, :]
        obs = real_obs

        model = self.virtual_model[task]
        model.add_traj(obs, traj)
        total_punishment = 0

        # TODO: RULES
        #######################################
        if egoID == 'UD' or egoID == 'DU':
            action_safe_set = ([[-action_bound, action_bound]],)

        #######################################
        for step in range(3):
            action = self.TASK2MODEL[task].run(obs)
            _, _, _, _, veh2veh4real, _ = model.rollout_out(action)
            total_punishment += veh2veh4real

        if total_punishment != 0:
            sa = None
            print('original action will cause collision within three steps!!!')
            for safe_action in action_safe_set:
                obs = real_obs
                model.add_traj(obs, traj)
                total_punishment = 0
                for step in range(1):
                    _, _, _, _, veh2veh4real, _ = model.rollout_out(safe_action)
                    total_punishment += veh2veh4real
                    if veh2veh4real != 0:   # collide
                        break
                if total_punishment == 0:
                    print('found the safe action', safe_action)
                    sa = np.array(safe_action[0])
                    break
                else:
                    print('still collide')
                    sa = np.array(action_safe_set[0][0])
            print(sa)
        else:
            sa = self.TASK2MODEL[task].run(real_obs).numpy().squeeze(0)
        return sa


class Simulation(object):
    def __init__(self, init_n_ego_dict, logdir):
        self.init_n_ego_dict = init_n_ego_dict
        self.multiego = MultiEgo(copy.deepcopy(self.init_n_ego_dict))
        self.traffic = Traffic(100, 'display', self.init_n_ego_dict)
        self.n_ego_traj_trans = {}
        plt.ion()
        self.fig = plt.figure(figsize=(8, 8))
        self.hist_posi = {egoID: [] for egoID in self.init_n_ego_dict.keys()}
        self.episode_counter = -1
        self.step_counter = -1
        self.logdir = logdir
        self.reset()

    def reset(self):
        self.multiego.reset(copy.deepcopy(self.init_n_ego_dict))
        self.traffic.init_traffic(copy.deepcopy(self.init_n_ego_dict))
        self.traffic.sim_step()
        n_ego_traj = {egoID: self.multiego.mpp.generate_path(NAME2TASK[egoID[:2]]) for egoID in
                      self.multiego.n_ego_dynamics.keys()}
        self.n_ego_traj_trans = {}
        for egoID, ego_traj in n_ego_traj.items():
            traj_list = []
            for item in ego_traj:
                temp = np.array([rotate_coordination(x, y, 0, -ROTATE_ANGLE[egoID[0]])[0] for x, y in
                                 zip(item.path[0], item.path[1])]), \
                       np.array([rotate_coordination(x, y, 0, -ROTATE_ANGLE[egoID[0]])[1] for x, y in
                                 zip(item.path[0], item.path[1])])
                traj_list.append(temp)
            self.n_ego_traj_trans[egoID] = traj_list
        self.hist_posi = {egoID: [] for egoID in self.init_n_ego_dict.keys()}
        if self.logdir is not None:
            self.episode_counter += 1
            self.step_counter = -1
            os.makedirs(self.logdir + '/episode{}'.format(self.episode_counter))

    def step(self):
        self.step_counter += 1
        current_n_ego_vehicles = self.traffic.n_ego_vehicles
        current_v_light = self.traffic.v_light
        current_n_ego_collision_flag = self.traffic.n_ego_collision_flag
        start_time = time.time()
        next_n_ego_dynamics = self.multiego.get_next_n_ego_dynamics(current_n_ego_vehicles, current_v_light)
        end_time = time.time()
        # print('Time for all vehicles:', end_time - start_time)
        n_ego_done = self.multiego.judge_n_ego_done(current_n_ego_collision_flag)
        for egoID, flag_list in n_ego_done.items():
            if flag_list[0]:
                print('Ego {} collision!'.format(egoID))
                return 1
            elif flag_list[1]:
                print('Ego {} achieve goal!'.format(egoID))
                self.multiego.n_ego_dynamics.pop(egoID)
                self.traffic.n_ego_dict.pop(egoID)
                next_n_ego_dynamics.pop(egoID)

                self.multiego.n_ego_select_index.pop(egoID)
                self.n_ego_traj_trans.pop(egoID)
                if len(self.traffic.n_ego_dict) == 0:
                    print('All ego achieve goal!'.format(egoID))

                    return 1
        self.traffic.set_own_car(next_n_ego_dynamics)
        self.traffic.sim_step()
        return 0

    def render(self,):
        # plot basic map
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
        # plt.arrow(lane_width/2, -square_length / 2-10, 0, 5, color='b', head_width=1)
        # plt.arrow(lane_width/2, -square_length / 2-10+2.5, -0.5, 0, color='b', head_width=1)
        # plt.arrow(lane_width*1.5, -square_length / 2-10, 0, 5, color='b', head_width=1)
        # plt.arrow(lane_width*2.5, -square_length / 2 - 10, 0, 5, color='b')
        # plt.arrow(lane_width*2.5, -square_length / 2 - 10+5, 0.5, 0, color='b', head_width=1)

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

        v_light = self.traffic.v_light
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

        # plot other cars
        n_ego_vehicles = {egoID: self.multiego.n_ego_instance[egoID].all_vehicles for egoID in self.multiego.n_ego_dynamics.keys()}
        n_ego_dynamics = {egoID: self.multiego.n_ego_instance[egoID].ego_dynamics for egoID in self.multiego.n_ego_dynamics.keys()}
        n_ego_traj = {egoID: self.multiego.mpp.generate_path(NAME2TASK[egoID[:2]]) for egoID in self.multiego.n_ego_dynamics.keys()}

        some_egoID = list(n_ego_vehicles.keys())[0]
        all_vehicles = cal_info_in_transform_coordination(n_ego_vehicles[some_egoID], 0, 0,
                                                          -ROTATE_ANGLE[some_egoID[0]])
        n_ego_dynamics_trans = {}
        for egoID, ego_dynamics in n_ego_dynamics.items():
            n_ego_dynamics_trans[egoID] = cal_ego_info_in_transform_coordination(ego_dynamics, 0, 0,
                                                                             -ROTATE_ANGLE[egoID[0]])

        for veh in all_vehicles:
            x = veh['x']
            y = veh['y']
            a = veh['phi']
            l = veh['l']
            w = veh['w']
            if is_in_plot_area(x, y):
                draw_rotate_rec(x, y, a, l, w, 'black')
                plot_phi_line(x, y, a, 'black')

        # plot own car
        for egoID, ego_info in n_ego_dynamics_trans.items():
            ego_x = ego_info['x']
            ego_y = ego_info['y']
            ego_a = ego_info['phi']
            ego_l = ego_info['l']
            ego_w = ego_info['w']
            self.hist_posi[egoID].append((ego_x, ego_y))
            draw_rotate_rec(ego_x, ego_y, ego_a, ego_l, ego_w, 'fuchsia')
            plot_phi_line(ego_x, ego_y, ego_a, 'fuchsia')

        # plot history
        for egoID in self.init_n_ego_dict.keys():
            for hist_x, hist_y in self.hist_posi[egoID]:
                plt.scatter(hist_x, hist_y, color='fuchsia', alpha=0.1)

        # plot trajectory
        color = ['blue', 'coral', 'darkcyan']
        for egoID, planed_traj in self.n_ego_traj_trans.items():
            for i, path in enumerate(planed_traj):
                alpha = 1
                if v_color != 'green':
                    if egoID[:2] in ['DL', 'DU', 'UD', 'UR']:
                        alpha = 0.2
                if h_color != 'green':
                    if egoID[:2] in ['RD', 'RL', 'LR', 'LU']:
                        alpha = 0.2
                if planed_traj is not None:
                    if i == self.multiego.n_ego_select_index[egoID]:
                        ax.plot(path[0], path[1], color=color[i], alpha=alpha)
                    else:
                        ax.plot(path[0], path[1], color=color[i], alpha=0.2)
        # plt.show()
        plt.pause(0.001)
        if self.logdir is not None:
            plt.savefig(self.logdir + '/episode{}'.format(self.episode_counter) + '/step{}.pdf'.format(self.step_counter))



if __name__ == '__main__':
    init_n_ego_dict = dict(
        DL1=dict(v_x=5, v_y=0, r=0, x=0.5 * LANE_WIDTH, y=-30, phi=90, l=4.3, w=1.9, routeID='dl'),
        DU1=dict(v_x=8, v_y=0, r=0, x=1.5 * LANE_WIDTH, y=-45, phi=90, l=4.3, w=1.9, routeID='du'),
        DR1=dict(v_x=5, v_y=0, r=0, x=2.5 * LANE_WIDTH, y=-30, phi=90, l=4.3, w=1.9, routeID='dr'),
        RD1=dict(v_x=3, v_y=0, r=0, x=31.5, y=0.5 * LANE_WIDTH, phi=180, l=4.3, w=1.9, routeID='rd'),
        RL1=dict(v_x=5, v_y=0, r=0, x=33, y=1.5 * LANE_WIDTH, phi=180, l=4.3, w=1.9, routeID='rl'),
        RU1=dict(v_x=5, v_y=0, r=0, x=38, y=2.5 * LANE_WIDTH, phi=180, l=4.3, w=1.9, routeID='ru'),
        UR1=dict(v_x=5, v_y=0, r=0, x=-0.5 * LANE_WIDTH, y=32, phi=-90, l=4.3, w=1.9, routeID='ur'),
        UD1=dict(v_x=5, v_y=0, r=0, x=-1.5 * LANE_WIDTH, y=50, phi=-90, l=4.3, w=1.9, routeID='ud'),
        UL1=dict(v_x=5, v_y=0, r=0, x=-2.5 * LANE_WIDTH, y=50, phi=-90, l=4.3, w=1.9, routeID='ul'),
        LU1=dict(v_x=5, v_y=0, r=0, x=-34, y=-0.5 * LANE_WIDTH, phi=0, l=4.3, w=1.9, routeID='lu'),
        LR1=dict(v_x=5, v_y=0, r=0, x=-32, y=-1.5 * LANE_WIDTH, phi=0, l=4.3, w=1.9, routeID='lr'),
        LD1=dict(v_x=5, v_y=0, r=0, x=-30, y=-2.5 * LANE_WIDTH, phi=0, l=4.3, w=1.9, routeID='ld'),
    )
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = './results/{time}'.format(time=time_now)
    os.makedirs(logdir)
    simulation = Simulation(init_n_ego_dict, logdir)
    done = 0
    while 1:
        while not done:
            done = simulation.step()
            if not done:
                start_time = time.time()
                simulation.render()
                end_time = time.time()
                # print('render time:', end_time -start_time)
        simulation.reset()
        print('NEW EPISODE*********************************')
        done = 0
