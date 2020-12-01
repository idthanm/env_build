#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: multi_ego.py
# =====================================

import copy
import os

import matplotlib.pyplot as plt
import numpy as np

from endtoend import CrossroadEnd2end
from endtoend_env_utils import rotate_coordination, cal_ego_info_in_transform_coordination, \
    cal_info_in_transform_coordination
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
        self.TASK2MODEL = dict(left=LoadPolicy(dirname + '/models/left', 94000),
                               straight=LoadPolicy(dirname + '/models/straight', 94000),
                               right=LoadPolicy(dirname + '/models/right', 94000),)
        self.n_ego_instance = {}
        self.n_ego_dynamics = {}
        for egoID, ego_dict in init_n_ego_dict.items():
            self.n_ego_instance[egoID] = CrossroadEnd2end(training_task=NAME2TASK[egoID[:2]], display=True)

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
            obs = self.n_ego_instance[egoID]._get_obs(exit_=egoID[0])
            task = NAME2TASK[egoID[:2]]
            # select and safety shield -------------------
            logits = self.TASK2MODEL[task].run(obs).numpy()
            # select and safety shield -------------------
            action_trans = self.n_ego_instance[egoID]._action_transformation_for_end2end(logits)
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


class Simulation(object):
    def __init__(self, init_n_ego_dict):
        self.init_n_ego_dict = init_n_ego_dict
        self.multiego = MultiEgo(copy.deepcopy(self.init_n_ego_dict))
        self.traffic = Traffic(100, 'display', self.init_n_ego_dict)
        self.reset()
        plt.ion()

    def reset(self):
        self.multiego.reset(copy.deepcopy(self.init_n_ego_dict))
        self.traffic.init_traffic(copy.deepcopy(self.init_n_ego_dict))
        self.traffic.sim_step()

    def step(self):
        current_n_ego_vehicles = self.traffic.n_ego_vehicles
        current_v_light = self.traffic.v_light
        current_n_ego_collision_flag = self.traffic.n_ego_collision_flag
        next_n_ego_dynamics = self.multiego.get_next_n_ego_dynamics(current_n_ego_vehicles, current_v_light)
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
                if len(self.traffic.n_ego_dict) == 0:
                    print('All ego achieve goal!'.format(egoID))
                    return 1
        self.traffic.set_own_car(next_n_ego_dynamics)
        self.traffic.sim_step()
        return 0

    def render(self, mode='human'):
        if mode == 'human':
            # plot basic map
            square_length = 36
            extension = 40
            lane_width = 3.75
            dotted_line_style = '--'
            light_line_width = 3

            plt.cla()
            plt.title("Demo")
            ax = plt.axes(xlim=(-square_length / 2 - extension, square_length / 2 + extension),
                          ylim=(-square_length / 2 - extension, square_length / 2 + extension))
            plt.axis("equal")

            # ax.add_patch(plt.Rectangle((-square_length / 2, -square_length / 2),
            #                            square_length, square_length, edgecolor='black', facecolor='none'))
            ax.add_patch(plt.Rectangle((-square_length / 2 - extension, -square_length / 2 - extension),
                                       square_length + 2 * extension, square_length + 2 * extension, edgecolor='black',
                                       facecolor='none'))

            # ----------horizon--------------
            plt.plot([-square_length / 2 - extension, -square_length / 2], [0, 0], color='black')
            plt.plot([square_length / 2 + extension, square_length / 2], [0, 0], color='black')

            #
            plt.plot([-square_length / 2 - extension, -square_length / 2], [lane_width, lane_width],
                     linestyle=dotted_line_style, color='black')
            plt.plot([square_length / 2 + extension, square_length / 2], [lane_width, lane_width],
                     linestyle=dotted_line_style, color='black')

            plt.plot([-square_length / 2 - extension, -square_length / 2], [2 * lane_width, 2 * lane_width],
                     color='black')
            plt.plot([square_length / 2 + extension, square_length / 2], [2 * lane_width, 2 * lane_width],
                     color='black')
            #
            plt.plot([-square_length / 2 - extension, -square_length / 2], [-lane_width, -lane_width],
                     linestyle=dotted_line_style, color='black')
            plt.plot([square_length / 2 + extension, square_length / 2], [-lane_width, -lane_width],
                     linestyle=dotted_line_style, color='black')

            plt.plot([-square_length / 2 - extension, -square_length / 2], [-2 * lane_width, -2 * lane_width],
                     color='black')
            plt.plot([square_length / 2 + extension, square_length / 2], [-2 * lane_width, -2 * lane_width],
                     color='black')

            #
            plt.plot([-square_length / 2, -2 * lane_width], [-square_length / 2, -square_length / 2],
                     color='black')
            plt.plot([square_length / 2, 2 * lane_width], [-square_length / 2, -square_length / 2],
                     color='black')
            plt.plot([-square_length / 2, -2 * lane_width], [square_length / 2, square_length / 2],
                     color='black')
            plt.plot([square_length / 2, 2 * lane_width], [square_length / 2, square_length / 2],
                     color='black')

            # ----------vertical----------------
            plt.plot([0, 0], [-square_length / 2 - extension, -square_length / 2], color='black')
            plt.plot([0, 0], [square_length / 2 + extension, square_length / 2], color='black')

            #
            plt.plot([lane_width, lane_width], [-square_length / 2 - extension, -square_length / 2],
                     linestyle=dotted_line_style, color='black')
            plt.plot([lane_width, lane_width], [square_length / 2 + extension, square_length / 2],
                     linestyle=dotted_line_style, color='black')

            plt.plot([2 * lane_width, 2 * lane_width], [-square_length / 2 - extension, -square_length / 2],
                     color='black')
            plt.plot([2 * lane_width, 2 * lane_width], [square_length / 2 + extension, square_length / 2],
                     color='black')

            #
            plt.plot([-lane_width, -lane_width], [-square_length / 2 - extension, -square_length / 2],
                     linestyle=dotted_line_style, color='black')
            plt.plot([-lane_width, -lane_width], [square_length / 2 + extension, square_length / 2],
                     linestyle=dotted_line_style, color='black')

            plt.plot([-2 * lane_width, -2 * lane_width], [-square_length / 2 - extension, -square_length / 2],
                     color='black')
            plt.plot([-2 * lane_width, -2 * lane_width], [square_length / 2 + extension, square_length / 2],
                     color='black')

            #
            plt.plot([-square_length / 2, -square_length / 2], [-square_length / 2, -2 * lane_width],
                     color='black')
            plt.plot([-square_length / 2, -square_length / 2], [square_length / 2, 2 * lane_width],
                     color='black')
            plt.plot([square_length / 2, square_length / 2], [-square_length / 2, -2 * lane_width],
                     color='black')
            plt.plot([square_length / 2, square_length / 2], [square_length / 2, 2 * lane_width],
                     color='black')

            # ----------stop line--------------
            v_light = self.traffic.v_light
            if v_light == 0:
                v_color, h_color = 'green', 'red'
            elif v_light == 1:
                v_color, h_color = 'orange', 'red'
            elif v_light == 2:
                v_color, h_color = 'red', 'green'
            else:
                v_color, h_color = 'red', 'orange'

            plt.plot([0, lane_width], [-square_length / 2, -square_length / 2],
                     color=v_color, linewidth=light_line_width)
            plt.plot([lane_width, 2 * lane_width], [-square_length / 2, -square_length / 2],
                     color='green', linewidth=light_line_width)

            plt.plot([-2 * lane_width, -lane_width], [square_length / 2, square_length / 2],
                     color='green', linewidth=light_line_width)
            plt.plot([-lane_width, 0], [square_length / 2, square_length / 2],
                     color=v_color, linewidth=light_line_width)

            plt.plot([-square_length / 2, -square_length / 2], [0, -lane_width],
                     color=h_color, linewidth=light_line_width)
            plt.plot([-square_length / 2, -square_length / 2], [-lane_width, -2 * lane_width],
                     color='green', linewidth=light_line_width)

            plt.plot([square_length / 2, square_length / 2], [lane_width, 0],
                     color=h_color, linewidth=light_line_width)
            plt.plot([square_length / 2, square_length / 2], [2 * lane_width, lane_width],
                     color='green', linewidth=light_line_width)

            # # ----------Oblique--------------
            # plt.plot([2 * lane_width, square_length / 2], [-square_length / 2, -2 * lane_width],
            #          color='black')
            # plt.plot([2 * lane_width, square_length / 2], [square_length / 2, 2 * lane_width],
            #          color='black')
            # plt.plot([-2 * lane_width, -square_length / 2], [-square_length / 2, -2 * lane_width],
            #          color='black')
            # plt.plot([-2 * lane_width, -square_length / 2], [square_length / 2, 2 * lane_width],
            #          color='black')

            def is_in_plot_area(x, y, tolerance=5):
                if -square_length / 2 - extension + tolerance < x < square_length / 2 + extension - tolerance and \
                        -square_length / 2 - extension + tolerance < y < square_length / 2 + extension - tolerance:
                    return True
                else:
                    return False

            def draw_rotate_rec(x, y, a, l, w, color):
                RU_x, RU_y, _ = rotate_coordination(l / 2, w / 2, 0, -a)
                RD_x, RD_y, _ = rotate_coordination(l / 2, -w / 2, 0, -a)
                LU_x, LU_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
                LD_x, LD_y, _ = rotate_coordination(-l / 2, -w / 2, 0, -a)
                ax.plot([RU_x + x, RD_x + x], [RU_y + y, RD_y + y], color=color)
                ax.plot([RU_x + x, LU_x + x], [RU_y + y, LU_y + y], color=color)
                ax.plot([LD_x + x, RD_x + x], [LD_y + y, RD_y + y], color=color)
                ax.plot([LD_x + x, LU_x + x], [LD_y + y, LU_y + y], color=color)

            # plot other cars
            n_ego_vehicles = {egoID: self.multiego.n_ego_instance[egoID].all_vehicles for egoID in self.multiego.n_ego_dynamics.keys()}
            n_ego_dynamics = {egoID: self.multiego.n_ego_instance[egoID].ego_dynamics for egoID in self.multiego.n_ego_dynamics.keys()}
            n_ego_traj = {egoID: self.multiego.n_ego_instance[egoID].ref_path.path for egoID in self.multiego.n_ego_dynamics.keys()}

            some_egoID = list(n_ego_vehicles.keys())[0]
            all_vehicles = cal_info_in_transform_coordination(n_ego_vehicles[some_egoID], 0, 0,
                                                              -ROTATE_ANGLE[some_egoID[0]])
            n_ego_dynamics_trans = {}
            for egoID, ego_dynamics in n_ego_dynamics.items():
                n_ego_dynamics_trans[egoID] = cal_ego_info_in_transform_coordination(ego_dynamics, 0, 0,
                                                                                 -ROTATE_ANGLE[egoID[0]])

            n_ego_traj_trans = {}
            for egoID, ego_traj in n_ego_traj.items():
                n_ego_traj_trans[egoID] = np.array([rotate_coordination(x, y, 0, -ROTATE_ANGLE[egoID[0]])[0] for x, y in
                                                    zip(ego_traj[0], ego_traj[1])]), \
                                          np.array([rotate_coordination(x, y, 0, -ROTATE_ANGLE[egoID[0]])[1] for x, y in
                                                    zip(ego_traj[0], ego_traj[1])])

            for veh in all_vehicles:
                x = veh['x']
                y = veh['y']
                a = veh['phi']
                l = veh['l']
                w = veh['w']
                if is_in_plot_area(x, y):
                    draw_rotate_rec(x, y, a, l, w, 'black')

            # plot own car
            for egoID, ego_info in n_ego_dynamics_trans.items():
                ego_x = ego_info['x']
                ego_y = ego_info['y']
                ego_a = ego_info['phi']
                ego_l = ego_info['l']
                ego_w = ego_info['w']
                draw_rotate_rec(ego_x, ego_y, ego_a, ego_l, ego_w, 'red')

            # plot planed trj
            for egoID, planed_traj in n_ego_traj_trans.items():
                alpha = 1
                if v_color != 'green':
                    if egoID[:2] in ['DL', 'DU', 'UD', 'UR']:
                        alpha = 0.2
                if h_color != 'green':
                    if egoID[:2] in ['RD', 'RL', 'LR', 'LU']:
                        alpha = 0.2
                if planed_traj is not None:
                    ax.plot(planed_traj[0], planed_traj[1], color='g', alpha=alpha)

            plt.show()
            plt.pause(0.1)


if __name__ == '__main__':
    init_n_ego_dict = dict(
        DL1=dict(v_x=5, v_y=0, r=0, x=1.875, y=-30, phi=90, l=4.3, w=1.9, routeID='dl'),
        DU1=dict(v_x=8, v_y=0, r=0, x=1.875, y=-38, phi=90, l=4.3, w=1.9, routeID='du'),
        DR1=dict(v_x=5, v_y=0, r=0, x=5.625, y=-30, phi=90, l=4.3, w=1.9, routeID='dr'),
        RD1=dict(v_x=3, v_y=0, r=0, x=30, y=1.875, phi=180, l=4.3, w=1.9, routeID='rd'),
        RL1=dict(v_x=3, v_y=0, r=0, x=22, y=1.875, phi=180, l=4.3, w=1.9, routeID='rl'),
        RU1=dict(v_x=5, v_y=0, r=0, x=30, y=5.625, phi=180, l=4.3, w=1.9, routeID='ru'),
        UR1=dict(v_x=5, v_y=0, r=0, x=-1.875, y=30, phi=-90, l=4.3, w=1.9, routeID='ur'),
        UD1=dict(v_x=5, v_y=0, r=0, x=-1.875, y=22, phi=-90, l=4.3, w=1.9, routeID='ud'),
        UL1=dict(v_x=5, v_y=0, r=0, x=-5.625, y=22, phi=-90, l=4.3, w=1.9, routeID='ul'),
        LU1=dict(v_x=5, v_y=0, r=0, x=-30, y=-1.875, phi=0, l=4.3, w=1.9, routeID='lu'),
        LR1=dict(v_x=5, v_y=0, r=0, x=-38, y=-1.875, phi=0, l=4.3, w=1.9, routeID='lr'),
        LD1=dict(v_x=5, v_y=0, r=0, x=-30, y=-5.625, phi=0, l=4.3, w=1.9, routeID='ld'),
    )

    simulation = Simulation(init_n_ego_dict)
    done = 0
    while 1:
        while not done:
            done = simulation.step()
            if not done:
                simulation.render()
        simulation.reset()
        done = 0
