#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/10/12
# @Author  : Yang Guan; Yangang Ren (Tsinghua Univ.)
# @FileName: hier_decision.py
# =====================================
from dynamics_and_models import ReferencePath
import numpy as np
from math import pi, sqrt, sin, cos
import bezier

L, W = 4.8, 2.0
LANE_WIDTH = 3.75
LANE_NUMBER = 3
CROSSROAD_SIZE = 50
control_ext = 15
sl = 40
meter_pointnum_ratio = 30


class MultiPathGenerator(object):
    def __init__(self, ref_index=3):
        # state: [v_x, v_y, r, x, y, phi(°)]
        self.path_num = 3                                   # number of trajectories
        self.exp_v = 8.
        self.order = [0 for _ in range(self.path_num)]
        self.ego_info_dim = 6
        self.ref_index = ref_index
        self.path_list = []

    def generate_path(self, task):
        self.path_list = []
        for path_index in range(self.path_num):
            ref = ReferencePath(task)
            ref.set_path(path_index)
            self.path_list.append(ref)
        return self.path_list


class StaticTrajectoryGenerator_origin(object):
    def __init__(self, mode, ref_index=3, v_light=0):
        # state: [v_x, v_y, r, x, y, phi(°)]
        self.mode = mode
        self.path_num = 3                                   # number of trajectories
        self.exp_v = 8.
        self.order = [0 for _ in range(self.path_num)]
        self.ego_info_dim = 6
        self.ref_index = ref_index

    def generate_traj(self, task, state=None, v_light=0):
        """"generate reference trajectory in real time"""
        if self.mode == 'static_traj':
            self.path_list = []
            for path_index in range(self.path_num):
                ref = ReferencePath(task)
                ref.set_path(self.mode, path_index)
                self.path_list.append(ref)
        else:
            self._future_point_choice(state, task)
            self.construct_ref_path(task, state, v_light)
        return self.path_list, self._future_points_init(task)

    def _future_points_init(self, task):
        """only correlated with tasks"""
        self.feature_points = []
        if task == 'left':
            self.end_offsets = [LANE_WIDTH * 0.5, LANE_WIDTH * 1.5, LANE_WIDTH * 2.5]
            start_offsets = [LANE_WIDTH * 0.5]
            for start_offset in start_offsets:
                for end_offset in self.end_offsets:
                    control_point1 = start_offset,      -CROSSROAD_SIZE/2,               0.5 * pi
                    control_point2 = -CROSSROAD_SIZE/2, end_offset,                      -pi
                    control_point3 = -CROSSROAD_SIZE,   end_offset,                      -pi
                    control_point4 = -sl - CROSSROAD_SIZE/2,  end_offset,                -pi

                    node = [control_point1, control_point2, control_point3, control_point4]
                    self.feature_points.append(node)

        elif task == 'right':
            self.end_offsets = [-LANE_WIDTH * 2.5, -LANE_WIDTH * 1.5, -LANE_WIDTH * 0.5]
            start_offsets = [LANE_WIDTH * 2.5]
            for start_offset in start_offsets:
                for end_offset in self.end_offsets:
                    control_point1 = start_offset,      -CROSSROAD_SIZE/2,               0.5 * pi
                    control_point2 = CROSSROAD_SIZE/2, end_offset,                      0.
                    control_point3 = CROSSROAD_SIZE,   end_offset,                      0.
                    control_point4 = sl + CROSSROAD_SIZE/2,  end_offset,                0.

                    node = [control_point1, control_point2, control_point3, control_point4]
                    self.feature_points.append(node)

        elif task == 'straight':
            self.end_offsets = [LANE_WIDTH*(i+0.5) for i in range(LANE_NUMBER)]
            start_offsets = [LANE_WIDTH * 1.5]
            for start_offset in start_offsets:
                for end_offset in self.end_offsets:
                    control_point1 = start_offset, -CROSSROAD_SIZE/2,   0.5 * pi
                    control_point2 = end_offset, CROSSROAD_SIZE/2,      0.5 * pi
                    control_point3 = end_offset, CROSSROAD_SIZE,        0.5 * pi
                    control_point4 = end_offset, CROSSROAD_SIZE/2 + sl, 0.5 * pi

                    node = [control_point1, control_point2, control_point3, control_point4]
                    self.feature_points.append(node)

        return self.feature_points

    def _future_point_choice(self, state, task):
        # choose the forward feature points according to the current state
        self.feature_points = []
        self.feature_points_all = self._future_points_init(task)
        x, y, v, phi = state[3], state[4], state[0], state[5] / 180 * pi
        for i, item in enumerate(self.feature_points_all):
            if -CROSSROAD_SIZE/2 <= y:
                if x > -CROSSROAD_SIZE/2:
                    self.feature_points.append(list(item[1:]))
                elif -CROSSROAD_SIZE < x <= -CROSSROAD_SIZE/2:
                    self.feature_points.append(list(item[2:]))
                else:
                    self.feature_points.append(list(item[3:]))
            else:
                self.feature_points = self._future_points_init(task)

        # if distance is more than dist_bound, change the feature point priorly
        self.dist_bound = 4. * np.ones([self.path_num])
        self.L0, self.L3 = 15. * np.ones([self.path_num]), 15. * np.ones([self.path_num])

        # choose feature point set according to distance between vehicle and the closest point
        for path_index in range(0, self.path_num):
            feature_point = self.feature_points[path_index][0]
            dist = sqrt((x - feature_point[0])**2 + (y - feature_point[1])**2)
            if dist < self.dist_bound[path_index]:
                self.feature_points[path_index] = self.feature_points[path_index][1:]
                print('Updating feature point…')
            feature_point = self.feature_points[path_index][0]
            dist = sqrt((x - feature_point[0])**2 + (y - feature_point[1])**2)
            self.L0[path_index] = dist / 3.0
            self.L3[path_index] = dist / 3.0

    def construct_ref_path(self, task, state, v_light):
        x, y, phi = state[3], state[4], state[5] / 180 * pi
        state = [x, y, phi]

        # The total trajectory consists of three different parts.
        def traj_start_straight(path_index, state):
            if len(self.feature_points[path_index]) == len(self.feature_points_all[path_index]) and state[1] < -CROSSROAD_SIZE / 2:
                start_line_y = np.linspace(state[1], -CROSSROAD_SIZE / 2, int(abs(state[1]-(-CROSSROAD_SIZE / 2))) * meter_pointnum_ratio, dtype=np.float32)[:-1]
                start_line_x = np.linspace(state[0], LANE_WIDTH / 2, int(abs(state[1]-(-CROSSROAD_SIZE / 2))) * meter_pointnum_ratio, dtype=np.float32)[:-1]
            else:
                start_line_x, start_line_y = [], []
            return start_line_x, start_line_y

        def traj_cross(path_index, state):
            if len(self.feature_points[path_index]) == 4:     # the vehicle has not passed into the crossing
                feature_point1, feature_point2 = self.feature_points[path_index][0], self.feature_points[path_index][1]
                self.L0, self.L3 = 15. * np.ones([self.path_num]), 15. * np.ones([self.path_num])
                cross_x, cross_y = self.trajectory_planning(feature_point1, feature_point2, feature_point2, self.L0[path_index], self.L3[path_index])

            elif len(self.feature_points[path_index]) == 1:
                feature_point1 = self.feature_points[path_index][0]
                cross_x, cross_y = self.trajectory_planning(state, feature_point1, feature_point1, self.L0[path_index],
                                                            self.L3[path_index])
            else:
                feature_point1, feature_point2 = self.feature_points[path_index][0], self.feature_points[path_index][1]
                cross_x, cross_y = self.trajectory_planning(state, feature_point1, feature_point2, self.L0[path_index],
                                                            self.L3[path_index])
            return cross_x, cross_y

        def traj_end_straight(path_index):
            if len(self.feature_points[path_index]) >= 3:    # the vehicle has not arrived
                end_line_y = self.end_offsets[path_index] * np.ones(shape=(sl * meter_pointnum_ratio), dtype=np.float32)[1:]
                end_line_x = np.linspace(-CROSSROAD_SIZE / 2, -sl - CROSSROAD_SIZE/2, sl * meter_pointnum_ratio, dtype=np.float32)[1:]
            elif len(self.feature_points[path_index]) == 2:  # the vehicle has passed the intersection
                end_line_y = self.end_offsets[path_index] * np.ones(shape=((sl - CROSSROAD_SIZE//2) * meter_pointnum_ratio), dtype=np.float32)[1:]
                end_line_x = np.linspace(-CROSSROAD_SIZE, -sl - CROSSROAD_SIZE / 2, (sl - CROSSROAD_SIZE//2) * meter_pointnum_ratio, dtype=np.float32)[1:]
            else:                                           # the vehicle has arrived
                end_line_x, end_line_y = [], []
            return end_line_x, end_line_y

        self.path_list = []
        for path_index in range(0, self.path_num):
            start_line_x, start_line_y = traj_start_straight(path_index, state)
            cross_x, cross_y = traj_cross(path_index, state)
            end_line_x, end_line_y = traj_end_straight(path_index)

            total_x, total_y = np.append(start_line_x, np.append(cross_x, end_line_x)), np.append(start_line_y, np.append(cross_y, end_line_y))
            total_x = total_x.astype(np.float32)
            total_y = total_y.astype(np.float32)

            xs_1, ys_1 = total_x[:-1], total_y[:-1]
            xs_2, ys_2 = total_x[1:], total_y[1:]
            phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1) * 180 / pi
            planed_trj = xs_1, ys_1, phis_1

            ref = ReferencePath(task)
            ref.set_path(self.mode, path_index=path_index, path=planed_trj)
            self.path_list.append(ref)

    def trajectory_planning(self, state, point1, point2, L0, L3):
        X0 = state[0]; X1 = state[0] + L0 * cos(state[2]); X2 = point1[0] - L3 * cos(point1[2]); X3 = point1[0]
        Y0 = state[1]; Y1 = state[1] + L0 * sin(state[2]); Y2 = point1[1] - L3 * sin(point1[2]); Y3 = point1[1]
        node = np.asfortranarray([[X0, X1, X2, X3], [Y0, Y1, Y2, Y3]], dtype=np.float32)
        curve = bezier.Curve(node, degree=3)
        # s_vals = np.linspace(0, 1.0, int(pi / 2 * (CROSSROAD_SIZE / 2 + LANE_WIDTH / 2)) * 30) # size: 1260
        s_vals = np.linspace(0, 1.0, 500)
        traj_data = curve.evaluate_multi(s_vals)
        traj_data = traj_data.astype(np.float32)
        return traj_data[0, :], traj_data[1, :]
        # xs_1, ys_1 = trj_data[0, :-1], trj_data[1, :-1]
        # xs_2, ys_2 = trj_data[0, 1:], trj_data[1, 1:]
        # phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1) * 180 / pi
        # planed_trj = xs_1, ys_1, phis_1
        # return planed_trj


