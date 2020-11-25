#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/10/12
# @Author  : Yang Guan; Yangang Ren (Tsinghua Univ.)
# @FileName: hierarchical_decision.py
# =====================================
from dynamics_and_models import ReferencePath
import numpy as np
from math import pi, sqrt, sin, cos
import bezier
import matplotlib.pyplot as plt

L, W = 4.8, 2.0
LANE_WIDTH = 3.75
LANE_NUMBER = 3
CROSSROAD_SIZE = 50
control_ext = 15
sl = 40
meter_pointnum_ratio = 30


class StaticTrajectoryGenerator(object):
    def __init__(self, task, state, mode, v_light=0):
        # state: [v_x, v_y, r, x, y, phi(°)]
        self.mode = mode
        self.path_num = 3  # the number of static trajectories
        self.exp_v = 8.
        self.N = 25
        self.order = [0 for _ in range(self.path_num)]
        self.task = task
        self.ego_info_dim = 6
        self.feature_points_all = self._future_points_init(self.task)
        self.state = state[:self.ego_info_dim]
        self.ref_index = np.random.choice(list(range(self.path_num)))
        self._future_point_choice(self.state)
        self.construct_ref_path(self.task, self.state, v_light)

    def generate_traj(self, task, state, v_light=0):
        """"generate two reference trajectory in real time"""
        if self.mode == 'static_traj':
            self.path_list = []
            for path_index in range(self.path_num):
                ref = ReferencePath(self.task)
                ref.set_path(self.mode, path_index)
                self.path_list.append(ref)
        else:
            self._future_point_choice(state)
            self.construct_ref_path(task, state, v_light)
        return self.path_list, self.feature_points

    def _future_points_init(self, task):
        feature_points = []
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
                    feature_points.append(node)

        elif task == 'right':
            raise SystemExit('unfinished for right turn tasks!')

        elif task == 'straight':
            raise SystemExit('unfinished for right go straight tasks!')

        return feature_points

    def _future_point_choice(self, state):
        # choose the forward feature points according to the current state
        self.feature_points = []
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
                self.feature_points = self._future_points_init(self.task)

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


        if self.ref_index == 0:
            self.new_point_bound = [0.16, 0.06]
        else:
            self.new_point_bound = [0.06, 0.16]

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

            ref = ReferencePath(self.task)
            ref.set_path(self.mode, path_index=path_index, path=planed_trj)
            self.path_list.append(ref)

    def trajectory_planning(self, state, point1, point2, L0, L3):
        X0 = state[0]; X1 = state[0] + L0 * cos(state[2]); X2 = point1[0] - L3 * cos(point1[2]); X3 = point1[0]
        Y0 = state[1]; Y1 = state[1] + L0 * sin(state[2]); Y2 = point1[1] - L3 * sin(point1[2]); Y3 = point1[1]
        plt.clf()
        plt.scatter([X1, X2], [Y1, Y2], marker='D', c='coral')
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

    # def trajectory_planning(self, x0, x_des1, x_des2, L0, L3, new_method_bound, v=4.0, T=0.1, N=20):
    #     traj, L0, L3, flag = self.reference_trajectory(x0, x_des1, x_des2, L0, L3, v, T, N, new_method_bound)
    #     return traj, L0, L3, flag
    #
    # def reference_trajectory(self, x0, x_des1, x_des2, L0, L3, v, T, N, NEW_PATH_BOUND, RELATIVE_ERR=0.02):
    #     # used for tracking!
    #     # only some of state info is necessary, e.g. x and y locations!
    #     traj = np.zeros([N + 1, len(x0)])
    #     traj_t = np.zeros(N + 1)
    #     ErrorBound = []
    #
    #     X0 = x0[0]; X1 = x0[0] + L0 * cos(x0[3]); X2 = x_des1[0] - L3 * cos(x_des1[3]); X3 = x_des1[0]; X4 = x_des2[0];
    #     Y0 = x0[1]; Y1 = x0[1] + L0 * sin(x0[3]); Y2 = x_des1[1] - L3 * sin(x_des1[3]); Y3 = x_des1[1]; Y4 = x_des2[1];
    #
    #     phi_3 = x_des1[-1]
    #     phi_4 = x_des2[-1]
    #     traj[0, :] = x0[0], x0[1], x0[2], x0[3] * 180 /pi
    #
    #     tt0 = 0
    #
    #     for i in range(N):
    #         delta_L = v * T
    #         ErrorBound.append(RELATIVE_ERR * delta_L)
    #         tt = max(2 * delta_L, 1)  # 前提：t=1对应的弧长>1!
    #         delta_tt = tt / 2.
    #         X_temp, Y_temp = self.bezier_generator(X0, X1, X2, X3, X4, Y0, Y1, Y2, Y3, Y4, phi_3, phi_4, tt0 + tt)
    #
    #         distance = ((X_temp - traj[i, 0]) ** 2 + (Y_temp - traj[i, 1]) ** 2) ** 0.5
    #
    #         while abs(distance - delta_L) > ErrorBound[i]:  # binary search!
    #             if distance - delta_L > 0:
    #                 tt = tt - delta_tt
    #             else:
    #                 tt = tt + delta_tt
    #             delta_tt = delta_tt / 2.
    #             X_temp, Y_temp = self.bezier_generator(X0, X1, X2, X3, X4, Y0, Y1, Y2, Y3, Y4, phi_3, phi_4, tt0 + tt)
    #             distance = ((X_temp - traj[i, 0]) ** 2 + (Y_temp - traj[i, 1]) ** 2) ** 0.5
    #
    #         traj_t[i + 1] = tt0 + tt
    #         traj[i + 1, 0] = X_temp
    #         traj[i + 1, 1] = Y_temp
    #
    #         tt0 = tt0 + tt
    #
    #     t1 = traj_t[1]  # t1: relative position of current state on last trajectory!
    #
    #     if t1 >= NEW_PATH_BOUND:
    #         flag = 0
    #         L0 = 0
    #         L3 = 0
    #     else:
    #         tau1 = t1 + 0.5 * (1 - t1)
    #         tau2 = t1 + 1. / 3. * (1 - t1)
    #         # A_tempx = np.array([ [3./8., 3./8.], [4./9., 2./9.] ])
    #         b_tempx = np.array([
    #             X0 * (1 - tau1) ** 3 + 3 * X1 * tau1 * (1 - tau1) ** 2 + 3 * X2 * tau1 ** 2 * (1 - tau1) + X3 * tau1 ** 3 -
    #             traj[1, 0] / 8. - X3 / 8.,
    #             X0 * (1 - tau2) ** 3 + 3 * X1 * tau2 * (1 - tau2) ** 2 + 3 * X2 * tau2 ** 2 * (1 - tau2) + X3 * tau2 ** 3 -
    #             traj[1, 0] / 27. * 8. - X3 / 27.
    #         ])
    #         # X_new = np.dot(np.linalg.inv(A_tempx),b_tempx)得到的向量第一个元素为新的X1，第二个元素为新的X2
    #         X_new = np.dot(np.array([[-2.667, 4.5], [5.333, -4.5]]), b_tempx)
    #
    #         # A_tempy = np.array([ [3./8., 3./8.], [4./9., 2./9.] ])
    #         b_tempy = np.array([
    #             Y0 * (1 - tau1) ** 3 + 3 * Y1 * tau1 * (1 - tau1) ** 2 + 3 * Y2 * tau1 ** 2 * (1 - tau1) + Y3 * tau1 ** 3 -
    #             traj[1, 1] / 8. - Y3 / 8.,
    #             Y0 * (1 - tau2) ** 3 + 3 * Y1 * tau2 * (1 - tau2) ** 2 + 3 * Y2 * tau2 ** 2 * (1 - tau2) + Y3 * tau2 ** 3 -
    #             traj[1, 1] / 27. * 8. - Y3 / 27.
    #         ])
    #         # Y_new = np.dot(np.linalg.inv(A_tempy),b_tempy)#得到的向量第一个元素为新的X1，第二个元素为新的X2
    #         Y_new = np.dot(np.array([[-2.667, 4.5], [5.333, -4.5]]), b_tempy)
    #         # print('Y_new[0] = ',Y_new[0])
    #
    #         L0 = ((X_new[0] - traj[1, 0]) ** 2 + (Y_new[0] - traj[1, 1]) ** 2) ** 0.5
    #         L3 = ((X_new[1] - X3) ** 2 + (Y_new[1] - Y3) ** 2) ** 0.5
    #         flag = 1
    #     return traj, L0, L3, flag
    #
    # def bezier_generator(self, X0, X1, X2, X3, X4, Y0, Y1, Y2, Y3, Y4, phi_3, phi_4, t):
    #     # X0,Y0: current point
    #     # X1,Y1,X2,Y2: auxiliary point
    #     # X3,Y3: next feature point
    #     # X4,Y4: nextnext feature point
    #     HYPERPARA_BEZIER_1 = 2.0
    #     HYPERPARA_BEZIER_2 = 4.0
    #
    #     if t < 1:
    #         X_t = X0 * (1 - t) ** 3 + 3 * X1 * t * (1 - t) ** 2 + 3 * X2 * t ** 2 * (1 - t) + X3 * t ** 3
    #         Y_t = Y0 * (1 - t) ** 3 + 3 * Y1 * t * (1 - t) ** 2 + 3 * Y2 * t ** 2 * (1 - t) + Y3 * t ** 3
    #     else:  # 先判断下一段是直线还是1/4圆弧
    #         if phi_3 == phi_4 and (X3 == X4 or Y3 == Y4):  # 下一段全局路径是直线
    #             X_t = X3 + (t - 1) * cos(phi_3)
    #             Y_t = Y3 + (t - 1) * sin(phi_3)
    #         else:
    #             L0 = ((X3 - X4) ** 2 + (Y3 - Y4) ** 2) ** 0.5 / HYPERPARA_BEZIER_1
    #             L3 = ((X3 - X4) ** 2 + (Y3 - Y4) ** 2) ** 0.5 / HYPERPARA_BEZIER_2
    #             X3_0 = X3 + L0 * cos(phi_3);
    #             X3_1 = X4 - L3 * cos(phi_4)
    #             Y3_0 = Y3 + L0 * sin(phi_3);
    #             Y3_1 = Y4 - L3 * sin(phi_4)
    #             X_t = X3 * (2 - t) ** 3 + 3 * X3_0 * (t - 1) * (2 - t) ** 2 + 3 * X3_1 * (t - 1) ** 2 * (2 - t) + X4 * (
    #                         t - 1) ** 3
    #             Y_t = Y3 * (2 - t) ** 3 + 3 * Y3_0 * (t - 1) * (2 - t) ** 2 + 3 * Y3_1 * (t - 1) ** 2 * (2 - t) + Y4 * (
    #                         t - 1) ** 3
    #     return X_t, Y_t

