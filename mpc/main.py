import time
from math import pi

import bezier
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import logical_and
from scipy.optimize import minimize

from multi_env.multi_ego import LoadPolicy


def deal_with_phi_diff(phi_diff):
    phi_diff = np.where(phi_diff > 180., phi_diff - 360., phi_diff)
    phi_diff = np.where(phi_diff < -180., phi_diff + 360., phi_diff)
    return phi_diff


# class VehicleDynamics(object):
#     def __init__(self, ):
#         # self.vehicle_params = dict(C_f=88000.,  # front wheel cornering stiffness [N/rad]
#         #                            C_r=94000.,  # rear wheel cornering stiffness [N/rad]
#         #                            a=1.14,  # distance from CG to front axle [m]
#         #                            b=1.40,  # distance from CG to rear axle [m]
#         #                            mass=1500.,  # mass [kg]
#         #                            I_z=2420.,  # Polar moment of inertia at CG [kg*m^2]
#         #                            miu=1.0,  # tire-road friction coefficient
#         #                            g=9.81,  # acceleration of gravity [m/s^2]
#         #                            )
#         self.vehicle_params = dict(C_f=88000.,  # front wheel cornering stiffness [N/rad]
#                                    C_r=94000.,  # rear wheel cornering stiffness [N/rad]
#                                    a=1.14,  # distance from CG to front axle [m]
#                                    b=1.40,  # distance from CG to rear axle [m]
#                                    mass=1500.,  # mass [kg]
#                                    I_z=2420.,  # Polar moment of inertia at CG [kg*m^2]
#                                    miu=1.0,  # tire-road friction coefficient
#                                    g=9.81,  # acceleration of gravity [m/s^2]
#                                    )
#         a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
#                         self.vehicle_params['mass'], self.vehicle_params['g']
#         F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
#         self.vehicle_params.update(dict(F_zf=F_zf,
#                                         F_zr=F_zr))
#
#     def f_xu(self, states, actions):  # states and actions are tensors, [[], [], ...]
#         v_x, v_y, r, x, y, phi = states[:,0],states[:,1],states[:,2],states[:,3], states[:,4],states[:,5]
#         phi = phi * np.pi / 180.
#         steer, a_x = actions[:,0],actions[:,1]
#         C_f = self.vehicle_params['C_f']
#         C_r = self.vehicle_params['C_r']
#         a = self.vehicle_params['a']
#         b = self.vehicle_params['b']
#         mass = self.vehicle_params['mass']
#         I_z = self.vehicle_params['I_z']
#         miu = self.vehicle_params['miu']
#         g = self.vehicle_params['g']
#
#         F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
#         F_xf = np.where(a_x < 0, mass * a_x / 2, np.zeros_like(a_x))
#         F_xr = np.where(a_x < 0, mass * a_x / 2, mass * a_x)
#         miu_f = np.sqrt(np.square(miu * F_zf) - np.square(F_xf)) / F_zf
#         miu_r = np.sqrt(np.square(miu * F_zr) - np.square(F_xr)) / F_zr
#         alpha_f = np.arctan((v_y + a * r) / v_x) - steer
#         alpha_r = np.arctan((v_y - b * r) / v_x)
#
#         Ff_w1 = np.square(C_f) / (3 * F_zf * miu_f)
#         Ff_w2 = np.power(C_f, 3) / (27 * np.power(F_zf * miu_f, 2))
#         F_yf_max = F_zf * miu_f
#
#         Fr_w1 = np.square(C_r) / (3 * F_zr * miu_r)
#         Fr_w2 = np.power(C_r, 3) / (27 * np.power(F_zr * miu_r, 2))
#         F_yr_max = F_zr * miu_r
#
#         F_yf = - C_f * np.tan(alpha_f) + Ff_w1 * np.tan(alpha_f) * np.abs(
#             np.tan(alpha_f)) - Ff_w2 * np.power(np.tan(alpha_f), 3)
#         F_yr = - C_r * np.tan(alpha_r) + Fr_w1 * np.tan(alpha_r) * np.abs(
#             np.tan(alpha_r)) - Fr_w2 * np.power(np.tan(alpha_r), 3)
#
#         F_yf = np.minimum(F_yf, F_yf_max)
#         F_yf = np.minimum(F_yf, -F_yf_max)
#
#         F_yr = np.minimum(F_yr, F_yr_max)
#         F_yr = np.minimum(F_yr, -F_yr_max)
#
#         state_deriv = [a_x + v_y * r,
#                        (F_yf * np.cos(steer) + F_yr) / mass - v_x * r,
#                        (a * F_yf * np.cos(steer) - b * F_yr) / I_z,
#                        v_x * np.cos(phi) - v_y * np.sin(phi),
#                        v_x * np.sin(phi) + v_y * np.cos(phi),
#                        r * 180 / np.pi,
#                        ]
#
#         state_deriv_stack = np.stack(state_deriv, 1)
#         params = np.stack([miu_f, miu_r], 1)
#
#         return state_deriv_stack, params
#
#     def prediction(self, x_1, u_1, frequency, RK):
#         f_xu_1, params = self.f_xu(x_1, u_1)
#         x_next = f_xu_1 / frequency + x_1
#         return x_next, params

class TimerStat:
    def __init__(self, window_size=10):
        self._window_size = window_size
        self._samples = []
        self._units_processed = []
        self._start_time = None
        self._total_time = 0.0
        self.count = 0

    def __enter__(self):
        assert self._start_time is None, "concurrent updates not supported"
        self._start_time = time.time()

    def __exit__(self, type, value, tb):
        assert self._start_time is not None
        time_delta = time.time() - self._start_time
        self.push(time_delta)
        self._start_time = None

    def push(self, time_delta):
        self._samples.append(time_delta)
        if len(self._samples) > self._window_size:
            self._samples.pop(0)
        self.count += 1
        self._total_time += time_delta

    @property
    def mean(self):
        if not self._samples:
            return 0.0
        return float(np.mean(self._samples))


class VehicleDynamics(object):
    def __init__(self, ):
        self.vehicle_params = dict(C_f=-128915.5,  # front wheel cornering stiffness [N/rad]
                                   C_r=-85943.6,  # rear wheel cornering stiffness [N/rad]
                                   a=1.06,  # distance from CG to front axle [m]
                                   b=1.85,  # distance from CG to rear axle [m]
                                   mass=1412.,  # mass [kg]
                                   I_z=1536.7,  # Polar moment of inertia at CG [kg*m^2]
                                   miu=1.0,  # tire-road friction coefficient
                                   g=9.81,  # acceleration of gravity [m/s^2]
                                   )
        a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
                        self.vehicle_params['mass'], self.vehicle_params['g']
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))

    def f_xu(self, states, actions, tau):  # states and actions are tensors, [[], [], ...]
        v_x, v_y, r, x, y, phi = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4], states[:, 5]
        phi = phi * np.pi / 180.
        steer, a_x = actions[:, 0], actions[:, 1]
        C_f = self.vehicle_params['C_f']
        C_r = self.vehicle_params['C_r']
        a = self.vehicle_params['a']
        b = self.vehicle_params['b']
        mass = self.vehicle_params['mass']
        I_z = self.vehicle_params['I_z']
        miu = self.vehicle_params['miu']
        g = self.vehicle_params['g']

        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        F_xf = np.where(a_x < 0, mass * a_x / 2, np.zeros_like(a_x))
        F_xr = np.where(a_x < 0, mass * a_x / 2, mass * a_x)
        miu_f = np.sqrt(np.square(miu * F_zf) - np.square(F_xf)) / F_zf
        miu_r = np.sqrt(np.square(miu * F_zr) - np.square(F_xr)) / F_zr

        next_state = [v_x + tau * (a_x + v_y * r),
                      (mass * v_y * v_x + tau * (
                                  a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * np.square(
                          v_x) * r) / (mass * v_x - tau * (C_f + C_r)),
                      (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (
                                  tau * (np.square(a) * C_f + np.square(b) * C_r) - I_z * v_x),
                      x + tau * (v_x * np.cos(phi) - v_y * np.sin(phi)),
                      y + tau * (v_x * np.sin(phi) + v_y * np.cos(phi)),
                      (phi + tau * r) * 180 / np.pi]

        return np.stack(next_state, 1), np.stack([miu_f, miu_r], 1)

    def prediction(self, x_1, u_1, frequency, RK):
        x_next, next_params = self.f_xu(x_1, u_1, 1 / frequency)
        return x_next, next_params


class ReferencePath(object):
    def __init__(self, task, ref_index=None, mode='no_train'):
        self.mode = mode
        self.task = task
        self.path_list = []
        self._construct_ref_path(self.task)
        self.ref_index = ref_index if ref_index is not None else np.random.choice([0, 1])
        self.path = self.path_list[self.ref_index]
        self.exp_v = 8

    def _construct_ref_path(self, task):
        sl = 40
        planed_trj = None
        meter_pointnum_ratio = 30
        if task == 'left':
            if self.mode == 'training':
                end_offsets = [3.75, 3.75]
            else:
                end_offsets = [1.875, 5.625]
            for i, end_offset in enumerate(end_offsets):
                control_point1 = 1.875, -18
                control_point2 = 1.875, -18 + 10
                control_point3 = -18 + 10, end_offset
                control_point4 = -18, end_offset

                node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                                          [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]],
                                         dtype=np.float32)
                curve = bezier.Curve(node, degree=3)
                s_vals = np.linspace(0, 1.0, 30 * meter_pointnum_ratio)
                trj_data = curve.evaluate_multi(s_vals)
                trj_data = trj_data.astype(np.float32)
                start_straight_line_x = 1.875 * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[:-1]
                start_straight_line_y = np.linspace(-18 - sl, -18, sl * meter_pointnum_ratio, dtype=np.float32)[:-1]
                end_straight_line_x = np.linspace(-18, -18 - sl, sl * meter_pointnum_ratio, dtype=np.float32)[1:]
                end_straight_line_y = end_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[1:]
                planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x), \
                             np.append(np.append(start_straight_line_y, trj_data[1]), end_straight_line_y)

                xs_1, ys_1 = planed_trj[0][:-1], planed_trj[1][:-1]
                xs_2, ys_2 = planed_trj[0][1:], planed_trj[1][1:]
                phis_1 = np.arctan2(ys_2 - ys_1,
                                    xs_2 - xs_1) * 180 / pi
                # if i == 1:
                #     phis_1[len(start_straight_line_x):len(start_straight_line_x)+len(s_vals)] = \
                #         phis_1[len(start_straight_line_x)+150:len(start_straight_line_x)+len(s_vals)+150]
                planed_trj = xs_1, ys_1, phis_1
                self.path_list.append(planed_trj)

        elif task == 'straight':
            if self.mode == 'training':
                end_offsets = [3.75, 3.75]
            else:
                end_offsets = [1.875, 5.625]

            for end_offset in end_offsets:
                control_point1 = 1.875, -18
                control_point2 = 1.875, -18 + 10
                control_point3 = end_offset, 18 - 10
                control_point4 = end_offset, 18

                node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                                          [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]]
                                         , dtype=np.float32)
                curve = bezier.Curve(node, degree=3)
                s_vals = np.linspace(0, 1.0, 36 * meter_pointnum_ratio)
                trj_data = curve.evaluate_multi(s_vals)
                trj_data = trj_data.astype(np.float32)
                start_straight_line_x = 1.875 * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[:-1]
                start_straight_line_y = np.linspace(-18 - sl, -18, sl * meter_pointnum_ratio, dtype=np.float32)[:-1]
                end_straight_line_x = end_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[1:]
                end_straight_line_y = np.linspace(18, 18 + sl, sl * meter_pointnum_ratio, dtype=np.float32)[1:]
                planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x), \
                             np.append(np.append(start_straight_line_y, trj_data[1]), end_straight_line_y)
                xs_1, ys_1 = planed_trj[0][:-1], planed_trj[1][:-1]
                xs_2, ys_2 = planed_trj[0][1:], planed_trj[1][1:]
                phis_1 = np.arctan2(ys_2 - ys_1,
                                    xs_2 - xs_1) * 180 / pi
                planed_trj = xs_1, ys_1, phis_1
                self.path_list.append(planed_trj)
        else:
            assert task == 'right'
            if self.mode == 'training':
                end_offsets = [-3.75, -3.75]
            else:
                end_offsets = [-1.875, -5.625]

            for i, end_offset in enumerate(end_offsets):
                control_point1 = 5.625, -18
                control_point2 = 5.625, -18 + 10
                control_point3 = 18 - 10, end_offset
                control_point4 = 18, end_offset

                node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                                          [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]],
                                         dtype=np.float32)
                curve = bezier.Curve(node, degree=3)
                s_vals = np.linspace(0, 1.0, 13 * meter_pointnum_ratio)
                trj_data = curve.evaluate_multi(s_vals)
                trj_data = trj_data.astype(np.float32)
                start_straight_line_x = 5.625 * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[:-1]
                start_straight_line_y = np.linspace(-18 - sl, -18, sl * meter_pointnum_ratio, dtype=np.float32)[:-1]
                end_straight_line_x = np.linspace(18, 18 + sl, sl * meter_pointnum_ratio, dtype=np.float32)[1:]
                end_straight_line_y = end_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[1:]
                planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x), \
                             np.append(np.append(start_straight_line_y, trj_data[1]), end_straight_line_y)
                xs_1, ys_1 = planed_trj[0][:-1], planed_trj[1][:-1]
                xs_2, ys_2 = planed_trj[0][1:], planed_trj[1][1:]
                phis_1 = np.arctan2(ys_2 - ys_1,
                                    xs_2 - xs_1) * 180 / pi
                if i == 0:
                    phis_1[len(start_straight_line_x):len(start_straight_line_x) + len(s_vals)] = \
                        phis_1[len(start_straight_line_x) + 100:len(start_straight_line_x) + len(s_vals) + 100]
                planed_trj = xs_1, ys_1, phis_1
                self.path_list.append(planed_trj)

    def find_closest_point(self, xs, ys):
        xs_tile = np.tile(np.reshape(xs, (-1, 1)), (1, len(self.path[0])))
        ys_tile = np.tile(np.reshape(ys, (-1, 1)), (1, len(self.path[0])))
        pathx_tile = np.tile(np.reshape(self.path[0], (1, -1)), ([len(xs), 1]))
        pathy_tile = np.tile(np.reshape(self.path[1], (1, -1)), ([len(xs), 1]))

        dist_array = np.square(xs_tile - pathx_tile) + np.square(ys_tile - pathy_tile)

        indexs = np.argmin(dist_array, 1)
        return indexs, self.indexs2points(indexs)

    def future_n_data(self, current_indexs, n):
        future_data_list = []
        current_indexs = current_indexs.astype(np.int32)
        for _ in range(n):
            current_indexs += 80
            current_indexs = np.where(current_indexs >= len(self.path[0]) - 2, len(self.path[0]) - 2, current_indexs)
            future_data_list.append(self.indexs2points(current_indexs))
        return future_data_list

    def indexs2points(self, indexs):
        points = self.path[0][indexs], \
                 self.path[1][indexs], \
                 self.path[2][indexs]

        return points[0], points[1], points[2]

    def tracking_error_vector(self, ego_xs, ego_ys, ego_phis, ego_vs, n):
        indexs, current_points = self.find_closest_point(ego_xs, ego_ys)
        n_future_data = self.future_n_data(indexs, n)
        all_ref = [current_points] + n_future_data

        def two2one(ref_xs, ref_ys):
            if self.task == 'left':
                delta_ = np.sqrt(np.square(ego_xs - (-18)) + np.square(ego_ys - (-18))) - \
                         np.sqrt(np.square(ref_xs - (-18)) + np.square(ref_ys - (-18)))
                delta_ = np.where(ego_ys < -18, ego_xs - ref_xs, delta_)
                delta_ = np.where(ego_xs < -18, ego_ys - ref_ys, delta_)
                return delta_

        tracking_error = np.concatenate([np.stack([two2one(ref_point[0], ref_point[1]),
                                                   deal_with_phi_diff(ego_phis - ref_point[2]),
                                                   ego_vs - self.exp_v], 1)
                                         for ref_point in all_ref], 1)
        return tracking_error


class ModelPredictiveControl:
    def __init__(self, init_x, horizon):
        self.fre = 10
        self.horizon = horizon
        self.init_x = init_x
        self.vehicle_dynamics = VehicleDynamics()
        self.task = 'left'
        self.exp_v = 8
        self.ref_path = None
        self.ego_info_dim = 6
        self.per_veh_info_dim = 4

    def reset_init_x(self, init_x, ref_index):
        self.init_x = init_x
        self.ref_path = ReferencePath('left', ref_index)

    def compute_next_obses(self, obses, actions):
        ego_infos, tracking_infos, veh_infos = obses[:, :self.ego_info_dim], \
                                               obses[:, self.ego_info_dim:self.ego_info_dim + 3], \
                                               obses[:, self.ego_info_dim + 3:]

        next_ego_infos = self.ego_predict(ego_infos, actions)

        next_tracking_infos = self.ref_path.tracking_error_vector(next_ego_infos[:, 3],
                                                                  next_ego_infos[:, 4],
                                                                  next_ego_infos[:, 5],
                                                                  next_ego_infos[:, 0],
                                                                  0)
        next_veh_infos = self.veh_predict(veh_infos)
        next_obses = np.concatenate([next_ego_infos, next_tracking_infos, next_veh_infos], 1)
        return next_obses

    def ego_predict(self, ego_infos, actions):
        ego_next_infos, _ = self.vehicle_dynamics.prediction(ego_infos[:, :6], actions,
                                                             self.fre, 1)

        return ego_next_infos

    def veh_predict(self, veh_infos):
        if self.task == 'left':
            veh_mode_list = ['dl'] * 1 + ['du'] * 1 + ['ud'] * 2 + ['ul'] * 2
        elif self.task == 'straight':
            veh_mode_list = ['dl'] * 2 + ['du'] * 2 + ['ud'] * 2 + ['ru'] * 3 + ['ur'] * 3
        else:
            assert self.task == 'right'
            veh_mode_list = ['dr'] * 2 + ['ur'] * 3 + ['lr'] * 3

        predictions_to_be_concat = []

        for vehs_index in range(len(veh_mode_list)):
            predictions_to_be_concat.append(self.predict_for_a_mode(
                veh_infos[:, vehs_index * self.per_veh_info_dim:(vehs_index + 1) * self.per_veh_info_dim],
                veh_mode_list[vehs_index]))
        return np.concatenate(predictions_to_be_concat, 1)

    def predict_for_a_mode(self, vehs, mode):
        veh_xs, veh_ys, veh_vs, veh_phis = vehs[:, 0], vehs[:, 1], vehs[:, 2], vehs[:, 3]
        veh_phis_rad = veh_phis * np.pi / 180.

        zeros = np.zeros_like(veh_xs)

        veh_xs_delta = veh_vs / self.fre * np.cos(veh_phis_rad)
        veh_ys_delta = veh_vs / self.fre * np.sin(veh_phis_rad)

        if mode in ['dl', 'rd', 'ur', 'lu']:
            veh_phis_rad_delta = np.where(-18 < veh_xs < 18, (veh_vs / 19.875) / self.fre, zeros)
        elif mode in ['dr', 'ru', 'ul', 'ld']:
            veh_phis_rad_delta = np.where(-18 < veh_ys < 18, -(veh_vs / 12.375) / self.fre, zeros)
        else:
            veh_phis_rad_delta = zeros
        next_veh_xs, next_veh_ys, next_veh_vs, next_veh_phis_rad = \
            veh_xs + veh_xs_delta, veh_ys + veh_ys_delta, veh_vs, veh_phis_rad + veh_phis_rad_delta
        next_veh_phis_rad = np.where(next_veh_phis_rad > np.pi, next_veh_phis_rad - 2 * np.pi, next_veh_phis_rad)
        next_veh_phis_rad = np.where(next_veh_phis_rad <= -np.pi, next_veh_phis_rad + 2 * np.pi, next_veh_phis_rad)
        next_veh_phis = next_veh_phis_rad * 180 / np.pi
        return np.stack([next_veh_xs, next_veh_ys, next_veh_vs, next_veh_phis], 1)

    def plant_model(self, u, x):
        x_copy = x.copy()
        x_copy = self.compute_next_obses(x_copy[np.newaxis, :], u[np.newaxis, :])[0]
        return x_copy

    def compute_rew(self, obses, actions):
        ego_infos, tracking_infos, veh_infos = obses[:, :self.ego_info_dim], \
                                               obses[:, self.ego_info_dim:self.ego_info_dim + 3], \
                                               obses[:, self.ego_info_dim + 3:]
        steers, a_xs = actions[:, 0], actions[:, 1]
        # rewards related to action
        punish_steer = -np.square(steers)
        punish_a_x = -np.square(a_xs)

        # rewards related to ego stability
        punish_yaw_rate = -np.square(ego_infos[:, 2])

        # rewards related to tracking error
        devi_y = -np.square(tracking_infos[:, 0])
        devi_phi = -np.square(tracking_infos[:, 1] * np.pi / 180.)
        devi_v = -np.square(tracking_infos[:, 2])

        L, W = 4.8, 2.
        veh2veh = np.zeros_like(veh_infos[:, 0])
        for veh_index in range(int(np.shape(veh_infos)[1] / self.per_veh_info_dim)):
            vehs = veh_infos[:, veh_index * self.per_veh_info_dim:(veh_index + 1) * self.per_veh_info_dim]
            rela_phis_rad = np.arctan2(vehs[:, 1] - ego_infos[:, 4], vehs[:, 0] - ego_infos[:, 3])
            ego_phis_rad = ego_infos[:, 5] * np.pi / 180.
            cos_values, sin_values = np.cos(rela_phis_rad - ego_phis_rad), np.sin(rela_phis_rad - ego_phis_rad)
            dists = np.sqrt(np.square(vehs[:, 0] - ego_infos[:, 3]) + np.square(vehs[:, 1] - ego_infos[:, 4]))
            punish_cond = logical_and(logical_and(dists * cos_values > -5., dists * np.abs(sin_values) < (L + W) / 2),
                                      dists < 10.)
            veh2veh -= np.where(punish_cond, 10. - dists, np.zeros_like(veh_infos[:, 0]))

        rewards = 0.01 * devi_v + 0.04 * devi_y + 0.1 * devi_phi + 0.5 * veh2veh + 0.02 * punish_yaw_rate + \
                  0.1 * punish_steer + 0.005 * punish_a_x
        return rewards

    def cost_function(self, u):
        u = u.reshape(self.horizon, 2)
        loss = 0.
        x = self.init_x.copy()
        for i in range(0, self.horizon):
            u_i = u[i] * np.array([0.4, 3.])
            loss -= self.compute_rew(x[np.newaxis, :], u_i[np.newaxis, :])[0]
            x = self.plant_model(u_i, x)

        return loss


def plot_mpc_rl(file_dir):
    data = np.load(file_dir, allow_pickle=True)
    iteration = np.array([i for i in range(len(data))])
    mpc_steer = np.array([0.4*trunk['mpc_action'][0] for trunk in data])
    mpc_acc = np.array([3*trunk['mpc_action'][1] for trunk in data])
    mpc_time = np.array([trunk['mpc_time'] for trunk in data])
    rl_steer = np.array([0.4 * trunk['rl_action'][0] for trunk in data])
    rl_acc = np.array([3 * trunk['rl_action'][1] for trunk in data])
    rl_time = np.array([trunk['rl_time'] for trunk in data])
    print("mean_mpc_time: {}, mean_rl_time: {}".format(np.mean(mpc_time), np.mean(rl_time)))
    print("var_mpc_time: {}, var_rl_time: {}".format(np.var(mpc_time), np.var(rl_time)))

    df_mpc = pd.DataFrame({'algorithms': 'SLSQP',
                           'iteration': iteration,
                           'steer': mpc_steer,
                           'acc': mpc_acc,
                           'time': mpc_time})
    df_rl = pd.DataFrame({'algorithms': 'CAPO',
                          'iteration': iteration,
                          'steer': rl_steer,
                          'acc': rl_acc,
                          'time': rl_time})
    total_df = df_mpc.append(df_rl, ignore_index=True)
    f1 = plt.figure(1)
    ax1 = f1.add_axes([0.155, 0.12, 0.82, 0.86])
    sns.lineplot(x="iteration", y="steer", hue="algorithms", data=total_df, linewidth=2, palette="bright",)
    # ax1.set_ylabel('Average Q-value Estimation Bias', fontsize=15)
    # ax1.set_xlabel("Million iterations", fontsize=15)
    # plt.xlim(0, 3)
    # plt.ylim(-40, 80)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    f2 = plt.figure(2)
    ax2 = f2.add_axes([0.155, 0.12, 0.82, 0.86])
    sns.lineplot(x="iteration", y="acc", hue="algorithms", data=total_df, linewidth=2, palette="bright", )
    # ax2.set_ylabel('Average Q-value Estimation Bias', fontsize=15)
    # ax2.set_xlabel("Million iterations", fontsize=15)
    # plt.xlim(0, 3)
    # plt.ylim(-40, 80)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    f3 = plt.figure(3)
    ax3 = f3.add_axes([0.155, 0.12, 0.82, 0.86])
    sns.lineplot(x="iteration", y="time", hue="algorithms", data=total_df, linewidth=2, palette="bright", )
    # ax3.set_ylabel('Average Q-value Estimation Bias', fontsize=15)
    # ax3.set_xlabel("Million iterations", fontsize=15)
    # plt.xlim(0, 3)
    # plt.ylim(-40, 80)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.show()


def run_mpc():
    horizon_list = [25]
    done = 0
    mpc_timer, rl_timer = TimerStat(), TimerStat()
    # rl_policy = LoadPolicy('../multi_env/models/left', 94000)
    env = gym.make('CrossroadEnd2end-v0', training_task='left', num_future_data=0)

    for horizon in horizon_list:
        for i in range(1):
            data2plot = []
            obs = env.reset()
            mpc = ModelPredictiveControl(obs, horizon)
            bounds = [(-1., 1.), (-1., 1.)] * horizon
            u_init = np.zeros((horizon, 2))
            mpc.reset_init_x(obs, env.ref_path.ref_index)
            for _ in range(90):
                with mpc_timer:
                    results = minimize(mpc.cost_function,
                                       x0=u_init.flatten(),
                                       method='SLSQP',
                                       bounds=bounds,
                                       tol=1e-1,
                                       options={'disp': True}
                                       )
                mpc_action = results.x
                # with rl_timer:
                #     rl_action = rl_policy.run(obs).numpy()[0]
                # data2plot.append(dict(mpc_action=mpc_action,
                #                       rl_action=rl_action,
                #                       mpc_time=mpc_timer.mean,
                #                       rl_time=rl_timer.mean, ))

                # print(mpc_action)
                # print(results.success, results.message)
                # u_init = np.concatenate([mpc_action[2:], mpc_action[-2:]])
                if not results.success:
                    print('fail')
                    mpc_action = [0., 0.]
                obs, reward, done, info = env.step(mpc_action[:2])
                mpc.reset_init_x(obs, env.ref_path.ref_index)
                env.render()
            # np.save('mpc_rl.npy', np.array(data2plot))


if __name__ == '__main__':
    run_mpc()
    # plot_mpc_rl('mpc_rl_1.npy')


