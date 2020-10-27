#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: dynamics_and_models.py
# =====================================

from math import pi, cos, sin

import bezier
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import logical_and, logical_or

# gym.envs.user_defined.toyota_env.
from endtoend_env_utils import rotate_coordination

L, W = 4.8, 2.


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
        C_f = tf.convert_to_tensor(self.vehicle_params['C_f'], dtype=tf.float32)
        C_r = tf.convert_to_tensor(self.vehicle_params['C_r'], dtype=tf.float32)
        a = tf.convert_to_tensor(self.vehicle_params['a'], dtype=tf.float32)
        b = tf.convert_to_tensor(self.vehicle_params['b'], dtype=tf.float32)
        mass = tf.convert_to_tensor(self.vehicle_params['mass'], dtype=tf.float32)
        I_z = tf.convert_to_tensor(self.vehicle_params['I_z'], dtype=tf.float32)
        miu = tf.convert_to_tensor(self.vehicle_params['miu'], dtype=tf.float32)
        g = tf.convert_to_tensor(self.vehicle_params['g'], dtype=tf.float32)

        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        F_xf = tf.where(a_x < 0, mass * a_x / 2, tf.zeros_like(a_x))
        F_xr = tf.where(a_x < 0, mass * a_x / 2, mass * a_x)
        miu_f = tf.sqrt(tf.square(miu * F_zf) - tf.square(F_xf)) / F_zf
        miu_r = tf.sqrt(tf.square(miu * F_zr) - tf.square(F_xr)) / F_zr
        alpha_f = tf.atan((v_y + a * r) / (v_x+1e-8)) - steer
        alpha_r = tf.atan((v_y - b * r) / (v_x+1e-8))

        next_state = [v_x + tau * (a_x + v_y * r),
                      (mass * v_y * v_x + tau * (
                                  a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * tf.square(
                          v_x) * r) / (mass * v_x - tau * (C_f + C_r)),
                      (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (
                                  tau * (tf.square(a) * C_f + tf.square(b) * C_r) - I_z * v_x),
                      x + tau * (v_x * tf.cos(phi) - v_y * tf.sin(phi)),
                      y + tau * (v_x * tf.sin(phi) + v_y * tf.cos(phi)),
                      (phi + tau * r) * 180 / np.pi]

        return tf.stack(next_state, 1), tf.stack([alpha_f, alpha_r, miu_f, miu_r], 1)

    def prediction(self, x_1, u_1, frequency):
        x_next, next_params = self.f_xu(x_1, u_1, 1 / frequency)
        return x_next, next_params


# class VehicleDynamics(object):
#     def __init__(self, ):
#         self.vehicle_params = OrderedDict(C_f=88000.,  # front wheel cornering stiffness [N/rad]
#                                           C_r=94000.,  # rear wheel cornering stiffness [N/rad]
#                                           a=1.14,  # distance from CG to front axle [m]
#                                           b=1.40,  # distance from CG to rear axle [m]
#                                           mass=1500.,  # mass [kg]
#                                           I_z=2420.,  # Polar moment of inertia at CG [kg*m^2]
#                                           miu=1.0,  # tire-road friction coefficient
#                                           g=9.81,  # acceleration of gravity [m/s^2]
#                                           )
#         a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
#                         self.vehicle_params['mass'], self.vehicle_params['g']
#         F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
#         self.vehicle_params.update(dict(F_zf=F_zf,
#                                         F_zr=F_zr))
#
#     def f_xu(self, states, actions):  # states and actions are tensors, [[], [], ...]
#         with tf.name_scope('f_xu') as scope:
#             # veh_state = obs: v_xs, v_ys, rs, delta_ys, delta_phis, steers, a_xs
#             # 1, 2, 0.2, 2.4, 1, 2, 0.4
#
#             # 0.2 * torch.tensor([1, 5, 10, 12, 5, 10, 2]
#             # vx, vy, r, delta_phi, delta_y
#             # veh_full_state: v_ys, rs, v_xs, phis, ys, xs
#             v_x, v_y, r, x, y, phi = states[:, 0], states[:, 1], states[:, 2], \
#                                      states[:, 3], states[:, 4], states[:, 5]
#             phi = phi * np.pi / 180.
#             steer, a_x = actions[:, 0], actions[:, 1]
#
#             C_f = tf.convert_to_tensor(self.vehicle_params['C_f'], dtype=tf.float32)
#             C_r = tf.convert_to_tensor(self.vehicle_params['C_r'], dtype=tf.float32)
#             a = tf.convert_to_tensor(self.vehicle_params['a'], dtype=tf.float32)
#             b = tf.convert_to_tensor(self.vehicle_params['b'], dtype=tf.float32)
#             mass = tf.convert_to_tensor(self.vehicle_params['mass'], dtype=tf.float32)
#             I_z = tf.convert_to_tensor(self.vehicle_params['I_z'], dtype=tf.float32)
#             miu = tf.convert_to_tensor(self.vehicle_params['miu'], dtype=tf.float32)
#             g = tf.convert_to_tensor(self.vehicle_params['g'], dtype=tf.float32)
#
#             F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
#             F_xf = tf.where(a_x < 0, mass * a_x / 2, tf.zeros_like(a_x, dtype=tf.float32))
#             F_xr = tf.where(a_x < 0, mass * a_x / 2, mass * a_x)
#             miu_f = tf.sqrt(tf.square(miu * F_zf) - tf.square(F_xf)) / F_zf
#             miu_r = tf.sqrt(tf.square(miu * F_zr) - tf.square(F_xr)) / F_zr
#             alpha_f = tf.atan((v_y + a * r) / v_x) - steer
#             alpha_r = tf.atan((v_y - b * r) / v_x)
#
#             Ff_w1 = tf.square(C_f) / (3 * F_zf * miu_f)
#             Ff_w2 = tf.pow(C_f, 3) / (27 * tf.pow(F_zf * miu_f, 2))
#             F_yf_max = F_zf * miu_f
#
#             Fr_w1 = tf.square(C_r) / (3 * F_zr * miu_r)
#             Fr_w2 = tf.pow(C_r, 3) / (27 * tf.pow(F_zr * miu_r, 2))
#             F_yr_max = F_zr * miu_r
#
#             F_yf = - C_f * tf.tan(alpha_f) + Ff_w1 * tf.tan(alpha_f) * tf.abs(
#                 tf.tan(alpha_f)) - Ff_w2 * tf.pow(tf.tan(alpha_f), 3)
#             F_yr = - C_r * tf.tan(alpha_r) + Fr_w1 * tf.tan(alpha_r) * tf.abs(
#                 tf.tan(alpha_r)) - Fr_w2 * tf.pow(tf.tan(alpha_r), 3)
#
#             F_yf = tf.minimum(F_yf, F_yf_max)
#             F_yf = tf.maximum(F_yf, -F_yf_max)
#
#             F_yr = tf.minimum(F_yr, F_yr_max)
#             F_yr = tf.maximum(F_yr, -F_yr_max)
#
#             state_deriv = [a_x + v_y * r,
#                            (F_yf * tf.cos(steer) + F_yr) / mass - v_x * r,
#                            (a * F_yf * tf.cos(steer) - b * F_yr) / I_z,
#                            v_x * tf.cos(phi) - v_y * tf.sin(phi),
#                            v_x * tf.sin(phi) + v_y * tf.cos(phi),
#                            r * 180 / np.pi,
#                            ]
#
#             state_deriv_stack = tf.stack(state_deriv, axis=1)
#             ego_params = tf.stack([alpha_f, alpha_r, miu_f, miu_r], axis=1)
#         return state_deriv_stack, ego_params
#
#     def prediction(self, x_1, u_1, frequency):
#         f_xu_1, params = self.f_xu(x_1, u_1)
#         x_next = f_xu_1 / frequency + x_1
#
#         return x_next, params


class EnvironmentModel(object):  # all tensors
    def __init__(self, task, num_future_data=0):
        self.task = task
        self.vehicle_dynamics = VehicleDynamics()
        self.base_frequency = 10.
        self.obses = None
        self.ego_params = None
        self.actions = None
        self.task = None
        # self.ref_path = None
        self.num_future_data = num_future_data
        self.exp_v = 8.
        self.reward_info = None
        self.ego_info_dim = 6
        self.per_veh_info_dim = 4
        self.per_tracking_info_dim = 3

    def reset(self, obses, task):
        self.obses = obses
        self.actions = None
        self.task = task
        # self.ref_path = ReferencePath(task, mode='training')
        self.reward_info = None

    def rollout_out(self, actions):  # obses and actions are tensors, think of actions are in range [-1, 1]
        with tf.name_scope('model_step') as scope:
            self.actions = self._action_transformation_for_end2end(actions)
            rewards, punish_term = self.compute_rewards(self.obses, self.actions)
            self.obses = self.compute_next_obses(self.obses, self.actions)
            # self.reward_info.update({'final_rew': rewards.numpy()[0]})

        return self.obses, rewards, punish_term

    def _action_transformation_for_end2end(self, actions):  # [-1, 1]
        actions = tf.clip_by_value(actions, -1.05, 1.05)
        steer_norm, a_xs_norm = actions[:, 0], actions[:, 1]
        steer_scale, a_xs_scale = 0.4 * steer_norm, 3. * a_xs_norm-1
        return tf.stack([steer_scale, a_xs_scale], 1)

    def compute_rewards(self, obses, actions):
        with tf.name_scope('compute_reward') as scope:
            ego_infos, tracking_infos, veh_infos = obses[:, :self.ego_info_dim], \
                                                   obses[:,
                                                   self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (
                                                               self.num_future_data + 1)], \
                                                   obses[:, self.ego_info_dim + self.per_tracking_info_dim * (
                                                               self.num_future_data + 1):]
            steers, a_xs = actions[:, 0], actions[:, 1]
            # rewards related to action
            punish_steer = -tf.square(steers)
            punish_a_x = -tf.square(a_xs)

            # rewards related to ego stability
            punish_yaw_rate = -tf.square(ego_infos[:, 2])

            # rewards related to tracking error
            devi_y = -tf.square(tracking_infos[:, 0])
            devi_phi = -tf.cast(tf.square(tracking_infos[:, 1] * np.pi / 180.), dtype=tf.float32)
            devi_v = -tf.square(tracking_infos[:, 2])

            # rewards related to veh2veh collision
            ego_lws = (L - W) / 2.
            ego_front_points = tf.cast(ego_infos[:, 3] + ego_lws * tf.cos(ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32), \
                               tf.cast(ego_infos[:, 4] + ego_lws * tf.sin(ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32)
            ego_rear_points = tf.cast(ego_infos[:, 3] - ego_lws * tf.cos(ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32), \
                              tf.cast(ego_infos[:, 4] - ego_lws * tf.sin(ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32)
            veh2veh = tf.zeros_like(veh_infos[:, 0])
            # for veh_index in range(int(tf.shape(veh_infos)[1] / self.per_veh_info_dim)):
            #     vehs = veh_infos[:, veh_index * self.per_veh_info_dim:(veh_index + 1) * self.per_veh_info_dim]
            #     rela_phis_rad = tf.atan2(vehs[:, 1] - ego_infos[:, 4], vehs[:, 0] - ego_infos[:, 3])
            #     ego_phis_rad = ego_infos[:, 5] * np.pi / 180.
            #     cos_values, sin_values = tf.cos(rela_phis_rad - ego_phis_rad), tf.sin(rela_phis_rad - ego_phis_rad)
            #     dists = tf.sqrt(tf.square(vehs[:, 0] - ego_infos[:, 3]) + tf.square(vehs[:, 1] - ego_infos[:, 4]))
            #     punish_cond = logical_or(logical_and(
            #         logical_and(cos_values > 0., dists * tf.abs(sin_values) < (L + W) / 2),
            #         dists < 7), dists<3.)
            #     veh2veh += tf.where(punish_cond, tf.square(7 - dists), tf.zeros_like(veh_infos[:, 0]))

            for veh_index in range(int(tf.shape(veh_infos)[1] / self.per_veh_info_dim)):
                vehs = veh_infos[:, veh_index * self.per_veh_info_dim:(veh_index + 1) * self.per_veh_info_dim]
                veh_lws = (L - W) / 2.
                veh_front_points = tf.cast(vehs[:, 0] + veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
                                   tf.cast(vehs[:, 1] + veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
                veh_rear_points = tf.cast(vehs[:, 0] - veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
                                  tf.cast(vehs[:, 1] - veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
                for ego_point in [ego_front_points, ego_rear_points]:
                    for veh_point in [veh_front_points, veh_rear_points]:
                        veh2veh_dist = tf.sqrt(
                            tf.square(ego_point[0] - veh_point[0]) + tf.square(ego_point[1] - veh_point[1])) - 3.5
                        veh2veh += tf.where(veh2veh_dist < 0, tf.square(veh2veh_dist), tf.zeros_like(veh_infos[:, 0]))

            veh2road = tf.zeros_like(veh_infos[:, 0])
            if self.task == 'left':
                for ego_point in [ego_front_points, ego_rear_points]:
                    veh2road += tf.where(logical_and(ego_point[1] < -18, ego_point[0] < 1),
                                         tf.square(ego_point[0]-1), tf.zeros_like(veh_infos[:, 0]))
                    veh2road += tf.where(logical_and(ego_point[1] < -18, 3.75-ego_point[0] < 1),
                                         tf.square(3.75-ego_point[0] - 1), tf.zeros_like(veh_infos[:, 0]))
                    veh2road += tf.where(logical_and(ego_point[0] > 0, ego_point[1] > -5),
                                         tf.square(ego_point[1]+5), tf.zeros_like(veh_infos[:, 0]))
                    veh2road += tf.where(logical_and(ego_point[1] > -18, 3.75 - ego_point[0] < 1),
                                         tf.square(3.75 - ego_point[0] - 1), tf.zeros_like(veh_infos[:, 0]))
                    veh2road += tf.where(logical_and(ego_point[0] < 0, 7.5 - ego_point[1] < 1),
                                         tf.square(7.5 - ego_point[1] - 1), tf.zeros_like(veh_infos[:, 0]))
                    veh2road += tf.where(logical_and(ego_point[0] < -18, ego_point[1] - 0 < 1),
                                         tf.square(ego_point[1] - 0 - 1), tf.zeros_like(veh_infos[:, 0]))

            rewards = 0.1 * devi_v + 0.8 * devi_y + 0.8 * devi_phi + 0.02 * punish_yaw_rate + \
                      5 * punish_steer + 0.05 * punish_a_x
            punish_term = veh2veh + veh2road
            # self.reward_info = dict(punish_steer=punish_steer.numpy()[0],
            #                         punish_a_x=punish_a_x.numpy()[0],
            #                         punish_yaw_rate=punish_yaw_rate.numpy()[0],
            #                         devi_v=devi_v.numpy()[0],
            #                         devi_y=devi_y.numpy()[0],
            #                         devi_phi=devi_phi.numpy()[0],
            #                         veh2veh=veh2veh.numpy()[0],
            #                         scaled_punish_steer=5 * punish_steer.numpy()[0],
            #                         scaled_punish_a_x=0.05 * punish_a_x.numpy()[0],
            #                         scaled_punish_yaw_rate=0.02 * punish_yaw_rate.numpy()[0],
            #                         scaled_devi_v=0.01 * devi_v.numpy()[0],
            #                         scaled_devi_y=0.04 * devi_y.numpy()[0],
            #                         scaled_devi_phi=0.1 * devi_phi.numpy()[0],
            #                         scaled_veh2veh=0.5 * veh2veh.numpy()[0],
            #                         reward=rewards.numpy()[0])
            return rewards, punish_term

    def compute_next_obses(self, obses, actions):
        ego_infos, tracking_infos, veh_infos = obses[:, :self.ego_info_dim], obses[:,
                                                                             self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (
                                                                                         self.num_future_data + 1)], \
                                               obses[:, self.ego_info_dim + self.per_tracking_info_dim * (
                                                           self.num_future_data + 1):]

        next_ego_infos = self.ego_predict(ego_infos, actions)

        # next_tracking_infos = self.ref_path.tracking_error_vector(next_ego_infos[:, 3],
        #                                                           next_ego_infos[:, 4],
        #                                                           next_ego_infos[:, 5],
        #                                                           next_ego_infos[:, 0],
        #                                                           self.num_future_data)
        next_tracking_infos = self.tracking_error_predict(ego_infos, tracking_infos, actions)
        next_veh_infos = self.veh_predict(veh_infos)
        next_obses = tf.concat([next_ego_infos, next_tracking_infos, next_veh_infos], 1)
        return next_obses

    def ego_predict(self, ego_infos, actions):
        ego_next_infos, _ = self.vehicle_dynamics.prediction(ego_infos[:, :6], actions, self.base_frequency)
        v_xs, v_ys, rs, xs, ys, phis = ego_next_infos[:, 0], ego_next_infos[:, 1], ego_next_infos[:, 2], \
                                       ego_next_infos[:, 3], ego_next_infos[:, 4], ego_next_infos[:, 5]
        v_xs = tf.clip_by_value(v_xs, 0., 35.)
        ego_next_infos = tf.stack([v_xs, v_ys, rs, xs, ys, phis], axis=1)
        return ego_next_infos

    def tracking_error_predict(self, ego_infos, tracking_infos, actions):
        v_xs, v_ys, rs, xs, ys, phis = ego_infos[:, 0], ego_infos[:, 1], ego_infos[:, 2],\
                                       ego_infos[:, 3], ego_infos[:, 4], ego_infos[:, 5]
        delta_ys, delta_phis, delta_vs = tracking_infos[:, 0], tracking_infos[:, 1], tracking_infos[:, 2]
        rela_obs = tf.stack([v_xs, v_ys, rs, xs, delta_ys, delta_phis], axis=1)
        rela_obs_tp1, _ = self.vehicle_dynamics.prediction(rela_obs, actions, self.base_frequency)
        v_xs_tp1, v_ys_tp1, rs_tp1, xs_tp1, delta_ys_tp1, delta_phis_tp1 = rela_obs_tp1[:, 0], rela_obs_tp1[:, 1], rela_obs_tp1[:, 2], \
                                                                           rela_obs_tp1[:, 3], rela_obs_tp1[:, 4], rela_obs_tp1[:, 5]
        next_tracking_infos = tf.stack([delta_ys_tp1, delta_phis_tp1, v_xs_tp1-self.exp_v], axis=1)
        return next_tracking_infos

    def veh_predict(self, veh_infos):
        if self.task == 'left':
            veh_mode_list = ['dl'] * 1 + ['du'] * 1 + ['ud'] * 2 + ['ul'] * 2
        elif self.task == 'straight':
            veh_mode_list = ['dl'] * 1 + ['du'] * 1 + ['ud'] * 1 + ['ru'] * 2 + ['ur'] * 2
        else:
            assert self.task == 'right'
            veh_mode_list = ['dr'] * 1 + ['ur'] * 2 + ['lr'] * 2

        predictions_to_be_concat = []

        for vehs_index in range(len(veh_mode_list)):
            predictions_to_be_concat.append(self.predict_for_a_mode(
                veh_infos[:, vehs_index * self.per_veh_info_dim:(vehs_index + 1) * self.per_veh_info_dim],
                veh_mode_list[vehs_index]))
        return tf.concat(predictions_to_be_concat, 1)

    def predict_for_a_mode(self, vehs, mode):
        veh_xs, veh_ys, veh_vs, veh_phis = vehs[:, 0], vehs[:, 1], vehs[:, 2], vehs[:, 3]
        veh_phis_rad = veh_phis * np.pi / 180.

        middle_cond = logical_and(logical_and(veh_xs > -18, veh_xs < 18),
                                  logical_and(veh_ys > -18, veh_ys < 18))
        zeros = tf.zeros_like(veh_xs)

        veh_xs_delta = veh_vs / self.base_frequency * tf.cos(veh_phis_rad)
        veh_ys_delta = veh_vs / self.base_frequency * tf.sin(veh_phis_rad)

        if mode in ['dl', 'rd', 'ur', 'lu']:
            veh_phis_rad_delta = tf.where(middle_cond, (veh_vs / 19.875) / self.base_frequency, zeros)
        elif mode in ['dr', 'ru', 'ul', 'ld']:
            veh_phis_rad_delta = tf.where(middle_cond, -(veh_vs / 12.375) / self.base_frequency, zeros)
        else:
            veh_phis_rad_delta = zeros
        next_veh_xs, next_veh_ys, next_veh_vs, next_veh_phis_rad = \
            veh_xs + veh_xs_delta, veh_ys + veh_ys_delta, veh_vs, veh_phis_rad + veh_phis_rad_delta
        next_veh_phis_rad = tf.where(next_veh_phis_rad > np.pi, next_veh_phis_rad - 2 * np.pi, next_veh_phis_rad)
        next_veh_phis_rad = tf.where(next_veh_phis_rad <= -np.pi, next_veh_phis_rad + 2 * np.pi, next_veh_phis_rad)
        next_veh_phis = next_veh_phis_rad * 180 / np.pi
        return tf.stack([next_veh_xs, next_veh_ys, next_veh_vs, next_veh_phis], 1)

    def render(self, mode='human'):
        if mode == 'human':
            # plot basic map
            square_length = 36
            extension = 40
            lane_width = 3.75
            dotted_line_style = '--'

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
            plt.plot([0, 2 * lane_width], [-square_length / 2, -square_length / 2],
                     color='black')
            plt.plot([-2 * lane_width, 0], [square_length / 2, square_length / 2],
                     color='black')
            plt.plot([-square_length / 2, -square_length / 2], [0, -2 * lane_width],
                     color='black')
            plt.plot([square_length / 2, square_length / 2], [2 * lane_width, 0],
                     color='black')

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

            def plot_phi_line(x, y, phi, color):
                line_length = 3
                x_forw, y_forw = x + line_length * cos(phi * pi / 180.), \
                                 y + line_length * sin(phi * pi / 180.)
                plt.plot([x, x_forw], [y, y_forw], color=color, linewidth=0.5)

            obses = self.obses.numpy()
            ego_info, tracing_info, vehs_info = obses[0, :self.ego_info_dim], obses[0,
                                                                              self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (
                                                                                          self.num_future_data + 1)], \
                                                obses[0, self.ego_info_dim + self.per_tracking_info_dim * (
                                                            self.num_future_data + 1):]
            # plot cars
            for veh_index in range(int(len(vehs_info) / self.per_veh_info_dim)):
                veh = vehs_info[self.per_veh_info_dim * veh_index:self.per_veh_info_dim * (veh_index + 1)]
                veh_x, veh_y, veh_v, veh_phi = veh

                if is_in_plot_area(veh_x, veh_y):
                    # plt.text(veh_x, veh_y, '{:.1f}'.format(min([dist1, dist2, dist3, dist4])))
                    plot_phi_line(veh_x, veh_y, veh_phi, 'black')
                    draw_rotate_rec(veh_x, veh_y, veh_phi, L, W, 'black')

            # plot own car
            # ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi, ego_l, ego_w, \
            # ego_alpha_f, ego_alpha_r, ego_miu_f, ego_miu_r,\
            # up1, down1, left1, right1, point11x, point11y, point12x, point12y, \
            # up2, down2, left2, right2, point21x, point21y, point22x, point22y= ego_info
            delta_y, delta_phi = tracing_info[0], tracing_info[1]
            ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi = ego_info

            plot_phi_line(ego_x, ego_y, ego_phi, 'red')
            draw_rotate_rec(ego_x, ego_y, ego_phi, L, W, 'red')

            # plot planed trj
            # ax.plot(self.ref_path.path[0], self.ref_path.path[1], color='g')
            # indexs, points = self.ref_path.find_closest_point(np.array([ego_x], np.float32),
            #                                                   np.array([ego_y], np.float32))
            # path_x, path_y, path_phi = points[0][0], points[1][0], points[2][0]
            # delta_x, delta_y, delta_phi, delta_v = tracing_info[:4]
            # # delta_x, delta_y, delta_phi = ego_x - path_x, ego_y - path_y, ego_phi - path_phi
            # plt.plot(path_x, path_y, 'go')
            # plot_phi_line(path_x, path_y, path_phi, 'g')

            # plot text
            text_x, text_y_start = -110, 60
            ge = iter(range(0, 1000, 4))
            plt.text(text_x, text_y_start - next(ge), 'ego_x: {:.2f}m'.format(ego_x))
            plt.text(text_x, text_y_start - next(ge), 'ego_y: {:.2f}m'.format(ego_y))
            # plt.text(text_x, text_y_start - next(ge), 'UDLR: {:.2f} {:.2f} {:.2f} {:.2f}'.format(min([up1, up2]),
            #                                                                                          min([down1, down2]),
            #                                                                                          min([left1, left2]),
            #                                                                                          min([right1, right2])))
            # plt.text(text_x, text_y_start - next(ge), '1deltas {:.2f} {:.2f}'.format(point11x, point11y))
            # plt.text(text_x, text_y_start - next(ge), '2deltas {:.2f} {:.2f}'.format(point12x, point12y))
            # plt.text(text_x, text_y_start - next(ge), 'path_x: {:.2f}m'.format(path_x))
            # plt.text(text_x, text_y_start - next(ge), 'path_y: {:.2f}m'.format(path_y))
            # plt.text(text_x, text_y_start - next(ge), 'delta_x: {:.2f}m'.format(delta_x))
            plt.text(text_x, text_y_start - next(ge), 'delta_y: {:.2f}m'.format(delta_y))
            plt.text(text_x, text_y_start - next(ge), r'ego_phi: ${:.2f}\degree$'.format(ego_phi))
            # plt.text(text_x, text_y_start - next(ge), r'path_phi: ${:.2f}\degree$'.format(path_phi))
            plt.text(text_x, text_y_start - next(ge), r'delta_phi: ${:.2f}\degree$'.format(delta_phi))

            plt.text(text_x, text_y_start - next(ge), 'v_x: {:.2f}m/s'.format(ego_v_x))
            plt.text(text_x, text_y_start - next(ge), 'exp_v: {:.2f}m/s'.format(self.exp_v))
            plt.text(text_x, text_y_start - next(ge), 'v_y: {:.2f}m/s'.format(ego_v_y))
            plt.text(text_x, text_y_start - next(ge), 'yaw_rate: {:.2f}rad/s'.format(ego_r))

            if self.actions is not None:
                steer, a_x = self.actions[0, 0], self.actions[0, 1]
                plt.text(text_x, text_y_start - next(ge),
                         r'steer: {:.2f}rad (${:.2f}\degree$)'.format(steer, steer * 180 / np.pi))
                plt.text(text_x, text_y_start - next(ge), 'a_x: {:.2f}m/s^2'.format(a_x))

            text_x, text_y_start = 70, 60
            ge = iter(range(0, 1000, 4))

            # reward info
            if self.reward_info is not None:
                for key, val in self.reward_info.items():
                    plt.text(text_x, text_y_start - next(ge), '{}: {:.4f}'.format(key, val))

            plt.show()
            plt.pause(0.1)


def deal_with_phi_diff(phi_diff):
    phi_diff = tf.where(phi_diff > 180., phi_diff - 360., phi_diff)
    phi_diff = tf.where(phi_diff < -180., phi_diff + 360., phi_diff)
    return phi_diff


class ReferencePath(object):
    def __init__(self, task, mode=None):
        self.mode = mode
        self.exp_v = 8.
        self.task = task
        self.path_list = []
        self._construct_ref_path(self.task)
        self.ref_index = np.random.choice([0, 1])
        self.path = self.path_list[self.ref_index]

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
                planed_trj = xs_1, ys_1, phis_1
                self.path_list.append(planed_trj)

    def find_closest_point(self, xs, ys):
        xs_tile = tf.tile(tf.reshape(xs, (-1, 1)), tf.constant([1, len(self.path[0])]))
        ys_tile = tf.tile(tf.reshape(ys, (-1, 1)), tf.constant([1, len(self.path[0])]))
        pathx_tile = tf.tile(tf.reshape(self.path[0], (1, -1)), tf.constant([len(xs), 1]))
        pathy_tile = tf.tile(tf.reshape(self.path[1], (1, -1)), tf.constant([len(xs), 1]))

        dist_array = tf.square(xs_tile - pathx_tile) + tf.square(ys_tile - pathy_tile)

        indexs = tf.argmin(dist_array, 1)
        return indexs, self.indexs2points(indexs)

    def future_n_data(self, current_indexs, n):
        future_data_list = []
        current_indexs = tf.cast(current_indexs, tf.int32)
        for _ in range(n):
            current_indexs += 80
            current_indexs = tf.where(current_indexs >= len(self.path[0]) - 2, len(self.path[0]) - 2, current_indexs)
            future_data_list.append(self.indexs2points(current_indexs))
        return future_data_list

    def indexs2points(self, indexs):
        tf.assert_equal(tf.reduce_all(indexs < len(self.path[0])), tf.constant(True, dtype=tf.bool))
        points = tf.gather(self.path[0], indexs), \
                 tf.gather(self.path[1], indexs), \
                 tf.gather(self.path[2], indexs)

        return points[0], points[1], points[2]

    def tracking_error_vector(self, ego_xs, ego_ys, ego_phis, ego_vs, n):
        indexs, current_points = self.find_closest_point(ego_xs, ego_ys)
        tracking_points = self.indexs2points(indexs+80)
        n_future_data = self.future_n_data(indexs+80, n)
        all_ref = [(current_points[0], current_points[1], tracking_points[2])] + n_future_data

        def two2one(ref_xs, ref_ys):
            if self.task == 'left':
                delta_ = tf.sqrt(tf.square(ego_xs - (-18)) + tf.square(ego_ys - (-18))) - \
                         tf.sqrt(tf.square(ref_xs - (-18)) + tf.square(ref_ys - (-18)))
                delta_ = tf.where(ego_ys < -18, ego_xs - ref_xs, delta_)
                delta_ = tf.where(ego_xs < -18, ego_ys - ref_ys, delta_)
                return -delta_
            elif self.task == 'straight':
                delta_ = ego_xs - ref_xs
                return -delta_
            else:
                assert self.task == 'right'
                delta_ = -(tf.sqrt(tf.square(ego_xs - (18)) + tf.square(ego_ys - (-18))) -
                           tf.sqrt(tf.square(ref_xs - (18)) + tf.square(ref_ys - (-18))))
                delta_ = tf.where(ego_ys < -18, ego_xs - ref_xs, delta_)
                delta_ = tf.where(ego_xs > 18, -(ego_ys - ref_ys), delta_)
                return -delta_

        tracking_error = tf.concat([tf.stack([two2one(ref_point[0], ref_point[1]),
                                              deal_with_phi_diff(ego_phis - ref_point[2]),
                                              ego_vs - self.exp_v], 1)
                                    for ref_point in all_ref], 1)
        return tracking_error

    def plot_path(self, x, y):
        plt.axis('equal')
        plt.plot(self.path_list[0][0], self.path_list[0][1])
        plt.plot(self.path_list[1][0], self.path_list[1][1], 'r')
        index, closest_point = self.find_closest_point(np.array([x], np.float32),
                                                       np.array([y], np.float32))
        plt.plot(x, y, 'b*')
        plt.plot(closest_point[0], closest_point[1], 'ro')
        plt.show()


def test_ref_path():
    path = ReferencePath('left')
    path.plot_path(1.875, 0)


def test_future_n_data():
    path = ReferencePath('straight')
    plt.axis('equal')
    current_i = 600
    plt.plot(path.path[0], path.path[1])
    future_data_list = path.future_n_data(current_i, 5)
    plt.plot(path.indexs2points(current_i)[0], path.indexs2points(current_i)[1], 'go')
    for point in future_data_list:
        plt.plot(point[0], point[1], 'r*')
    plt.show()


def test_tracking_error_vector():
    path = ReferencePath('straight')
    xs = np.array([1.875, 1.875, -10, -20], np.float32)
    ys = np.array([-20, 0, -10, -1], np.float32)
    phis = np.array([90, 135, 135, 180], np.float32)
    vs = np.array([10, 12, 10, 10], np.float32)

    tracking_error_vector = path.tracking_error_vector(xs, ys, phis, vs, 0)
    print(tracking_error_vector)


def test_model():
    from endtoend import CrossroadEnd2end
    env = CrossroadEnd2end('left', 0)
    model = EnvironmentModel('left', 0)
    obs_list = []
    obs = env.reset()
    done = 0
    # while not done:
    for i in range(10):
        obs_list.append(obs)
        action = np.array([0, -1], dtype=np.float32)
        obs, reward, done, info = env.step(action)
        env.render()
    obses = np.stack(obs_list, 0)
    model.reset(obses, 'left')
    print(obses.shape)
    for rollout_step in range(100):
        actions = tf.tile(tf.constant([[0.5, 0]], dtype=tf.float32), tf.constant([len(obses), 1]))
        obses, rewards, punish = model.rollout_out(actions)
        print(rewards.numpy()[0], punish.numpy()[0])
        model.render()


if __name__ == '__main__':
    test_model()
