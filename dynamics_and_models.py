#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: dynamics_and_models.py
# =====================================

from collections import OrderedDict
from math import pi, cos, sin

import bezier
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.math import logical_and, logical_or

# gym.envs.user_defined.toyota_env.
from endtoend_env_utils import rotate_coordination


class VehicleDynamics(object):
    def __init__(self, ):
        self.vehicle_params = OrderedDict(C_f=88000.,  # front wheel cornering stiffness [N/rad]
                                          C_r=94000.,  # rear wheel cornering stiffness [N/rad]
                                          a=1.14,  # distance from CG to front axle [m]
                                          b=1.40,  # distance from CG to rear axle [m]
                                          mass=1500.,  # mass [kg]
                                          I_z=2420.,  # Polar moment of inertia at CG [kg*m^2]
                                          miu=1.0,  # tire-road friction coefficient
                                          g=9.81,  # acceleration of gravity [m/s^2]
                                          )
        a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
                        self.vehicle_params['mass'], self.vehicle_params['g']
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))

    def f_xu(self, states, actions):  # states and actions are tensors, [[], [], ...]
        with tf.name_scope('f_xu') as scope:
            # veh_state = obs: v_xs, v_ys, rs, delta_ys, delta_phis, steers, a_xs
            # 1, 2, 0.2, 2.4, 1, 2, 0.4

            # 0.2 * torch.tensor([1, 5, 10, 12, 5, 10, 2]
            # vx, vy, r, delta_phi, delta_y
            # veh_full_state: v_ys, rs, v_xs, phis, ys, xs
            v_x, v_y, r, x, y, phi = states[:, 0], states[:, 1], states[:, 2], \
                                     states[:, 3], states[:, 4], states[:, 5]
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
            F_xf = tf.where(a_x < 0, mass * a_x / 2, tf.zeros_like(a_x, dtype=tf.float32))
            F_xr = tf.where(a_x < 0, mass * a_x / 2, mass * a_x)
            miu_f = tf.sqrt(tf.square(miu * F_zf) - tf.square(F_xf)) / F_zf
            miu_r = tf.sqrt(tf.square(miu * F_zr) - tf.square(F_xr)) / F_zr
            alpha_f = tf.atan((v_y + a * r) / v_x) - steer
            alpha_r = tf.atan((v_y - b * r) / v_x)

            Ff_w1 = tf.square(C_f) / (3 * F_zf * miu_f)
            Ff_w2 = tf.pow(C_f, 3) / (27 * tf.pow(F_zf * miu_f, 2))
            F_yf_max = F_zf * miu_f

            Fr_w1 = tf.square(C_r) / (3 * F_zr * miu_r)
            Fr_w2 = tf.pow(C_r, 3) / (27 * tf.pow(F_zr * miu_r, 2))
            F_yr_max = F_zr * miu_r

            F_yf = - C_f * tf.tan(alpha_f) + Ff_w1 * tf.tan(alpha_f) * tf.abs(
                tf.tan(alpha_f)) - Ff_w2 * tf.pow(tf.tan(alpha_f), 3)
            F_yr = - C_r * tf.tan(alpha_r) + Fr_w1 * tf.tan(alpha_r) * tf.abs(
                tf.tan(alpha_r)) - Fr_w2 * tf.pow(tf.tan(alpha_r), 3)

            F_yf = tf.minimum(F_yf, F_yf_max)
            F_yf = tf.maximum(F_yf, -F_yf_max)

            F_yr = tf.minimum(F_yr, F_yr_max)
            F_yr = tf.maximum(F_yr, -F_yr_max)

            # tmp_f = tf.square(C_f * tf.tan(alpha_f)) / (27 * tf.square(miu_f * F_zf)) - C_f * tf.abs(tf.tan(alpha_f)) / (
            #         3 * miu_f * F_zf) + 1
            # tmp_r = tf.square(C_r * tf.tan(alpha_r)) / (27 * tf.square(miu_r * F_zr)) - C_r * tf.abs(tf.tan(alpha_r)) / (
            #         3 * miu_r * F_zr) + 1
            #
            # F_yf = -tf.sign(alpha_f) * tf.minimum(tf.abs(C_f * tf.tan(alpha_f) * tmp_f), tf.abs(miu_f * F_zf))
            # F_yr = -tf.sign(alpha_r) * tf.minimum(tf.abs(C_r * tf.tan(alpha_r) * tmp_r), tf.abs(miu_r * F_zr))

            state_deriv = [a_x + v_y * r,
                           (F_yf * tf.cos(steer) + F_yr) / mass - v_x * r,
                           (a * F_yf * tf.cos(steer) - b * F_yr) / I_z,
                           v_x * tf.cos(phi) - v_y * tf.sin(phi),
                           v_x * tf.sin(phi) + v_y * tf.cos(phi),
                           r * 180 / np.pi,
                           ]

            state_deriv_stack = tf.stack(state_deriv, axis=1)
            ego_params = tf.stack([alpha_f, alpha_r, miu_f, miu_r], axis=1)
        return state_deriv_stack, ego_params

    def prediction(self, x_1, u_1, frequency, RK):
        if RK == 1:
            f_xu_1, params = self.f_xu(x_1, u_1)
            x_next = f_xu_1 / frequency + x_1

        elif RK == 2:
            f_xu_1, params = self.f_xu(x_1, u_1)
            K1 = (1 / frequency) * f_xu_1
            x_2 = x_1 + K1
            f_xu_2, _ = self.f_xu(x_2, u_1)
            K2 = (1 / frequency) * f_xu_2
            x_next = x_1 + (K1 + K2) / 2
        else:
            assert RK == 4
            f_xu_1, params = self.f_xu(x_1, u_1)
            K1 = (1 / frequency) * f_xu_1
            x_2 = x_1 + K1 / 2
            f_xu_2, _ = self.f_xu(x_2, u_1)
            K2 = (1 / frequency) * f_xu_2
            x_3 = x_1 + K2 / 2
            f_xu_3, _ = self.f_xu(x_3, u_1)
            K3 = (1 / frequency) * f_xu_3
            x_4 = x_1 + K3
            f_xu_4, _ = self.f_xu(x_4, u_1)
            K4 = (1 / frequency) * f_xu_4
            x_next = x_1 + (K1 + 2 * K2 + 2 * K3 + K4) / 6
        return x_next, params


class EnvironmentModel(object):  # all tensors
    def __init__(self, task, num_future_data=5):
        self.task = task
        self.vehicle_dynamics = VehicleDynamics()
        self.base_frequency = 10.
        self.obses = None
        self.ego_params = None
        self.actions = None
        self.dones = None
        self.dones_type = None
        self.task = None
        self.ref_path = None
        self.num_future_data = num_future_data
        self.exp_v = 8.
        self.alpha_f_bounds = None
        self.alpha_r_bounds = None
        self.r_bounds = None
        self.reward_info = None
        self.ego_info_dim = 12
        self.per_veh_info_dim = 6

    def reset(self, obses, task):
        self.obses = obses
        self.actions = None
        self.dones = tf.cast(tf.zeros_like(self.obses[:, 0]), tf.bool)
        self.dones_type = tf.constant(['not_done_yet']*len(self.obses[:, 0]), tf.string)
        self.task = task
        self.ref_path = ReferencePath(task, mode='training')
        self._compute_bounds(obses)
        self.reward_info = None

    def rollout_out(self, actions):  # obses and actions are tensors, think of actions are in range [-1, 1]
        with tf.name_scope('model_step') as scope:
            prev_dones = self.dones
            self.actions = self._action_transformation_for_end2end2(actions)
            rewards = self.compute_rewards3(self.obses, self.actions, prev_dones)
            self.obses = self.compute_next_obses(self.obses, self.actions)
            self.dones, self.dones_type = self.judge_dones(self.obses)
            dones_reward = self.compute_dones_reward(self.dones_type, self.dones, prev_dones)
            rewards += dones_reward
            # self.reward_info.update({'done_rew':dones_reward.numpy()[0],
            #                          'final_rew': rewards.numpy()[0]})

        return self.obses, rewards, self.dones

    def compute_dones_reward(self, dones_type, dones, prev_dones):
        dones_reward = tf.zeros_like(dones, dtype=tf.float32)
        dones_reward = tf.where(dones_type=='collision', -20., dones_reward)
        dones_reward = tf.where(dones_type=='break_road_constrain', -20., dones_reward)
        dones_reward = tf.where(dones_type=='deviation_too_much', -20., dones_reward)
        dones_reward = tf.where(dones_type=='break_stability', -20., dones_reward)
        dones_reward = tf.where(dones_type=='good_done', 50., dones_reward)
        this_step_dones = tf.cast(dones, tf.float32) - tf.cast(prev_dones, tf.float32)
        dones_reward = dones_reward * this_step_dones
        return dones_reward

    # def _action_transformation_for_end2end(self, actions):  # [-1, 1]
    #     # steer_norm, a_xs_norm = actions[:, 0], actions[:, 1]
    #     # steer_scale, a_xs_scale = 0.2 * steer_norm, 3. * a_xs_norm
    #     # ego_v_xs = self.obses[:, 0]
    #     # acc_lower_bound = tf.maximum(-3.*tf.ones_like(a_xs_scale), -ego_v_xs/3.)
    #     # a_xs_scale = tf.clip_by_value(a_xs_scale, acc_lower_bound, 5.*tf.ones_like(a_xs_scale))
    #     # return tf.stack([steer_scale, a_xs_scale], 1)
    #     steers_norm, a_xs_norm = actions[:, 0], actions[:, 1]
    #     # steer_scale = tf.where(self.obses[:, 4]<-18., 0., 0.2 * steers_norm)
    #     steer_scale = 0.2 * steers_norm
    #     ego_v_xs = self.obses[:, 0]
    #     acc_lower_bounds = tf.maximum(-3., -ego_v_xs/3.)
    #     acc_upper_bounds = tf.maximum(1., tf.minimum(3., -2*ego_v_xs + 21.))
    #
    #     a_xs_scale = (a_xs_norm + 1.) / 2. * (acc_upper_bounds - acc_lower_bounds) + acc_lower_bounds
    #     return tf.stack([steer_scale, a_xs_scale], 1)

    def _action_transformation_for_end2end2(self, actions):  # [-1, 1]
        steer_norm, a_xs_norm = actions[:, 0], actions[:, 1]
        steer_scale, a_xs_scale = 0.2 * steer_norm, 3. * a_xs_norm
        return tf.stack([steer_scale, a_xs_scale], 1)

    def _compute_bounds(self, obses):
        F_zf, F_zr = self.vehicle_dynamics.vehicle_params['F_zf'], self.vehicle_dynamics.vehicle_params['F_zr']
        C_f, C_r = self.vehicle_dynamics.vehicle_params['C_f'], self.vehicle_dynamics.vehicle_params['C_r']
        miu_fs, miu_rs = obses[:, 10], obses[:, 11]
        self.alpha_f_bounds, self.alpha_r_bounds = 3 * miu_fs * F_zf / C_f, 3 * miu_rs * F_zr / C_r
        self.r_bounds = miu_rs * self.vehicle_dynamics.vehicle_params['g'] / tf.abs(obses[:, 0])

    def judge_dones(self, obses):
        ego_infos, tracking_infos, veh_infos = obses[:, :self.ego_info_dim], obses[:, self.ego_info_dim:self.ego_info_dim + 4 * (self.num_future_data+1)], \
                                               obses[:, self.ego_info_dim + 4 * (self.num_future_data+1):]
        # dones related to ego stability
        self._compute_bounds(obses)
        alpha_fs, alpha_rs = ego_infos[:, 8], ego_infos[:, 9]
        dones_alpha_f = logical_or(alpha_fs > self.alpha_f_bounds, alpha_fs < -self.alpha_f_bounds)
        dones_alpha_r = logical_or(alpha_rs > self.alpha_r_bounds, alpha_rs < -self.alpha_r_bounds)
        dones_r = logical_or(ego_infos[:, 2] > self.r_bounds, ego_infos[:, 2] < -self.r_bounds)
        stability_dones = logical_or(logical_or(dones_alpha_f, dones_alpha_r), dones_r)
        self.dones_type = tf.where(stability_dones, 'break_stability', self.dones_type)
        self.dones = tf.math.logical_or(self.dones, stability_dones)

        # dones related to deviation
        delta_xs, delta_ys, delta_phis = tracking_infos[:, 0], tracking_infos[:, 1], tracking_infos[:, 2]
        dists = tf.sqrt(tf.square(delta_xs)+tf.square(delta_ys))
        deviation_dones = logical_or(dists>10., tf.abs(delta_phis)>30.)
        self.dones_type = tf.where(deviation_dones, 'deviation_too_much', self.dones_type)
        self.dones = tf.math.logical_or(self.dones, deviation_dones)

        # dones related to veh2road collision
        ego_lws = (ego_infos[:, 6] - ego_infos[:, 7]) / 2.
        ego_front_points = tf.cast(ego_infos[:, 3] + ego_lws * tf.cos(ego_infos[:, 5] * np.pi / 180.),
                                   dtype=tf.float32), \
                           tf.cast(ego_infos[:, 4] + ego_lws * tf.sin(ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32)
        ego_rear_points = tf.cast(ego_infos[:, 3] - ego_lws * tf.cos(ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32), \
                          tf.cast(ego_infos[:, 4] - ego_lws * tf.sin(ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32)
        coeff = 1.14
        rho_ego = ego_infos[0, 7] / 2 * coeff
        veh2road_dones = tf.cast(tf.zeros_like(obses[:, 0]), tf.bool)
        # dones_up = obses[:, 12] < 0
        # dones_down = obses[:, 13] < 0
        # dones_left = obses[:, 14] < 0
        # dones_right = obses[:, 15] < 0
        # dones_front_corner1 = tf.sqrt(tf.square(obses[:, 16])+tf.square(obses[:, 17])) - rho_ego < 0
        # dones_front_corner2 = tf.sqrt(tf.square(obses[:, 18])+tf.square(obses[:, 19])) - rho_ego < 0
        # dones_rear_corner1 = tf.sqrt(tf.square(obses[:, 20]) + tf.square(obses[:, 21])) - rho_ego < 0
        # dones_rear_corner2 = tf.sqrt(tf.square(obses[:, 22]) + tf.square(obses[:, 23])) - rho_ego < 0
        #
        # for dones in [dones_up, dones_down, dones_left, dones_right,
        #               dones_front_corner1, dones_front_corner2, dones_rear_corner1, dones_rear_corner2]:
        #     veh2road_dones = logical_or(veh2road_dones, dones)
        #
        # if self.task == 'left':
        #     dones_good_done = logical_and(logical_and(ego_infos[:, 4] > 0, ego_infos[:, 4] < 7.5),
        #                                   ego_infos[:, 3] < -18 - 5)
        # elif self.task == 'straight':
        #     dones_good_done = logical_and(logical_and(ego_infos[:, 3] > 0, ego_infos[:, 3] < 7.5),
        #                                   ego_infos[:, 4] > 18 + 5)
        # else:
        #     assert self.task == 'right'
        #     dones_good_done = logical_and(logical_and(ego_infos[:, 4] < 0, ego_infos[:, 4] < -7.5),
        #                                   ego_infos[:, 3] > 18 + 5)
        # self.dones_type = tf.where(dones_good_done, 'good_done', self.dones_type)
        # self.dones = tf.math.logical_or(self.dones, dones_good_done)

        if self.task == 'left':
            dones_good_done = logical_and(logical_and(ego_infos[:, 4] > 0, ego_infos[:, 4] < 7.5),
                                          ego_infos[:, 3] < -18 - 2)
            self.dones_type = tf.where(dones_good_done, 'good_done', self.dones_type)
            self.dones = tf.math.logical_or(self.dones, dones_good_done)

            for ego_point in [ego_front_points, ego_rear_points]:
                dones_before1 = logical_and(ego_point[1] < -18, ego_point[0] - 0 < rho_ego)
                dones_before2 = logical_and(ego_point[1] < -18, 3.75 - ego_point[0] < rho_ego)

                middle_cond = logical_and(logical_and(ego_point[0] > -18, ego_point[0] < 18),
                                          logical_and(ego_point[1] > -18, ego_point[1] < 18))
                dones_middle1 = logical_and(middle_cond, 7.5 - ego_point[1] < rho_ego)
                dones_middle2 = logical_and(middle_cond, 7.5 - ego_point[0] < rho_ego)
                dones_middle3 = logical_and(logical_and(middle_cond, ego_point[1] > 7.5),
                                            ego_point[0] - (-18) < rho_ego)
                dones_middle4 = logical_and(logical_and(middle_cond, ego_point[1] < 0),
                                            ego_point[0] - (-18) < rho_ego)
                dones_middle5 = logical_and(logical_and(middle_cond, ego_point[0] < 0),
                                            ego_point[1] - (-18) < rho_ego)
                dones_middle6 = logical_and(logical_and(middle_cond, ego_point[0] > 3.75),
                                            ego_point[1] - (-18) < rho_ego)

                dones_middle7 = logical_and(middle_cond, tf.sqrt(tf.square(ego_point[0] - (-18)) + tf.square(
                        ego_point[1] - 0)) < rho_ego)
                dones_middle8 = logical_and(middle_cond, tf.sqrt(tf.square(ego_point[0] - (-18)) + tf.square(
                        ego_point[1] - 7.5)) < rho_ego)

                dones_after1 = logical_and(ego_point[0] < -18, ego_point[1] - 0 < rho_ego)
                dones_after2 = logical_and(ego_point[0] < -18, 7.5 - ego_point[1] < rho_ego)

                for dones in [dones_before1, dones_before2, dones_middle1, dones_middle2, dones_middle3, dones_middle4,
                              dones_middle5, dones_middle6, dones_middle7, dones_middle8,
                              dones_after1, dones_after2]:
                    veh2road_dones = logical_or(veh2road_dones, dones)

        elif self.task == 'straight':
            dones_good_done = logical_and(logical_and(ego_infos[:, 3] > 0, ego_infos[:, 3] < 7.5),
                                          ego_infos[:, 4] > 18 + 2)
            self.dones_type = tf.where(dones_good_done, 'good_done', self.dones_type)
            self.dones = tf.math.logical_or(self.dones, dones_good_done)

            for ego_point in [ego_front_points, ego_rear_points]:
                dones_before1 = logical_and(ego_point[1] < -18, ego_point[0] - 0 < rho_ego)
                dones_before2 = logical_and(ego_point[1] < -18, 3.75 - ego_point[0] < rho_ego)

                middle_cond = logical_and(logical_and(ego_point[0] > -18, ego_point[0] < 18),
                                          logical_and(ego_point[1] > -18, ego_point[1] < 18))
                dones_middle1 = logical_and(middle_cond, ego_point[0] - (-18) < rho_ego)
                dones_middle2 = logical_and(middle_cond, 18 - ego_point[0] < rho_ego)
                dones_middle3 = logical_and(logical_and(middle_cond, ego_point[0] < 0),
                                            18 - ego_point[1] < rho_ego)
                dones_middle4 = logical_and(logical_and(middle_cond, ego_point[0] > 7.5),
                                            18 - ego_point[1] < rho_ego)
                dones_middle5 = logical_and(logical_and(middle_cond, ego_point[0] < 0),
                                            ego_point[1] - (-18) < rho_ego)
                dones_middle6 = logical_and(logical_and(middle_cond, ego_point[0] > 3.75),
                                            ego_point[1] - (-18) < rho_ego)

                dones_middle7 = logical_and(middle_cond, tf.sqrt(tf.square(ego_point[0] - 0) + tf.square(
                    ego_point[1] - 18)) < rho_ego)
                dones_middle8 = logical_and(middle_cond, tf.sqrt(tf.square(ego_point[0] - 7.5) + tf.square(
                    ego_point[1] - 18)) < rho_ego)

                dones_after1 = logical_and(ego_point[1] > 18, ego_point[0] - 0 < rho_ego)
                dones_after2 = logical_and(ego_point[1] > 18, 7.5 - ego_point[0] < rho_ego)

                for dones in [dones_before1, dones_before2, dones_middle1, dones_middle2, dones_middle3, dones_middle4,
                              dones_middle5, dones_middle6, dones_middle7, dones_middle8,
                              dones_after1, dones_after2]:
                    veh2road_dones = logical_or(veh2road_dones, dones)

        else:
            assert self.task == 'right'
            dones_good_done = logical_and(logical_and(ego_infos[:, 4] < 0, ego_infos[:, 4] < -7.5),
                                          ego_infos[:, 3] > 18 + 2)
            self.dones_type = tf.where(dones_good_done, 'good_done', self.dones_type)
            self.dones = tf.math.logical_or(self.dones, dones_good_done)

            for ego_point in [ego_front_points, ego_rear_points]:
                dones_before1 = logical_and(ego_point[1] < -18, ego_point[0] - 3.75 < rho_ego)
                dones_before2 = logical_and(ego_point[1] < -18, 7.5 - ego_point[0] < rho_ego)

                middle_cond = logical_and(logical_and(ego_point[0] > -18, ego_point[0] < 18),
                                          logical_and(ego_point[1] > -18, ego_point[1] < 18))
                dones_middle1 = logical_and(middle_cond, ego_point[0] - (-18) < rho_ego)
                dones_middle2 = logical_and(middle_cond, 18 - ego_point[1] < rho_ego)
                dones_middle3 = logical_and(logical_and(middle_cond, ego_point[1] > 0),
                                            18 - ego_point[0] < rho_ego)
                dones_middle4 = logical_and(logical_and(middle_cond, ego_point[1] < -7.5),
                                            18 - ego_point[0] < rho_ego)
                dones_middle5 = logical_and(logical_and(middle_cond, ego_point[0] > 7.5),
                                            ego_point[1] - (-18) < rho_ego)
                dones_middle6 = logical_and(logical_and(middle_cond, ego_point[0] < 3.75),
                                            ego_point[1] - (-18) < rho_ego)
                dones_middle7 = logical_and(middle_cond, tf.sqrt(tf.square(ego_point[0] - 18) + tf.square(
                    ego_point[1] - 0)) < rho_ego)
                dones_middle8 = logical_and(middle_cond, tf.sqrt(tf.square(ego_point[0] - 18) + tf.square(
                    ego_point[1] - (-7.5))) < rho_ego)

                dones_after1 = logical_and(ego_point[0] > 18, 0 - ego_point[1] < rho_ego)
                dones_after2 = logical_and(ego_point[0] > 18, ego_point[1] - (-7.5) < rho_ego)

                for dones in [dones_before1, dones_before2, dones_middle1, dones_middle2, dones_middle3, dones_middle4,
                              dones_middle5, dones_middle6, dones_middle7, dones_middle8,
                              dones_after1, dones_after2]:
                    veh2road_dones = logical_or(veh2road_dones, dones)

        self.dones_type = tf.where(veh2road_dones, 'break_road_constrain', self.dones_type)
        self.dones = logical_or(self.dones, veh2road_dones)

        # dones related to veh2veh collision
        veh2veh_dones = tf.cast(tf.zeros_like(obses[:, 0]), tf.bool)
        for veh_index in range(int(tf.shape(veh_infos)[1] / self.per_veh_info_dim)):
            vehs = veh_infos[:, veh_index * self.per_veh_info_dim: (veh_index + 1)*self.per_veh_info_dim]
            # for i in [6, 7, 8, 9]:
            #     dones = vehs[:, i] < 0
            #     veh2veh_dones = logical_or(veh2veh_dones, dones)
            veh_lws = (vehs[:, 4] - vehs[:, 5]) / 2.
            rho_vehs = vehs[:, 5] / 2. * coeff
            veh_front_points = tf.cast(vehs[:, 0] + veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
                               tf.cast(vehs[:, 1] + veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
            veh_rear_points = tf.cast(vehs[:, 0] - veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
                              tf.cast(vehs[:, 1] - veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
            for ego_point in [ego_front_points, ego_rear_points]:
                for veh_point in [veh_front_points, veh_rear_points]:
                    veh2veh_square_dist = tf.square(ego_point[0] - veh_point[0]) + tf.square(
                        ego_point[1] - veh_point[1])
                    dones = tf.sqrt(veh2veh_square_dist) < rho_ego + rho_vehs
                    veh2veh_dones = logical_or(veh2veh_dones, dones)

        self.dones_type = tf.where(veh2veh_dones, 'collision', self.dones_type)
        self.dones = logical_or(self.dones, veh2veh_dones)
        return self.dones, self.dones_type

    # def compute_rewards(self, obses, actions, prev_dones):
    #     with tf.name_scope('compute_reward') as scope:
    #
    #         ego_infos, tracking_infos, veh_infos = obses[:, :12], obses[:, 12:12 + 3 + 3 * self.num_future_data], \
    #                                                obses[:, 12 + 3 + 3 * self.num_future_data:]
    #         steers, a_xs = actions[:, 0], actions[:, 1]
    #
    #         # rewards related to ego stability
    #         alpha_fs, alpha_rs, miu_fs, miu_rs = ego_infos[:, 8], ego_infos[:, 9], ego_infos[:, 10], ego_infos[:, 11]
    #         # rew_alpha_f = -1 / tf.cast(tf.square(alpha_fs - self.alpha_f_bounds), dtype=tf.float32)
    #         # rew_alpha_r = -1 / tf.cast(tf.square(alpha_rs - self.alpha_r_bounds), dtype=tf.float32)
    #         # rew_r = -1 / tf.cast(tf.square(ego_infos[:, 2] - self.r_bounds), dtype=tf.float32)
    #
    #         rew_alpha_f = - tf.cast(tf.nn.relu(tf.abs(alpha_fs) - self.alpha_f_bounds), dtype=tf.float32)
    #         rew_alpha_r = - tf.cast(tf.nn.relu(tf.abs(alpha_rs) - self.alpha_r_bounds), dtype=tf.float32)
    #         rew_r = - tf.cast(tf.nn.relu(tf.abs(ego_infos[:, 2]) - self.r_bounds), dtype=tf.float32)
    #
    #         # rewards related to action
    #         punish_steer = -tf.square(steers)
    #         punish_a_x = -tf.square(a_xs)
    #
    #         # rewards related to ego stability
    #         punish_yaw_rate = -tf.square(ego_infos[:, 2])
    #
    #         # rewards related to tracking error
    #         devi_v = -tf.cast(tf.square(ego_infos[:, 0] - self.exp_v), dtype=tf.float32)
    #         devi_y = -tf.square(tracking_infos[:, 0]) - tf.square(tracking_infos[:, 1])
    #         devi_phi = -tf.cast(tf.square(tracking_infos[:, 2] * np.pi / 180.), dtype=tf.float32)
    #
    #         # rewards related to veh2road collision
    #         ego_lws = (ego_infos[:, 6] - ego_infos[:, 7]) / 2.
    #         ego_front_points = tf.cast(ego_infos[:, 3] + ego_lws * tf.cos(ego_infos[:, 5] * np.pi / 180.),
    #                                    dtype=tf.float32), \
    #                            tf.cast(ego_infos[:, 4] + ego_lws * tf.sin(ego_infos[:, 5] * np.pi / 180.),
    #                                    dtype=tf.float32)
    #         ego_rear_points = tf.cast(ego_infos[:, 3] - ego_lws * tf.cos(ego_infos[:, 5] * np.pi / 180.),
    #                                   dtype=tf.float32), \
    #                           tf.cast(ego_infos[:, 4] - ego_lws * tf.sin(ego_infos[:, 5] * np.pi / 180.),
    #                                   dtype=tf.float32)
    #         rho_ego = ego_infos[0, 7] / 2.
    #         zeros = tf.zeros_like(ego_front_points[0])
    #         if self.task == 'left':
    #             veh2road = tf.zeros_like(ego_front_points[0])
    #             for ego_point in [ego_front_points, ego_rear_points]:
    #                 before1 = tf.where(ego_point[1] < -18, 1 / tf.square(ego_point[0] - 0 - rho_ego), zeros)
    #                 before2 = tf.where(ego_point[1] < -18, 1 / tf.square(3.75 - ego_point[0] - rho_ego), zeros)
    #                 middle_cond = logical_and(logical_and(ego_point[0] > -18, ego_point[0] < 18),
    #                                           logical_and(ego_point[1] > -18, ego_point[1] < 18))
    #                 middle1 = tf.where(middle_cond, 1 / tf.square(18 - ego_point[1] - rho_ego), zeros)
    #                 middle2 = tf.where(middle_cond, 1 / tf.square(18 - ego_point[0] - rho_ego), zeros)
    #                 middle3 = tf.where(logical_and(middle_cond, ego_point[1] > 7.5),
    #                                    1 / tf.square(ego_point[0] - (-18) - rho_ego), zeros)
    #                 middle4 = tf.where(logical_and(middle_cond, ego_point[1] < 0),
    #                                    1 / tf.square(ego_point[0] - (-18) - rho_ego), zeros)
    #                 middle5 = tf.where(logical_and(middle_cond, ego_point[0] < 0),
    #                                    1 / tf.square(ego_point[1] - (-18) - rho_ego), zeros)
    #                 middle6 = tf.where(logical_and(middle_cond, ego_point[0] > 3.75),
    #                                    1 / tf.square(ego_point[1] - (-18) - rho_ego), zeros)
    #                 after1 = tf.where(ego_point[0] < -18, 1 / tf.square(ego_point[1] - 0 - rho_ego), zeros)
    #                 after2 = tf.where(ego_point[0] < -18, 1 / tf.square(7.5 - ego_point[1] - rho_ego), zeros)
    #
    #                 this_point = before1 + before2 + middle1 + middle2 + middle3 + middle4 + middle5 + middle6 + after1 + after2
    #                 veh2road -= this_point
    #
    #         elif self.task == 'straight':
    #             veh2road = tf.zeros_like(ego_front_points[0])
    #             for ego_point in [ego_front_points, ego_rear_points]:
    #                 before1 = tf.where(ego_point[1] < -18, 1 / tf.square(ego_point[0] - 0 - rho_ego), zeros)
    #                 before2 = tf.where(ego_point[1] < -18, 1 / tf.square(3.75 - ego_point[0] - rho_ego), zeros)
    #                 middle_cond = logical_and(logical_and(ego_point[0] > -18, ego_point[0] < 18),
    #                                           logical_and(ego_point[1] > -18, ego_point[1] < 18))
    #                 middle1 = tf.where(middle_cond, 1 / tf.square(ego_point[0] - (-18) - rho_ego), zeros)
    #                 middle2 = tf.where(middle_cond, 1 / tf.square(18 - ego_point[0] - rho_ego), zeros)
    #                 middle3 = tf.where(logical_and(middle_cond, ego_point[0] < 0),
    #                                    1 / tf.square(18 - ego_point[1] - rho_ego), zeros)
    #                 middle4 = tf.where(logical_and(middle_cond, ego_point[0] > 7.5),
    #                                    1 / tf.square(18 - ego_point[1] - rho_ego), zeros)
    #                 middle5 = tf.where(logical_and(middle_cond, ego_point[0] < 0),
    #                                    1 / tf.square(ego_point[1] - (-18) - rho_ego), zeros)
    #                 middle6 = tf.where(logical_and(middle_cond, ego_point[0] > 3.75),
    #                                    1 / tf.square(ego_point[1] - (-18) - rho_ego), zeros)
    #                 after1 = tf.where(ego_point[1] > 18, 1 / tf.square(ego_point[0] - 0 - rho_ego), zeros)
    #                 after2 = tf.where(ego_point[1] > 18, 1 / tf.square(7.5 - ego_point[0] - rho_ego), zeros)
    #                 this_point = before1 + before2 + middle1 + middle2 + middle3 + middle4 + middle5 + middle6 + after1 + after2
    #                 veh2road -= this_point
    #
    #         else:
    #             veh2road = tf.zeros_like(ego_front_points[0])
    #             assert self.task == 'right'
    #             for ego_point in [ego_front_points, ego_rear_points]:
    #                 before1 = tf.where(ego_point[1] < -18, 1 / tf.square(ego_point[0] - 3.75 - rho_ego), zeros)
    #                 before2 = tf.where(ego_point[1] < -18, 1 / tf.square(7.5 - ego_point[0] - rho_ego), zeros)
    #                 middle_cond = logical_and(logical_and(ego_point[0] > -18, ego_point[0] < 18),
    #                                           logical_and(ego_point[1] > -18, ego_point[1] < 18))
    #                 middle1 = tf.where(middle_cond, 1 / tf.square(ego_point[0] - (-18) - rho_ego), zeros)
    #                 middle2 = tf.where(middle_cond, 1 / tf.square(18 - ego_point[1] - rho_ego), zeros)
    #                 middle3 = tf.where(logical_and(middle_cond, ego_point[1] > 0),
    #                                    1 / tf.square(18 - ego_point[0] - rho_ego), zeros)
    #                 middle4 = tf.where(logical_and(middle_cond, ego_point[1] < -7.5),
    #                                    1 / tf.square(18 - ego_point[0] - rho_ego), zeros)
    #                 middle5 = tf.where(logical_and(middle_cond, ego_point[0] > 7.5),
    #                                    1 / tf.square(ego_point[1] - (-18) - rho_ego), zeros)
    #                 middle6 = tf.where(logical_and(middle_cond, ego_point[0] < 3.75),
    #                                    1 / tf.square(ego_point[1] - (-18) - rho_ego), zeros)
    #                 after1 = tf.where(ego_point[0] > 18, 1 / tf.square(0 - ego_point[1] - rho_ego), zeros)
    #                 after2 = tf.where(ego_point[0] > 18, 1 / tf.square(ego_point[1] - (-7.5) - rho_ego), zeros)
    #
    #                 this_point = before1 + before2 + middle1 + middle2 + middle3 + middle4 + middle5 + middle6 + after1 + after2
    #                 veh2road -= this_point
    #
    #         # rewards related to veh2veh collision
    #         veh2veh = tf.zeros_like(ego_front_points[0])
    #         for veh_index in range(int(tf.shape(veh_infos)[1] / 6)):
    #             vehs = veh_infos[:, veh_index * 6:6 * (veh_index + 1)]
    #             veh_lws = (vehs[:, 4] - vehs[:, 5]) / 2.
    #             rho_vehs = vehs[:, 5] / 2.
    #             veh_front_points = tf.cast(vehs[:, 0] + veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
    #                                tf.cast(vehs[:, 1] + veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
    #             veh_rear_points = tf.cast(vehs[:, 0] - veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
    #                               tf.cast(vehs[:, 1] - veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
    #             for ego_point in [ego_front_points, ego_rear_points]:
    #                 for veh_point in [veh_front_points, veh_rear_points]:
    #                     veh2veh_dist = tf.sqrt(
    #                         tf.square(ego_point[0] - veh_point[0]) + tf.square(ego_point[1] - veh_point[1])) - \
    #                                    tf.convert_to_tensor(rho_ego + rho_vehs, dtype=tf.float32)
    #                     veh2veh -= 1 / tf.square(veh2veh_dist)
    #
    #         # self.reward_info = dict(punish_steer=punish_steer.numpy()[0],
    #         #                    punish_a_x=punish_a_x.numpy()[0],
    #         #                    punish_yaw_rate=punish_yaw_rate.numpy()[0],
    #         #                    devi_v=devi_v.numpy()[0],
    #         #                    devi_y=devi_y.numpy()[0],
    #         #                    devi_phi=devi_phi.numpy()[0],
    #         #                    veh2road=veh2road.numpy()[0],
    #         #                    veh2veh=veh2veh.numpy()[0],
    #         #                    rew_alpha_f=rew_alpha_f.numpy()[0],
    #         #                    rew_alpha_r=rew_alpha_r.numpy()[0],
    #         #                    rew_r=rew_r.numpy()[0]
    #         #                    )
    #
    #         # rew_alpha_f = tf.where(rew_alpha_f < -10000., -10000. * tf.ones_like(rew_alpha_f), rew_alpha_f)
    #         # rew_alpha_r = tf.where(rew_alpha_r < -10000., -10000. * tf.ones_like(rew_alpha_r), rew_alpha_r)
    #         # rew_r = tf.where(rew_r < -10000., -10000. * tf.ones_like(rew_r), rew_r)
    #         veh2road = tf.where(veh2road < -10000., -10000. * tf.ones_like(veh2road), veh2road)
    #         veh2veh = tf.where(veh2road < -10000., -10000. * tf.ones_like(veh2veh), veh2veh)
    #
    #         rewards = 0.01 * devi_v + 0.04 * devi_y + devi_phi + 0.02 * punish_yaw_rate + \
    #                   0.05 * punish_steer + 0.0005 * punish_a_x + 0.1 * veh2road + 0.1 * veh2veh + \
    #                   100 * rew_alpha_f + 100 * rew_alpha_r + 100 * rew_r
    #         rewards = tf.cast(tf.math.logical_not(prev_dones), tf.float32) * rewards
    #         return rewards

    # def compute_rewards2(self, obses, actions, prev_dones):
    #     with tf.name_scope('compute_reward') as scope:
    #
    #         ego_infos, tracking_infos, veh_infos = obses[:, :12], obses[:, 12:12 + 3 + 3 * self.num_future_data], \
    #                                                obses[:, 12 + 3 + 3 * self.num_future_data:]
    #         steers, a_xs = actions[:, 0], actions[:, 1]
    #
    #         # rewards related to ego stability
    #         alpha_fs, alpha_rs, miu_fs, miu_rs = ego_infos[:, 8], ego_infos[:, 9], ego_infos[:, 10], ego_infos[:, 11]
    #         # rew_alpha_f = -1 / tf.cast(tf.square(alpha_fs - self.alpha_f_bounds), dtype=tf.float32)
    #         # rew_alpha_r = -1 / tf.cast(tf.square(alpha_rs - self.alpha_r_bounds), dtype=tf.float32)
    #         # rew_r = -1 / tf.cast(tf.square(ego_infos[:, 2] - self.r_bounds), dtype=tf.float32)
    #
    #         rew_alpha_f = - tf.cast(tf.nn.relu(tf.abs(alpha_fs) - self.alpha_f_bounds), dtype=tf.float32)
    #         rew_alpha_r = - tf.cast(tf.nn.relu(tf.abs(alpha_rs) - self.alpha_r_bounds), dtype=tf.float32)
    #         rew_r = - tf.cast(tf.nn.relu(tf.abs(ego_infos[:, 2]) - self.r_bounds), dtype=tf.float32)
    #
    #         # rewards related to action
    #         punish_steer = -tf.square(steers)
    #         punish_a_x = -tf.square(a_xs)
    #
    #         # rewards related to ego stability
    #         punish_yaw_rate = -tf.square(ego_infos[:, 2])
    #
    #         # rewards related to tracking error
    #         devi_v = -tf.cast(tf.square(ego_infos[:, 0] - self.exp_v), dtype=tf.float32)
    #         devi_y = -tf.square(tracking_infos[:, 0]) - tf.square(tracking_infos[:, 1])
    #         devi_phi = -tf.cast(tf.square(tracking_infos[:, 2] * np.pi / 180.), dtype=tf.float32)
    #
    #         # rewards related to veh2road collision
    #         ego_lws = (ego_infos[:, 6] - ego_infos[:, 7]) / 2.
    #         ego_front_points = tf.cast(ego_infos[:, 3] + ego_lws * tf.cos(ego_infos[:, 5] * np.pi / 180.),
    #                                    dtype=tf.float32), \
    #                            tf.cast(ego_infos[:, 4] + ego_lws * tf.sin(ego_infos[:, 5] * np.pi / 180.),
    #                                    dtype=tf.float32)
    #         ego_rear_points = tf.cast(ego_infos[:, 3] - ego_lws * tf.cos(ego_infos[:, 5] * np.pi / 180.),
    #                                   dtype=tf.float32), \
    #                           tf.cast(ego_infos[:, 4] - ego_lws * tf.sin(ego_infos[:, 5] * np.pi / 180.),
    #                                   dtype=tf.float32)
    #         coeff = 1.14
    #         rho_ego = ego_infos[0, 7] / 2. * coeff
    #         zeros = tf.zeros_like(ego_front_points[0])
    #         if self.task == 'left':
    #             veh2road = tf.zeros_like(ego_front_points[0])
    #             for ego_point in [ego_front_points, ego_rear_points]:
    #                 before1 = tf.where(ego_point[1] < -18, tf.nn.relu(-(ego_point[0] - 0 - rho_ego)), zeros)
    #                 before2 = tf.where(ego_point[1] < -18, tf.nn.relu(-(3.75 - ego_point[0] - rho_ego)), zeros)
    #                 middle_cond = logical_and(logical_and(ego_point[0] > -18, ego_point[0] < 18),
    #                                           logical_and(ego_point[1] > -18, ego_point[1] < 18))
    #                 middle1 = tf.where(middle_cond, tf.nn.relu(-(18 - ego_point[1] - rho_ego)), zeros)
    #                 middle2 = tf.where(middle_cond, tf.nn.relu(-(18 - ego_point[0] - rho_ego)), zeros)
    #                 middle3 = tf.where(logical_and(middle_cond, ego_point[1] > 7.5),
    #                                    tf.nn.relu(-(ego_point[0] - (-18) - rho_ego)), zeros)
    #                 middle4 = tf.where(logical_and(middle_cond, ego_point[1] < 0),
    #                                    tf.nn.relu(-(ego_point[0] - (-18) - rho_ego)), zeros)
    #                 middle5 = tf.where(logical_and(middle_cond, ego_point[0] < 0),
    #                                    tf.nn.relu(-(ego_point[1] - (-18) - rho_ego)), zeros)
    #                 middle6 = tf.where(logical_and(middle_cond, ego_point[0] > 3.75),
    #                                    tf.nn.relu(-(ego_point[1] - (-18) - rho_ego)), zeros)
    #
    #                 middle7 = tf.where(middle_cond, tf.nn.relu(-(tf.sqrt(tf.square(ego_point[0] - (-18)) + tf.square(
    #                     ego_point[1] - 0)) - rho_ego)), zeros)
    #                 middle8 = tf.where(middle_cond, tf.nn.relu(-(tf.sqrt(tf.square(ego_point[0] - (-18)) + tf.square(
    #                     ego_point[1] - 7.5)) - rho_ego)), zeros)
    #
    #                 after1 = tf.where(ego_point[0] < -18, tf.nn.relu(-(ego_point[1] - 0 - rho_ego)), zeros)
    #                 after2 = tf.where(ego_point[0] < -18, tf.nn.relu(-(7.5 - ego_point[1] - rho_ego)), zeros)
    #
    #                 this_point = before1 + before2 +\
    #                              middle1 + middle2 + middle3 + middle4 + middle5 + middle6 + middle7 + middle8 +\
    #                              after1 + after2
    #                 veh2road -= this_point
    #
    #         elif self.task == 'straight':
    #             veh2road = tf.zeros_like(ego_front_points[0])
    #             for ego_point in [ego_front_points, ego_rear_points]:
    #                 before1 = tf.where(ego_point[1] < -18, tf.nn.relu(-(ego_point[0] - 0 - rho_ego)), zeros)
    #                 before2 = tf.where(ego_point[1] < -18, tf.nn.relu(-(3.75 - ego_point[0] - rho_ego)), zeros)
    #                 middle_cond = logical_and(logical_and(ego_point[0] > -18, ego_point[0] < 18),
    #                                           logical_and(ego_point[1] > -18, ego_point[1] < 18))
    #                 middle1 = tf.where(middle_cond, tf.nn.relu(-(ego_point[0] - (-18) - rho_ego)), zeros)
    #                 middle2 = tf.where(middle_cond, tf.nn.relu(-(18 - ego_point[0] - rho_ego)), zeros)
    #                 middle3 = tf.where(logical_and(middle_cond, ego_point[0] < 0),
    #                                    tf.nn.relu(-(18 - ego_point[1] - rho_ego)), zeros)
    #                 middle4 = tf.where(logical_and(middle_cond, ego_point[0] > 7.5),
    #                                    tf.nn.relu(-(18 - ego_point[1] - rho_ego)), zeros)
    #                 middle5 = tf.where(logical_and(middle_cond, ego_point[0] < 0),
    #                                    tf.nn.relu(-(ego_point[1] - (-18) - rho_ego)), zeros)
    #                 middle6 = tf.where(logical_and(middle_cond, ego_point[0] > 3.75),
    #                                    tf.nn.relu(-(ego_point[1] - (-18) - rho_ego)), zeros)
    #
    #                 middle7 = tf.where(middle_cond, tf.nn.relu(-(tf.sqrt(tf.square(ego_point[0] - 0) + tf.square(
    #                     ego_point[1] - 18)) - rho_ego)), zeros)
    #                 middle8 = tf.where(middle_cond, tf.nn.relu(-(tf.sqrt(tf.square(ego_point[0] - 7.5) + tf.square(
    #                     ego_point[1] - 18)) - rho_ego)), zeros)
    #
    #                 after1 = tf.where(ego_point[1] > 18, tf.nn.relu(-(ego_point[0] - 0 - rho_ego)), zeros)
    #                 after2 = tf.where(ego_point[1] > 18, tf.nn.relu(-(7.5 - ego_point[0] - rho_ego)), zeros)
    #                 this_point = before1 + before2 + \
    #                              middle1 + middle2 + middle3 + middle4 + middle5 + middle6 + middle7 + middle8 +\
    #                              after1 + after2
    #                 veh2road -= this_point
    #
    #         else:
    #             veh2road = tf.zeros_like(ego_front_points[0])
    #             assert self.task == 'right'
    #             for ego_point in [ego_front_points, ego_rear_points]:
    #                 before1 = tf.where(ego_point[1] < -18, tf.nn.relu(-(ego_point[0] - 3.75 - rho_ego)), zeros)
    #                 before2 = tf.where(ego_point[1] < -18, tf.nn.relu(-(7.5 - ego_point[0] - rho_ego)), zeros)
    #                 middle_cond = logical_and(logical_and(ego_point[0] > -18, ego_point[0] < 18),
    #                                           logical_and(ego_point[1] > -18, ego_point[1] < 18))
    #                 middle1 = tf.where(middle_cond, tf.nn.relu(-(ego_point[0] - (-18) - rho_ego)), zeros)
    #                 middle2 = tf.where(middle_cond, tf.nn.relu(-(18 - ego_point[1] - rho_ego)), zeros)
    #                 middle3 = tf.where(logical_and(middle_cond, ego_point[1] > 0),
    #                                    tf.nn.relu(-(18 - ego_point[0] - rho_ego)), zeros)
    #                 middle4 = tf.where(logical_and(middle_cond, ego_point[1] < -7.5),
    #                                    tf.nn.relu(-(18 - ego_point[0] - rho_ego)), zeros)
    #                 middle5 = tf.where(logical_and(middle_cond, ego_point[0] > 7.5),
    #                                    tf.nn.relu(-(ego_point[1] - (-18) - rho_ego)), zeros)
    #                 middle6 = tf.where(logical_and(middle_cond, ego_point[0] < 3.75),
    #                                    tf.nn.relu(-(ego_point[1] - (-18) - rho_ego)), zeros)
    #                 middle7 = tf.where(middle_cond, tf.nn.relu(-(tf.sqrt(tf.square(ego_point[0] - 18) + tf.square(
    #                     ego_point[1] - 0)) - rho_ego)), zeros)
    #                 middle8 = tf.where(middle_cond, tf.nn.relu(-(tf.sqrt(tf.square(ego_point[0] - 18) + tf.square(
    #                     ego_point[1] - (-7.5))) - rho_ego)), zeros)
    #
    #                 after1 = tf.where(ego_point[0] > 18, tf.nn.relu(-(0 - ego_point[1] - rho_ego)), zeros)
    #                 after2 = tf.where(ego_point[0] > 18, tf.nn.relu(-(ego_point[1] - (-7.5) - rho_ego)), zeros)
    #
    #                 this_point = before1 + before2 + \
    #                              middle1 + middle2 + middle3 + middle4 + middle5 + middle6 + middle7 + middle8 + \
    #                              after1 + after2
    #                 veh2road -= this_point
    #
    #         # rewards related to veh2veh collision
    #         veh2veh = tf.zeros_like(ego_front_points[0])
    #         for veh_index in range(int(tf.shape(veh_infos)[1] / 6)):
    #             vehs = veh_infos[:, veh_index * 6:6 * (veh_index + 1)]
    #             veh_lws = (vehs[:, 4] - vehs[:, 5]) / 2.
    #             rho_vehs = vehs[:, 5] / 2. * coeff
    #             veh_front_points = tf.cast(vehs[:, 0] + veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
    #                                tf.cast(vehs[:, 1] + veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
    #             veh_rear_points = tf.cast(vehs[:, 0] - veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
    #                               tf.cast(vehs[:, 1] - veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
    #             for ego_point in [ego_front_points, ego_rear_points]:
    #                 for veh_point in [veh_front_points, veh_rear_points]:
    #                     veh2veh_dist = tf.sqrt(
    #                         tf.square(ego_point[0] - veh_point[0]) + tf.square(ego_point[1] - veh_point[1])) - \
    #                                    tf.convert_to_tensor(rho_ego + rho_vehs, dtype=tf.float32)
    #                     veh2veh -= 1 / tf.square(veh2veh_dist)
    #
    #         # self.reward_info = dict(punish_steer=punish_steer.numpy()[0],
    #         #                    punish_a_x=punish_a_x.numpy()[0],
    #         #                    punish_yaw_rate=punish_yaw_rate.numpy()[0],
    #         #                    devi_v=devi_v.numpy()[0],
    #         #                    devi_y=devi_y.numpy()[0],
    #         #                    devi_phi=devi_phi.numpy()[0],
    #         #                    veh2road=veh2road.numpy()[0],
    #         #                    veh2veh=veh2veh.numpy()[0],
    #         #                    rew_alpha_f=rew_alpha_f.numpy()[0],
    #         #                    rew_alpha_r=rew_alpha_r.numpy()[0],
    #         #                    rew_r=rew_r.numpy()[0]
    #         #                    )
    #
    #         # rew_alpha_f = tf.where(rew_alpha_f < -10000., -10000. * tf.ones_like(rew_alpha_f), rew_alpha_f)
    #         # rew_alpha_r = tf.where(rew_alpha_r < -10000., -10000. * tf.ones_like(rew_alpha_r), rew_alpha_r)
    #         # rew_r = tf.where(rew_r < -10000., -10000. * tf.ones_like(rew_r), rew_r)
    #         # veh2road = tf.where(veh2road < -10000., -10000. * tf.ones_like(veh2road), veh2road)
    #         veh2veh = tf.where(veh2veh < -100., -100. * tf.ones_like(veh2veh), veh2veh)
    #
    #         rewards = 0.01 * devi_v + 0.04 * devi_y + 5 * devi_phi + 0.02 * punish_yaw_rate + \
    #                   0.05 * punish_steer + 0.0005 * punish_a_x + 100 * veh2road + veh2veh + \
    #                   100 * rew_alpha_f + 100 * rew_alpha_r + 100 * rew_r
    #         rewards = tf.cast(tf.math.logical_not(prev_dones), tf.float32) * rewards
    #         return rewards

    def compute_rewards3(self, obses, actions, prev_dones):
        with tf.name_scope('compute_reward') as scope:
            ego_infos, tracking_infos, veh_infos = obses[:, :self.ego_info_dim], obses[:, self.ego_info_dim:self.ego_info_dim + 4 * (self.num_future_data+1)], \
                                                   obses[:, self.ego_info_dim + 4 * (self.num_future_data+1):]
            steers, a_xs = actions[:, 0], actions[:, 1]
            # rewards related to action
            punish_steer = -tf.square(steers)
            punish_a_x = -tf.square(a_xs)

            # rewards related to ego stability
            punish_yaw_rate = -tf.square(ego_infos[:, 2])

            # rewards related to tracking error
            devi_v = -tf.cast(tf.square(ego_infos[:, 0] - self.exp_v), dtype=tf.float32)
            devi_y = -tf.square(tracking_infos[:, 0]) - tf.square(tracking_infos[:, 1])
            devi_phi = -tf.cast(tf.square(tracking_infos[:, 4+2] * np.pi / 180.), dtype=tf.float32)

            # rewards related to veh2veh collision
            ego_lws = (ego_infos[:, 6] - ego_infos[:, 7]) / 2.
            ego_front_points = tf.cast(ego_infos[:, 3] + ego_lws * tf.cos(ego_infos[:, 5] * np.pi / 180.),
                                       dtype=tf.float32), \
                               tf.cast(ego_infos[:, 4] + ego_lws * tf.sin(ego_infos[:, 5] * np.pi / 180.),
                                       dtype=tf.float32)
            ego_rear_points = tf.cast(ego_infos[:, 3] - ego_lws * tf.cos(ego_infos[:, 5] * np.pi / 180.),
                                      dtype=tf.float32), \
                              tf.cast(ego_infos[:, 4] - ego_lws * tf.sin(ego_infos[:, 5] * np.pi / 180.),
                                      dtype=tf.float32)
            coeff = 1.14
            rho_ego = ego_infos[0, 7] / 2. * coeff
            # rewards related to veh2road collision

            zeros = tf.zeros_like(ego_front_points[0])
            if self.task == 'left':
                veh2road = tf.zeros_like(ego_front_points[0])
                for ego_point in [ego_front_points, ego_rear_points]:
                    before1 = tf.where(ego_point[1] < -18, 0./tf.square(ego_point[0] - 0 - rho_ego), zeros)
                    before2 = tf.where(ego_point[1] < -18, 0./tf.square(3.75 - ego_point[0] - rho_ego), zeros)
                    middle_cond = logical_and(logical_and(ego_point[0] > -18, ego_point[0] < 18),
                                              logical_and(ego_point[1] > -18, ego_point[1] < 18))
                    middle1 = tf.where(middle_cond, 1./tf.square(7.5 - ego_point[1] - rho_ego), zeros)
                    middle2 = tf.where(middle_cond, 1./tf.square(7.5 - ego_point[0] - rho_ego), zeros)
                    middle3 = tf.where(logical_and(middle_cond, ego_point[1] < 0),
                                       1./tf.square(ego_point[0] - (-18) - rho_ego), zeros)
                    middle4 = tf.where(logical_and(middle_cond, ego_point[0] < 0),
                                       1./tf.square(ego_point[1] - (-18) - rho_ego), zeros)

                    after1 = tf.where(ego_point[0] < -18, 0./tf.square(ego_point[1] - 0 - rho_ego), zeros)
                    after2 = tf.where(ego_point[0] < -18, 0./tf.square(7.5 - ego_point[1] - rho_ego), zeros)

                    this_point = before1 + before2 +\
                                 middle1 + middle2 + middle3 + middle4 +\
                                 after1 + after2
                    veh2road -= this_point

            veh2veh = tf.zeros_like(veh_infos[:, 0])
            for veh_index in range(int(tf.shape(veh_infos)[1] / self.per_veh_info_dim)):
                vehs = veh_infos[:, veh_index * self.per_veh_info_dim:(veh_index + 1)*self.per_veh_info_dim]
                # for i in [6, 7, 8, 9]:
                #     veh2veh -= 1. / tf.square(vehs[:, i])
                veh_lws = (vehs[:, 4] - vehs[:, 5]) / 2.
                rho_vehs = vehs[:, 5] / 2. * coeff
                veh_front_points = tf.cast(vehs[:, 0] + veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
                                   tf.cast(vehs[:, 1] + veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
                veh_rear_points = tf.cast(vehs[:, 0] - veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
                                  tf.cast(vehs[:, 1] - veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
                for ego_point in [ego_front_points, ego_rear_points]:
                    for veh_point in [veh_front_points, veh_rear_points]:
                        veh2veh_dist = tf.sqrt(
                            tf.square(ego_point[0] - veh_point[0]) + tf.square(ego_point[1] - veh_point[1])) - \
                                       tf.convert_to_tensor(rho_ego + rho_vehs, dtype=tf.float32)
                        veh2veh -= 1 / veh2veh_dist
                        # veh2veh -= tf.nn.relu(-(veh2veh_dist - 10.))

            veh2road = tf.where(veh2road < -3., -3. * tf.ones_like(veh2road), veh2road)
            veh2veh = tf.where(veh2veh < -3., -3. * tf.ones_like(veh2veh), veh2veh)
            rewards = 0.01 * devi_v + 0.1 * devi_y + 5 * devi_phi + 0.02 * punish_yaw_rate + \
                      0.05 * punish_steer + 0.0005 * punish_a_x + veh2veh + veh2road
            rewards = tf.cast(tf.math.logical_not(prev_dones), tf.float32) * rewards
            # self.reward_info = dict(punish_steer=punish_steer.numpy()[0],
            #                         punish_a_x=punish_a_x.numpy()[0],
            #                         punish_yaw_rate=punish_yaw_rate.numpy()[0],
            #                         devi_v=devi_v.numpy()[0],
            #                         devi_y=devi_y.numpy()[0],
            #                         devi_phi=devi_phi.numpy()[0],
            #                         veh2road=veh2road.numpy()[0],
            #                         veh2veh=veh2veh.numpy()[0],
            #                         rew_alpha_f=0.,
            #                         rew_alpha_r=0.,
            #                         rew_r=0.,
            #                         scaled_punish_steer=0.05 * punish_steer.numpy()[0],
            #                         scaled_punish_a_x=0.0005 * punish_a_x.numpy()[0],
            #                         scaled_punish_yaw_rate=0.02 * punish_yaw_rate.numpy()[0],
            #                         scaled_devi_v=0.01 * devi_v.numpy()[0],
            #                         scaled_devi_y=0.1 * devi_y.numpy()[0],
            #                         scaled_devi_phi=5 * devi_phi.numpy()[0],
            #                         scaled_veh2road=veh2road.numpy()[0],
            #                         scaled_veh2veh=veh2veh.numpy()[0],
            #                         scaled_rew_alpha_f=0.,
            #                         scaled_rew_alpha_r=0.,
            #                         scaled_rew_r=0.,
            #                         reward=rewards.numpy()[0]
            #                         )
            return rewards

    def compute_next_obses(self, obses, actions):
        ego_infos, tracking_infos, veh_infos = obses[:, :self.ego_info_dim], obses[:, self.ego_info_dim:self.ego_info_dim + 4 * (self.num_future_data+1)], \
                                               obses[:, self.ego_info_dim + 4 * (self.num_future_data+1):]

        next_ego_infos = self.ego_predict(ego_infos, actions)

        next_tracking_infos = self.ref_path.tracking_error_vector(next_ego_infos[:, 3],
                                                                  next_ego_infos[:, 4],
                                                                  next_ego_infos[:, 5],
                                                                  next_ego_infos[:, 0],
                                                                  self.num_future_data)
        next_veh_infos = self.veh_predict(next_ego_infos, veh_infos)
        next_obses = tf.concat([next_ego_infos, next_tracking_infos, next_veh_infos], 1)
        return next_obses

    def ego_predict(self, ego_infos, actions):
        ego_next_infos_except_lw, ego_next_params = self.vehicle_dynamics.prediction(ego_infos[:, :6], actions,
                                                                                     self.base_frequency, 1)
        ego_next_lw = ego_infos[:, 6:8]

        # ego_v_xs, ego_v_ys, ego_rs, ego_xs, ego_ys, ego_phis = ego_next_infos_except_lw[:, 0], \
        #                                                        ego_next_infos_except_lw[:, 1], \
        #                                                        ego_next_infos_except_lw[:, 2], \
        #                                                        ego_next_infos_except_lw[:, 3], \
        #                                                        ego_next_infos_except_lw[:, 4], \
        #                                                        ego_next_infos_except_lw[:, 5]
        # ego_ls, ego_ws = ego_next_lw[:, 0], ego_next_lw[:, 1]
        # ego_lws = (ego_ls - ego_ws) / 2.
        # coeff = 1.14
        # rho_egos = ego_ws / 2. * coeff
        # ego_front_points = tf.cast(ego_xs + ego_lws * tf.cos(ego_phis * np.pi / 180.), dtype=tf.float32), \
        #                    tf.cast(ego_ys + ego_lws * tf.sin(ego_phis * np.pi / 180.), dtype=tf.float32)
        # ego_rear_points = tf.cast(ego_xs - ego_lws * tf.cos(ego_phis * np.pi / 180.), dtype=tf.float32), \
        #                   tf.cast(ego_ys - ego_lws * tf.sin(ego_phis * np.pi / 180.), dtype=tf.float32)

        # other_features = []
        # ones = 50. * tf.ones_like(ego_front_points[0])
        # if self.task == 'left':
        #     for ego_points in [ego_front_points, ego_rear_points]:
        #         ups = downs = lefts = rights = ones
        #         middle_cond = logical_and(logical_and(ego_points[0] > -18, ego_points[0] < 18),
        #                                   logical_and(ego_points[1] > -18, ego_points[1] < 18))
        #         ups = tf.where(ego_points[1] <= -18., 18. - ego_points[1] - rho_egos, ups)
        #         ups = tf.where(middle_cond, 18. - ego_points[1] - rho_egos, ups)
        #         ups = tf.where(ego_points[0] <= -18., 7.5 - ego_points[1] - rho_egos, ups)
        #
        #         downs = tf.where(ego_points[1] <= -18., ones, downs)
        #         downs = tf.where(middle_cond, ego_points[1] - (-18.) - rho_egos, downs)
        #         downs = tf.where(logical_and(middle_cond, logical_and(0 < ego_points[0], ego_points[0] < 3.75)), ones, downs)
        #         downs = tf.where(ego_points[0] <= -18., ego_points[1] - 0 - rho_egos, downs)
        #
        #         lefts = tf.where(ego_points[1] <= -18., ego_points[0] - 0 - rho_egos, lefts)
        #         lefts = tf.where(middle_cond, ego_points[0] - (-18) - rho_egos, lefts)
        #         lefts = tf.where(logical_and(middle_cond, logical_and(0 < ego_points[1], ego_points[1] < 7.5)), ones, lefts)
        #         lefts = tf.where(ego_points[0] <= -18., ones, lefts)
        #
        #         rights = tf.where(ego_points[1] <= -18., 3.75 - ego_points[0] - rho_egos, rights)
        #         rights = tf.where(middle_cond, 18. - ego_points[0] - rho_egos, rights)
        #         rights = tf.where(ego_points[0] <= -18., 18. - ego_points[0] - rho_egos, rights)
        #
        #         other_features.extend([ups, downs, lefts, rights])
        #         for corner in [(-18., 7.5), (-18., 0.)]:
        #             other_features.extend([ego_points[0] - corner[0], ego_points[1] - corner[1]])
        # elif self.task == 'straight':
        #     for ego_points in [ego_front_points, ego_rear_points]:
        #         ups = downs = lefts = rights = ones
        #         middle_cond = logical_and(logical_and(ego_points[0] > -18, ego_points[0] < 18),
        #                                   logical_and(ego_points[1] > -18, ego_points[1] < 18))
        #
        #         ups = tf.where(ego_points[1] <= -18., ones, ups)
        #         ups = tf.where(middle_cond, 18. - ego_points[1] - rho_egos, ups)
        #         ups = tf.where(logical_and(middle_cond, logical_and(0. < ego_points[1], ego_points[1] < 7.5)), ones, ups)
        #         ups = tf.where(ego_points[1] >= 18., ones, ups)
        #
        #         downs = tf.where(ego_points[1] <= -18., ones, downs)
        #         downs = tf.where(middle_cond, ego_points[1] - (-18.) - rho_egos, downs)
        #         downs = tf.where(logical_and(middle_cond, logical_and(0 < ego_points[0], ego_points[0] < 3.75)), ones, downs)
        #         downs = tf.where(ego_points[1] >= 18., ones, downs)
        #
        #         lefts = tf.where(ego_points[1] <= -18., ego_points[0] - 0 - rho_egos, lefts)
        #         lefts = tf.where(middle_cond, ego_points[0] - (-18) - rho_egos, lefts)
        #         lefts = tf.where(ego_points[1] >= 18., ego_points[0] - 0 - rho_egos, lefts)
        #
        #         rights = tf.where(ego_points[1] <= -18., 3.75 - ego_points[0] - rho_egos, rights)
        #         rights = tf.where(middle_cond, 18. - ego_points[0] - rho_egos, rights)
        #         rights = tf.where(ego_points[1] >= 18., 7.5 - ego_points[0] - rho_egos, rights)
        #
        #         other_features.extend([ups, downs, lefts, rights])
        #         for corner in [(0., 18.), (7.5, 18.)]:
        #             other_features.extend([ego_points[0] - corner[0], ego_points[1] - corner[1]])
        # else:
        #     assert self.task == 'right'
        #     for ego_points in [ego_front_points, ego_rear_points]:
        #         ups = downs = lefts = rights = ones
        #         middle_cond = logical_and(logical_and(ego_points[0] > -18, ego_points[0] < 18),
        #                                   logical_and(ego_points[1] > -18, ego_points[1] < 18))
        #         ups = tf.where(ego_points[1] <= -18., 18 - ego_points[1] - rho_egos, ups)
        #         ups = tf.where(middle_cond, 18. - ego_points[1] - rho_egos, ups)
        #         ups = tf.where(ego_points[0] >= 18., 0. - ego_points[1] - rho_egos, ups)
        #
        #         downs = tf.where(ego_points[1] <= -18., ones, downs)
        #         downs = tf.where(middle_cond, ego_points[1] - (-18.) - rho_egos, downs)
        #         downs = tf.where(logical_and(middle_cond, logical_and(3.75 < ego_points[0], ego_points[0] < 7.5)), ones, downs)
        #         downs = tf.where(ego_points[0] >= 18., ego_points[1] - (-7.5) - rho_egos, downs)
        #
        #         lefts = tf.where(ego_points[1] <= -18., ego_points[0] - 3.75 - rho_egos, lefts)
        #         lefts = tf.where(middle_cond, ego_points[0] - (-18) - rho_egos, lefts)
        #         lefts = tf.where(ego_points[0] >= 18., ego_points[0] - (-18.) - rho_egos, lefts)
        #
        #         rights = tf.where(ego_points[1] <= -18., 7.5 - ego_points[0] - rho_egos, rights)
        #         rights = tf.where(middle_cond, 18. - ego_points[0] - rho_egos, rights)
        #         rights = tf.where(logical_and(middle_cond, logical_and(-7.5 < ego_points[1], ego_points[1] < 0)), ones, rights)
        #         rights = tf.where(ego_points[0] >= 18., ones, rights)
        #
        #         other_features.extend([ups, downs, lefts, rights])
        #         for corner in [(18., 0), (18., -7.5)]:
        #             other_features.extend([ego_points[0] - corner[0], ego_points[1] - corner[1]])
        # ego_other_features = tf.stack(other_features, axis=1)
        # return tf.concat([ego_next_infos_except_lw, ego_next_lw, ego_next_params, ego_other_features], 1)

        return tf.concat([ego_next_infos_except_lw, ego_next_lw, ego_next_params], 1)

    def veh_predict(self, next_ego_infos, veh_infos):
        if self.task == 'left':
            veh_mode_list = ['dl'] * 2 + ['du'] * 2 + ['ud'] * 3 + ['ul'] * 3
        elif self.task == 'straight':
            veh_mode_list = ['dl'] * 2 + ['du'] * 2 + ['ud'] * 2 + ['ru'] * 3 + ['ur'] * 3
        else:
            assert self.task == 'right'
            veh_mode_list = ['dr'] * 2 + ['ur'] * 3 + ['lr'] * 3

        predictions_to_be_concat = []

        for vehs_index in range(len(veh_mode_list)):
            predictions_to_be_concat.append(self.predict_for_a_mode(next_ego_infos,
                                                                    veh_infos[:, vehs_index * self.per_veh_info_dim:(vehs_index + 1) * self.per_veh_info_dim],
                                                                    veh_mode_list[vehs_index]))
        return tf.concat(predictions_to_be_concat, 1)

    def predict_for_a_mode(self, next_ego_infos, vehs, mode):
        # next_ego_xs, next_ego_ys, next_ego_phis, next_ego_ls, next_ego_ws = next_ego_infos[:, 3],\
        #                                            next_ego_infos[:, 4],\
        #                                            next_ego_infos[:, 5],\
        #                                            next_ego_infos[:, 6],\
        #                                            next_ego_infos[:, 7]
        # next_ego_lws = (next_ego_ls - next_ego_ws) / 2.
        # coeff = 1.14
        # next_rho_egos = next_ego_ws / 2. * coeff
        # next_ego_front_points = tf.cast(next_ego_xs + next_ego_lws * tf.cos(next_ego_phis * np.pi / 180.), dtype=tf.float32), \
        #                   tf.cast(next_ego_ys + next_ego_lws * tf.sin(next_ego_phis * np.pi / 180.), dtype=tf.float32)
        # next_ego_rear_points = tf.cast(next_ego_xs - next_ego_lws * tf.cos(next_ego_phis * np.pi / 180.), dtype=tf.float32), \
        #                  tf.cast(next_ego_ys - next_ego_lws * tf.sin(next_ego_phis * np.pi / 180.), dtype=tf.float32)

        veh_xs, veh_ys, veh_vs, veh_phis, veh_ls, veh_ws = \
            vehs[:, 0], vehs[:, 1], vehs[:, 2], vehs[:, 3], vehs[:, 4], vehs[:, 5]
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
        next_veh_xs, next_veh_ys, next_veh_vs, next_veh_phis_rad, next_veh_ls, next_veh_ws =\
            veh_xs + veh_xs_delta, veh_ys + veh_ys_delta, veh_vs, veh_phis_rad + veh_phis_rad_delta, veh_ls, veh_ws
        next_veh_phis_rad = tf.where(next_veh_phis_rad > np.pi, next_veh_phis_rad - 2 * np.pi, next_veh_phis_rad)
        next_veh_phis_rad = tf.where(next_veh_phis_rad <= -np.pi, next_veh_phis_rad + 2 * np.pi, next_veh_phis_rad)
        next_veh_phis = next_veh_phis_rad * 180 / np.pi

        # next_veh_lws = (next_veh_ls - next_veh_ws) / 2.
        # next_rho_vehs = next_veh_ws / 2. * coeff
        # next_veh_front_points = tf.cast(next_veh_xs + next_veh_lws * tf.cos(next_veh_phis * np.pi / 180.), dtype=tf.float32), \
        #                    tf.cast(next_veh_ys + next_veh_lws * tf.sin(next_veh_phis * np.pi / 180.), dtype=tf.float32)
        # next_veh_rear_points = tf.cast(next_veh_xs - next_veh_lws * tf.cos(next_veh_phis * np.pi / 180.), dtype=tf.float32), \
        #                   tf.cast(next_veh_ys - next_veh_lws * tf.sin(next_veh_phis * np.pi / 180.), dtype=tf.float32)

        # next_veh2veh_dist = []
        # for next_ego_point in [next_ego_front_points, next_ego_rear_points]:
        #     for next_veh_point in [next_veh_front_points, next_veh_rear_points]:
        #         point_dist = tf.sqrt(tf.square(next_ego_point[0] - next_veh_point[0]) + tf.square(
        #             next_ego_point[1] - next_veh_point[1])) - (next_rho_egos + next_rho_vehs)
        #         next_veh2veh_dist.append(point_dist)
        # return tf.stack([next_veh_xs, next_veh_ys, next_veh_vs, next_veh_phis, next_veh_ls, next_veh_ws]
        #                 + next_veh2veh_dist, 1)
        return tf.stack([next_veh_xs, next_veh_ys, next_veh_vs, next_veh_phis, next_veh_ls, next_veh_ws], 1)

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
            ego_info, tracing_info, vehs_info = obses[0, :self.ego_info_dim], obses[0, self.ego_info_dim:self.ego_info_dim + 4 * (self.num_future_data+1)], \
                                                obses[0, self.ego_info_dim + 4 * (self.num_future_data+1):]
            # plot cars
            for veh_index in range(int(len(vehs_info) / self.per_veh_info_dim)):
                veh = vehs_info[self.per_veh_info_dim * veh_index:self.per_veh_info_dim * (veh_index + 1)]
                # veh_x, veh_y, veh_v, veh_phi, veh_l, veh_w, dist1, dist2, dist3, dist4 = veh
                veh_x, veh_y, veh_v, veh_phi, veh_l, veh_w = veh

                if is_in_plot_area(veh_x, veh_y):
                    # plt.text(veh_x, veh_y, '{:.1f}'.format(min([dist1, dist2, dist3, dist4])))
                    plot_phi_line(veh_x, veh_y, veh_phi, 'black')
                    draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, 'black')

            # plot own car
            # ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi, ego_l, ego_w, \
            # ego_alpha_f, ego_alpha_r, ego_miu_f, ego_miu_r,\
            # up1, down1, left1, right1, point11x, point11y, point12x, point12y, \
            # up2, down2, left2, right2, point21x, point21y, point22x, point22y= ego_info
            ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi, ego_l, ego_w, \
            ego_alpha_f, ego_alpha_r, ego_miu_f, ego_miu_r = ego_info

            plot_phi_line(ego_x, ego_y, ego_phi, 'red')
            draw_rotate_rec(ego_x, ego_y, ego_phi, ego_l, ego_w, 'red')

            # plot planed trj
            ax.plot(self.ref_path.path[0], self.ref_path.path[1], color='g')
            indexs, points = self.ref_path.find_closest_point(np.array([ego_x], np.float32),
                                                              np.array([ego_y], np.float32))
            path_x, path_y, path_phi = points[0][0], points[1][0], points[2][0]
            delta_x, delta_y, delta_phi, delta_v = tracing_info[:4]
            # delta_x, delta_y, delta_phi = ego_x - path_x, ego_y - path_y, ego_phi - path_phi
            plt.plot(path_x, path_y, 'go')
            plot_phi_line(path_x, path_y, path_phi, 'g')

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
            plt.text(text_x, text_y_start - next(ge), 'path_x: {:.2f}m'.format(path_x))
            plt.text(text_x, text_y_start - next(ge), 'path_y: {:.2f}m'.format(path_y))
            plt.text(text_x, text_y_start - next(ge), 'delta_x: {:.2f}m'.format(delta_x))
            plt.text(text_x, text_y_start - next(ge), 'delta_y: {:.2f}m'.format(delta_y))
            plt.text(text_x, text_y_start - next(ge), r'ego_phi: ${:.2f}\degree$'.format(ego_phi))
            plt.text(text_x, text_y_start - next(ge), r'path_phi: ${:.2f}\degree$'.format(path_phi))
            plt.text(text_x, text_y_start - next(ge), r'delta_phi: ${:.2f}\degree$'.format(delta_phi))

            plt.text(text_x, text_y_start - next(ge), 'v_x: {:.2f}m/s'.format(ego_v_x))
            plt.text(text_x, text_y_start - next(ge), 'exp_v: {:.2f}m/s'.format(self.exp_v))
            plt.text(text_x, text_y_start - next(ge), 'v_y: {:.2f}m/s'.format(ego_v_y))
            plt.text(text_x, text_y_start - next(ge), 'yaw_rate: {:.2f}rad/s'.format(ego_r))
            plt.text(text_x, text_y_start - next(ge),
                     'yaw_rate bound: [{:.2f}, {:.2f}]'.format(-self.r_bounds[0], self.r_bounds[0]))

            plt.text(text_x, text_y_start - next(ge), r'$\alpha_f$: {:.2f} rad'.format(ego_alpha_f))
            plt.text(text_x, text_y_start - next(ge),
                     r'$\alpha_f$ bound: [{:.2f}, {:.2f}] '.format(-self.alpha_f_bounds[0],
                                                                   self.alpha_f_bounds[0]))
            plt.text(text_x, text_y_start - next(ge), r'$\alpha_r$: {:.2f} rad'.format(ego_alpha_r))
            plt.text(text_x, text_y_start - next(ge),
                     r'$\alpha_r$ bound: [{:.2f}, {:.2f}] '.format(-self.alpha_r_bounds[0],self.alpha_r_bounds[0]))

            if self.actions is not None:
                steer, a_x = self.actions[0, 0], self.actions[0, 1]
                plt.text(text_x, text_y_start - next(ge),
                         r'steer: {:.2f}rad (${:.2f}\degree$)'.format(steer, steer * 180 / np.pi))
                plt.text(text_x, text_y_start - next(ge), 'a_x: {:.2f}m/s^2'.format(a_x))

            text_x, text_y_start = 70, 60
            ge = iter(range(0, 1000, 4))

            # done info
            plt.text(text_x, text_y_start - next(ge), 'done: {}'.format(self.dones[0]))
            plt.text(text_x, text_y_start - next(ge), 'done info: {}'.format(self.dones_type[0]))

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
        self.task = task
        self.path_list = []
        self._construct_ref_path(self.task)
        self.ref_index = np.random.choice([0, 1])
        self.path = self.path_list[self.ref_index]

    def _construct_ref_path(self, task):
        sl = 20
        planed_trj = None
        meter_pointnum_ratio = 30
        if task == 'left':
            if self.mode == 'training':
                end_offsets = [3.75, 3.75]
            else:
                end_offsets = [1.875, 5.625]
            for end_offset in end_offsets:
                control_point1 = 1.875, -18
                control_point2 = 1.875, -18 + 20
                control_point3 = -18 + 20, end_offset
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

            for end_offset in end_offsets:
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
        n_future_data = self.future_n_data(indexs, n)
        all_ref = [current_points] + n_future_data
        tracking_error = tf.concat([tf.stack([ego_xs - ref_point[0],
                                              ego_ys - ref_point[1],
                                              deal_with_phi_diff(ego_phis - ref_point[2]),
                                              ego_vs - 10.], 1)
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
    path = ReferencePath('left')
    xs = np.array([1.875, 1.875], np.float32)
    ys = np.array([-20, 0], np.float32)
    phis = np.array([90, 135], np.float32)
    tracking_error_vector = path.tracking_error_vector(xs, ys, phis, 0)
    print(tracking_error_vector)


def test_model():
    from endtoend import CrossroadEnd2end
    env = CrossroadEnd2end('left', 5)
    model = EnvironmentModel('left', 5)
    obs_list = []
    obs = env.reset()
    done = 0
    while not done:
        obs_list.append(obs)
        action = np.array([0, 0], dtype=np.float32)
        obs, reward, done, info = env.step(action)
        env.render()
    obses = np.stack(obs_list, 0)
    model.reset(obses, 'left')
    print(obses.shape)
    for rollout_step in range(100):
        actions = tf.tile(tf.constant([[0, 0]], dtype=tf.float32), tf.constant([len(obses), 1]))
        obses, rewards, dones = model.rollout_out(actions)
        print(rewards.numpy()[0], dones.numpy()[0])
        model.render()


if __name__ == '__main__':
    test_ref_path()
