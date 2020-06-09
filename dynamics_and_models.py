from collections import OrderedDict
import numpy as np
import tensorflow as tf
import bezier
from math import cos, sin, fabs, pi, sqrt, atan2
import matplotlib.pyplot as plt
from tensorflow.math import logical_and, logical_or
from endtoend_env_utils import rotate_coordination


class VehicleDynamics(object):
    def __init__(self,):
        self.vehicle_params = OrderedDict(C_f=100000.,  # front wheel cornering stiffness [N/rad]
                                          C_r=100000.,  # rear wheel cornering stiffness [N/rad]
                                          a=1.5,  # distance from CG to front axle [m]
                                          b=1.5,  # distance from CG to rear axle [m]
                                          mass=3000.,  # mass [kg]
                                          I_z=4000.,  # Polar moment of inertia at CG [kg*m^2]
                                          miu=0.8,  # tire-road friction coefficient
                                          g=9.81,  # acceleration of gravity [m/s^2]
                                          )

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
        return state_deriv_stack

    def prediction(self, x_1, u_1, frequency, RK):
        if RK == 1:
            f_xu_1 = self.f_xu(x_1, u_1)
            x_next = f_xu_1 / frequency + x_1

        elif RK == 2:
            f_xu_1 = self.f_xu(x_1, u_1)
            K1 = (1 / frequency) * f_xu_1
            x_2 = x_1 + K1
            f_xu_2 = self.f_xu(x_2, u_1)
            K2 = (1 / frequency) * f_xu_2
            x_next = x_1 + (K1 + K2) / 2
        else:
            assert RK == 4
            f_xu_1 = self.f_xu(x_1, u_1)
            K1 = (1 / frequency) * f_xu_1
            x_2 = x_1 + K1 / 2
            f_xu_2 = self.f_xu(x_2, u_1)
            K2 = (1 / frequency) * f_xu_2
            x_3 = x_1 + K2 / 2
            f_xu_3 = self.f_xu(x_3, u_1)
            K3 = (1 / frequency) * f_xu_3
            x_4 = x_1 + K3
            f_xu_4 = self.f_xu(x_4, u_1)
            K4 = (1 / frequency) * f_xu_4
            x_next = x_1 + (K1 + 2 * K2 + 2 * K3 + K4) / 6
        return x_next


class EnvironmentModel(object):  # all tensors
    def __init__(self, task, num_future_data=5):
        self.task = task
        self.vehicle_dynamics = VehicleDynamics()
        self.base_frequency = 10.
        self.obses = None
        self.dones = None
        self.task = None
        self.ref_path = None
        self.num_future_data = num_future_data
        self.expected_vs = 10.

    def reset(self, obses, task):
        self.obses = obses
        self.dones = tf.cast(tf.zeros_like(self.obses[:, 0]), tf.bool)
        self.task = task
        self.ref_path = ReferencePath(task, mode='training')

    def rollout_out(self, actions):  # obses and actions are tensors, think of actions are in range [-1, 1]
        with tf.name_scope('model_step') as scope:
            steer_norm, a_xs_norm = actions[:, 0], actions[:, 1]
            actions = tf.stack([steer_norm * 1.2 * np.pi / 9, a_xs_norm * 3.], 1)
            self.obses = self.compute_next_obses(self.obses, actions)
            rewards = self.compute_rewards(self.obses, actions)
            self.judge_dones(self.obses)

        return self.obses, rewards, self.dones

    def judge_dones(self, obses):
        ego_infos, tracking_infos, veh_infos = obses[:, :8], obses[:, 8:8 + 3 + 3 * self.num_future_data], \
                                               obses[:, 8 + 3 + 3 * self.num_future_data:]
        # rewards related to veh2road collision
        ego_lws = (ego_infos[:, 6] - ego_infos[:, 7]) / 2.
        ego_front_points = tf.cast(ego_infos[:, 3] + ego_lws * tf.cos(ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32), \
                           tf.cast(ego_infos[:, 4] + ego_lws * tf.sin(ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32)
        ego_rear_points = tf.cast(ego_infos[:, 3] - ego_lws * tf.cos(ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32), \
                          tf.cast(ego_infos[:, 4] - ego_lws * tf.sin(ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32)
        rho_ego = ego_infos[0, 7] / 2
        if self.task == 'left':
            dones_good_done = logical_and(logical_and(ego_infos[:, 4]>0, ego_infos[:, 4]<7.5),
                                          ego_infos[:, 3] < -18-5)
            for ego_point in [ego_front_points, ego_rear_points]:
                dones_before1 = logical_and(ego_point[1] < -18, ego_point[0] - 0 < rho_ego)
                dones_before2 = logical_and(ego_point[1] < -18, 3.75 - ego_point[0] < rho_ego)

                middle_cond = logical_and(logical_and(ego_point[0] > -18, ego_point[0] < 18),
                                          logical_and(ego_point[1] > -18, ego_point[1] < 18))
                dones_middle1 = logical_and(middle_cond, 18 - ego_point[1] < rho_ego)
                dones_middle2 = logical_and(middle_cond, 18 - ego_point[0] < rho_ego)
                dones_middle3 = logical_and(logical_and(middle_cond, ego_point[1] > 7.5),
                                            ego_point[0] - (-18) < rho_ego)
                dones_middle4 = logical_and(logical_and(middle_cond, ego_point[1] < 0),
                                            ego_point[0] - (-18) < rho_ego)
                dones_middle5 = logical_and(logical_and(middle_cond, ego_point[0] < 0),
                                            ego_point[1] - (-18) < rho_ego)
                dones_middle6 = logical_and(logical_and(middle_cond, ego_point[0] > 3.75),
                                            ego_point[1] - (-18) < rho_ego)
                dones_after1 = logical_and(ego_point[0] < -18, ego_point[1] - 0 < rho_ego)
                dones_after2 = logical_and(ego_point[0] < -18, 7.5 - ego_point[1] < rho_ego)

                for dones in [dones_before1, dones_before2, dones_middle1, dones_middle2, dones_middle3, dones_middle4,
                              dones_middle5, dones_middle6, dones_after1, dones_after2, dones_good_done]:
                    self.dones = tf.math.logical_or(self.dones, dones)

        elif self.task == 'straight':
            dones_good_done = logical_and(logical_and(ego_infos[:, 3] > 0, ego_infos[:, 3] < 7.5),
                                          ego_infos[:, 4] > 18 + 5)
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
                dones_after1 = logical_and(ego_point[1] > 18, ego_point[0] - 0 < rho_ego)
                dones_after2 = logical_and(ego_point[1] > 18, 7.5 - ego_point[0] < rho_ego)

                for dones in [dones_before1, dones_before2, dones_middle1, dones_middle2, dones_middle3, dones_middle4,
                              dones_middle5, dones_middle6, dones_after1, dones_after2, dones_good_done]:
                    self.dones = tf.math.logical_or(self.dones, dones)
        else:
            assert self.task == 'right'
            dones_good_done = logical_and(logical_and(ego_infos[:, 4] < 0, ego_infos[:, 4] < -7.5),
                                          ego_infos[:, 3] > 18 + 5)
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
                dones_after1 = logical_and(ego_point[0] > 18, 0 - ego_point[1] < rho_ego)
                dones_after2 = logical_and(ego_point[0] > 18, ego_point[1] - (-7.5) < rho_ego)

                for dones in [dones_before1, dones_before2, dones_middle1, dones_middle2, dones_middle3, dones_middle4,
                              dones_middle5, dones_middle6, dones_after1, dones_after2, dones_good_done]:
                    self.dones = tf.math.logical_or(self.dones, dones)

        # rewards related to veh2veh collision
        for veh_index in range(int(tf.shape(veh_infos)[1] / 6)):
            vehs = veh_infos[:, veh_index*6:6 * (veh_index + 1)]
            veh_lws = (vehs[:, 4] - vehs[:, 5]) / 2.
            rho_vehs = vehs[:, 5] / 2.
            veh_front_points = tf.cast(vehs[:, 0] + veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
                               tf.cast(vehs[:, 1] + veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
            veh_rear_points = tf.cast(vehs[:, 0] - veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
                              tf.cast(vehs[:, 1] - veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
            for ego_point in [ego_front_points, ego_rear_points]:
                for veh_point in [veh_front_points, veh_rear_points]:
                    veh2veh_square_dist = tf.square(ego_point[0] - veh_point[0]) + tf.square(
                        ego_point[1] - veh_point[1])
                    self.dones = logical_or(self.dones, tf.sqrt(veh2veh_square_dist) < rho_ego + rho_vehs)

    def compute_rewards(self, obses, actions):
        with tf.name_scope('compute_reward') as scope:

            ego_infos, tracking_infos, veh_infos = obses[:, :8], obses[:, 8:8+3+3*self.num_future_data],\
                                                   obses[:, 8+3+3*self.num_future_data:]
            steers, a_xs = actions[:, 0], actions[:, 1]

            # rewards related to action
            punish_steer = -tf.square(steers)
            punish_a_x = -tf.square(a_xs)

            # rewards related to ego stability
            punish_yaw_rate = -tf.square(ego_infos[:, 2])

            # rewards related to tracking error
            devi_v = -tf.cast(tf.square(ego_infos[:, 0] - self.expected_vs), dtype=tf.float32)
            devi_y = -tf.square(tracking_infos[:, 0])-tf.square(tracking_infos[:, 1])
            devi_phi = -tf.cast(tf.square(tracking_infos[:, 2]*np.pi/180.), dtype=tf.float32)

            # rewards related to veh2road collision
            ego_lws = (ego_infos[:, 6] - ego_infos[:, 7]) / 2.
            ego_front_points = tf.cast(ego_infos[:, 3] + ego_lws * tf.cos(ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32),\
                               tf.cast(ego_infos[:, 4] + ego_lws * tf.sin(ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32)
            ego_rear_points = tf.cast(ego_infos[:, 3] - ego_lws * tf.cos(ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32), \
                               tf.cast(ego_infos[:, 4] - ego_lws * tf.sin(ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32)
            rho_ego = ego_infos[0, 7] / 2.
            zeros = tf.zeros_like(ego_front_points[0])
            if self.task == 'left':
                veh2road = tf.zeros_like(ego_front_points[0])
                for ego_point in [ego_front_points, ego_rear_points]:
                    before1 = tf.where(ego_point[1] < -18, 1/tf.square(ego_point[0] - 0 - rho_ego), zeros)
                    before2 = tf.where(ego_point[1] < -18, 1/tf.square(3.75 - ego_point[0] - rho_ego), zeros)
                    middle_cond = logical_and(logical_and(ego_point[0]>-18, ego_point[0]<18),
                                              logical_and(ego_point[1]>-18, ego_point[1]<18))
                    middle1 = tf.where(middle_cond, 1/tf.square(18 - ego_point[1] - rho_ego), zeros)
                    middle2 = tf.where(middle_cond, 1/tf.square(18 - ego_point[0] - rho_ego), zeros)
                    middle3 = tf.where(logical_and(middle_cond, ego_point[1] > 7.5), 1/tf.square(ego_point[0] - (-18) - rho_ego), zeros)
                    middle4 = tf.where(logical_and(middle_cond, ego_point[1] < 0), 1/tf.square(ego_point[0] - (-18) - rho_ego), zeros)
                    middle5 = tf.where(logical_and(middle_cond, ego_point[0] < 0), 1/tf.square(ego_point[1] - (-18) - rho_ego), zeros)
                    middle6 = tf.where(logical_and(middle_cond, ego_point[0] > 3.75), 1/tf.square(ego_point[1] - (-18) - rho_ego), zeros)
                    after1 = tf.where(ego_point[0] < -18, 1 / tf.square(ego_point[1] - 0 - rho_ego), zeros)
                    after2 = tf.where(ego_point[0] < -18, 1 / tf.square(7.5 - ego_point[1] - rho_ego), zeros)

                    this_point = before1 + before2 + middle1 + middle2 + middle3 + middle4 + middle5 + middle6 + after1 + after2
                    veh2road += this_point

            elif self.task == 'straight':
                veh2road = tf.zeros_like(ego_front_points[0])
                for ego_point in [ego_front_points, ego_rear_points]:
                    before1 = tf.where(ego_point[1] < -18, 1 / tf.square(ego_point[0] - 0 - rho_ego), zeros)
                    before2 = tf.where(ego_point[1] < -18, 1 / tf.square(3.75 - ego_point[0] - rho_ego), zeros)
                    middle_cond = logical_and(logical_and(ego_point[0] > -18, ego_point[0] < 18),
                                              logical_and(ego_point[1] > -18, ego_point[1] < 18))
                    middle1 = tf.where(middle_cond, 1 / tf.square(ego_point[0] - (-18) - rho_ego), zeros)
                    middle2 = tf.where(middle_cond, 1 / tf.square(18 - ego_point[0] - rho_ego), zeros)
                    middle3 = tf.where(logical_and(middle_cond, ego_point[0] < 0), 1 / tf.square(18 - ego_point[1] - rho_ego), zeros)
                    middle4 = tf.where(logical_and(middle_cond, ego_point[0] > 7.5), 1 / tf.square(18 - ego_point[1] - rho_ego), zeros)
                    middle5 = tf.where(logical_and(middle_cond, ego_point[0] < 0), 1 / tf.square(ego_point[1] - (-18) - rho_ego), zeros)
                    middle6 = tf.where(logical_and(middle_cond, ego_point[0] > 3.75), 1 / tf.square(ego_point[1] - (-18) - rho_ego), zeros)
                    after1 = tf.where(ego_point[1] > 18, 1 / tf.square(ego_point[0] - 0 - rho_ego), zeros)
                    after2 = tf.where(ego_point[1] > 18, 1 / tf.square(7.5 - ego_point[0] - rho_ego), zeros)
                    this_point = before1 + before2 + middle1 + middle2 + middle3 + middle4 + middle5 + middle6 + after1 + after2
                    veh2road += this_point

            else:
                veh2road = tf.zeros_like(ego_front_points[0])
                assert self.task == 'right'
                for ego_point in [ego_front_points, ego_rear_points]:
                    before1 = tf.where(ego_point[1] < -18, 1 / tf.square(ego_point[0] - 3.75 - rho_ego), zeros)
                    before2 = tf.where(ego_point[1] < -18, 1 / tf.square(7.5 - ego_point[0] - rho_ego), zeros)
                    middle_cond = logical_and(logical_and(ego_point[0] > -18, ego_point[0] < 18),
                                              logical_and(ego_point[1] > -18, ego_point[1] < 18))
                    middle1 = tf.where(middle_cond, 1 / tf.square(ego_point[0] - (-18) - rho_ego), zeros)
                    middle2 = tf.where(middle_cond, 1 / tf.square(18 - ego_point[1] - rho_ego), zeros)
                    middle3 = tf.where(logical_and(middle_cond, ego_point[1]>0), 1 / tf.square(18 - ego_point[0] - rho_ego), zeros)
                    middle4 = tf.where(logical_and(middle_cond, ego_point[1]<-7.5), 1 / tf.square(18 - ego_point[0] - rho_ego), zeros)
                    middle5 = tf.where(logical_and(middle_cond, ego_point[0]>7.5), 1 / tf.square(ego_point[1] - (-18) - rho_ego), zeros)
                    middle6 = tf.where(logical_and(middle_cond, ego_point[0]<3.75), 1 / tf.square(ego_point[1] - (-18) - rho_ego), zeros)
                    after1 = tf.where(ego_point[0]>18, 1 / tf.square(0 - ego_point[1] - rho_ego), zeros)
                    after2 = tf.where(ego_point[0]>18, 1 / tf.square(ego_point[1] - (-7.5) - rho_ego), zeros)

                    this_point = before1 + before2 + middle1 + middle2 + middle3 + middle4 + middle5 + middle6 + after1 + after2
                    veh2road += this_point

            # rewards related to veh2veh collision
            veh2veh = tf.zeros_like(ego_front_points[0])
            for veh_index in range(int(tf.shape(veh_infos)[1]/6)):
                vehs = veh_infos[:, veh_index*6:6*(veh_index+1)]
                veh_lws = (vehs[:, 4] - vehs[:, 5]) / 2.
                rho_vehs = vehs[:, 5] / 2.
                veh_front_points = tf.cast(vehs[:, 0] + veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32),\
                                   tf.cast(vehs[:, 1] + veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
                veh_rear_points = tf.cast(vehs[:, 0] - veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
                                  tf.cast(vehs[:, 1] - veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
                for ego_point in [ego_front_points, ego_rear_points]:
                    for veh_point in [veh_front_points, veh_rear_points]:
                        veh2veh_dist = tf.sqrt(tf.square(ego_point[0] - veh_point[0])+tf.square(ego_point[1] - veh_point[1])) -\
                                       tf.convert_to_tensor(rho_ego + rho_vehs, dtype=tf.float32)
                        veh2veh += 1/tf.square(veh2veh_dist)

            # reward_dict = dict(punish_steer=punish_steer.numpy()[0],
            #                    punish_a_x=punish_a_x.numpy()[0],
            #                    punish_yaw_rate=punish_yaw_rate.numpy()[0],
            #                    devi_v=devi_v.numpy()[0],
            #                    devi_y=devi_y.numpy()[0],
            #                    devi_phi=devi_phi.numpy()[0],
            #                    veh2road=-veh2road.numpy()[0],
            #                    veh2veh=-veh2veh.numpy()[0])
            # print(reward_dict)
            veh2road = tf.where(veh2road > 10000., 10000.*tf.ones_like(veh2road), veh2road)
            veh2veh = tf.where(veh2road > 10000., 10000.*tf.ones_like(veh2veh), veh2veh)

            rewards = 0.01 * devi_v + 0.04 * devi_y + 0.1 * devi_phi + 0.02 * punish_yaw_rate + \
                      0.05 * punish_steer + 0.0005 * punish_a_x - 0.1*veh2road - 0.1*veh2veh
            rewards = tf.cast(tf.math.logical_not(self.dones), tf.float32) * rewards
            return rewards

    def compute_next_obses(self, obses, actions):
        ego_infos, tracking_infos, veh_infos = obses[:, :8], obses[:, 8:8 + 3 + 3 * self.num_future_data], \
                                               obses[:, 8 + 3 + 3 * self.num_future_data:]
        ego_next_infos_except_lw = self.vehicle_dynamics.prediction(ego_infos[:, :6], actions, self.base_frequency, 1)
        ego_next_lw = ego_infos[:, 6:8]
        next_tracking_infos = self.ref_path.tracking_error_vector(ego_next_infos_except_lw[:, 3],
                                                                  ego_next_infos_except_lw[:, 4],
                                                                  ego_next_infos_except_lw[:, 5],
                                                                  self.num_future_data)
        next_veh_infos = self.veh_predict(veh_infos)
        next_obses = tf.concat([ego_next_infos_except_lw, ego_next_lw, next_tracking_infos, next_veh_infos], 1)
        return next_obses

    def veh_predict(self, veh_infos):
        if self.task == 'left':
            veh_mode_list = ['dl']*2 + ['du']*2 + ['ud']*3 + ['ul']*3
        elif self.task == 'straight':
            veh_mode_list = ['dl']*2 + ['du']*2 + ['ud']*2 + ['ru']*3 + ['ur']*3
        else:
            assert self.task == 'right'
            veh_mode_list = ['dr']*2 + ['ur']*3 + ['lr']*3

        predictions_to_be_concat = []

        for vehs_index in range(len(veh_mode_list)):
            predictions_to_be_concat.append(self.predict_for_a_mode(veh_infos[:, vehs_index*6:(vehs_index+1)*6],
                                                                    veh_mode_list[vehs_index]))
        return tf.concat(predictions_to_be_concat, 1)

    def predict_for_a_mode(self, vehs, mode):
        xs, ys, vs, phis, ls, ws = vehs[:, 0], vehs[:, 1], vehs[:, 2], vehs[:, 3], vehs[:, 4], vehs[:, 5]
        phis_rad = phis * np.pi / 180.
        middle_cond = logical_and(logical_and(xs > -18, xs < 18),
                                  logical_and(ys > -18, ys < 18))
        zeros = tf.zeros_like(xs)

        xs_delta = vs / self.base_frequency * tf.cos(phis_rad)
        ys_delta = vs / self.base_frequency * tf.sin(phis_rad)

        if mode in ['dl', 'rd', 'ur', 'lu']:
            phis_rad_delta = tf.where(middle_cond, (vs/19.875)/self.base_frequency, zeros)
        elif mode in ['dr', 'ru', 'ul', 'ld']:
            phis_rad_delta = tf.where(middle_cond, -(vs/12.375)/self.base_frequency, zeros)
        else:
            phis_rad_delta = zeros
        next_xs, next_ys, next_vs, next_phis_rad, next_ls, next_ws = xs + xs_delta, ys + ys_delta, vs, phis_rad + phis_rad_delta, ls, ws
        next_phis_rad = tf.where(next_phis_rad > np.pi, next_phis_rad - 2 * np.pi, next_phis_rad)
        next_phis_rad = tf.where(next_phis_rad <= -np.pi, next_phis_rad + 2 * np.pi, next_phis_rad)
        next_phis = next_phis_rad * 180 / np.pi
        return tf.stack([next_xs, next_ys, next_vs, next_phis, next_ls, next_ws], 1)

    def render(self, mode='human'):
        if mode == 'human':
            # plot basic map
            square_length = 36
            extension = 40
            lane_width = 3.75
            dotted_line_style = '--'

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

            obses = self.obses.numpy()
            ego_info, tracing_info, vehs_info = obses[0, :8], obses[0, 8:8+3+3*self.num_future_data],\
                                                obses[0, 8+3+3*self.num_future_data:]
            # plot cars
            for veh_index in range(int(len(vehs_info)/6)):
                veh = vehs_info[6*veh_index:6*(veh_index+1)]
                x = veh[0]
                y = veh[1]
                a = veh[3]
                l = veh[4]
                w = veh[5]
                if is_in_plot_area(x, y):
                    draw_rotate_rec(x, y, a, l, w, 'black')

            # plot own car
            ego_x = ego_info[3]
            ego_y = ego_info[4]
            ego_a = ego_info[5]
            ego_l = ego_info[6]
            ego_w = ego_info[7]
            draw_rotate_rec(ego_x, ego_y, ego_a, ego_l, ego_w, 'red')

            # plot planed trj
            ax.plot(self.ref_path.path[0], self.ref_path.path[1], color='g')

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
                start_straight_line_x = 1.875 * np.ones(shape=(sl*meter_pointnum_ratio,), dtype=np.float32)
                start_straight_line_y = np.linspace(-18-sl, -18, sl*meter_pointnum_ratio, dtype=np.float32)
                end_straight_line_x = np.linspace(-18, -18-sl, sl*meter_pointnum_ratio, dtype=np.float32)
                end_straight_line_y = end_offset * np.ones(shape=(sl*meter_pointnum_ratio,), dtype=np.float32)
                planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x),\
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
                control_point3 = end_offset, 18-10
                control_point4 = end_offset, 18

                node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                                          [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]]
                                         , dtype=np.float32)
                curve = bezier.Curve(node, degree=3)
                s_vals = np.linspace(0, 1.0, 36*meter_pointnum_ratio)
                trj_data = curve.evaluate_multi(s_vals)
                trj_data = trj_data.astype(np.float32)
                start_straight_line_x = 1.875 * np.ones(shape=(sl*meter_pointnum_ratio,), dtype=np.float32)
                start_straight_line_y = np.linspace(-18-sl, -18, sl*meter_pointnum_ratio, dtype=np.float32)
                end_straight_line_x = end_offset * np.ones(shape=(sl*meter_pointnum_ratio,), dtype=np.float32)
                end_straight_line_y = np.linspace(18, 18+sl, sl*meter_pointnum_ratio, dtype=np.float32)
                planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x),\
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
                control_point3 = 18-10, end_offset
                control_point4 = 18, end_offset

                node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                                          [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]],
                                         dtype=np.float32)
                curve = bezier.Curve(node, degree=3)
                s_vals = np.linspace(0, 1.0, 13*meter_pointnum_ratio)
                trj_data = curve.evaluate_multi(s_vals)
                trj_data = trj_data.astype(np.float32)
                start_straight_line_x = 5.625 * np.ones(shape=(sl*meter_pointnum_ratio,), dtype=np.float32)
                start_straight_line_y = np.linspace(-18-sl, -18, sl*meter_pointnum_ratio, dtype=np.float32)
                end_straight_line_x = np.linspace(18, 18+sl, sl*meter_pointnum_ratio, dtype=np.float32)
                end_straight_line_y = end_offset * np.ones(shape=(sl*meter_pointnum_ratio,), dtype=np.float32)
                planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x),\
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
            current_indexs += 50
            current_indexs = tf.where(current_indexs >= len(self.path[0])-2, len(self.path[0])-2, current_indexs)
            future_data_list.append(self.indexs2points(current_indexs))
        return future_data_list

    def indexs2points(self, indexs):
        points = tf.gather(self.path[0], indexs), \
                 tf.gather(self.path[1], indexs), \
                 tf.gather(self.path[2], indexs)

        return points[0], points[1], points[2]

    def tracking_error_vector(self, ego_xs, ego_ys, ego_phis, n):
        indexs, current_points = self.find_closest_point(ego_xs, ego_ys)
        n_future_data = self.future_n_data(indexs, n)
        all_ref = [current_points] + n_future_data
        tracking_error = tf.concat([tf.stack([ego_xs - ref_point[0], ego_ys - ref_point[1],
                                              deal_with_phi_diff(ego_phis - ref_point[2])], 1)
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
    path.plot_path(1.875,0)


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
    env = CrossroadEnd2end('right', 5)
    model = EnvironmentModel('right', 5)
    obs_list = []
    obs = env.reset()
    done = 0
    while not done:
        obs_list.append(obs)
        action = np.array([0, 0], dtype=np.float32)
        obs, reward, done, info = env.step(action)
        # env.render()
    obses = np.stack(obs_list, 0)
    model.reset(obses, 'right')
    for rollout_step in range(100):
        actions = tf.zeros_like(obses)
        obses, rewards, dones = model.rollout_out(actions)
        print(rewards.numpy()[0], dones.numpy()[0])
        model.render()


if __name__ == '__main__':
    test_ref_path()




