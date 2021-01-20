#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/08
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: dynamics_and_models.py
# =====================================

from math import pi, cos, sin

import bezier
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import logical_and

# gym.envs.user_defined.toyota_env.
from endtoend_env_utils import rotate_coordination, L, W, VEHICLE_MODE_LIST, EXPECTED_V,\
    CROSSROAD_D_HEIGHT,CROSSROAD_U_HEIGHT,CROSSROAD_HALF_WIDTH,\
    LANE_WIDTH_LR,LANE_WIDTH_UD, LANE_NUMBER_LR, LANE_NUMBER_UD, T, VEH_NUM



class VehicleDynamics(object):
    def __init__(self, ):
        self.vehicle_params = dict(C_f=-155495.0,  # front wheel cornering stiffness [N/rad]
                                   C_r=-155495.0,  # rear wheel cornering stiffness [N/rad]
                                   a=1.19,  # distance from CG to front axle [m]
                                   b=1.46,  # distance from CG to rear axle [m]
                                   mass=1520.,  # mass [kg]
                                   I_z=2642.,  # Polar moment of inertia at CG [kg*m^2]
                                   miu=0.8,  # tire-road friction coefficient
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


class EnvironmentModel(object):  # all tensors
    def __init__(self, training_task, num_future_data=0):
        self.task = training_task
        self.mode = None
        self.vehicle_dynamics = VehicleDynamics()
        self.base_frequency = 10.
        self.obses = None
        self.ego_params = None
        self.actions = None
        self.ref_path = ReferencePath(self.task)
        self.ref_indexes = None
        self.num_future_data = num_future_data
        self.exp_v = EXPECTED_V
        self.reward_info = None
        self.ego_info_dim = 6
        self.per_veh_info_dim = 4
        self.per_tracking_info_dim = 3
        self.per_veh_constraint_dim = 4
        self.vehicle_num = VEH_NUM[self.task]
        self.road_constraints_num = 4
        self.collision_constraints_num = self.per_veh_constraint_dim * self.vehicle_num
        self.constraints_num = self.collision_constraints_num + self.road_constraints_num

    def reset(self, obses, ref_indexes=None):  # input are all tensors
        self.obses = obses
        self.ref_indexes = ref_indexes
        self.actions = None
        self.reward_info = None

    def add_traj(self, obses, trajectory, mode=None):
        self.obses = obses
        self.ref_path = trajectory
        self.mode = mode

    def rollout_out(self, actions) -> object:  # obses and actions are tensors, think of actions are in range [-1, 1]
        with tf.name_scope('model_step') as scope:
            self.actions = self._action_transformation_for_end2end(actions)
            rewards, constraints, real_punish_term, veh2veh4real, veh2road4real, _ \
                = self.compute_rewards(self.obses, self.actions)
            self.obses = self.compute_next_obses(self.obses, self.actions)
            # self.reward_info.update({'final_rew': rewards.numpy()[0]})

        return self.obses, rewards, constraints, real_punish_term, veh2veh4real, veh2road4real

    def safety_calculation(self, obs, actions):
        # judge collision
        actions = self._action_transformation_for_end2end(actions)
        obs_next = self.compute_next_obses(obs, actions)           # for relative state
        # obs_next = self.convert_vehs_to_abso(obs_next)             # for absolute state
        with tf.name_scope('compute_reward') as scope:
            ego_infos, tracking_infos, veh_infos = obs_next[:, :self.ego_info_dim], \
                                                   obs_next[:,
                                                   self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (
                                                               self.num_future_data + 1)], \
                                                   obs_next[:, self.ego_info_dim + self.per_tracking_info_dim * (
                                                               self.num_future_data + 1):]
            # rewards related to veh2veh collision
            ego_lws = (L - W) / 2.
            ego_front_points = tf.cast(ego_infos[:, 3] + ego_lws * tf.cos(ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32), \
                               tf.cast(ego_infos[:, 4] + ego_lws * tf.sin(ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32)
            ego_rear_points = tf.cast(ego_infos[:, 3] - ego_lws * tf.cos(ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32), \
                              tf.cast(ego_infos[:, 4] - ego_lws * tf.sin(ego_infos[:, 5] * np.pi / 180.), dtype=tf.float32)
            veh2veh4real = tf.zeros_like(veh_infos[:, 0])
            veh2veh4training = tf.zeros_like(veh_infos[:, 0])

            for veh_index in range(int(tf.shape(veh_infos)[1] / self.per_veh_info_dim)):
                vehs = veh_infos[:, veh_index * self.per_veh_info_dim:(veh_index + 1) * self.per_veh_info_dim]
                veh_lws = (L - W) / 2.
                veh_front_points = tf.cast(vehs[:, 0] + veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
                                   tf.cast(vehs[:, 1] + veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
                veh_rear_points = tf.cast(vehs[:, 0] - veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
                                  tf.cast(vehs[:, 1] - veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
                for ego_point in [ego_front_points, ego_rear_points]:
                    for veh_point in [veh_front_points, veh_rear_points]:
                        veh2veh_dist = tf.sqrt(tf.square(ego_point[0] - veh_point[0]) + tf.square(ego_point[1] - veh_point[1]))
                        veh2veh4training += tf.where(veh2veh_dist-3.5 < 0, tf.square(veh2veh_dist-3.5), tf.zeros_like(veh_infos[:, 0]))
                        veh2veh4real += tf.where(veh2veh_dist-2.5 < 0, tf.square(veh2veh_dist-2.5), tf.zeros_like(veh_infos[:, 0]))
        # obs_next = self.convert_vehs_to_rela(obs_next)
        return obs_next, veh2veh4real

    def _action_transformation_for_end2end(self, actions):  # [-1, 1] # TODO:
        actions = tf.clip_by_value(actions, -1.05, 1.05)
        steer_norm, a_xs_norm = actions[:, 0], actions[:, 1]
        steer_scale, a_xs_scale = 0.4 * steer_norm, 2.25 * a_xs_norm-0.75
        return tf.stack([steer_scale, a_xs_scale], 1)


    def compute_rewards(self, obses, actions):
        # obses = self.convert_vehs_to_abso(obses)
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
            ego_v = ego_infos[:, 0]

            punish_absolute_v = tf.where(ego_v>EXPECTED_V, -tf.square(ego_infos[:, 0]), tf.zeros_like(ego_v) )
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
            veh2veh4real = tf.zeros_like(veh_infos[:, 0])
            veh2veh4training = tf.zeros_like(veh_infos[:, 0])

            # for veh_index in range(int(tf.shape(veh_infos)[1] / self.per_veh_info_dim)):
            #     vehs = veh_infos[:, veh_index * self.per_veh_info_dim:(veh_index + 1) * self.per_veh_info_dim]
            #     veh_lws = (L - W) / 2.
            #     veh_front_points = tf.cast(vehs[:, 0] + veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
            #                        tf.cast(vehs[:, 1] + veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
            #     veh_rear_points = tf.cast(vehs[:, 0] - veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
            #                       tf.cast(vehs[:, 1] - veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
            #     for ego_point in [ego_front_points, ego_rear_points]:
            #         for veh_point in [veh_front_points, veh_rear_points]:
            #             veh2veh_dist = tf.sqrt(tf.square(ego_point[0] - veh_point[0]) + tf.square(ego_point[1] - veh_point[1]))
            #             veh2veh4training += tf.where(veh2veh_dist-3.5 < 0, tf.square(veh2veh_dist-3.5), tf.zeros_like(veh_infos[:, 0]))
            #             veh2veh4real += tf.where(veh2veh_dist-2.5 < 0, tf.square(veh2veh_dist-2.5), tf.zeros_like(veh_infos[:, 0]))

            constraints_list = tf.TensorArray(tf.float32, size=self.constraints_num)
            veh2veh_dist = tf.zeros_like(veh_infos[:, 0])
            constraint_index = 0
            for veh_index in range(int(tf.shape(veh_infos)[1] / self.per_veh_info_dim)):
                vehs = veh_infos[:, veh_index * self.per_veh_info_dim:(veh_index + 1) * self.per_veh_info_dim]
                veh_lws = (L - W) / 2.
                veh_front_points = tf.cast(vehs[:, 0] + veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
                                   tf.cast(vehs[:, 1] + veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
                veh_rear_points = tf.cast(vehs[:, 0] - veh_lws * tf.cos(vehs[:, 3] * np.pi / 180.), dtype=tf.float32), \
                                  tf.cast(vehs[:, 1] - veh_lws * tf.sin(vehs[:, 3] * np.pi / 180.), dtype=tf.float32)
                for ego_point_index, ego_point in enumerate([ego_front_points, ego_rear_points]):
                    for other_point_index, veh_point in enumerate([veh_front_points, veh_rear_points]):
                        constraint_index = veh_index * self.per_veh_constraint_dim + ego_point_index * 2 + other_point_index
                        veh2veh_dist = -tf.sqrt(
                            tf.square(ego_point[0] - veh_point[0]) + tf.square(ego_point[1] - veh_point[1])) + 2.5
                        veh2veh4real += tf.where(veh2veh_dist > 1, tf.square(veh2veh_dist - 1), tf.zeros_like(veh_infos[:, 0]))
                        # update veh2veh dist last
                        constraints_list = constraints_list.write(constraint_index, veh2veh_dist)
                        # print(constraint_index)
                        # tf.print(veh2veh_dist)
                        # tf.print(constraint_index)
                        # tf.print(constraints_list.read(constraint_index))


            veh2road4real = tf.zeros_like(veh_infos[:, 0])
            veh2road4training = tf.zeros_like(veh_infos[:, 0])
            veh2road_dist = tf.zeros_like(veh_infos[:, 0])
            if self.task == 'left':
                for i,ego_point in enumerate([ego_front_points, ego_rear_points]):

                    constraint_index = self.collision_constraints_num + 2 * i
                    veh2road_dist = tf.where(ego_point[1] < - CROSSROAD_D_HEIGHT, -ego_point[0]+1, tf.zeros_like(veh2road_dist))
                    constraints_list = constraints_list.write(constraint_index, veh2road_dist)

                    constraint_index = self.collision_constraints_num + 2 * i + 1
                    veh2road_dist = tf.where(ego_point[1] < - CROSSROAD_D_HEIGHT, -LANE_WIDTH_UD + ego_point[0] + 1,
                                             tf.zeros_like(veh2road_dist))
                    constraints_list = constraints_list.write(constraint_index, veh2road_dist)

                    veh2road4real += tf.where(logical_and(ego_point[1] < -CROSSROAD_D_HEIGHT, ego_point[0] < 1),
                                         tf.square(ego_point[0] - 1), tf.zeros_like(veh_infos[:, 0]))
                    veh2road4real += tf.where(logical_and(ego_point[1] < -CROSSROAD_D_HEIGHT, LANE_WIDTH_UD - ego_point[0] < 1),
                                         tf.square(LANE_WIDTH_UD - ego_point[0] - 1), tf.zeros_like(veh_infos[:, 0]))

            elif self.task == 'straight':
                for ego_point in [ego_front_points, ego_rear_points]:
                    veh2road4training += tf.where(logical_and(ego_point[1] < -CROSSROAD_D_HEIGHT, ego_point[0] - 0 < 1),
                                         tf.square(ego_point[0] -1), tf.zeros_like(veh_infos[:, 0]))
                    veh2road4training += tf.where(logical_and(ego_point[1] < -CROSSROAD_D_HEIGHT, LANE_WIDTH_UD-ego_point[0] < 1),
                                         tf.square(LANE_WIDTH_UD-ego_point[0] - 1), tf.zeros_like(veh_infos[:, 0]))
                    # veh2road4training += tf.where(logical_and(ego_point[1] > CROSSROAD_U_HEIGHT, LANE_WIDTH_UD * LANE_NUMBER_UD - ego_point[0] < 1),
                    #                      tf.square(LANE_WIDTH_UD * LANE_NUMBER_UD - ego_point[0] - 1), tf.zeros_like(veh_infos[:, 0]))
                    # veh2road4training += tf.where(logical_and(ego_point[1] > CROSSROAD_U_HEIGHT, ego_point[0] - 0 < 1),
                    #                      tf.square(ego_point[0] - 0 - 1), tf.zeros_like(veh_infos[:, 0]))

                    veh2road4real += tf.where(logical_and(ego_point[1] < -CROSSROAD_D_HEIGHT, ego_point[0]-0 < 1),
                                                  tf.square(ego_point[0] - 1), tf.zeros_like(veh_infos[:, 0]))
                    veh2road4real += tf.where(
                        logical_and(ego_point[1] < -CROSSROAD_D_HEIGHT, LANE_WIDTH_UD - ego_point[0] < 1),
                        tf.square(LANE_WIDTH_UD - ego_point[0] - 1), tf.zeros_like(veh_infos[:, 0]))
                    # veh2road4real += tf.where(
                    #     logical_and(ego_point[1] > CROSSROAD_U_HEIGHT, LANE_WIDTH_UD * LANE_NUMBER_UD - ego_point[0] < 1),
                    #     tf.square(LANE_WIDTH_UD * LANE_NUMBER_UD - ego_point[0] - 1), tf.zeros_like(veh_infos[:, 0]))
                    # veh2road4real += tf.where(logical_and(ego_point[1] > CROSSROAD_U_HEIGHT, ego_point[0] - 0 < 1),
                    #                               tf.square(ego_point[0] - 0 - 1), tf.zeros_like(veh_infos[:, 0]))
            else:
                assert self.task == 'right'
                for ego_point in [ego_front_points, ego_rear_points]:
                    veh2road4training += tf.where(logical_and(ego_point[1] < -CROSSROAD_D_HEIGHT, ego_point[0] - (LANE_NUMBER_UD-1) * LANE_WIDTH_UD < 1),
                                         tf.square(ego_point[0] - (LANE_NUMBER_UD-1) * LANE_WIDTH_UD - 1), tf.zeros_like(veh_infos[:, 0]))
                    veh2road4training += tf.where(logical_and(ego_point[1] < -CROSSROAD_D_HEIGHT, LANE_NUMBER_UD * LANE_WIDTH_UD-ego_point[0] < 1),
                                         tf.square(LANE_NUMBER_UD * LANE_WIDTH_UD-ego_point[0] - 1), tf.zeros_like(veh_infos[:, 0]))
                    # veh2road4training += tf.where(logical_and(ego_point[0] > CROSSROAD_HALF_WIDTH, 0 - ego_point[1] < 1),
                    #                      tf.square(0 - ego_point[1] - 1), tf.zeros_like(veh_infos[:, 0]))
                    # veh2road4training += tf.where(logical_and(ego_point[0] > CROSSROAD_HALF_WIDTH, ego_point[1] - (-LANE_WIDTH_LR * LANE_NUMBER_LR) < 1),
                    #                      tf.square(ego_point[1] - (-LANE_WIDTH_LR * LANE_NUMBER_LR) - 1), tf.zeros_like(veh_infos[:, 0]))

                    veh2road4real += tf.where(
                        logical_and(ego_point[1] < -CROSSROAD_D_HEIGHT, ego_point[0] - (LANE_NUMBER_UD-1) * LANE_WIDTH_UD < 1),
                        tf.square(ego_point[0] - (LANE_NUMBER_UD-1) * LANE_WIDTH_UD), tf.zeros_like(veh_infos[:, 0]))
                    veh2road4real += tf.where(
                        logical_and(ego_point[1] < -CROSSROAD_D_HEIGHT, LANE_NUMBER_UD * LANE_WIDTH_UD - ego_point[0] < 1),
                        tf.square(LANE_NUMBER_UD * LANE_WIDTH_UD - ego_point[0] - 1), tf.zeros_like(veh_infos[:, 0]))
                    # veh2road4real += tf.where(logical_and(ego_point[0] > CROSSROAD_HALF_WIDTH, 0 - ego_point[1] < 1),
                    #                               tf.square(0 - ego_point[1] - 1), tf.zeros_like(veh_infos[:, 0]))
                    # veh2road4real += tf.where(
                    #     logical_and(ego_point[0] > CROSSROAD_HALF_WIDTH, ego_point[1] - (-LANE_WIDTH_LR * LANE_NUMBER_LR) < 1),
                    #     tf.square(ego_point[1] - (-LANE_WIDTH_LR * LANE_NUMBER_LR) - 1), tf.zeros_like(veh_infos[:, 0]))
            constraints = tf.transpose(constraints_list.stack())
            abs_v_coeffi = 0.01
            rewards = 0.2 * devi_v + 0.8 * devi_y + 30 * devi_phi + 0.02 * punish_yaw_rate + \
                      5 * punish_steer + 0.05 * punish_a_x + abs_v_coeffi * punish_absolute_v
            real_punish_term = veh2veh4real + veh2road4real

            reward_dict = dict(punish_steer=punish_steer,
                               punish_a_x=punish_a_x,
                               punish_yaw_rate=punish_yaw_rate,
                               devi_v=devi_v,
                               devi_y=devi_y,
                               devi_phi=devi_phi,
                               abs_v=punish_absolute_v,
                               scaled_punish_steer=5 * punish_steer,
                               scaled_punish_a_x=0.05 * punish_a_x,
                               scaled_punish_yaw_rate=0.02 * punish_yaw_rate,
                               scaled_devi_v=0.2 * devi_v,
                               scaled_devi_y=0.8 * devi_y,
                               scaled_devi_phi=30 * devi_phi,
                               scaled_abs_v=abs_v_coeffi * punish_absolute_v,
                               # veh2veh4training=veh2veh4training,
                               # veh2road4training=veh2road4training,
                               veh2veh4real=veh2veh4real,
                               veh2road4real=veh2road4real,
                               )

            return rewards, constraints, real_punish_term, veh2veh4real, veh2road4real, reward_dict

    def compute_next_obses(self, obses, actions):
        # obses = self.convert_vehs_to_abso(obses)
        ego_infos, tracking_infos, veh_infos = obses[:, :self.ego_info_dim],\
                                               obses[:, self.ego_info_dim:
                                                        self.ego_info_dim + self.per_tracking_info_dim * (
                                                                                         self.num_future_data + 1)], \
                                               obses[:, self.ego_info_dim + self.per_tracking_info_dim * (
                                                           self.num_future_data + 1):]

        next_ego_infos = self.ego_predict(ego_infos, actions)
        # different for training and selecting
        if self.mode == 'selecting':
            next_tracking_infos = self.ref_path.tracking_error_vector(next_ego_infos[:, 3],
                                                                      next_ego_infos[:, 4],
                                                                      next_ego_infos[:, 5],
                                                                      next_ego_infos[:, 0],
                                                                      self.num_future_data)
        else:
            # next_tracking_infos = self.tracking_error_predict(ego_infos, tracking_infos, actions)
            next_tracking_infos = tf.zeros(shape=(len(next_ego_infos),
                                                  (self.num_future_data+1)*self.per_tracking_info_dim))
            ref_indexes = tf.expand_dims(self.ref_indexes, axis=1)
            for ref_idx, path in enumerate(self.ref_path.path_list):
                self.ref_path.path = path
                tracking_info_4_this_ref_idx = self.ref_path.tracking_error_vector(next_ego_infos[:, 3],
                                                                                   next_ego_infos[:, 4],
                                                                                   next_ego_infos[:, 5],
                                                                                   next_ego_infos[:, 0],
                                                                                   self.num_future_data)
                next_tracking_infos = tf.where(ref_indexes == ref_idx, tracking_info_4_this_ref_idx,
                                               next_tracking_infos)

        next_veh_infos = self.veh_predict(veh_infos)
        next_obses = tf.concat([next_ego_infos, next_tracking_infos, next_veh_infos], 1)
        # next_obses = self.convert_vehs_to_rela(next_obses)
        return next_obses

    # def convert_vehs_to_rela(self, obs_abso):
    #     ego_infos, tracking_infos, veh_infos = obs_abso[:, :self.ego_info_dim], \
    #                                            obs_abso[:, self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (
    #                                                      self.num_future_data + 1)], \
    #                                            obs_abso[:, self.ego_info_dim + self.per_tracking_info_dim * (
    #                                                        self.num_future_data + 1):]
    #     ego_x, ego_y = ego_infos[:, 3], ego_infos[:, 4]
    #     ego = tf.tile(tf.stack([ego_x, ego_y, tf.zeros_like(ego_x), tf.zeros_like(ego_x)], 1),
    #                   (1, int(tf.shape(veh_infos)[1]/self.per_veh_info_dim)))
    #     vehs_rela = veh_infos - ego
    #     out = tf.concat([ego_infos, tracking_infos, vehs_rela], 1)
    #     return out

    # def convert_vehs_to_abso(self, obs_rela):
    #     ego_infos, tracking_infos, veh_rela = obs_rela[:, :self.ego_info_dim], \
    #                                            obs_rela[:, self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (
    #                                                    self.num_future_data + 1)], \
    #                                            obs_rela[:, self.ego_info_dim + self.per_tracking_info_dim * (
    #                                                    self.num_future_data + 1):]
    #     ego_x, ego_y = ego_infos[:, 3], ego_infos[:, 4]
    #     ego = tf.tile(tf.stack([ego_x, ego_y, tf.zeros_like(ego_x), tf.zeros_like(ego_x)], 1),
    #                   (1, int(tf.shape(veh_rela)[1] / self.per_veh_info_dim)))
    #     vehs_abso = veh_rela + ego
    #     out = tf.concat([ego_infos, tracking_infos, vehs_abso], 1)
    #     return out

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
        veh_mode_list = VEHICLE_MODE_LIST[self.task] #TODO: temp
        predictions_to_be_concat = []

        for vehs_index in range(len(veh_mode_list)):
            predictions_to_be_concat.append(self.predict_for_a_mode(
                veh_infos[:, vehs_index * self.per_veh_info_dim:(vehs_index + 1) * self.per_veh_info_dim],
                veh_mode_list[vehs_index]))
        return tf.concat(predictions_to_be_concat, 1)

    def predict_for_a_mode(self, vehs, mode):
        veh_xs, veh_ys, veh_vs, veh_phis = vehs[:, 0], vehs[:, 1], vehs[:, 2], vehs[:, 3]
        veh_phis_rad = veh_phis * np.pi / 180.
        zeros = tf.zeros_like(veh_xs)

        veh_xs_delta = veh_vs / self.base_frequency * tf.cos(veh_phis_rad)
        veh_ys_delta = veh_vs / self.base_frequency * tf.sin(veh_phis_rad)

        if mode in ['dl', 'rd', 'ur', 'lu']: #TODO: temp predict Psi
            middle_cond = logical_and(logical_and(veh_xs > -CROSSROAD_HALF_WIDTH, veh_xs < CROSSROAD_HALF_WIDTH),
                                      logical_and(veh_ys > -CROSSROAD_HALF_WIDTH, veh_ys < CROSSROAD_HALF_WIDTH))
            veh_phis_rad_delta = tf.where(middle_cond, (veh_vs / (CROSSROAD_HALF_WIDTH+0.5*LANE_WIDTH_UD)) / self.base_frequency, zeros)

        elif mode in ['dr', 'ld']:
            middle_cond = logical_and(logical_and(veh_xs > -20.55, veh_xs < 20.55),
                                      logical_and(veh_ys > -CROSSROAD_D_HEIGHT, veh_ys < 0))
            veh_phis_rad_delta = tf.where(middle_cond, -(veh_vs / 15.3) / self.base_frequency, zeros)
        elif mode in ['ru', 'ul']:
            middle_cond = logical_and(logical_and(veh_xs > -CROSSROAD_HALF_WIDTH, veh_xs < CROSSROAD_HALF_WIDTH),
                                      logical_and(veh_ys > 0, veh_ys < 27.6))
            veh_phis_rad_delta = tf.where(middle_cond, -(veh_vs / 16.4) / self.base_frequency, zeros)
        else:
            veh_phis_rad_delta = zeros
        next_veh_xs, next_veh_ys, next_veh_vs, next_veh_phis_rad = \
            veh_xs + veh_xs_delta, veh_ys + veh_ys_delta, veh_vs, veh_phis_rad + veh_phis_rad_delta
        next_veh_phis_rad = tf.where(next_veh_phis_rad > np.pi, next_veh_phis_rad - 2 * np.pi, next_veh_phis_rad)
        next_veh_phis_rad = tf.where(next_veh_phis_rad <= -np.pi, next_veh_phis_rad + 2 * np.pi, next_veh_phis_rad)
        next_veh_phis = next_veh_phis_rad * 180 / np.pi
        return tf.stack([next_veh_xs, next_veh_ys, next_veh_vs, next_veh_phis], 1)

    def render(self, mode='human'): #TODO:for debug
        if mode == 'human':
            # plot basic map
            extension = 40
            light_line_width = 3
            dotted_line_style = '--'
            solid_line_style = '-'

            plt.cla()
            plt.title("Crossroad")
            ax = plt.axes(xlim=(-CROSSROAD_HALF_WIDTH - extension, CROSSROAD_HALF_WIDTH + extension),
                          ylim=(-CROSSROAD_D_HEIGHT - extension, CROSSROAD_U_HEIGHT + extension))
            plt.axis("equal")
            plt.axis('off')
            ax.add_patch(plt.Rectangle((-CROSSROAD_HALF_WIDTH - extension, -CROSSROAD_D_HEIGHT - extension),
                                       2 * CROSSROAD_HALF_WIDTH + 2 * extension, CROSSROAD_D_HEIGHT+CROSSROAD_U_HEIGHT + 2 * extension, edgecolor='black',
                                       facecolor='none'))

            # ----------horizon--------------
            plt.plot([-CROSSROAD_HALF_WIDTH - extension, -CROSSROAD_HALF_WIDTH], [0, 0], color='black')
            plt.plot([CROSSROAD_HALF_WIDTH + extension, CROSSROAD_HALF_WIDTH], [0, 0], color='black')

            #
            for i in range(1, LANE_NUMBER_LR + 1):
                linestyle = dotted_line_style if i < LANE_NUMBER_LR else solid_line_style
                plt.plot([-CROSSROAD_HALF_WIDTH - extension, -CROSSROAD_HALF_WIDTH], [i * LANE_WIDTH_LR, i * LANE_WIDTH_LR],
                         linestyle=linestyle, color='black')
                plt.plot([CROSSROAD_HALF_WIDTH + extension, CROSSROAD_HALF_WIDTH], [i * LANE_WIDTH_LR, i * LANE_WIDTH_LR],
                         linestyle=linestyle, color='black')
                plt.plot([-CROSSROAD_HALF_WIDTH - extension, -CROSSROAD_HALF_WIDTH], [-i * LANE_WIDTH_LR, -i * LANE_WIDTH_LR],
                         linestyle=linestyle, color='black')
                plt.plot([CROSSROAD_HALF_WIDTH + extension, CROSSROAD_HALF_WIDTH], [-i * LANE_WIDTH_LR, -i * LANE_WIDTH_LR],
                         linestyle=linestyle, color='black')

            # ----------vertical----------------
            plt.plot([0, 0], [-CROSSROAD_D_HEIGHT - extension, -CROSSROAD_D_HEIGHT], color='black')
            plt.plot([0, 0], [CROSSROAD_U_HEIGHT + extension, CROSSROAD_U_HEIGHT], color='black')

            #
            for i in range(1, LANE_NUMBER_UD + 1):
                linestyle = dotted_line_style if i < LANE_NUMBER_UD else solid_line_style
                plt.plot([i * LANE_WIDTH_UD, i * LANE_WIDTH_UD], [-CROSSROAD_D_HEIGHT - extension, -CROSSROAD_D_HEIGHT],
                         linestyle=linestyle, color='black')
                plt.plot([i * LANE_WIDTH_UD, i * LANE_WIDTH_UD], [CROSSROAD_U_HEIGHT + extension, CROSSROAD_U_HEIGHT],
                         linestyle=linestyle, color='black')
                plt.plot([-i * LANE_WIDTH_UD, -i * LANE_WIDTH_UD], [-CROSSROAD_D_HEIGHT - extension, -CROSSROAD_D_HEIGHT],
                         linestyle=linestyle, color='black')
                plt.plot([-i * LANE_WIDTH_UD, -i * LANE_WIDTH_UD], [CROSSROAD_U_HEIGHT + extension, CROSSROAD_U_HEIGHT],
                         linestyle=linestyle, color='black')

            # ----------Oblique--------------
            plt.plot([LANE_NUMBER_UD * LANE_WIDTH_UD, CROSSROAD_HALF_WIDTH],
                     [-CROSSROAD_D_HEIGHT, -LANE_NUMBER_LR * LANE_WIDTH_LR],
                     color='black')
            plt.plot([LANE_NUMBER_UD * LANE_WIDTH_UD, CROSSROAD_HALF_WIDTH],
                     [CROSSROAD_U_HEIGHT, LANE_NUMBER_LR * LANE_WIDTH_LR],
                     color='black')
            plt.plot([-LANE_NUMBER_UD * LANE_WIDTH_UD, -CROSSROAD_HALF_WIDTH],
                     [-CROSSROAD_D_HEIGHT, -LANE_NUMBER_LR * LANE_WIDTH_LR],
                     color='black')
            plt.plot([-LANE_NUMBER_UD * LANE_WIDTH_UD, -CROSSROAD_HALF_WIDTH],
                     [CROSSROAD_U_HEIGHT, LANE_NUMBER_LR * LANE_WIDTH_LR],
                     color='black')

            def is_in_plot_area(x, y, tolerance=5):
                if -CROSSROAD_HALF_WIDTH - extension + tolerance < x < CROSSROAD_HALF_WIDTH + extension - tolerance and \
                        -CROSSROAD_D_HEIGHT - extension + tolerance < y < CROSSROAD_U_HEIGHT + extension - tolerance:
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

            # abso_obs = self.convert_vehs_to_abso(self.obses)
            obses = self.obses.numpy()
            ego_info, tracing_info, vehs_info = obses[0, :self.ego_info_dim], \
                                                obses[0, self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (
                                                                                          self.num_future_data + 1)], \
                                                obses[0, self.ego_info_dim + self.per_tracking_info_dim * (
                                                            self.num_future_data + 1):]
            # plot cars
            for veh_index in range(int(len(vehs_info) / self.per_veh_info_dim)):
                veh = vehs_info[self.per_veh_info_dim * veh_index:self.per_veh_info_dim * (veh_index + 1)]
                veh_x, veh_y, veh_v, veh_phi = veh

                if is_in_plot_area(veh_x, veh_y):
                    plot_phi_line(veh_x, veh_y, veh_phi, 'black')
                    draw_rotate_rec(veh_x, veh_y, veh_phi, L, W, 'black')

            # plot own car
            delta_y, delta_phi = tracing_info[0], tracing_info[1]
            ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi = ego_info

            plot_phi_line(ego_x, ego_y, ego_phi, 'red')
            draw_rotate_rec(ego_x, ego_y, ego_phi, L, W, 'red')

            # plot text
            text_x, text_y_start = -110, 60
            ge = iter(range(0, 1000, 4))
            plt.text(text_x, text_y_start - next(ge), 'ego_x: {:.2f}m'.format(ego_x))
            plt.text(text_x, text_y_start - next(ge), 'ego_y: {:.2f}m'.format(ego_y))
            plt.text(text_x, text_y_start - next(ge), 'delta_y: {:.2f}m'.format(delta_y))
            plt.text(text_x, text_y_start - next(ge), r'ego_phi: ${:.2f}\degree$'.format(ego_phi))
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
    def __init__(self, task, mode=None, ref_index=None):
        self.mode = mode
        self.traj_mode = None
        self.exp_v = EXPECTED_V #TODO: temp
        self.task = task
        self.path_list = []
        self.path_len_list = []
        self._construct_ref_path(self.task)
        self.ref_index = np.random.choice(len(self.path_list)) if ref_index is None else ref_index
        self.path = self.path_list[self.ref_index]

    def set_path(self, traj_mode, path_index=None, path=None):
        self.traj_mode = traj_mode
        if traj_mode == 'dyna_traj':
            self.path = path
        elif traj_mode == 'static_traj':
            self.ref_index = path_index
            self.path = self.path_list[self.ref_index]

    def _construct_ref_path(self, task):
        sl = 40  # straight length
        meter_pointnum_ratio = 30
        control_ext_x = 2 * CROSSROAD_HALF_WIDTH / 3. + 5.
        control_ext_y = 2 * CROSSROAD_D_HEIGHT / 3. + 3.
        if task == 'left':
            end_offsets = [LANE_WIDTH_LR*(i+0.5) for i in range(LANE_NUMBER_LR)] #TODO: temp
            start_offsets = [LANE_WIDTH_UD*0.5] #TODO: temp
            for start_offset in start_offsets:
                for i, end_offset in enumerate(end_offsets):
                    if i == 0:
                        end_offset += 0.2
                    control_point1 = start_offset, -CROSSROAD_D_HEIGHT
                    control_point2 = start_offset, -CROSSROAD_D_HEIGHT + control_ext_y
                    control_point3 = -CROSSROAD_HALF_WIDTH + control_ext_x, end_offset
                    control_point4 = -CROSSROAD_HALF_WIDTH, end_offset

                    node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                                              [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]],
                                             dtype=np.float32)
                    curve = bezier.Curve(node, degree=3)
                    s_vals = np.linspace(0, 1.0, int(T * 2 * (control_ext_x + control_ext_y) / 4) * meter_pointnum_ratio)
                    trj_data = curve.evaluate_multi(s_vals)
                    trj_data = trj_data.astype(np.float32)
                    start_straight_line_x = LANE_WIDTH_UD/2 * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[:-1]
                    start_straight_line_y = np.linspace(-CROSSROAD_D_HEIGHT - sl, -CROSSROAD_D_HEIGHT, sl * meter_pointnum_ratio, dtype=np.float32)[:-1]
                    end_straight_line_x = np.linspace(-CROSSROAD_HALF_WIDTH, -CROSSROAD_HALF_WIDTH - sl, sl * meter_pointnum_ratio, dtype=np.float32)[1:]
                    end_straight_line_y = end_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[1:]
                    planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x), \
                                 np.append(np.append(start_straight_line_y, trj_data[1]), end_straight_line_y)

                    xs_1, ys_1 = planed_trj[0][:-1], planed_trj[1][:-1]
                    xs_2, ys_2 = planed_trj[0][1:], planed_trj[1][1:]
                    phis_1 = np.arctan2(ys_2 - ys_1,
                                        xs_2 - xs_1) * 180 / pi
                    planed_trj = xs_1, ys_1, phis_1
                    self.path_list.append(planed_trj)
                    self.path_len_list.append((sl * meter_pointnum_ratio, len(trj_data[0]), len(xs_1)))

        elif task == 'straight':
            end_offsets = [LANE_WIDTH_UD*(i+0.5)-0.2 for i in range(LANE_NUMBER_UD)]
            start_offsets = [LANE_WIDTH_UD*0.5]
            for start_offset in start_offsets:
                for end_offset in end_offsets:
                    control_point1 = start_offset, -CROSSROAD_D_HEIGHT
                    control_point2 = start_offset, -CROSSROAD_D_HEIGHT + control_ext_y
                    control_point3 = end_offset, CROSSROAD_U_HEIGHT - control_ext_y
                    control_point4 = end_offset, CROSSROAD_U_HEIGHT

                    node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                                              [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]]
                                             , dtype=np.float32)
                    curve = bezier.Curve(node, degree=3)
                    s_vals = np.linspace(0, 1.0, int(CROSSROAD_U_HEIGHT + CROSSROAD_D_HEIGHT) * meter_pointnum_ratio)
                    trj_data = curve.evaluate_multi(s_vals)
                    trj_data = trj_data.astype(np.float32)
                    start_straight_line_x = start_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[:-1]
                    start_straight_line_y = np.linspace(-CROSSROAD_D_HEIGHT - sl, -CROSSROAD_D_HEIGHT, sl * meter_pointnum_ratio, dtype=np.float32)[:-1]
                    end_straight_line_x = end_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[1:]
                    end_straight_line_y = np.linspace(CROSSROAD_U_HEIGHT, CROSSROAD_U_HEIGHT + sl, sl * meter_pointnum_ratio, dtype=np.float32)[1:]
                    planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x), \
                                 np.append(np.append(start_straight_line_y, trj_data[1]), end_straight_line_y)
                    xs_1, ys_1 = planed_trj[0][:-1], planed_trj[1][:-1]
                    xs_2, ys_2 = planed_trj[0][1:], planed_trj[1][1:]
                    phis_1 = np.arctan2(ys_2 - ys_1,
                                        xs_2 - xs_1) * 180 / pi
                    planed_trj = xs_1, ys_1, phis_1
                    self.path_list.append(planed_trj)
                    self.path_len_list.append((sl * meter_pointnum_ratio, len(trj_data[0]), len(xs_1)))

        else:
            assert task == 'right'
            control_ext_y = 2 * CROSSROAD_D_HEIGHT / 5.+ 3.
            control_ext_x = 2 * CROSSROAD_HALF_WIDTH / 5.+ 3.
            end_offsets = [-LANE_WIDTH_LR * (i + 0.5) for i in range(LANE_NUMBER_LR)]
            start_offsets = [LANE_WIDTH_UD*(LANE_NUMBER_UD-0.5)]

            for start_offset in start_offsets:
                for i, end_offset in enumerate(end_offsets):
                    if i == 0:
                        end_offset -= 0.2
                    control_point1 = start_offset, -CROSSROAD_D_HEIGHT
                    control_point2 = start_offset, -CROSSROAD_D_HEIGHT + control_ext_y
                    control_point3 = CROSSROAD_HALF_WIDTH - control_ext_x, end_offset
                    control_point4 = CROSSROAD_HALF_WIDTH, end_offset

                    node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                                              [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]],
                                             dtype=np.float32)
                    curve = bezier.Curve(node, degree=3)
                    s_vals = np.linspace(0, 1.0, int(pi/2*(CROSSROAD_D_HEIGHT-LANE_WIDTH_LR*(LANE_NUMBER_LR/2-0.5))) * meter_pointnum_ratio)
                    trj_data = curve.evaluate_multi(s_vals)
                    trj_data = trj_data.astype(np.float32)
                    start_straight_line_x = start_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[:-1]
                    start_straight_line_y = np.linspace(-CROSSROAD_D_HEIGHT - sl, -CROSSROAD_D_HEIGHT, sl * meter_pointnum_ratio, dtype=np.float32)[:-1]
                    end_straight_line_x = np.linspace(CROSSROAD_HALF_WIDTH, CROSSROAD_HALF_WIDTH + sl, sl * meter_pointnum_ratio, dtype=np.float32)[1:]
                    end_straight_line_y = end_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[1:]
                    planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x), \
                                 np.append(np.append(start_straight_line_y, trj_data[1]), end_straight_line_y)
                    xs_1, ys_1 = planed_trj[0][:-1], planed_trj[1][:-1]
                    xs_2, ys_2 = planed_trj[0][1:], planed_trj[1][1:]
                    phis_1 = np.arctan2(ys_2 - ys_1,
                                        xs_2 - xs_1) * 180 / pi
                    planed_trj = xs_1, ys_1, phis_1
                    self.path_list.append(planed_trj)
                    self.path_len_list.append((sl * meter_pointnum_ratio, len(trj_data[0]), len(xs_1)))

    def find_closest_point(self, xs, ys, ratio=10):
        path_len = len(self.path[0])
        reduced_idx = np.arange(0, path_len, ratio)
        reduced_len = len(reduced_idx)
        reduced_path_x, reduced_path_y = self.path[0][reduced_idx], self.path[1][reduced_idx]
        xs_tile = tf.tile(tf.reshape(xs, (-1, 1)), tf.constant([1, reduced_len]))
        ys_tile = tf.tile(tf.reshape(ys, (-1, 1)), tf.constant([1, reduced_len]))
        pathx_tile = tf.tile(tf.reshape(reduced_path_x, (1, -1)), tf.constant([len(xs), 1]))
        pathy_tile = tf.tile(tf.reshape(reduced_path_y, (1, -1)), tf.constant([len(xs), 1]))

        dist_array = tf.square(xs_tile - pathx_tile) + tf.square(ys_tile - pathy_tile)

        indexs = tf.argmin(dist_array, 1) * ratio
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
        indexs = tf.where(indexs >= 0, indexs, 0)
        indexs = tf.where(indexs < len(self.path[0]), indexs, len(self.path[0])-1)
        points = tf.gather(self.path[0], indexs), \
                 tf.gather(self.path[1], indexs), \
                 tf.gather(self.path[2], indexs)

        return points[0], points[1], points[2]

    def tracking_error_vector(self, ego_xs, ego_ys, ego_phis, ego_vs, n, func=None):
        def two2one(ref_xs, ref_ys):
            if self.task == 'left':
                delta_ = tf.sqrt(tf.square(ego_xs - (-CROSSROAD_HALF_WIDTH)) + tf.square(ego_ys - (-CROSSROAD_D_HEIGHT))) - \
                         tf.sqrt(tf.square(ref_xs - (-CROSSROAD_HALF_WIDTH)) + tf.square(ref_ys - (-CROSSROAD_D_HEIGHT)))
                delta_ = tf.where(ego_ys < -CROSSROAD_D_HEIGHT, ego_xs - ref_xs, delta_)
                delta_ = tf.where(ego_xs < -CROSSROAD_HALF_WIDTH, ego_ys - ref_ys, delta_)
                return -delta_
            elif self.task == 'straight':
                delta_ = ego_xs - ref_xs
                return -delta_
            else:
                assert self.task == 'right'
                delta_ = -(tf.sqrt(tf.square(ego_xs - CROSSROAD_HALF_WIDTH) + tf.square(ego_ys - (-CROSSROAD_D_HEIGHT))) -
                           tf.sqrt(tf.square(ref_xs - CROSSROAD_HALF_WIDTH) + tf.square(ref_ys - (-CROSSROAD_D_HEIGHT))))
                delta_ = tf.where(ego_ys < -CROSSROAD_D_HEIGHT, ego_xs - ref_xs, delta_)
                delta_ = tf.where(ego_xs > CROSSROAD_HALF_WIDTH, -(ego_ys - ref_ys), delta_)
                return -delta_

        if self.traj_mode == 'dyna_traj':
            if func == 'tracking':
                indexs = tf.constant([1], dtype=tf.int32)
                current_points = self.indexs2points(indexs)
                n_future_data = self.future_n_data(indexs, n)
                all_ref = [current_points] + n_future_data
                print(current_points)

                tracking_error = tf.concat([tf.stack([two2one(ref_point[0], ref_point[1]),
                                                      deal_with_phi_diff(ego_phis - ref_point[2]),
                                                      ego_vs - self.exp_v], 1)
                                            for ref_point in all_ref], 1)

            else:
                indexs, current_points = self.find_closest_point(ego_xs, ego_ys)
                # print('Index:', indexs.numpy(), 'points:', current_points[:])
                n_future_data = self.future_n_data(indexs, n)
                all_ref = [current_points] + n_future_data

                tracking_error = tf.concat([tf.stack([two2one(ref_point[0], ref_point[1]),
                                                      deal_with_phi_diff(ego_phis - ref_point[2]),
                                                      ego_vs - self.exp_v], 1)
                                            for ref_point in all_ref], 1)
            final = None
        else:
            indexs, current_points = self.find_closest_point(ego_xs, ego_ys)
            # print('Index:', indexs.numpy(), 'points:', current_points[:])
            n_future_data = self.future_n_data(indexs, n)

            tracking_error = tf.stack([two2one(current_points[0], current_points[1]),
                                               deal_with_phi_diff(ego_phis - current_points[2]),
                                               ego_vs - self.exp_v], 1)

            final = tracking_error
            if n > 0:
                future_points = tf.concat([tf.stack([ref_point[0] - ego_xs,
                                                     ref_point[1] - ego_ys,
                                                     deal_with_phi_diff(ego_phis - ref_point[2])], 1)
                                           for ref_point in n_future_data], 1)
                final = tf.concat([final, future_points], 1)

        return final

    def plot_path(self, x, y):
        plt.axis('equal')
        plt.plot(self.path_list[0][0], self.path_list[0][1], 'b')
        plt.plot(self.path_list[1][0], self.path_list[1][1], 'r')
        plt.plot(self.path_list[2][0], self.path_list[2][1], 'g')
        print(self.path_len_list)

        index, closest_point = self.find_closest_point(np.array([x], np.float32),
                                                       np.array([y], np.float32))
        plt.plot(x, y, 'b*')
        plt.plot(closest_point[0], closest_point[1], 'ro')
        plt.show()


def test_ref_path():
    path = ReferencePath('right')
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

    tracking_error_vector = path.tracking_error_vector(xs, ys, phis, vs, 10)
    print(tracking_error_vector)


def test_model():
    from endtoend import CrossroadEnd2end
    env = CrossroadEnd2end('left', 0)
    model = EnvironmentModel('left', 0)
    model.ref_indexes = 0
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
    model.reset(obses, ref_indexes=tf.convert_to_tensor([0.]))
    print(obses.shape)
    for rollout_step in range(100):
        # print('model ref index', model.ref_indexes)
        actions = tf.tile(tf.constant([[0.5, 0]], dtype=tf.float32), tf.constant([len(obses), 1]))
        obses, rewards, constraints, real_punish, _, _ = model.rollout_out(actions)
        print(constraints.numpy, constraints.numpy().shape)
        model.render()


def test_tf_function():
    class Test2():
        def __init__(self):
            self.c = 2

        def step1(self, a):
            print('trace')
            self.c = a

        def step2(self):
            return self.c

    test2 = Test2()

    @tf.function#(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
    def f(a):
        test2.step1(a)
        return test2.step2()

    print(f(2), type(test2.c))
    print(f(2), test2.c)

    print(f(tf.constant(2)), type(test2.c))
    print(f(tf.constant(3)), test2.c)

    # print(f(2), test2.c)
    # print(f(3), test2.c)
    # print(f(2), test2.c)
    # print(f())
    # print(f())
    #
    # test2.c.assign_add(12)
    # print(test2.c)
    # print(f())





    # b= test2.create_test1(1)
    # print(test2.b,b, test2.b.a)
    # b=test2.create_test1(2)
    # print(test2.b,b,test2.b.a)
    # b=test2.create_test1(1)
    # print(test2.b,b,test2.b.a)
    # test2.create_test1(1)
    # test2.pc()
    # test2.create_test1(1)
    # test2.pc()
@tf.function
def test_tffunc(inttt):
    print(22)
    if inttt=='1':
        a = 2
    elif inttt == '2':
        a = 233
    else:
        a=22
    return a

def test_ref():
    import numpy as np
    import matplotlib.pyplot as plt
    # ref = ReferencePath('left')
    # path1, path2, path3 = ref.path_list
    # path1, path2, path3 = [ite[1200:-1200] for ite in path1],\
    #                       [ite[1200:-1200] for ite in path2], \
    #                       [ite[1200:-1200] for ite in path3]
    # x1, y1, phi1 = path1
    # x2, y2, phi2 = path2
    # x3, y3, phi3 = path3
    # p1, p2, p3 = np.arctan2(y1-(-CROSSROAD_SIZE/2), x1 - (-CROSSROAD_SIZE/2)), \
    #              np.arctan2(y2 - (-CROSSROAD_SIZE / 2), x2 - (-CROSSROAD_SIZE / 2)), \
    #              np.arctan2(y3 - (-CROSSROAD_SIZE / 2), x3 - (-CROSSROAD_SIZE / 2))
    # d1, d2, d3 = np.sqrt(np.square(x1-(-CROSSROAD_SIZE/2))+np.square(y1-(-CROSSROAD_SIZE/2))),\
    #              np.sqrt(np.square(x2-(-CROSSROAD_SIZE/2))+np.square(y2-(-CROSSROAD_SIZE/2))),\
    #              np.sqrt(np.square(x3-(-CROSSROAD_SIZE/2))+np.square(y3-(-CROSSROAD_SIZE/2)))
    #
    # plt.plot(p1, d1, 'r')
    # plt.plot(p2, d2, 'g')
    # plt.plot(p3, d3, 'b')
    # z1 = np.polyfit(p1, d1, 3, rcond=None, full=False, w=None, cov=False)
    # p1_fit = np.poly1d(z1)
    # plt.plot(p1, p1_fit(p1), 'r*')
    #
    # z2 = np.polyfit(p2, d2, 3, rcond=None, full=False, w=None, cov=False)
    # p2_fit = np.poly1d(z2)
    # plt.plot(p2, p2_fit(p2), 'g*')
    #
    # z3 = np.polyfit(p3, d3, 3, rcond=None, full=False, w=None, cov=False)
    # p3_fit = np.poly1d(z3)
    # plt.plot(p3, p3_fit(p3), 'b*')

    # ref = ReferencePath('straight')
    # path1, path2, path3, path4, path5, path6 = ref.path_list
    # path1, path2, path3 = [ite[1200:-1200] for ite in path1], \
    #                       [ite[1200:-1200] for ite in path2], \
    #                       [ite[1200:-1200] for ite in path3]
    # x1, y1, phi1 = path1
    # x2, y2, phi2 = path2
    # x3, y3, phi3 = path3
    #
    # plt.plot(y1, x1, 'r')
    # plt.plot(y2, x2, 'g')
    # plt.plot(y3, x3, 'b')
    # z1 = np.polyfit(y1, x1, 3, rcond=None, full=False, w=None, cov=False)
    # print(type(list(z1)))
    # p1_fit = np.poly1d(z1)
    # print(z1, p1_fit)
    # plt.plot(y1, p1_fit(y1), 'r*')
    # plt.show()

    ref = ReferencePath('right')
    path1 = ref.path_list[0]
    path1 = [ite[1200:-1200] for ite in path1]

    x1, y1, phi1 = path1


    plt.plot(y1, x1, 'r')

    z1 = np.polyfit(y1, x1, 3, rcond=None, full=False, w=None, cov=False)
    print(type(list(z1)))
    p1_fit = np.poly1d(z1)
    print(z1, p1_fit)
    plt.plot(y1, p1_fit(y1), 'r*')
    plt.show()

def test_gen_ref_and_save():
    import matplotlib.pyplot as plt
    plt.figure()
    for task in ['left','straight','right']:
        ref = ReferencePath(task)
        name = task + '_ref.npy'
        np.save(name,np.array(ref.path_list))
        for path in ref.path_list:
            plt.plot(path[0],path[1])
    plt.show()



if __name__ == '__main__':
    # test_gen_ref_and_save()
    test_model()


