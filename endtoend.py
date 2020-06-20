#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: endtoend.py
# =====================================

import warnings
from collections import OrderedDict
from math import sqrt, cos, sin, pi

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gym.utils import seeding

# gym.envs.user_defined.toyota_env.
from dynamics_and_models import VehicleDynamics, ReferencePath
from endtoend_env_utils import shift_coordination, rotate_coordination
from traffic import Traffic

warnings.filterwarnings("ignore")


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = gym.spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        space = gym.spaces.Box(low, high, dtype=np.float32)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


def judge_feasible(orig_x, orig_y, task):  # map dependant
    # return True if -900 < orig_x < 900 and -150 - 3.75 * 4 < orig_y < -150 else False
    def is_in_straight_before1(orig_x, orig_y):
        return 0 < orig_x < 3.75 and orig_y <= -18

    def is_in_straight_before2(orig_x, orig_y):
        return 3.75 < orig_x < 7.5 and orig_y <= -18

    def is_in_straight_after(orig_x, orig_y):
        return 0 < orig_x < 3.75 * 2 and orig_y >= 18

    def is_in_left(orig_x, orig_y):
        return 0 < orig_y < 3.75 * 2 and orig_x < -18

    def is_in_right(orig_x, orig_y):
        return -3.75 * 2 < orig_y < 0 and orig_x > 18

    def is_in_middle(orig_x, orig_y):
        return True if -18 < orig_y < 18 and -18 < orig_x < 18 else False

    def is_in_middle_left(orig_x, orig_y):
        return True if -18 < orig_y < 7.5 and -18 < orig_x < 7.5 else False
        # if -18 < orig_y < 18 and -18 < orig_x < 18:
        #     if -3.75 * 2 < orig_x < 3.75 * 2:
        #         return True if -18 < orig_y < 18 else False
        #     elif orig_x > 3.75 * 2:
        #         return True if orig_x - (18 + 3.75 * 2) < orig_y < -orig_x + (18 + 3.75 * 2) else False
        #     else:
        #         return True if -orig_x - (18 + 3.75 * 2) < orig_y < orig_x + (18 + 3.75 * 2) else False
        # else:
        #     return False

    # judge feasible for turn left
    if task == 'left':
        return True if is_in_straight_before1(orig_x, orig_y) or is_in_left(orig_x, orig_y) \
                       or is_in_middle_left(orig_x, orig_y) else False
    elif task == 'straight':
        return True if is_in_straight_before1(orig_x, orig_y) or is_in_straight_after(orig_x, orig_y) \
                       or is_in_middle(orig_x, orig_y) else False
    else:
        assert task == 'right'
        return True if is_in_straight_before2(orig_x, orig_y) or is_in_right(orig_x, orig_y) \
                       or is_in_middle(orig_x, orig_y) else False


ROUTE2MODE = {('1o', '2i'): 'dr', ('1o', '3i'): 'du', ('1o', '4i'): 'dl',
              ('2o', '1i'): 'rd', ('2o', '3i'): 'ru', ('2o', '4i'): 'rl',
              ('3o', '1i'): 'ud', ('3o', '2i'): 'ur', ('3o', '4i'): 'ul',
              ('4o', '1i'): 'ld', ('4o', '2i'): 'lr', ('4o', '3i'): 'lu'}
MODE2TASK = {'dr': 'right', 'du': 'straight', 'dl': 'left',
             'rd': 'left', 'ru': 'right', 'rl': ' straight',
             'ud': 'straight', 'ur': 'left', 'ul': 'right',
             'ld': 'right', 'lr': 'straight', 'lu': 'left'}


class CrossroadEnd2end(gym.Env):
    def __init__(self,
                 training_task,  # 'left', 'straight', 'right'
                 num_future_data=5,
                 display=False):
        metadata = {'render.modes': ['human']}
        self.dynamics = VehicleDynamics()
        self.interested_vehs = None
        self.training_task = training_task
        self.ref_path = ReferencePath(self.training_task)
        self.detected_vehicles = None
        self.all_vehicles = None
        self.ego_dynamics = None
        self.num_future_data = num_future_data
        self.init_state = {}
        self.action_number = 2
        self.exp_v = 8.
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_number,), dtype=np.float32)

        self.seed()
        self.v_light = None
        self.step_length = 100  # ms

        self.step_time = self.step_length / 1000.0
        self.init_state = self._reset_init_state()
        self.traffic = Traffic(self.step_length,
                               mode='training',
                               init_n_ego_dict=self.init_state,
                               training_task=self.training_task)
        if not display:
            self.reset()
            action = self.action_space.sample()
            observation, _reward, done, _info = self.step(action)
            self._set_observation_space(observation)
            plt.ion()
        self.obs = None
        self.action = None
        self.veh_mode_list = []

        self.done_type = 'not_done_yet'
        self.reward_info = None
        self.ego_info_dim = None
        self.tracking_info_dim = None
        self.per_veh_info_dim = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, **kwargs):  # kwargs include three keys
        self.ref_path = ReferencePath(self.training_task)
        self.init_state = self._reset_init_state()
        self.traffic.init_traffic(self.init_state)
        self.traffic.sim_step()
        self._get_all_info()
        self.obs = self._get_obs()
        self.action = None
        self.reward_info = None
        self.done_type = 'not_done_yet'
        return self.obs

    def step(self, action):
        self.action = self._action_transformation_for_end2end3(action)
        reward, self.reward_info = self.compute_reward3(self.obs, self.action)
        next_ego_state, next_ego_params = self._get_next_ego_state(self.action)
        self.traffic.set_own_car(dict(ego=dict(v_x=next_ego_state[0],
                                               v_y=next_ego_state[1],
                                               r=next_ego_state[2],
                                               x=next_ego_state[3],
                                               y=next_ego_state[4],
                                               phi=next_ego_state[5],
                                               alpha_f=next_ego_params[0],
                                               alpha_r=next_ego_params[1],
                                               miu_f=next_ego_params[2],
                                               miu_r=next_ego_params[3],)
                                      ))
        self.traffic.sim_step()
        all_info = self._get_all_info()
        self.obs = self._get_obs()
        self.done_type, done = self._judge_done()
        done_rew = self._compute_done_rew(self.done_type)
        # reward, self.reward_info = self.compute_reward2(self.obs, self.action)
        reward += done_rew
        self.reward_info.update({'done_rew': done_rew,
                                 'final_rew': reward})
        # if done:
        #     print(self.done_type)
        all_info.update(self.reward_info)
        return self.obs, reward, done, all_info

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def _compute_done_rew(self, done_type):
        if done_type == 'not_done_yet':
            return 0
        elif done_type == 'good_done':
            return 50
        else:
            return -20

    def _get_all_info(self):  # used to update info, must be called every timestep before _get_obs
        # to fetch info
        self.all_vehicles = self.traffic.n_ego_vehicles['ego']  # coordination 2
        self.ego_dynamics = self.traffic.n_ego_info['ego']  # coordination 2
        self.v_light = self.traffic.v_light

        miu_f, miu_r = self.ego_dynamics['miu_f'], self.ego_dynamics['miu_r']
        F_zf, F_zr = self.dynamics.vehicle_params['F_zf'], self.dynamics.vehicle_params['F_zr']
        C_f, C_r = self.dynamics.vehicle_params['C_f'], self.dynamics.vehicle_params['C_r']
        alpha_f_bound, alpha_r_bound = 3 * miu_f * F_zf / C_f, 3 * miu_r * F_zr / C_r
        r_bound = miu_r * self.dynamics.vehicle_params['g'] / abs(self.ego_dynamics['v_x'])
        self.ego_dynamics.update(dict(alpha_f_bound=alpha_f_bound,
                                      alpha_r_bound=alpha_r_bound,
                                      r_bound=r_bound))
        # ego_dynamics
        # dict(v_x=ego_dict['v_x'],
        #      v_y=ego_dict['v_y'],
        #      r=ego_dict['r'],
        #      x=ego_dict['x'],
        #      y=ego_dict['y'],
        #      phi=ego_dict['phi'],
        #      l=ego_dict['l'],
        #      w=ego_dict['w'],
        #      alpha_f = ego_dict['alpha_f'],
        #      alpha_r = ego_dict['alpha_r'],
        #      miu_f = ego_dict['miu_f'],
        #      miu_r = ego_dict['miu_r'],
        #      Corner_point=self.cal_corner_point_of_ego_car(ego_dict)
        #      alpha_f_bound=alpha_f_bound
        #      alpha_r_bound=alpha_r_bound
        #      r_bound=r_bound)

        # all_vehicles
        # dict(x=x, y=y, v=v, phi=a, l=length,
        #      w=width, route=route)

        all_info = dict(all_vehicles=self.all_vehicles,
                        ego_dynamics=self.ego_dynamics,
                        v_light=self.v_light)
        return all_info

    def _judge_done(self):
        """
        :return:
         1: bad done: collision
         2: bad done: break_road_constrain
         3: good done: task succeed
         4: not done
        """
        if self.traffic.collision_flag:
            return 'collision', 1
        elif self._break_road_constrain():
            return 'break_road_constrain', 1
        elif self._deviate_too_much():
            return 'deviate_too_much', 1
        elif self._break_stability():
            return 'break_stability', 1
        elif self._break_red_light():
            return 'break_red_light', 1
        elif self._is_achieve_goal():
            return 'good_done', 1
        else:
            return 'not_done_yet', 0

    def _deviate_too_much(self):
        delta_x, delta_y, delta_phi = self.obs[self.ego_info_dim:self.ego_info_dim+3]
        dist = np.sqrt(np.square(delta_x) + np.square(delta_y))
        return True if dist > 10 or abs(delta_phi) > 30 else False

    def _break_road_constrain(self):
        results = list(map(lambda x: judge_feasible(*x, self.training_task), self.ego_dynamics['Corner_point']))
        return not all(results)

    def _break_stability(self):
        alpha_f, alpha_r, miu_f, miu_r = self.ego_dynamics['alpha_f'], self.ego_dynamics['alpha_r'], \
                                         self.ego_dynamics['miu_f'], self.ego_dynamics['miu_r']
        alpha_f_bound, alpha_r_bound = self.ego_dynamics['alpha_f_bound'], self.ego_dynamics['alpha_r_bound']
        r_bound = self.ego_dynamics['r_bound']
        # if -alpha_f_bound < alpha_f < alpha_f_bound \
        #         and -alpha_r_bound < alpha_r < alpha_r_bound and \
        #         -r_bound < self.ego_dynamics['r'] < r_bound:
        if -r_bound < self.ego_dynamics['r'] < r_bound:
            return False
        else:
            return True

    def _break_red_light(self):
        return True if self.v_light != 0 and self.ego_dynamics['y'] > -18 and self.training_task != 'right' else False

    def _is_achieve_goal(self):  # for now, only support turn left with the specific map
        x = self.ego_dynamics['x']
        y = self.ego_dynamics['y']
        if self.training_task == 'left':
            return True if x < -18 - 2 and 0 < y < 7.5 else False
        elif self.training_task == 'right':
            return True if x > 18 + 2 and -7.5 < y < 0 else False
        else:
            assert self.training_task == 'straight'
            return True if y > 18 + 2 and 0 < x < 7.5 else False

    # def _action_transformation_for_end2end(self, action):  # [-1, 1]
    #     # scaled_action = action * np.array([0.2, 3.], dtype=np.float32)
    #     # ego_v = self.ego_dynamics['v_x']
    #     # acc_lower_bound = max(-3., -ego_v/3.)
    #     # return np.clip(scaled_action,
    #     #                np.array([-3., acc_lower_bound], dtype=np.float32),
    #     #                np.array([3., 5], dtype=np.float32))
    #     steer_norm, a_x_norm = action[0], action[1]
    #     scaled_steer = 0. if self.obs[4]< -18. else 0.2 * steer_norm
    #     if self.obs[3] < -18+10:
    #         ego_x, ego_y = self.obs[3], self.obs[4]
    #         index, _ = self.ref_path.find_closest_point(np.array([ego_x], dtype=np.float32),
    #                                          np.array([ego_y], dtype=np.float32),)
    #         n_future_data = self.ref_path.future_n_data(index, 5)
    #         ego_phi = self.obs[5]
    #         ref_phi = n_future_data[-1][2]
    #         delta_phi = ego_phi - ref_phi
    #         scaled_steer = - 0.2/30. * delta_phi
    #         print(ego_phi, ref_phi.numpy(), delta_phi.numpy(), scaled_steer.numpy())
    #     ego_v = self.ego_dynamics['v_x']
    #     acc_lower_bound = max(-3., -ego_v/3.)
    #     acc_upper_bound = max(1., min(3, -2*ego_v+21.))
    #     scaled_a_x = (a_x_norm + 1.) / 2. * (acc_upper_bound - acc_lower_bound) + acc_lower_bound
    #     return np.array([scaled_steer, scaled_a_x], dtype=np.float32)
    #
    # def _action_transformation_for_end2end2(self, action):  # [-1, 1]
    #     # scaled_action = action * np.array([0.2, 3.], dtype=np.float32)
    #     # ego_v = self.ego_dynamics['v_x']
    #     # acc_lower_bound = max(-3., -ego_v/3.)
    #     # return np.clip(scaled_action,
    #     #                np.array([-3., acc_lower_bound], dtype=np.float32),
    #     #                np.array([3., 5], dtype=np.float32))
    #     steer_norm, a_x_norm = action[0], action[1]
    #     scaled_steer = 0. if self.obs[4]< -18. else 0.2 * steer_norm
    #     ego_v = self.ego_dynamics['v_x']
    #     acc_lower_bound = max(-3., -ego_v/3.)
    #     acc_upper_bound = max(1., min(3, -2*ego_v+21.))
    #     scaled_a_x = (a_x_norm + 1.) / 2. * (acc_upper_bound - acc_lower_bound) + acc_lower_bound
    #     return np.array([scaled_steer, scaled_a_x], dtype=np.float32)

    def _action_transformation_for_end2end3(self, action):  # [-1, 1]
        steer_norm, a_x_norm = action[0], action[1]
        scaled_steer = 0. if self.obs[4]< -18. else 0.2 * steer_norm
        scaled_a_x = 3.*a_x_norm
        ego_v = self.ego_dynamics['v_x']
        acc_lower_bound = max(-3., -ego_v/3.)
        acc_upper_bound = max(1., min(3, -2 * ego_v + 21.))

        ego_infos, tracking_infos, veh_infos = self.obs[:self.ego_info_dim], self.obs[self.ego_info_dim:self.ego_info_dim + 4 * (
                    self.num_future_data + 1)], self.obs[self.ego_info_dim + 4 * (self.num_future_data + 1):]
        if self.training_task == 'left':
            # veh_mode_list = [('dl', 2), ('du', 2), ('ud', 3), ('ul', 3)]
            first_ud = veh_infos[self.per_veh_info_dim*4:self.per_veh_info_dim*5]
            veh_x, veh_y, veh_v, veh_phi, veh_l, veh_w = first_ud
            if ego_infos[3] > 1. and ego_infos[4] > -18. and veh_y < 18.-5.:
                scaled_steer, scaled_a_x = 0., -3.

        scaled_action = np.array([scaled_steer, scaled_a_x], dtype=np.float32)
        return np.clip(scaled_action,
                       np.array([-3., acc_lower_bound], dtype=np.float32),
                       np.array([3., acc_upper_bound], dtype=np.float32))

    def _get_next_ego_state(self, trans_action):
        current_v_x = self.ego_dynamics['v_x']
        current_v_y = self.ego_dynamics['v_y']
        current_r = self.ego_dynamics['r']
        current_x = self.ego_dynamics['x']
        current_y = self.ego_dynamics['y']
        current_phi = self.ego_dynamics['phi']
        steer, a_x = trans_action
        state = np.array([[current_v_x, current_v_y, current_r, current_x, current_y, current_phi]], dtype=np.float32)
        action = np.array([[steer, a_x]], dtype=np.float32)
        next_ego_state, next_ego_params = self.dynamics.prediction(state, action, 10, 1)
        next_ego_state, next_ego_params = next_ego_state.numpy()[0],  next_ego_params.numpy()[0]
        return next_ego_state, next_ego_params

    def _get_obs(self, exit_='D'):
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_phi = self.ego_dynamics['phi']
        ego_v_x = self.ego_dynamics['v_x']

        vehs_vector = self._construct_veh_vector_short(exit_)
        ego_vector = self._construct_ego_vector_short()
        tracking_error = self.ref_path.tracking_error_vector(np.array([ego_x], dtype=np.float32),
                                                             np.array([ego_y], dtype=np.float32),
                                                             np.array([ego_phi], dtype=np.float32),
                                                             np.array([ego_v_x], dtype=np.float32),
                                                             self.num_future_data).numpy()[0]

        vector = np.concatenate((ego_vector, tracking_error, vehs_vector), axis=0)

        return vector

    def _construct_ego_vector_short(self):
        ego_v_x = self.ego_dynamics['v_x']
        ego_v_y = self.ego_dynamics['v_y']
        ego_r = self.ego_dynamics['r']
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_phi = self.ego_dynamics['phi']
        ego_l = self.ego_dynamics['l']
        ego_w = self.ego_dynamics['w']
        ego_alpha_f = self.ego_dynamics['alpha_f']
        ego_alpha_r = self.ego_dynamics['alpha_r']
        ego_miu_f = self.ego_dynamics['miu_f']
        ego_miu_r = self.ego_dynamics['miu_r']
        ego_feature = [ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi,
                            ego_l, ego_w, ego_alpha_f, ego_alpha_r, ego_miu_f, ego_miu_r]
        self.ego_info_dim = 12
        return np.array(ego_feature, dtype=np.float32)

    def _construct_ego_vector(self):
        ego_v_x = self.ego_dynamics['v_x']
        ego_v_y = self.ego_dynamics['v_y']
        ego_r = self.ego_dynamics['r']
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_phi = self.ego_dynamics['phi']
        ego_l = self.ego_dynamics['l']
        ego_w = self.ego_dynamics['w']
        ego_alpha_f = self.ego_dynamics['alpha_f']
        ego_alpha_r = self.ego_dynamics['alpha_r']
        ego_miu_f = self.ego_dynamics['miu_f']
        ego_miu_r = self.ego_dynamics['miu_r']
        ego_lw = (ego_l - ego_w) / 2.
        coeff = 1.14
        rho_ego = ego_w / 2. * coeff
        ego_front_point = tf.cast(ego_x + ego_lw * tf.cos(ego_phi * np.pi / 180.), dtype=tf.float32),\
                          tf.cast(ego_y + ego_lw * tf.sin(ego_phi * np.pi / 180.), dtype=tf.float32)
        ego_rear_point = tf.cast(ego_x - ego_lw * tf.cos(ego_phi * np.pi / 180.), dtype=tf.float32), \
                         tf.cast(ego_y - ego_lw * tf.sin(ego_phi * np.pi / 180.), dtype=tf.float32)
        ego_feature = [ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi,
                            ego_l, ego_w, ego_alpha_f, ego_alpha_r, ego_miu_f, ego_miu_r]
        if self.training_task == 'left':
            for ego_point in [ego_front_point, ego_rear_point]:
                if ego_point[1] <= -18:
                    up = 18 - ego_point[1] - rho_ego
                    down = 50
                    left = ego_point[0] - 0 - rho_ego
                    right = 3.75 - ego_point[0] - rho_ego
                elif -18 < ego_point[0] < 18 and -18 < ego_point[1] < 18:
                    up = 18 - ego_point[1] - rho_ego
                    down = 50 if 0 < ego_point[0] < 3.75 else ego_point[1] - (-18) - rho_ego
                    left = 50 if 0 < ego_point[1] < 7.5 else ego_point[0] - (-18) - rho_ego
                    right = 18 - ego_point[0] - rho_ego
                else:
                    up = 7.5 - ego_point[1] - rho_ego
                    down = ego_point[1] - 0 - rho_ego
                    left = 50
                    right = 18 - ego_point[0] - rho_ego
                ego_feature.extend([up, down, left, right])
                for corner in [(-18., 7.5), (-18, 0)]:
                    ego_feature.extend([ego_point[0]-corner[0], ego_point[1]-corner[1]])
        elif self.training_task == 'straight':
            for ego_point in [ego_front_point, ego_rear_point]:
                if ego_point[1] <= -18:
                    up = 50
                    down = 50
                    left = ego_point[0] - 0 - rho_ego
                    right = 3.75 - ego_point[0] - rho_ego
                elif -18 < ego_point[0] < 18 and -18 < ego_point[1] < 18:
                    up = 50 if 0 < ego_point[1] < 7.5 else 18 - ego_point[1] - rho_ego
                    down = 50 if 0 < ego_point[0] < 3.75 else ego_point[1] - (-18) - rho_ego
                    left = ego_point[0] - (-18) - rho_ego
                    right = 18 - ego_point[0] - rho_ego
                else:
                    up = 50
                    down = 50
                    left = ego_point[0] - 0 - rho_ego
                    right = 7.5 - ego_point[0] - rho_ego
                ego_feature.extend([up, down, left, right])
                for corner in [(0., 18.), (7.5, 18.)]:
                    ego_feature.extend([ego_point[0] - corner[0], ego_point[1] - corner[1]])
        else:
            assert self.training_task == 'right'
            for ego_point in [ego_front_point, ego_rear_point]:
                if ego_point[1] <= -18:
                    up = 18 - ego_point[1] - rho_ego
                    down = 50
                    left = ego_point[0] - 3.75 - rho_ego
                    right = 7.5 - ego_point[0] - rho_ego
                elif -18 < ego_point[0] < 18 and -18 < ego_point[1] < 18:
                    up = 18 - ego_point[1] - rho_ego
                    down = 50 if 3.75 < ego_point[0] < 7.5 else ego_point[1] - (-18) - rho_ego
                    left = ego_point[0] - (-18) - rho_ego
                    right = 50 if -7.5 < ego_point[1] < 0 else 18 - ego_point[0] - rho_ego
                else:
                    up = 0 - ego_point[1] - rho_ego
                    down = ego_point[1] - (-7.5) - rho_ego
                    left = ego_point[0] - (-18) - rho_ego
                    right = 50
                ego_feature.extend([up, down, left, right])
                for corner in [(18., 0), (18., -7.5)]:
                    ego_feature.extend([ego_point[0] - corner[0], ego_point[1] - corner[1]])
        self.ego_info_dim = 28
        return np.array(ego_feature, dtype=np.float32)

    def _construct_veh_vector_short(self, exit_='D'):
        ego_v_x = self.ego_dynamics['v_x']
        ego_v_y = self.ego_dynamics['v_y']
        ego_r = self.ego_dynamics['r']
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_phi = self.ego_dynamics['phi']
        ego_l = self.ego_dynamics['l']
        ego_w = self.ego_dynamics['w']
        ego_alpha_f = self.ego_dynamics['alpha_f']
        ego_alpha_r = self.ego_dynamics['alpha_r']
        ego_miu_f = self.ego_dynamics['miu_f']
        ego_miu_r = self.ego_dynamics['miu_r']
        v_light = self.v_light
        vehs_vector = []
        ego_lw = (ego_l - ego_w) / 2.
        coeff = 1.14
        rho_ego = ego_w / 2. * coeff
        ego_front_point = tf.cast(ego_x + ego_lw * tf.cos(ego_phi * np.pi / 180.), dtype=tf.float32), \
                          tf.cast(ego_y + ego_lw * tf.sin(ego_phi * np.pi / 180.), dtype=tf.float32)
        ego_rear_point = tf.cast(ego_x - ego_lw * tf.cos(ego_phi * np.pi / 180.), dtype=tf.float32), \
                         tf.cast(ego_y - ego_lw * tf.sin(ego_phi * np.pi / 180.), dtype=tf.float32)

        name_settings = dict(D=dict(do='1o', di='1i', ro='2o', ri='2i', uo='3o', ui='3i', lo='4o', li='4i'),
                             R=dict(do='2o', di='2i', ro='3o', ri='3i', uo='4o', ui='4i', lo='1o', li='1i'),
                             U=dict(do='3o', di='3i', ro='4o', ri='4i', uo='1o', ui='1i', lo='2o', li='2i'),
                             L=dict(do='4o', di='4i', ro='1o', ri='1i', uo='2o', ui='2i', lo='3o', li='3i'))

        name_setting = name_settings[exit_]

        def filter_interested_vehicles(vs, task):
            dl, du, dr, rd, rl, ru, ur, ud, ul, lu, lr, ld = [], [], [], [], [], [], [], [], [], [], [], []
            for v in vs:
                route_list = v['route']
                start = route_list[0]
                end = route_list[1]
                if start == name_setting['do'] and end == name_setting['li']:
                    dl.append(v)
                elif start == name_setting['do'] and end == name_setting['ui']:
                    du.append(v)
                elif start == name_setting['do'] and end == name_setting['ri']:
                    dr.append(v)

                elif start == name_setting['ro'] and end == name_setting['di']:
                    rd.append(v)
                elif start == name_setting['ro'] and end == name_setting['li']:
                    rl.append(v)
                elif start == name_setting['ro'] and end == name_setting['ui']:
                    ru.append(v)

                elif start == name_setting['uo'] and end == name_setting['ri']:
                    ur.append(v)
                elif start == name_setting['uo'] and end == name_setting['di']:
                    ud.append(v)
                elif start == name_setting['uo'] and end == name_setting['li']:
                    ul.append(v)

                elif start == name_setting['lo'] and end == name_setting['ui']:
                    lu.append(v)
                elif start == name_setting['lo'] and end == name_setting['ri']:
                    lr.append(v)
                elif start == name_setting['lo'] and end == name_setting['di']:
                    ld.append(v)
            # fetch veh in range
            dl = list(filter(lambda v: v['x'] > -28 and v['y'] > ego_y, dl))  # interest of left straight
            du = list(filter(lambda v: ego_y < v['y'] < 28, du))  # interest of left straight
            if v_light != 0 and ego_y < -18:
                du.append(dict(x=1.875, y=-18, v=0, phi=90, l=5, w=2.5, route=None))

            dr = list(filter(lambda v: v['x'] < 28 and v['y'] > ego_y, dr))  # interest of right

            rd = rd  # not interest in case of traffic light
            rl = rl  # not interest in case of traffic light
            ru = list(filter(lambda v: v['x'] < 28 and v['y'] < 28, ru))  # interest of straight

            ur_straight = list(filter(lambda v: v['x'] < ego_x + 7 and ego_y < v['y'] < 28, ur))  # interest of straight
            ur_right = list(filter(lambda v: v['x'] < 28 and v['y'] < 18, ur))  # interest of right
            ud = list(filter(lambda v: ego_y < v['y'] < 18 and ego_x > v['x'], ud))  # interest of left
            ul = list(filter(lambda v: v['x'] > -28 and v['y'] < 28, ul))  # interest of left

            lu = lu  # not interest in case of traffic light
            lr = list(filter(lambda v: -28 < v['x'] < 28, lr))  # interest of right
            ld = ld  # not interest in case of traffic light

            # sort
            dl = sorted(dl, key=lambda v: (v['y'], -v['x']))
            du = sorted(du, key=lambda v: v['y'])
            dr = sorted(dr, key=lambda v: (v['y'], v['x']))

            ru = sorted(ru, key=lambda v: (-v['x'], v['y']), reverse=True)

            ur_straight = sorted(ur_straight, key=lambda v: v['y'])
            ur_right = sorted(ur_right, key=lambda v: (-v['y'], v['x']), reverse=True)

            ud = sorted(ud, key=lambda v: v['y'])
            ul = sorted(ul, key=lambda v: (-v['y'], -v['x']), reverse=True)

            lr = sorted(lr, key=lambda v: -v['x'])

            # slice or fill to some number
            def slice_or_fill(sorted_list, fill_value, num):
                if len(sorted_list) >= num:
                    return sorted_list[:num]
                else:
                    while len(sorted_list) < num:
                        sorted_list.append(fill_value)
                    return sorted_list

            fill_value_for_dl = dict(x=-40, y=1.875, v=0, phi=180, w=2.5, l=5, route=('1o', '4i'))
            fill_value_for_du = dict(x=1.875, y=40, v=0, phi=90, w=2.5, l=5, route=('1o', '3i'))
            fill_value_for_dr = dict(x=40, y=-5.625, v=0, phi=0, w=2.5, l=5, route=('1o', '2i'))

            fill_value_for_ru = dict(x=35, y=5.625, v=0, phi=180, w=2.5, l=5, route=('2o', '3i'))

            fill_value_for_ur_straight = dict(x=-1.875, y=40, v=0, phi=-90, w=2.5, l=5, route=('3o', '2i'))
            fill_value_for_ur_right = dict(x=-1.875, y=40, v=0, phi=-90, w=2.5, l=5, route=('3o', '2i'))

            fill_value_for_ud = dict(x=-1.875, y=40, v=0, phi=-90, w=2.5, l=5, route=('3o', '1i'))
            fill_value_for_ul = dict(x=-5.625, y=40, v=0, phi=-90, w=2.5, l=5, route=('3o', '4i'))

            fill_value_for_lr = dict(x=-40, y=-1.875, v=0, phi=0, w=2.5, l=5, route=('4o', '2i'))

            tmp = OrderedDict()
            veh_mode_list = []
            if task == 'left':
                tmp['dl'] = slice_or_fill(dl, fill_value_for_dl, 2)
                tmp['du'] = slice_or_fill(du, fill_value_for_du, 2)
                tmp['ud'] = slice_or_fill(ud, fill_value_for_ud, 3)
                tmp['ul'] = slice_or_fill(ul, fill_value_for_ul, 3)
                veh_mode_list = [('dl', 2), ('du', 2), ('ud', 3), ('ul', 3)]
            elif task == 'straight':
                tmp['dl'] = slice_or_fill(dl, fill_value_for_dl, 2)
                tmp['du'] = slice_or_fill(du, fill_value_for_du, 2)
                tmp['ud'] = slice_or_fill(ud, fill_value_for_ud, 2)
                tmp['ru'] = slice_or_fill(ru, fill_value_for_ru, 3)
                tmp['ur'] = slice_or_fill(ur_straight, fill_value_for_ur_straight, 3)
                veh_mode_list = [('dl', 2), ('du', 2), ('ud', 2), ('ru', 3), ('ur', 3)]
            elif task == 'right':
                tmp['dr'] = slice_or_fill(dr, fill_value_for_dr, 2)
                tmp['ur'] = slice_or_fill(ur_right, fill_value_for_ur_right, 3)
                tmp['lr'] = slice_or_fill(lr, fill_value_for_lr, 3)
                veh_mode_list = [('dr', 2), ('ur', 3), ('lr', 3)]

            return tmp, veh_mode_list

        list_of_interested_veh_dict = []
        self.interested_vehs, self.veh_mode_list = filter_interested_vehicles(self.all_vehicles, self.training_task)
        for part in list(self.interested_vehs.values()):
            list_of_interested_veh_dict.extend(part)

        for veh in list_of_interested_veh_dict:
            veh_x, veh_y, veh_v, veh_phi, veh_l, veh_w = veh['x'], veh['y'], veh['v'],\
                                                         veh['phi'], veh['l'], veh['w']
            vehs_vector.extend([veh_x, veh_y, veh_v, veh_phi, veh_l, veh_w])
        self.per_veh_info_dim = 6
        return np.array(vehs_vector, dtype=np.float32)

    def _construct_veh_vector(self, exit_='D'):
        ego_v_x = self.ego_dynamics['v_x']
        ego_v_y = self.ego_dynamics['v_y']
        ego_r = self.ego_dynamics['r']
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_phi = self.ego_dynamics['phi']
        ego_l = self.ego_dynamics['l']
        ego_w = self.ego_dynamics['w']
        ego_alpha_f = self.ego_dynamics['alpha_f']
        ego_alpha_r = self.ego_dynamics['alpha_r']
        ego_miu_f = self.ego_dynamics['miu_f']
        ego_miu_r = self.ego_dynamics['miu_r']
        v_light = self.v_light
        vehs_vector = []
        ego_lw = (ego_l - ego_w) / 2.
        coeff = 1.14
        rho_ego = ego_w / 2. * coeff
        ego_front_point = tf.cast(ego_x + ego_lw * tf.cos(ego_phi * np.pi / 180.), dtype=tf.float32), \
                          tf.cast(ego_y + ego_lw * tf.sin(ego_phi * np.pi / 180.), dtype=tf.float32)
        ego_rear_point = tf.cast(ego_x - ego_lw * tf.cos(ego_phi * np.pi / 180.), dtype=tf.float32), \
                         tf.cast(ego_y - ego_lw * tf.sin(ego_phi * np.pi / 180.), dtype=tf.float32)

        name_settings = dict(D=dict(do='1o', di='1i', ro='2o', ri='2i', uo='3o', ui='3i', lo='4o', li='4i'),
                             R=dict(do='2o', di='2i', ro='3o', ri='3i', uo='4o', ui='4i', lo='1o', li='1i'),
                             U=dict(do='3o', di='3i', ro='4o', ri='4i', uo='1o', ui='1i', lo='2o', li='2i'),
                             L=dict(do='4o', di='4i', ro='1o', ri='1i', uo='2o', ui='2i', lo='3o', li='3i'))

        name_setting = name_settings[exit_]

        def filter_interested_vehicles(vs, task):
            dl, du, dr, rd, rl, ru, ur, ud, ul, lu, lr, ld = [], [], [], [], [], [], [], [], [], [], [], []
            for v in vs:
                route_list = v['route']
                start = route_list[0]
                end = route_list[1]
                if start == name_setting['do'] and end == name_setting['li']:
                    dl.append(v)
                elif start == name_setting['do'] and end == name_setting['ui']:
                    du.append(v)
                elif start == name_setting['do'] and end == name_setting['ri']:
                    dr.append(v)

                elif start == name_setting['ro'] and end == name_setting['di']:
                    rd.append(v)
                elif start == name_setting['ro'] and end == name_setting['li']:
                    rl.append(v)
                elif start == name_setting['ro'] and end == name_setting['ui']:
                    ru.append(v)

                elif start == name_setting['uo'] and end == name_setting['ri']:
                    ur.append(v)
                elif start == name_setting['uo'] and end == name_setting['di']:
                    ud.append(v)
                elif start == name_setting['uo'] and end == name_setting['li']:
                    ul.append(v)

                elif start == name_setting['lo'] and end == name_setting['ui']:
                    lu.append(v)
                elif start == name_setting['lo'] and end == name_setting['ri']:
                    lr.append(v)
                elif start == name_setting['lo'] and end == name_setting['di']:
                    ld.append(v)
            # fetch veh in range
            dl = list(filter(lambda v: v['x'] > -28 and v['y'] > ego_y, dl))  # interest of left straight
            du = list(filter(lambda v: ego_y < v['y'] < 28, du))  # interest of left straight
            if v_light != 0 and ego_y < -18:
                du.append(dict(x=1.875, y=-18, v=0, phi=90, l=5, w=2.5, route=None))

            dr = list(filter(lambda v: v['x'] < 28 and v['y'] > ego_y, dr))  # interest of right

            rd = rd  # not interest in case of traffic light
            rl = rl  # not interest in case of traffic light
            ru = list(filter(lambda v: v['x'] < 28 and v['y'] < 28, ru))  # interest of straight

            ur_straight = list(filter(lambda v: v['x'] < ego_x + 7 and ego_y < v['y'] < 28, ur))  # interest of straight
            ur_right = list(filter(lambda v: v['x'] < 28 and v['y'] < 18, ur))  # interest of right
            ud = list(filter(lambda v: ego_y < v['y'] < 28, ud))  # interest of left
            ul = list(filter(lambda v: v['x'] > -28 and v['y'] < 28, ul))  # interest of left

            lu = lu  # not interest in case of traffic light
            lr = list(filter(lambda v: -28 < v['x'] < 28, lr))  # interest of right
            ld = ld  # not interest in case of traffic light

            # sort
            dl = sorted(dl, key=lambda v: (v['y'], -v['x']))
            du = sorted(du, key=lambda v: v['y'])
            dr = sorted(dr, key=lambda v: (v['y'], v['x']))

            ru = sorted(ru, key=lambda v: (-v['x'], v['y']), reverse=True)

            ur_straight = sorted(ur_straight, key=lambda v: v['y'])
            ur_right = sorted(ur_right, key=lambda v: (-v['y'], v['x']), reverse=True)

            ud = sorted(ud, key=lambda v: v['y'])
            ul = sorted(ul, key=lambda v: (-v['y'], -v['x']), reverse=True)

            lr = sorted(lr, key=lambda v: -v['x'])

            # slice or fill to some number
            def slice_or_fill(sorted_list, fill_value, num):
                if len(sorted_list) >= num:
                    return sorted_list[:num]
                else:
                    while len(sorted_list) < num:
                        sorted_list.append(fill_value)
                    return sorted_list

            fill_value_for_dl = dict(x=-40, y=1.875, v=0, phi=180, w=2.5, l=5, route=('1o', '4i'))
            fill_value_for_du = dict(x=1.875, y=40, v=0, phi=90, w=2.5, l=5, route=('1o', '3i'))
            fill_value_for_dr = dict(x=40, y=-5.625, v=0, phi=0, w=2.5, l=5, route=('1o', '2i'))

            fill_value_for_ru = dict(x=35, y=5.625, v=0, phi=180, w=2.5, l=5, route=('2o', '3i'))

            fill_value_for_ur_straight = dict(x=-1.875, y=40, v=0, phi=-90, w=2.5, l=5, route=('3o', '2i'))
            fill_value_for_ur_right = dict(x=-1.875, y=40, v=0, phi=-90, w=2.5, l=5, route=('3o', '2i'))

            fill_value_for_ud = dict(x=-1.875, y=40, v=0, phi=-90, w=2.5, l=5, route=('3o', '1i'))
            fill_value_for_ul = dict(x=-5.625, y=40, v=0, phi=-90, w=2.5, l=5, route=('3o', '4i'))

            fill_value_for_lr = dict(x=-40, y=-1.875, v=0, phi=0, w=2.5, l=5, route=('4o', '2i'))

            tmp = OrderedDict()
            veh_mode_list = []
            if task == 'left':
                tmp['dl'] = slice_or_fill(dl, fill_value_for_dl, 2)
                tmp['du'] = slice_or_fill(du, fill_value_for_du, 2)
                tmp['ud'] = slice_or_fill(ud, fill_value_for_ud, 3)
                tmp['ul'] = slice_or_fill(ul, fill_value_for_ul, 3)
                veh_mode_list = [('dl', 2), ('du', 2), ('ud', 3), ('ul', 3)]
            elif task == 'straight':
                tmp['dl'] = slice_or_fill(dl, fill_value_for_dl, 2)
                tmp['du'] = slice_or_fill(du, fill_value_for_du, 2)
                tmp['ud'] = slice_or_fill(ud, fill_value_for_ud, 2)
                tmp['ru'] = slice_or_fill(ru, fill_value_for_ru, 3)
                tmp['ur'] = slice_or_fill(ur_straight, fill_value_for_ur_straight, 3)
                veh_mode_list = [('dl', 2), ('du', 2), ('ud', 2), ('ru', 3), ('ur', 3)]
            elif task == 'right':
                tmp['dr'] = slice_or_fill(dr, fill_value_for_dr, 2)
                tmp['ur'] = slice_or_fill(ur_right, fill_value_for_ur_right, 3)
                tmp['lr'] = slice_or_fill(lr, fill_value_for_lr, 3)
                veh_mode_list = [('dr', 2), ('ur', 3), ('lr', 3)]

            return tmp, veh_mode_list

        list_of_interested_veh_dict = []
        self.interested_vehs, self.veh_mode_list = filter_interested_vehicles(self.all_vehicles, self.training_task)
        for part in list(self.interested_vehs.values()):
            list_of_interested_veh_dict.extend(part)

        for veh in list_of_interested_veh_dict:
            veh_x, veh_y, veh_v, veh_phi, veh_l, veh_w = veh['x'], veh['y'], veh['v'],\
                                                         veh['phi'], veh['l'], veh['w']
            veh_lw = (veh_l - veh_w) / 2.
            rho_veh = veh_w / 2. * coeff
            veh_front_point = tf.cast(veh_x + veh_lw * tf.cos(veh_phi * np.pi / 180.), dtype=tf.float32), \
                               tf.cast(veh_y + veh_lw * tf.sin(veh_phi * np.pi / 180.), dtype=tf.float32)
            veh_rear_point = tf.cast(veh_x - veh_lw * tf.cos(veh_phi * np.pi / 180.), dtype=tf.float32), \
                              tf.cast(veh_y - veh_lw * tf.sin(veh_phi * np.pi / 180.), dtype=tf.float32)
            veh2veh_dist = []
            for ego_point in [ego_front_point, ego_rear_point]:
                for veh_point in [veh_front_point, veh_rear_point]:
                    point_dist = tf.sqrt(
                        tf.square(ego_point[0] - veh_point[0]) + tf.square(ego_point[1] - veh_point[1])) - \
                                   tf.convert_to_tensor(rho_ego + rho_veh, dtype=tf.float32)
                    veh2veh_dist.append(point_dist)
            vehs_vector.extend([veh_x, veh_y, veh_v, veh_phi, veh_l, veh_w]+veh2veh_dist)
        self.per_veh_info_dim = 10
        return np.array(vehs_vector, dtype=np.float32)

    def recover_orig_position_fn(self, transformed_x, transformed_y, x, y, d):  # x, y, d are used to transform
        # coordination
        transformed_x, transformed_y, _ = rotate_coordination(transformed_x, transformed_y, 0, -d)
        orig_x, orig_y = shift_coordination(transformed_x, transformed_y, -x, -y)
        return orig_x, orig_y

    def _reset_init_state(self):
        random_index = int(np.random.random()*(len(self.ref_path.path[0])-600)) + 100
        # random_index = 1200
        x, y, phi = self.ref_path.indexs2points(random_index)
        # v = 7 + 6 * np.random.random()
        v = 8 * np.random.random()
        if self.training_task == 'left':
            routeID = 'dl'
        elif self.training_task == 'straight':
            routeID = 'du'
        else:
            assert self.training_task == 'right'
            routeID = 'dr'
        return dict(ego=dict(x=x.numpy(),
                             y=y.numpy(),
                             v_x=v,
                             v_y=0,
                             r=0,
                             phi=phi.numpy(),
                             l=4.8,
                             w=1.8,
                             routeID=routeID,
                             alpha_f=0.,
                             alpha_r=0.,
                             miu_f=self.dynamics.vehicle_params['miu'],
                             miu_r=self.dynamics.vehicle_params['miu']))

    # def compute_reward(self, obs, action):
    #     ego_infos, tracking_infos, veh_infos = obs[:12], obs[12:12 + 3 + 3 * self.num_future_data], \
    #                                            obs[12 + 3 + 3 * self.num_future_data:]
    #     steers, a_xs = action[0], action[1]
    #
    #     # rewards related to ego stability
    #     alpha_f, alpha_r, miu_f, miu_r = ego_infos[8], ego_infos[9], ego_infos[10], ego_infos[11]
    #     alpha_f_bound, alpha_r_bound = self.ego_dynamics['alpha_f_bound'], self.ego_dynamics['alpha_r_bound']
    #     r_bound = self.ego_dynamics['r_bound']
    #
    #     rew_alpha_f = - tf.cast(tf.nn.relu(tf.abs(alpha_f) - alpha_f_bound), dtype=tf.float32)
    #     rew_alpha_r = - tf.cast(tf.nn.relu(tf.abs(alpha_r) - alpha_r_bound), dtype=tf.float32)
    #     rew_r = - tf.cast(tf.nn.relu(tf.abs(ego_infos[2]) - r_bound), dtype=tf.float32)
    #
    #     # rew_alpha_f = -1 / tf.cast(tf.square(alpha_f - alpha_f_bound), dtype=tf.float32)
    #     # rew_alpha_r = -1 / tf.cast(tf.square(alpha_r - alpha_r_bound), dtype=tf.float32)
    #     # rew_r = -1 / tf.cast(tf.square(ego_infos[2] - r_bound), dtype=tf.float32)
    #
    #     # rewards related to action
    #     punish_steer = -tf.square(steers)
    #     punish_a_x = -tf.square(a_xs)
    #
    #     # rewards related to ego stability
    #     punish_yaw_rate = -tf.square(ego_infos[2])
    #
    #     # rewards related to tracking error
    #     devi_v = -tf.cast(tf.square(ego_infos[0] - self.exp_v), dtype=tf.float32)
    #     devi_y = -tf.square(tracking_infos[0]) - tf.square(tracking_infos[1])
    #     devi_phi = -tf.cast(tf.square(tracking_infos[2] * np.pi / 180.), dtype=tf.float32)
    #
    #     # rewards related to veh2road collision
    #     ego_lw = (ego_infos[6] - ego_infos[7]) / 2.
    #     rho_ego = ego_infos[7] / 2.
    #     ego_front_point = tf.cast(ego_infos[3] + ego_lw * tf.cos(ego_infos[5] * np.pi / 180.), dtype=tf.float32),\
    #                       tf.cast(ego_infos[4] + ego_lw * tf.sin(ego_infos[5] * np.pi / 180.), dtype=tf.float32)
    #     ego_rear_point = tf.cast(ego_infos[3] - ego_lw * tf.cos(ego_infos[5] * np.pi / 180.), dtype=tf.float32), \
    #                      tf.cast(ego_infos[4] - ego_lw * tf.sin(ego_infos[5] * np.pi / 180.), dtype=tf.float32)
    #     if self.training_task == 'left':
    #         veh2road = tf.constant(0.)
    #         for ego_point in [ego_front_point, ego_rear_point]:
    #             before1 = 1 / tf.square(ego_point[0] - 0 - rho_ego) if ego_point[1] < -18 else tf.constant(0.)
    #             before2 = 1 / tf.square(3.75 - ego_point[0] - rho_ego) if ego_point[1] < -18 else tf.constant(0.)
    #             middle_cond = True if -18 < ego_point[0] < 18 and -18 < ego_point[1] < 18 else False
    #             middle1 = 1 / tf.square(18 - ego_point[1] - rho_ego) if middle_cond else tf.constant(0.)
    #             middle2 = 1 / tf.square(18 - ego_point[0] - rho_ego) if middle_cond else tf.constant(0.)
    #             middle3 = 1 / tf.square(ego_point[0] - (-18) - rho_ego) if middle_cond and ego_point[1] > 7.5 else tf.constant(0.)
    #             middle4 = 1 / tf.square(ego_point[0] - (-18) - rho_ego) if middle_cond and ego_point[1] < 0 else tf.constant(0.)
    #             middle5 = 1 / tf.square(ego_point[1] - (-18) - rho_ego) if middle_cond and ego_point[0] < 0 else tf.constant(0.)
    #             middle6 = 1 / tf.square(ego_point[1] - (-18) - rho_ego) if middle_cond and ego_point[0] > 3.75 else tf.constant(0.)
    #             after1 = 1 / tf.square(ego_point[1] - 0 - rho_ego) if ego_point[0] < -18 else tf.constant(0.)
    #             after2 = 1 / tf.square(7.5 - ego_point[1] - rho_ego) if ego_point[0] < -18 else tf.constant(0.)
    #
    #             this_point = before1 + before2 + middle1 + middle2 + middle3 + middle4 + middle5 + middle6 + after1 + after2
    #             veh2road -= this_point
    #
    #     elif self.training_task == 'straight':
    #         veh2road = tf.constant(0.)
    #         for ego_point in [ego_front_point, ego_rear_point]:
    #             before1 = 1 / tf.square(ego_point[0] - 0 - rho_ego) if ego_point[1] < -18 else tf.constant(0.)
    #             before2 = 1 / tf.square(3.75 - ego_point[0] - rho_ego) if ego_point[1] < -18 else tf.constant(0.)
    #             middle_cond = True if -18 < ego_point[0] < 18 and -18 < ego_point[1] < 18 else False
    #             middle1 = 1 / tf.square(ego_point[0] - (-18) - rho_ego) if middle_cond else tf.constant(0.)
    #             middle2 = 1 / tf.square(18 - ego_point[0] - rho_ego) if middle_cond else tf.constant(0.)
    #             middle3 = 1 / tf.square(18 - ego_point[1] - rho_ego) if middle_cond and ego_point[0] < 0 else tf.constant(0.)
    #             middle4 = 1 / tf.square(18 - ego_point[1] - rho_ego) if middle_cond and ego_point[0] > 7.5 else tf.constant(0.)
    #             middle5 = 1 / tf.square(ego_point[1] - (-18) - rho_ego) if middle_cond and ego_point[0] < 0 else tf.constant(0.)
    #             middle6 = 1 / tf.square(ego_point[1] - (-18) - rho_ego) if middle_cond and ego_point[0] > 3.75 else tf.constant(0.)
    #             after1 = 1 / tf.square(ego_point[0] - 0 - rho_ego) if ego_point[1] > 18 else tf.constant(0.)
    #             after2 = 1 / tf.square(7.5 - ego_point[0] - rho_ego) if ego_point[1] > 18 else tf.constant(0.)
    #
    #             this_point = before1 + before2 + middle1 + middle2 + middle3 + middle4 + middle5 + middle6 + after1 + after2
    #             veh2road -= this_point
    #
    #     else:
    #         veh2road = tf.constant(0.)
    #         assert self.training_task == 'right'
    #         for ego_point in [ego_front_point, ego_rear_point]:
    #             before1 = 1 / tf.square(ego_point[0] - 3.75 - rho_ego) if ego_point[1] < -18 else tf.constant(0.)
    #             before2 = 1 / tf.square(7.5 - ego_point[0] - rho_ego) if ego_point[1] < -18 else tf.constant(0.)
    #             middle_cond = True if -18 < ego_point[0] < 18 and -18 < ego_point[1] < 18 else False
    #             middle1 = 1 / tf.square(ego_point[0] - (-18) - rho_ego) if middle_cond else tf.constant(0.)
    #             middle2 = 1 / tf.square(18 - ego_point[1] - rho_ego) if middle_cond else tf.constant(0.)
    #             middle3 = 1 / tf.square(18 - ego_point[0] - rho_ego) if middle_cond and ego_point[1] > 0 else tf.constant(0.)
    #             middle4 = 1 / tf.square(18 - ego_point[0] - rho_ego) if middle_cond and ego_point[1] < -7.5 else tf.constant(0.)
    #             middle5 = 1 / tf.square(ego_point[1] - (-18) - rho_ego) if middle_cond and ego_point[0] > 7.5 else tf.constant(0.)
    #             middle6 = 1 / tf.square(ego_point[1] - (-18) - rho_ego) if middle_cond and ego_point[0] < 3.75 else tf.constant(0.)
    #             after1 = 1 / tf.square(0 - ego_point[1] - rho_ego) if ego_point[0] > 18 else tf.constant(0.)
    #             after2 = 1 / tf.square(ego_point[1] - (-7.5) - rho_ego) if ego_point[0] > 18 else tf.constant(0.)
    #
    #             this_point = before1 + before2 + middle1 + middle2 + middle3 + middle4 + middle5 + middle6 + after1 + after2
    #             veh2road -= this_point
    #     # rewards related to veh2veh collision
    #     veh2veh = tf.constant(0.)
    #     for veh_index in range(int(len(veh_infos) / 6)):
    #         veh = veh_infos[veh_index*6:6 * (veh_index + 1)]
    #         veh_lw = (veh[4] - veh[5]) / 2.
    #         rho_veh = veh[5] / 2.
    #         veh_front_point = tf.cast(veh[0] + veh_lw * tf.cos(veh[3] * np.pi / 180.), dtype=tf.float32), \
    #                            tf.cast(veh[1] + veh_lw * tf.sin(veh[3] * np.pi / 180.), dtype=tf.float32)
    #         veh_rear_point = tf.cast(veh[0] - veh_lw * tf.cos(veh[3] * np.pi / 180.), dtype=tf.float32), \
    #                           tf.cast(veh[1] - veh_lw * tf.sin(veh[3] * np.pi / 180.), dtype=tf.float32)
    #         for ego_point in [ego_front_point, ego_rear_point]:
    #             for veh_point in [veh_front_point, veh_rear_point]:
    #                 veh2veh_dist = tf.sqrt(tf.square(ego_point[0] - veh_point[0]) + tf.square(
    #                     ego_point[1] - veh_point[1])) - tf.convert_to_tensor(rho_ego + rho_veh, dtype=tf.float32)
    #                 veh2veh -= 1. / tf.square(veh2veh_dist)
    #
    #     reward_dict = dict(punish_steer=punish_steer.numpy(),
    #                        punish_a_x=punish_a_x.numpy(),
    #                        punish_yaw_rate=punish_yaw_rate.numpy(),
    #                        devi_v=devi_v.numpy(),
    #                        devi_y=devi_y.numpy(),
    #                        devi_phi=devi_phi.numpy(),
    #                        veh2road=veh2road.numpy(),
    #                        veh2veh=veh2veh.numpy(),
    #                        rew_alpha_f=rew_alpha_f.numpy(),
    #                        rew_alpha_r=rew_alpha_r.numpy(),
    #                        rew_r=rew_r.numpy()
    #                        )
    #     # print(reward_dict)
    #     # rew_alpha_f = -10000. if rew_alpha_f < -10000. else rew_alpha_f
    #     # rew_alpha_r = -10000. if rew_alpha_r < -10000. else rew_alpha_r
    #     # rew_r = -10000. if rew_r < -10000. else rew_r
    #     veh2road = -10000. if veh2road < -10000. else veh2road
    #     veh2veh = -10000. if veh2veh < -10000. else veh2veh
    #
    #     rewards = 0.01 * devi_v + 0.04 * devi_y + devi_phi + 0.02 * punish_yaw_rate + \
    #               0.05 * punish_steer + 0.0005 * punish_a_x + 0.1 * veh2road + 0.1 * veh2veh + \
    #               100 * rew_alpha_f + 100 * rew_alpha_r + 100 * rew_r
    #     return rewards.numpy(), reward_dict

    # def compute_reward2(self, obs, action):
    #     ego_infos, tracking_infos, veh_infos = obs[:12], obs[12:12 + 3 + 3 * self.num_future_data], \
    #                                            obs[12 + 3 + 3 * self.num_future_data:]
    #     steers, a_xs = action[0], action[1]
    #
    #     # rewards related to ego stability
    #     alpha_f, alpha_r, miu_f, miu_r = ego_infos[8], ego_infos[9], ego_infos[10], ego_infos[11]
    #     alpha_f_bound, alpha_r_bound = self.ego_dynamics['alpha_f_bound'], self.ego_dynamics['alpha_r_bound']
    #     r_bound = self.ego_dynamics['r_bound']
    #
    #     rew_alpha_f = - tf.cast(tf.nn.relu(tf.abs(alpha_f) - alpha_f_bound), dtype=tf.float32)
    #     rew_alpha_r = - tf.cast(tf.nn.relu(tf.abs(alpha_r) - alpha_r_bound), dtype=tf.float32)
    #     rew_r = - tf.cast(tf.nn.relu(tf.abs(ego_infos[2]) - r_bound), dtype=tf.float32)
    #
    #     # rew_alpha_f = -1 / tf.cast(tf.square(alpha_f - alpha_f_bound), dtype=tf.float32)
    #     # rew_alpha_r = -1 / tf.cast(tf.square(alpha_r - alpha_r_bound), dtype=tf.float32)
    #     # rew_r = -1 / tf.cast(tf.square(ego_infos[2] - r_bound), dtype=tf.float32)
    #
    #     # rewards related to action
    #     punish_steer = -tf.square(steers)
    #     punish_a_x = -tf.square(a_xs)
    #
    #     # rewards related to ego stability
    #     punish_yaw_rate = -tf.square(ego_infos[2])
    #
    #     # rewards related to tracking error
    #     devi_v = -tf.cast(tf.square(ego_infos[0] - self.exp_v), dtype=tf.float32)
    #     devi_y = -tf.square(tracking_infos[0]) - tf.square(tracking_infos[1])
    #     devi_phi = -tf.cast(tf.square(tracking_infos[2] * np.pi / 180.), dtype=tf.float32)
    #
    #     # rewards related to veh2road collision
    #     ego_lw = (ego_infos[6] - ego_infos[7]) / 2.
    #     coeff = 1.14
    #     rho_ego = ego_infos[7] / 2. * coeff
    #     ego_front_point = tf.cast(ego_infos[3] + ego_lw * tf.cos(ego_infos[5] * np.pi / 180.), dtype=tf.float32),\
    #                       tf.cast(ego_infos[4] + ego_lw * tf.sin(ego_infos[5] * np.pi / 180.), dtype=tf.float32)
    #     ego_rear_point = tf.cast(ego_infos[3] - ego_lw * tf.cos(ego_infos[5] * np.pi / 180.), dtype=tf.float32), \
    #                      tf.cast(ego_infos[4] - ego_lw * tf.sin(ego_infos[5] * np.pi / 180.), dtype=tf.float32)
    #     if self.training_task == 'left':
    #         veh2road = tf.constant(0.)
    #         for ego_point in [ego_front_point, ego_rear_point]:
    #             before1 = tf.nn.relu(-(ego_point[0] - 0 - rho_ego)) if ego_point[1] < -18 else tf.constant(0.)
    #             before2 = tf.nn.relu(-(3.75 - ego_point[0] - rho_ego)) if ego_point[1] < -18 else tf.constant(0.)
    #             middle_cond = True if -18 < ego_point[0] < 18 and -18 < ego_point[1] < 18 else False
    #             middle1 = tf.nn.relu(-(18 - ego_point[1] - rho_ego)) if middle_cond else tf.constant(0.)
    #             middle2 = tf.nn.relu(-(18 - ego_point[0] - rho_ego)) if middle_cond else tf.constant(0.)
    #             middle3 = tf.nn.relu(-(ego_point[0] - (-18) - rho_ego)) if middle_cond and ego_point[1] > 7.5 else tf.constant(0.)
    #             middle4 = tf.nn.relu(-(ego_point[0] - (-18) - rho_ego)) if middle_cond and ego_point[1] < 0 else tf.constant(0.)
    #             middle5 = tf.nn.relu(-(ego_point[1] - (-18) - rho_ego)) if middle_cond and ego_point[0] < 0 else tf.constant(0.)
    #             middle6 = tf.nn.relu(-(ego_point[1] - (-18) - rho_ego)) if middle_cond and ego_point[0] > 3.75 else tf.constant(0.)
    #
    #             middle7 = tf.nn.relu(-(tf.sqrt(tf.square(ego_point[0]-(-18)) + tf.square(ego_point[1]-0)) - rho_ego)) if middle_cond else tf.constant(0.)
    #             middle8 = tf.nn.relu(-(tf.sqrt(tf.square(ego_point[0]-(-18)) + tf.square(ego_point[1]-7.5)) - rho_ego)) if middle_cond else tf.constant(0.)
    #
    #             after1 = tf.nn.relu(-(ego_point[1] - 0 - rho_ego)) if ego_point[0] < -18 else tf.constant(0.)
    #             after2 = tf.nn.relu(-(7.5 - ego_point[1] - rho_ego)) if ego_point[0] < -18 else tf.constant(0.)
    #
    #             this_point = before1 + before2 +\
    #                          middle1 + middle2 + middle3 + middle4 + middle5 + middle6 + middle7 + middle8 +\
    #                          after1 + after2
    #             veh2road -= this_point
    #
    #     elif self.training_task == 'straight':
    #         veh2road = tf.constant(0.)
    #         for ego_point in [ego_front_point, ego_rear_point]:
    #             before1 = tf.nn.relu(-(ego_point[0] - 0 - rho_ego)) if ego_point[1] < -18 else tf.constant(0.)
    #             before2 = tf.nn.relu(-(3.75 - ego_point[0] - rho_ego)) if ego_point[1] < -18 else tf.constant(0.)
    #             middle_cond = True if -18 < ego_point[0] < 18 and -18 < ego_point[1] < 18 else False
    #             middle1 = tf.nn.relu(-(ego_point[0] - (-18) - rho_ego)) if middle_cond else tf.constant(0.)
    #             middle2 = tf.nn.relu(-(18 - ego_point[0] - rho_ego)) if middle_cond else tf.constant(0.)
    #             middle3 = tf.nn.relu(-(18 - ego_point[1] - rho_ego)) if middle_cond and ego_point[0] < 0 else tf.constant(0.)
    #             middle4 = tf.nn.relu(-(18 - ego_point[1] - rho_ego)) if middle_cond and ego_point[0] > 7.5 else tf.constant(0.)
    #             middle5 = tf.nn.relu(-(ego_point[1] - (-18) - rho_ego)) if middle_cond and ego_point[0] < 0 else tf.constant(0.)
    #             middle6 = tf.nn.relu(-(ego_point[1] - (-18) - rho_ego)) if middle_cond and ego_point[0] > 3.75 else tf.constant(0.)
    #             middle7 = tf.nn.relu(-(tf.sqrt(tf.square(ego_point[0] - 0) + tf.square(
    #                 ego_point[1] - 18)) - rho_ego)) if middle_cond else tf.constant(0.)
    #             middle8 = tf.nn.relu(-(tf.sqrt(tf.square(ego_point[0] - 7.5) + tf.square(
    #                 ego_point[1] - 18)) - rho_ego)) if middle_cond else tf.constant(0.)
    #
    #             after1 = tf.nn.relu(-(ego_point[0] - 0 - rho_ego)) if ego_point[1] > 18 else tf.constant(0.)
    #             after2 = tf.nn.relu(-(7.5 - ego_point[0] - rho_ego)) if ego_point[1] > 18 else tf.constant(0.)
    #
    #             this_point = before1 + before2 +\
    #                          middle1 + middle2 + middle3 + middle4 + middle5 + middle6 + middle7 + middle8 + \
    #                          after1 + after2
    #             veh2road -= this_point
    #
    #     else:
    #         veh2road = tf.constant(0.)
    #         assert self.training_task == 'right'
    #         for ego_point in [ego_front_point, ego_rear_point]:
    #             before1 = tf.nn.relu(-(ego_point[0] - 3.75 - rho_ego)) if ego_point[1] < -18 else tf.constant(0.)
    #             before2 = tf.nn.relu(-(7.5 - ego_point[0] - rho_ego)) if ego_point[1] < -18 else tf.constant(0.)
    #             middle_cond = True if -18 < ego_point[0] < 18 and -18 < ego_point[1] < 18 else False
    #             middle1 = tf.nn.relu(-(ego_point[0] - (-18) - rho_ego)) if middle_cond else tf.constant(0.)
    #             middle2 = tf.nn.relu(-(18 - ego_point[1] - rho_ego)) if middle_cond else tf.constant(0.)
    #             middle3 = tf.nn.relu(-(18 - ego_point[0] - rho_ego)) if middle_cond and ego_point[1] > 0 else tf.constant(0.)
    #             middle4 = tf.nn.relu(-(18 - ego_point[0] - rho_ego)) if middle_cond and ego_point[1] < -7.5 else tf.constant(0.)
    #             middle5 = tf.nn.relu(-(ego_point[1] - (-18) - rho_ego)) if middle_cond and ego_point[0] > 7.5 else tf.constant(0.)
    #             middle6 = tf.nn.relu(-(ego_point[1] - (-18) - rho_ego)) if middle_cond and ego_point[0] < 3.75 else tf.constant(0.)
    #             middle7 = tf.nn.relu(-(tf.sqrt(tf.square(ego_point[0] - 18) + tf.square(
    #                 ego_point[1] - 0)) - rho_ego)) if middle_cond else tf.constant(0.)
    #             middle8 = tf.nn.relu(-(tf.sqrt(tf.square(ego_point[0] - 18) + tf.square(
    #                 ego_point[1] - (-7.5))) - rho_ego)) if middle_cond else tf.constant(0.)
    #
    #             after1 = tf.nn.relu(-(0 - ego_point[1] - rho_ego)) if ego_point[0] > 18 else tf.constant(0.)
    #             after2 = tf.nn.relu(-(ego_point[1] - (-7.5) - rho_ego)) if ego_point[0] > 18 else tf.constant(0.)
    #
    #             this_point = before1 + before2 + \
    #                          middle1 + middle2 + middle3 + middle4 + middle5 + middle6 + middle7 + middle8+\
    #                          after1 + after2
    #             veh2road -= this_point
    #     # rewards related to veh2veh collision
    #     veh2veh = tf.constant(0.)
    #     for veh_index in range(int(len(veh_infos) / 6)):
    #         veh = veh_infos[veh_index*6:6 * (veh_index + 1)]
    #         veh_lw = (veh[4] - veh[5]) / 2.
    #         rho_veh = veh[5] / 2. * coeff
    #         veh_front_point = tf.cast(veh[0] + veh_lw * tf.cos(veh[3] * np.pi / 180.), dtype=tf.float32), \
    #                            tf.cast(veh[1] + veh_lw * tf.sin(veh[3] * np.pi / 180.), dtype=tf.float32)
    #         veh_rear_point = tf.cast(veh[0] - veh_lw * tf.cos(veh[3] * np.pi / 180.), dtype=tf.float32), \
    #                           tf.cast(veh[1] - veh_lw * tf.sin(veh[3] * np.pi / 180.), dtype=tf.float32)
    #         for ego_point in [ego_front_point, ego_rear_point]:
    #             for veh_point in [veh_front_point, veh_rear_point]:
    #                 veh2veh_dist = tf.sqrt(tf.square(ego_point[0] - veh_point[0]) + tf.square(
    #                     ego_point[1] - veh_point[1])) - tf.convert_to_tensor(rho_ego + rho_veh, dtype=tf.float32)
    #                 veh2veh -= 1. / tf.square(veh2veh_dist)
    #
    #     reward_dict = dict(punish_steer=punish_steer.numpy(),
    #                        punish_a_x=punish_a_x.numpy(),
    #                        punish_yaw_rate=punish_yaw_rate.numpy(),
    #                        devi_v=devi_v.numpy(),
    #                        devi_y=devi_y.numpy(),
    #                        devi_phi=devi_phi.numpy(),
    #                        veh2road=veh2road.numpy(),
    #                        veh2veh=veh2veh.numpy(),
    #                        rew_alpha_f=rew_alpha_f.numpy(),
    #                        rew_alpha_r=rew_alpha_r.numpy(),
    #                        rew_r=rew_r.numpy()
    #                        )
    #     # print(reward_dict)
    #     # rew_alpha_f = -10000. if rew_alpha_f < -10000. else rew_alpha_f
    #     # rew_alpha_r = -10000. if rew_alpha_r < -10000. else rew_alpha_r
    #     # rew_r = -10000. if rew_r < -10000. else rew_r
    #     # veh2road = -10000. if veh2road < -10000. else veh2road
    #     veh2veh = -100. if veh2veh < -100. else veh2veh
    #
    #     rewards = 0.01 * devi_v + 0.04 * devi_y + 5 * devi_phi + 0.02 * punish_yaw_rate + \
    #               0.05 * punish_steer + 0.0005 * punish_a_x + 100 * veh2road + veh2veh + \
    #               100 * rew_alpha_f + 100 * rew_alpha_r + 100 * rew_r
    #     return rewards.numpy(), reward_dict

    def compute_reward3(self, obs, action):
        ego_infos, tracking_infos, veh_infos = obs[:self.ego_info_dim], obs[self.ego_info_dim:self.ego_info_dim + 4 * (self.num_future_data+1)], \
                                               obs[self.ego_info_dim + 4 * (self.num_future_data+1):]
        steers, a_xs = action[0], action[1]

        # rewards related to action
        punish_steer = -tf.square(steers)
        punish_a_x = -tf.square(a_xs)

        # rewards related to ego stability
        punish_yaw_rate = -tf.square(ego_infos[2])

        # rewards related to tracking error
        devi_v = -tf.cast(tf.square(ego_infos[0] - self.exp_v), dtype=tf.float32)
        devi_y = -tf.square(tracking_infos[0]) - tf.square(tracking_infos[1])
        devi_phi = -tf.cast(tf.square(tracking_infos[4+2] * np.pi / 180.), dtype=tf.float32)

        ego_lw = (ego_infos[6] - ego_infos[7]) / 2.
        coeff = 1.14
        rho_ego = ego_infos[7] / 2. * coeff
        ego_front_point = tf.cast(ego_infos[3] + ego_lw * tf.cos(ego_infos[5] * np.pi / 180.), dtype=tf.float32), \
                          tf.cast(ego_infos[4] + ego_lw * tf.sin(ego_infos[5] * np.pi / 180.), dtype=tf.float32)
        ego_rear_point = tf.cast(ego_infos[3] - ego_lw * tf.cos(ego_infos[5] * np.pi / 180.), dtype=tf.float32), \
                         tf.cast(ego_infos[4] - ego_lw * tf.sin(ego_infos[5] * np.pi / 180.), dtype=tf.float32)

        if self.training_task == 'left':
            veh2road = tf.constant(0.)
            for ego_point in [ego_front_point, ego_rear_point]:
                before1 = 0./tf.square(ego_point[0] - 0 - rho_ego) if ego_point[1] < -18 else tf.constant(0.)
                before2 = 0./tf.square(3.75 - ego_point[0] - rho_ego) if ego_point[1] < -18 else tf.constant(0.)
                middle_cond = True if -18 < ego_point[0] < 18 and -18 < ego_point[1] < 18 else False
                middle1 = 1./tf.square(7.5 - ego_point[1] - rho_ego) if middle_cond else tf.constant(0.)
                middle2 = 1./tf.square(7.5 - ego_point[0] - rho_ego) if middle_cond else tf.constant(0.)
                middle3 = 1./tf.square(ego_point[0] - (-18) - rho_ego) if middle_cond and ego_point[1] < 0 else tf.constant(0.)
                middle4 = 1./tf.square(ego_point[1] - (-18) - rho_ego) if middle_cond and ego_point[0] < 0 else tf.constant(0.)

                after1 = 0./tf.square(ego_point[1] - 0 - rho_ego) if ego_point[0] < -18 else tf.constant(0.)
                after2 = 0./tf.square(7.5 - ego_point[1] - rho_ego) if ego_point[0] < -18 else tf.constant(0.)

                this_point = before1 + before2 +\
                             middle1 + middle2 + middle3 + middle4 + \
                             after1 + after2
                veh2road -= this_point

        # rewards related to veh2veh collision
        # veh2veh = tf.constant(0.)
        # for veh_index in range(int(len(veh_infos) / self.per_veh_info_dim)):
        #     veh = veh_infos[veh_index*self.per_veh_info_dim:(veh_index + 1)*self.per_veh_info_dim]
        #     for i in [6,7,8,9]:
        #         veh2veh -= 1. / tf.square(veh[i])
        veh2veh = tf.constant(0.)
        for veh_index in range(int(len(veh_infos) / self.per_veh_info_dim)):
            veh = veh_infos[veh_index * self.per_veh_info_dim:(veh_index + 1)*self.per_veh_info_dim]
            veh_lw = (veh[4] - veh[5]) / 2.
            rho_veh = veh[5] / 2. * coeff
            veh_front_point = tf.cast(veh[0] + veh_lw * tf.cos(veh[3] * np.pi / 180.), dtype=tf.float32), \
                              tf.cast(veh[1] + veh_lw * tf.sin(veh[3] * np.pi / 180.), dtype=tf.float32)
            veh_rear_point = tf.cast(veh[0] - veh_lw * tf.cos(veh[3] * np.pi / 180.), dtype=tf.float32), \
                             tf.cast(veh[1] - veh_lw * tf.sin(veh[3] * np.pi / 180.), dtype=tf.float32)
            for ego_point in [ego_front_point, ego_rear_point]:
                for veh_point in [veh_front_point, veh_rear_point]:
                    veh2veh_dist = tf.sqrt(tf.square(ego_point[0] - veh_point[0]) + tf.square(
                        ego_point[1] - veh_point[1])) - tf.convert_to_tensor(rho_ego + rho_veh, dtype=tf.float32)
                    veh2veh -= 1. / tf.square(veh2veh_dist)
                    # veh2veh -= tf.nn.relu(-(veh2veh_dist-10.))

        veh2road = tf.constant(-3., dtype=tf.float32) if veh2road < -3. else veh2road
        veh2veh = tf.constant(-3., dtype=tf.float32) if veh2veh < -3. else veh2veh

        reward = 0.01 * devi_v + 0.1 * devi_y + 5 * devi_phi + 0.02 * punish_yaw_rate + \
                  0.05 * punish_steer + 0.0005 * punish_a_x + veh2veh + veh2road
        reward_dict = dict(punish_steer=punish_steer.numpy(),
                           punish_a_x=punish_a_x.numpy(),
                           punish_yaw_rate=punish_yaw_rate.numpy(),
                           devi_v=devi_v.numpy(),
                           devi_y=devi_y.numpy(),
                           devi_phi=devi_phi.numpy(),
                           veh2road=veh2road.numpy(),
                           veh2veh=veh2veh.numpy(),
                           rew_alpha_f=0.,
                           rew_alpha_r=0.,
                           rew_r=0.,
                           scaled_punish_steer=0.05 * punish_steer.numpy(),
                           scaled_punish_a_x=0.0005 * punish_a_x.numpy(),
                           scaled_punish_yaw_rate=0.02 * punish_yaw_rate.numpy(),
                           scaled_devi_v=0.01 * devi_v.numpy(),
                           scaled_devi_y=0.1 * devi_y.numpy(),
                           scaled_devi_phi=5 * devi_phi.numpy(),
                           scaled_veh2road=veh2road.numpy(),
                           scaled_veh2veh=veh2veh.numpy(),
                           scaled_rew_alpha_f=0.,
                           scaled_rew_alpha_r=0.,
                           scaled_rew_r=0.,
                           reward_except_done=reward)
        return reward.numpy(), reward_dict

    def render(self, mode='human'):
        if mode == 'human':
            # plot basic map
            square_length = 36
            extension = 40
            lane_width = 3.75
            dotted_line_style = '--'
            light_line_width = 3

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
            # plt.plot([0, 2 * lane_width], [-square_length / 2, -square_length / 2],
            #          color='black')
            # plt.plot([-2 * lane_width, 0], [square_length / 2, square_length / 2],
            #          color='black')
            # plt.plot([-square_length / 2, -square_length / 2], [0, -2 * lane_width],
            #          color='black')
            # plt.plot([square_length / 2, square_length / 2], [2 * lane_width, 0],
            #          color='black')
            v_light = self.v_light
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

            # ----------Oblique--------------
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
                line_length = 3
                x_forw, y_forw = x + line_length * cos(phi*pi/180.),\
                                 y + line_length * sin(phi*pi/180.)
                plt.plot([x, x_forw], [y, y_forw], color=color, linewidth=0.5)

            # plot cars
            for veh in self.all_vehicles:
                veh_x = veh['x']
                veh_y = veh['y']
                veh_phi = veh['phi']
                veh_l = veh['l']
                veh_w = veh['w']
                if is_in_plot_area(veh_x, veh_y):
                    plot_phi_line(veh_x, veh_y, veh_phi, 'black')
                    draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, 'black')

            # plot_interested vehs
            for mode, num in self.veh_mode_list:
                for i in range(num):
                    veh = self.interested_vehs[mode][i]
                    veh_x = veh['x']
                    veh_y = veh['y']
                    veh_phi = veh['phi']
                    veh_l = veh['l']
                    veh_w = veh['w']
                    task2color = {'left': 'b', 'straight': 'c', 'right': 'm'}

                    if is_in_plot_area(veh_x, veh_y):
                        plot_phi_line(veh_x, veh_y, veh_phi, 'black')
                        task = MODE2TASK[mode]
                        color = task2color[task]
                        draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, color, linestyle=':')

            # plot own car
            # dict(v_x=ego_dict['v_x'],
            #      v_y=ego_dict['v_y'],
            #      r=ego_dict['r'],
            #      x=ego_dict['x'],
            #      y=ego_dict['y'],
            #      phi=ego_dict['phi'],
            #      l=ego_dict['l'],
            #      w=ego_dict['w'],
            #      Corner_point=self.cal_corner_point_of_ego_car(ego_dict)
            #      alpha_f_bound=alpha_f_bound,
            #      alpha_r_bound=alpha_r_bound,
            #      r_bound=r_bound)

            ego_v_x = self.ego_dynamics['v_x']
            ego_v_y = self.ego_dynamics['v_y']
            ego_r = self.ego_dynamics['r']
            ego_x = self.ego_dynamics['x']
            ego_y = self.ego_dynamics['y']
            ego_phi = self.ego_dynamics['phi']
            ego_l = self.ego_dynamics['l']
            ego_w = self.ego_dynamics['w']
            ego_alpha_f = self.ego_dynamics['alpha_f']
            ego_alpha_r = self.ego_dynamics['alpha_r']
            ego_miu_f = self.ego_dynamics['miu_f']
            ego_miu_r = self.ego_dynamics['miu_r']
            alpha_f_bound = self.ego_dynamics['alpha_f_bound']
            alpha_r_bound = self.ego_dynamics['alpha_r_bound']
            r_bound = self.ego_dynamics['r_bound']

            #===================================
            # ego_info, tracing_info, vehs_info = self.obs[:28], self.obs[28:28 + 4 * (self.num_future_data+1)], \
            #                                     self.obs[28 + 4 * (self.num_future_data+1):]
            # ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi, ego_l, ego_w, \
            # ego_alpha_f, ego_alpha_r, ego_miu_f, ego_miu_r, \
            # up1, down1, left1, right1, point11x, point11y, point12x, point12y, \
            # up2, down2, left2, right2, point21x, point21y, point22x, point22y= ego_info
            # delta_x, delta_y, delta_phi, delta_v = tracing_info[:4]
            #
            # start = 0
            # for mode, num in self.veh_mode_list:
            #     for _ in range(num):
            #         veh = vehs_info[start*10:(start+1)*10]
            #         veh_x, veh_y, veh_v, veh_phi, veh_l, veh_w, \
            #         dist1, dist2, dist3, dist4 = veh
            #         start += 1
            #         task2color = {'left': 'b', 'straight': 'c', 'right': 'm'}
            #         if is_in_plot_area(veh_x, veh_y):
            #             plot_phi_line(veh_x, veh_y, veh_phi, 'black')
            #             plt.text(veh_x, veh_y, '{:.1f}'.format(min([dist1, dist2, dist3, dist4])))
            #             task = MODE2TASK[mode]
            #             color = task2color[task]
            #             draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, color, linestyle=':')

            #===================================

            plot_phi_line(ego_x, ego_y, ego_phi, 'red')
            draw_rotate_rec(ego_x, ego_y, ego_phi, ego_l, ego_w, 'red')

            # plot planed trj
            ego_info, tracking_info, vehs_info = self.obs[:self.ego_info_dim], self.obs[self.ego_info_dim:self.ego_info_dim + 4 * (self.num_future_data+1)], \
                                                self.obs[self.ego_info_dim + 4 * (self.num_future_data+1):]
            for i in range(self.num_future_data + 1):
                delta_x, delta_y, delta_phi, delta_v = tracking_info[i*4:(i+1)*4]
                path_x, path_y, path_phi = ego_x-delta_x, ego_y-delta_y, ego_phi-delta_phi
                plt.plot(path_x, path_y, 'g.')
                plot_phi_line(path_x, path_y, path_phi, 'g')


            ax.plot(self.ref_path.path[0], self.ref_path.path[1], color='g')
            indexs, points = self.ref_path.find_closest_point(np.array([ego_x], np.float32), np.array([ego_y],np.float32))
            path_x, path_y, path_phi = points[0][0], points[1][0], points[2][0]
            delta_x, delta_y, delta_phi = ego_x - path_x, ego_y - path_y, ego_phi - path_phi

            # plot ego dynamics
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
            plt.text(text_x, text_y_start - next(ge), 'yaw_rate bound: [{:.2f}, {:.2f}]'.format(-r_bound, r_bound))

            plt.text(text_x, text_y_start - next(ge), r'$\alpha_f$: {:.2f} rad'.format(ego_alpha_f))
            plt.text(text_x, text_y_start - next(ge), r'$\alpha_f$ bound: [{:.2f}, {:.2f}] '.format(-alpha_f_bound,
                                                                                                        alpha_f_bound))
            plt.text(text_x, text_y_start - next(ge), r'$\alpha_r$: {:.2f} rad'.format(ego_alpha_r))
            plt.text(text_x, text_y_start - next(ge), r'$\alpha_r$ bound: [{:.2f}, {:.2f}] '.format(-alpha_r_bound,
                                                                                                        alpha_r_bound))
            if self.action is not None:
                steer, a_x = self.action[0], self.action[1]
                plt.text(text_x, text_y_start - next(ge), r'steer: {:.2f}rad (${:.2f}\degree$)'.format(steer, steer * 180 / np.pi))
                plt.text(text_x, text_y_start - next(ge), 'a_x: {:.2f}m/s^2'.format(a_x))

            text_x, text_y_start = 70, 60
            ge = iter(range(0, 1000, 4))

            # done info
            plt.text(text_x, text_y_start - next(ge), 'done info: {}'.format(self.done_type))

            # reward info
            if self.reward_info is not None:
                for key, val in self.reward_info.items():
                    plt.text(text_x, text_y_start - next(ge), '{}: {:.4f}'.format(key, val))

            plt.show()
            plt.pause(0.1)


def test_end2end():
    import time
    env = CrossroadEnd2end('left')
    obs = env.reset()
    i = 0
    done = 0
    start = time.time()
    while i < 100000:
        while not done:
            # print(i)
            i += 1
            action = np.array([0, 0], dtype=np.float32)
            obs, reward, done, info = env.step(action)
            env.render()
        done = 0
        obs = env.reset()
        env.render()
    end = time.time()
    print(end-start)


if __name__ == '__main__':
    test_end2end()