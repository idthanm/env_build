import numpy as np
import gym
from scipy.optimize import minimize
import time
from math import pi, cos, sin
import bezier
from numpy import logical_and, logical_or


def deal_with_phi_diff(phi_diff):
    phi_diff = np.where(phi_diff > 180., phi_diff - 360., phi_diff)
    phi_diff = np.where(phi_diff < -180., phi_diff + 360., phi_diff)
    return phi_diff


class VehicleDynamics(object):
    def __init__(self, ):
        self.vehicle_params = dict(C_f=88000.,  # front wheel cornering stiffness [N/rad]
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
        v_x, v_y, r, x, y, phi = states[:, 0], states[:, 1], states[:, 2], \
                                 states[:, 3], states[:, 4], states[:, 5]
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
        F_xf = np.where(a_x < 0, mass * a_x / 2, np.zeros_like(a_x, dtype=np.float32))
        F_xr = np.where(a_x < 0, mass * a_x / 2, mass * a_x)
        miu_f = np.sqrt(np.square(miu * F_zf) - np.square(F_xf)) / F_zf
        miu_r = np.sqrt(np.square(miu * F_zr) - np.square(F_xr)) / F_zr
        alpha_f = np.arctan((v_y + a * r) / v_x) - steer
        alpha_r = np.arctan((v_y - b * r) / v_x)

        Ff_w1 = np.square(C_f) / (3 * F_zf * miu_f)
        Ff_w2 = np.power(C_f, 3) / (27 * np.power(F_zf * miu_f, 2))
        F_yf_max = F_zf * miu_f

        Fr_w1 = np.square(C_r) / (3 * F_zr * miu_r)
        Fr_w2 = np.power(C_r, 3) / (27 * np.power(F_zr * miu_r, 2))
        F_yr_max = F_zr * miu_r

        F_yf = - C_f * np.tan(alpha_f) + Ff_w1 * np.tan(alpha_f) * np.abs(
            np.tan(alpha_f)) - Ff_w2 * np.power(np.tan(alpha_f), 3)
        F_yr = - C_r * np.tan(alpha_r) + Fr_w1 * np.tan(alpha_r) * np.abs(
            np.tan(alpha_r)) - Fr_w2 * np.power(np.tan(alpha_r), 3)

        F_yf = np.minimum(F_yf, F_yf_max)
        F_yf = np.minimum(F_yf, -F_yf_max)

        F_yr = np.minimum(F_yr, F_yr_max)
        F_yr = np.minimum(F_yr, -F_yr_max)

        state_deriv = [a_x + v_y * r,
                       (F_yf * np.cos(steer) + F_yr) / mass - v_x * r,
                       (a * F_yf * np.cos(steer) - b * F_yr) / I_z,
                       v_x * np.cos(phi) - v_y * np.sin(phi),
                       v_x * np.sin(phi) + v_y * np.cos(phi),
                       r * 180 / np.pi,
                       ]

        state_deriv_stack = np.stack(state_deriv, axis=1)
        ego_params = np.stack([alpha_f, alpha_r, miu_f, miu_r], axis=1)

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
                if i == 1:
                    phis_1[len(start_straight_line_x):len(start_straight_line_x)+len(s_vals)] = \
                        phis_1[len(start_straight_line_x)+150:len(start_straight_line_x)+len(s_vals)+150]
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
                    phis_1[len(start_straight_line_x):len(start_straight_line_x)+len(s_vals)] = \
                        phis_1[len(start_straight_line_x)+100:len(start_straight_line_x)+len(s_vals)+100]
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
        tracking_error = np.concatenate([np.stack([ego_xs - ref_point[0],
                                                   ego_ys - ref_point[1],
                                                   deal_with_phi_diff(ego_phis - ref_point[2]),
                                                   ego_vs - 10.], 1)
                                         for ref_point in all_ref], 1)
        return tracking_error


class ModelPredictiveControl:
    def __init__(self, init_x, horizon):
        self.tau = 0.1
        self.horizon = horizon
        self.init_x = init_x
        self.v_d = 8
        self.v_max, self.v_min, self.r_min = 10., 0., 7.
        self.ego_info_dim = 12
        self.tracking_info_dim = None
        self.per_veh_info_dim = 6
        self.vehicle_dynamics = VehicleDynamics()
        self.ref_path = ReferencePath('left', mode='training')
        self.task = 'left'
        self.exp_v = 10

    def reset_init_x(self, init_x):
        self.init_x = init_x

    def compute_next_obses(self, obses, actions):
        ego_infos, tracking_infos, veh_infos = obses[:, :self.ego_info_dim], \
                                               obses[:, self.ego_info_dim:self.ego_info_dim + 4], \
                                               obses[:, self.ego_info_dim + 4:]

        next_ego_infos = self.ego_predict(ego_infos, actions)

        next_tracking_infos = self.ref_path.tracking_error_vector(next_ego_infos[:, 3],
                                                                  next_ego_infos[:, 4],
                                                                  next_ego_infos[:, 5],
                                                                  next_ego_infos[:, 0],
                                                                  0)
        next_veh_infos = self.veh_predict(next_ego_infos, veh_infos)
        next_obses = np.concatenate([next_ego_infos, next_tracking_infos, next_veh_infos], 1)
        return next_obses

    def ego_predict(self, ego_infos, actions):
        ego_next_infos_except_lw, ego_next_params = self.vehicle_dynamics.prediction(ego_infos[:, :6], actions,
                                                                                     self.tau, 1)
        ego_next_lw = ego_infos[:, 6:8]

        return np.concatenate([ego_next_infos_except_lw, ego_next_lw, ego_next_params], 1)

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
            predictions_to_be_concat.append(self.predict_for_a_mode(veh_infos[:, vehs_index * self.per_veh_info_dim:(vehs_index + 1) * self.per_veh_info_dim],
                                                                    veh_mode_list[vehs_index]))
        return np.concatenate(predictions_to_be_concat, 1)

    def predict_for_a_mode(self, vehs, mode):
        veh_xs, veh_ys, veh_vs, veh_phis, veh_ls, veh_ws = \
            vehs[:, 0], vehs[:, 1], vehs[:, 2], vehs[:, 3], vehs[:, 4], vehs[:, 5]
        veh_phis_rad = veh_phis * np.pi / 180.

        zeros = np.zeros_like(veh_xs)

        veh_xs_delta = veh_vs / self.tau * np.cos(veh_phis_rad)
        veh_ys_delta = veh_vs / self.tau * np.sin(veh_phis_rad)

        if mode in ['dl', 'rd', 'ur', 'lu']:
            veh_phis_rad_delta = np.where(-18<veh_xs<18, (veh_vs / 19.875) / self.tau, zeros)
        elif mode in ['dr', 'ru', 'ul', 'ld']:
            veh_phis_rad_delta = np.where(-18<veh_ys<18, -(veh_vs / 12.375) / self.tau, zeros)
        else:
            veh_phis_rad_delta = zeros
        next_veh_xs, next_veh_ys, next_veh_vs, next_veh_phis_rad, next_veh_ls, next_veh_ws =\
            veh_xs + veh_xs_delta, veh_ys + veh_ys_delta, veh_vs, veh_phis_rad + veh_phis_rad_delta, veh_ls, veh_ws
        next_veh_phis_rad = np.where(next_veh_phis_rad > np.pi, next_veh_phis_rad - 2 * np.pi, next_veh_phis_rad)
        next_veh_phis_rad = np.where(next_veh_phis_rad <= -np.pi, next_veh_phis_rad + 2 * np.pi, next_veh_phis_rad)
        next_veh_phis = next_veh_phis_rad * 180 / np.pi
        return np.stack([next_veh_xs, next_veh_ys, next_veh_vs, next_veh_phis, next_veh_ls, next_veh_ws], 1)

    def plant_model(self, u, x):
        x_copy = x.copy()
        x_copy = self.compute_next_obses(x_copy[np.newaxis, :], u[np.newaxis, :])[0]
        return x_copy

    def compute_rew(self, obses, actions, prev_done):
        ego_infos, tracking_infos, veh_infos = obses[:, :self.ego_info_dim], \
                                               obses[:, self.ego_info_dim:self.ego_info_dim + 4], \
                                               obses[:, self.ego_info_dim + 4:]
        steers, a_xs = actions[:, 0], actions[:, 1]
        # rewards related to action
        punish_steer = -np.square(steers)
        punish_a_x = -np.square(a_xs)

        # rewards related to ego stability
        punish_yaw_rate = -np.square(ego_infos[:, 2])

        # rewards related to tracking error
        devi_v = -np.square(ego_infos[:, 0] - self.exp_v)
        devi_y = -np.square(tracking_infos[:, 0]) - np.square(tracking_infos[:, 1])
        devi_phi = -np.square(tracking_infos[:, 2] * np.pi / 180.)

        # rewards related to veh2veh collision
        ego_lws = (ego_infos[:, 6] - ego_infos[:, 7]) / 2.
        ego_front_points = ego_infos[:, 3] + ego_lws * np.cos(ego_infos[:, 5] * np.pi / 180.), \
                           ego_infos[:, 4] + ego_lws * np.sin(ego_infos[:, 5] * np.pi / 180.)
        ego_rear_points = ego_infos[:, 3] - ego_lws * np.cos(ego_infos[:, 5] * np.pi / 180.), \
                          ego_infos[:, 4] - ego_lws * np.sin(ego_infos[:, 5] * np.pi / 180.)
        coeff = 1.14
        rho_ego = ego_infos[0, 7] / 2. * coeff

        veh2veh = np.zeros_like(veh_infos[:, 0])
        for veh_index in range(int(veh_infos.shape[1] / self.per_veh_info_dim)):
            vehs = veh_infos[:, veh_index * self.per_veh_info_dim:(veh_index + 1)*self.per_veh_info_dim]
            # for i in [6, 7, 8, 9]:
            #     veh2veh -= 1. / tf.square(vehs[:, i])
            veh_lws = (vehs[:, 4] - vehs[:, 5]) / 2.
            rho_vehs = vehs[:, 5] / 2. * coeff
            veh_front_points = vehs[:, 0] + veh_lws * np.cos(vehs[:, 3] * np.pi / 180.), \
                               vehs[:, 1] + veh_lws * np.sin(vehs[:, 3] * np.pi / 180.)
            veh_rear_points = vehs[:, 0] - veh_lws * np.cos(vehs[:, 3] * np.pi / 180.), \
                              vehs[:, 1] - veh_lws * np.sin(vehs[:, 3] * np.pi / 180.)
            for ego_point in [ego_front_points, ego_rear_points]:
                for veh_point in [veh_front_points, veh_rear_points]:
                    veh2veh_dist = np.sqrt(
                        np.square(ego_point[0] - veh_point[0]) + np.square(ego_point[1] - veh_point[1])) - \
                                   (rho_ego + rho_vehs)
                    veh2veh -= 1 / np.abs(veh2veh_dist)
                    # veh2veh -= tf.nn.relu(-(veh2veh_dist - 10.))

        veh2veh += 0.8
        veh2veh = np.where(veh2veh < -3., -3. * np.ones_like(veh2veh), veh2veh)
        rewards = 0.01 * devi_v + 0.1 * devi_y + 5 * devi_phi + 0.02 * punish_yaw_rate + \
                  0.05 * punish_steer + 0.0005 * punish_a_x + veh2veh
        rewards = (1-prev_done) * rewards
        return rewards

    def _compute_bounds(self, obses):
        F_zf, F_zr = self.vehicle_dynamics.vehicle_params['F_zf'], self.vehicle_dynamics.vehicle_params['F_zr']
        C_f, C_r = self.vehicle_dynamics.vehicle_params['C_f'], self.vehicle_dynamics.vehicle_params['C_r']
        miu_fs, miu_rs = obses[:, 10], obses[:, 11]
        alpha_f_bounds, alpha_r_bounds = 3 * miu_fs * F_zf / C_f, 3 * miu_rs * F_zr / C_r
        r_bounds = miu_rs * self.vehicle_dynamics.vehicle_params['g'] / np.abs(obses[:, 0])
        return alpha_f_bounds, alpha_r_bounds, r_bounds

    def judge_dones(self, obses):
        final_dones = np.zeros_like(obses[:, 0]).astype(np.bool)
        dones_type = np.array(['not_done_yet'] * len(obses[:, 0]), np.str)
        ego_infos, tracking_infos, veh_infos = obses[:, :self.ego_info_dim],\
                                               obses[:, self.ego_info_dim:self.ego_info_dim + 4], \
                                               obses[:, self.ego_info_dim + 4:]
        # dones related to ego stability
        alpha_f_bounds, alpha_r_bounds, r_bounds = self._compute_bounds(obses)
        alpha_fs, alpha_rs = ego_infos[:, 8], ego_infos[:, 9]
        dones_alpha_f = -alpha_f_bounds<alpha_fs<alpha_f_bounds
        dones_alpha_r = -alpha_r_bounds<alpha_rs<alpha_r_bounds
        dones_r = -r_bounds<ego_infos[:, 2]<r_bounds
        stability_dones = logical_or(logical_or(dones_alpha_f, dones_alpha_r), dones_r)
        dones_type = np.where(stability_dones, 'break_stability', dones_type)


        ego_lws = (ego_infos[:, 6] - ego_infos[:, 7]) / 2.
        ego_front_points = ego_infos[:, 3] + ego_lws * np.cos(ego_infos[:, 5] * np.pi / 180.), \
                           ego_infos[:, 4] + ego_lws * np.sin(ego_infos[:, 5] * np.pi / 180.)
        ego_rear_points = ego_infos[:, 3] - ego_lws * np.cos(ego_infos[:, 5] * np.pi / 180.), \
                          ego_infos[:, 4] - ego_lws * np.sin(ego_infos[:, 5] * np.pi / 180.)
        coeff = 1.14
        rho_ego = ego_infos[0, 7] / 2 * coeff
        veh2road_dones = np.zeros_like(obses[:, 0]).astype(np.bool)

        if self.task == 'left':
            dones_good_done = logical_and(logical_and(ego_infos[:, 4] > 0, ego_infos[:, 4] < 7.5),
                                          ego_infos[:, 3] < -18 - 2)
            dones_type = np.where(dones_good_done, 'good_done', dones_type)

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

                dones_middle7 = logical_and(middle_cond, np.sqrt(np.square(ego_point[0] - (-18)) + np.square(
                        ego_point[1] - 0)) < rho_ego)
                dones_middle8 = logical_and(middle_cond, np.sqrt(np.square(ego_point[0] - (-18)) + np.square(
                        ego_point[1] - 7.5)) < rho_ego)

                dones_after1 = logical_and(ego_point[0] < -18, ego_point[1] - 0 < rho_ego)
                dones_after2 = logical_and(ego_point[0] < -18, 7.5 - ego_point[1] < rho_ego)

                for dones in [dones_before1, dones_before2, dones_middle1, dones_middle2, dones_middle3, dones_middle4,
                              dones_middle5, dones_middle6, dones_middle7, dones_middle8,
                              dones_after1, dones_after2]:
                    veh2road_dones = logical_or(veh2road_dones, dones)

        dones_type = np.where(veh2road_dones, 'break_road_constrain', dones_type)

        # dones related to veh2veh collision
        veh2veh_dones = np.zeros_like(obses[:, 0]).astype(np.bool)
        for veh_index in range(int(veh_infos.shape[1] / self.per_veh_info_dim)):
            vehs = veh_infos[:, veh_index * self.per_veh_info_dim: (veh_index + 1)*self.per_veh_info_dim]
            veh_lws = (vehs[:, 4] - vehs[:, 5]) / 2.
            rho_vehs = vehs[:, 5] / 2. * coeff
            veh_front_points = vehs[:, 0] + veh_lws * np.cos(vehs[:, 3] * np.pi / 180.), \
                               vehs[:, 1] + veh_lws * np.sin(vehs[:, 3] * np.pi / 180.)
            veh_rear_points = vehs[:, 0] - veh_lws * np.cos(vehs[:, 3] * np.pi / 180.), \
                              vehs[:, 1] - veh_lws * np.sin(vehs[:, 3] * np.pi / 180.)
            for ego_point in [ego_front_points, ego_rear_points]:
                for veh_point in [veh_front_points, veh_rear_points]:
                    veh2veh_square_dist = np.square(ego_point[0] - veh_point[0]) + np.square(
                        ego_point[1] - veh_point[1])
                    dones = np.sqrt(veh2veh_square_dist) < rho_ego + rho_vehs
                    veh2veh_dones = logical_or(veh2veh_dones, dones)

        dones_type = np.where(veh2veh_dones, 'collision', dones_type)

        for dones in [stability_dones, veh2road_dones, dones_good_done, veh2veh_dones]:
            final_dones = logical_or(final_dones, dones)

        return final_dones, dones_type

    def compute_done_rew(self, done, prev_done, done_type):
        if done_type == 'good_done':
            return (float(done)-float(prev_done))*20.
        elif done_type == 'not_done_yet':
            return 0.
        else:
            return -(float(done)-float(prev_done))*20.

    def cost_function(self, u):
        u = u.reshape(self.horizon, 2)
        loss = 0.
        done = prev_done = 0
        done_type = 'not_done_yet'
        x = self.init_x.copy()
        for i in range(0, self.horizon):
            loss -= self.compute_rew(x[np.newaxis, :], u[i][np.newaxis, :], done)
            x = self.plant_model(u[i], x)
            if done == 0:
                dones, dones_type = self.judge_dones(x[np.newaxis, :])
                done, done_type = dones[0], dones_type[0]
            loss -= self.compute_done_rew(done, prev_done, done_type)
            prev_done = done

        return loss


if __name__ == '__main__':
    horizon_list = [20]
    env = gym.make('CrossroadEnd2end-v0', training_task='left', num_future_data=0)
    done = 0
    for horizon in horizon_list:
        for i in range(10):
            obs = env.reset()
            mpc = ModelPredictiveControl(obs, horizon)
            bounds = [(-0.2, 0.2), (-3., 3.)] * horizon
            u_init = np.zeros((horizon, 2))

            while not done:
                start_time = time.time()
                action = minimize(mpc.cost_function,
                                  x0=u_init.flatten(),
                                  method='SLSQP',
                                  bounds=bounds,
                                  tol=1e-1).x
                print(action)
                end_time = time.time()

                u_init = np.concatenate([action[2:], action[-2:]])
                obs, reward, done, info = env.step(action[:2])
                mpc.reset_init_x(obs)
                env.render()
            done = 0





