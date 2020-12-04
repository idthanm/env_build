#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/10/16
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: mpc_ipopt.py
# =====================================


import math

import gym
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from casadi import *

from dynamics_and_models import ReferencePath
from endtoend_env_utils import CROSSROAD_SIZE, LANE_WIDTH, L, W, VEHICLE_MODE_LIST
from mpc.main import TimerStat


def deal_with_phi_casa(phi):
    phi = if_else(phi > 180, phi - 360, if_else(phi < -180, phi + 360, phi))
    return phi


def deal_with_phi(phi):
    phi = if_else(phi > 180, phi - 360, if_else(phi < -180, phi + 360, phi))
    return phi


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

    def f_xu(self, x, u, tau):
        v_x, v_y, r, x, y, phi = x[0], x[1], x[2], x[3], x[4], x[5]
        phi = phi * np.pi / 180.
        steer, a_x = u[0] * 0.4, u[1] * 3 - 1
        C_f = self.vehicle_params['C_f']
        C_r = self.vehicle_params['C_r']
        a = self.vehicle_params['a']
        b = self.vehicle_params['b']
        mass = self.vehicle_params['mass']
        I_z = self.vehicle_params['I_z']
        miu = self.vehicle_params['miu']
        g = self.vehicle_params['g']
        next_v = v_x + tau * (a_x + v_y * r)

        next_state = [v_x + tau * (a_x + v_y * r),
                      (mass * v_y * v_x + tau * (
                              a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * power(
                          v_x, 2) * r) / (mass * v_x - tau * (C_f + C_r)),
                      (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (
                              tau * (power(a, 2) * C_f + power(b, 2) * C_r) - I_z * v_x),
                      x + tau * (v_x * cos(phi) - v_y * sin(phi)),
                      y + tau * (v_x * sin(phi) + v_y * cos(phi)),
                      (phi + tau * r) * 180 / np.pi,
                      ]

        return next_state


class Dynamics(object):
    def __init__(self, x_init, num_future_data, ref_index, task, exp_v, tau, veh_mode_list, per_veh_info_dim=4):
        self.task = task
        self.exp_v = exp_v
        self.tau = tau
        self.per_veh_info_dim = per_veh_info_dim
        self.vd = VehicleDynamics()
        self.veh_mode_list = veh_mode_list
        self.vehs = x_init[6+3*(1+num_future_data):]
        self.x_init = x_init
        path = ReferencePath(task)
        self.ref_index = ref_index
        path = path.path_list[self.ref_index]
        x, y, phi = [ite[1200:-1200] for ite in path]
        if self.task == 'left':
            self.start, self.end = x[0], y[-1]
            fit_x = np.arctan2(y - (-CROSSROAD_SIZE / 2), x - (-CROSSROAD_SIZE / 2))
            fit_y1 = np.sqrt(np.square(x - (-CROSSROAD_SIZE / 2)) + np.square(y - (-CROSSROAD_SIZE / 2)))
            fit_y2 = phi
        elif self.task == 'straight':
            self.start, self.end = x[0], x[-1]
            fit_x = y
            fit_y1 = x
            fit_y2 = phi
        else:
            self.start, self.end = x[0], y[-1]
            fit_x = np.arctan2(y - (-CROSSROAD_SIZE / 2), x - (CROSSROAD_SIZE / 2))
            fit_y1 = np.sqrt(np.square(x - (CROSSROAD_SIZE / 2)) + np.square(y - (-CROSSROAD_SIZE / 2)))
            fit_y2 = phi
        self.fit_y1_para = list(np.polyfit(fit_x, fit_y1, 3, rcond=None, full=False, w=None, cov=False))
        self.fit_y2_para = list(np.polyfit(fit_x, fit_y2, 3, rcond=None, full=False, w=None, cov=False))

    # def tracking_error_pred(self, x, u):
    #     v_x, v_y, r, x, y, phi, delta_y, delta_phi, delta_v = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]
    #     delta_phi = deal_with_phi_casa(delta_phi)
    #     next_x = self.vd.f_xu([v_x, v_y, r, x, delta_y, delta_phi], u, self.tau)
    #     next_tracking_error = next_x[-2:] + [next_x[0] - self.exp_v]
    #     return next_tracking_error

    def tracking_error_pred(self, next_ego):
        v_x, v_y, r, x, y, phi = next_ego[0], next_ego[1], next_ego[2], next_ego[3], next_ego[4], next_ego[5]
        if self.task == 'left':
            out1 = [-(y-self.end), deal_with_phi_casa(phi-180.), v_x-self.exp_v]
            out2 = [-(x - self.start), deal_with_phi_casa(phi - 90.), v_x - self.exp_v]
            fit_x = arctan2(y - (-CROSSROAD_SIZE / 2), x - (-CROSSROAD_SIZE / 2))
            ref_d = self.fit_y1_para[0] * power(fit_x, 3) + self.fit_y1_para[1] * power(fit_x, 2) + \
                    self.fit_y1_para[2] * fit_x + self.fit_y1_para[3]
            ref_phi = self.fit_y2_para[0] * power(fit_x, 3) + self.fit_y2_para[1] * power(fit_x, 2) + \
                      self.fit_y2_para[2] * fit_x + self.fit_y2_para[3]
            d = sqrt(power(x - (-CROSSROAD_SIZE / 2), 2) + power(y - (-CROSSROAD_SIZE / 2), 2))
            out3 = [-(d-ref_d), deal_with_phi_casa(phi-ref_phi), v_x-self.exp_v]
            return [if_else(x < -CROSSROAD_SIZE/2, out1[0], if_else(y < -CROSSROAD_SIZE/2, out2[0], out3[0])),
                    if_else(x < -CROSSROAD_SIZE/2, out1[1], if_else(y < -CROSSROAD_SIZE/2, out2[1], out3[1])),
                    if_else(x < -CROSSROAD_SIZE/2, out1[2], if_else(y < -CROSSROAD_SIZE/2, out2[2], out3[2]))]
        elif self.task == 'straight':
            out1 = [-(x - self.start), deal_with_phi_casa(phi - 90.), v_x - self.exp_v]
            out2 = [-(x - self.end), deal_with_phi_casa(phi - 90.), v_x - self.exp_v]
            fit_x = y
            ref_d = self.fit_y1_para[0] * power(fit_x, 3) + self.fit_y1_para[1] * power(fit_x, 2) + \
                    self.fit_y1_para[2] * fit_x + self.fit_y1_para[3]
            ref_phi = self.fit_y2_para[0] * power(fit_x, 3) + self.fit_y2_para[1] * power(fit_x, 2) + \
                      self.fit_y2_para[2] * fit_x + self.fit_y2_para[3]
            d = x
            out3 = [-(d-ref_d), deal_with_phi_casa(phi-ref_phi), v_x-self.exp_v]
            return [if_else(y < -CROSSROAD_SIZE/2, out1[0], if_else(y > CROSSROAD_SIZE/2, out2[0], out3[0])),
                    if_else(y < -CROSSROAD_SIZE/2, out1[1], if_else(y > CROSSROAD_SIZE/2, out2[1], out3[1])),
                    if_else(y < -CROSSROAD_SIZE/2, out1[2], if_else(y > CROSSROAD_SIZE/2, out2[2], out3[2]))]
        else:
            assert self.task == 'right'
            out1 = [-(x - self.start), deal_with_phi_casa(phi - 90.), v_x - self.exp_v]
            out2 = [y - self.end, deal_with_phi_casa(phi - 0.), v_x - self.exp_v]
            fit_x = arctan2(y - (-CROSSROAD_SIZE / 2), x - (CROSSROAD_SIZE / 2))
            ref_d = self.fit_y1_para[0] * power(fit_x, 3) + self.fit_y1_para[1] * power(fit_x, 2) + \
                    self.fit_y1_para[2] * fit_x + self.fit_y1_para[3]
            ref_phi = self.fit_y2_para[0] * power(fit_x, 3) + self.fit_y2_para[1] * power(fit_x, 2) + \
                      self.fit_y2_para[2] * fit_x + self.fit_y2_para[3]
            d = sqrt(power(x - (CROSSROAD_SIZE / 2), 2) + power(y - (-CROSSROAD_SIZE / 2), 2))
            out3 = [d - ref_d, deal_with_phi_casa(phi - ref_phi), v_x - self.exp_v]
            return [if_else(y < -CROSSROAD_SIZE / 2, out1[0], if_else(x > CROSSROAD_SIZE / 2, out2[0], out3[0])),
                    if_else(y < -CROSSROAD_SIZE / 2, out1[1], if_else(x > CROSSROAD_SIZE / 2, out2[1], out3[1])),
                    if_else(y < -CROSSROAD_SIZE / 2, out1[2], if_else(x > CROSSROAD_SIZE / 2, out2[2], out3[2]))]

    def vehs_pred(self):
        predictions = []
        for vehs_index in range(len(self.veh_mode_list)):
            predictions += \
                self.predict_for_a_mode(
                    self.vehs[vehs_index * self.per_veh_info_dim:(vehs_index + 1) * self.per_veh_info_dim],
                    self.veh_mode_list[vehs_index])
        self.vehs = predictions

    def predict_for_a_mode(self, vehs, mode):
        veh_x, veh_y, veh_v, veh_phi = vehs[0], vehs[1], vehs[2], vehs[3]
        veh_phis_rad = veh_phi * np.pi / 180.
        veh_x_delta = veh_v * self.tau * math.cos(veh_phis_rad)
        veh_y_delta = veh_v * self.tau * math.sin(veh_phis_rad)

        if mode in ['dl', 'rd', 'ur', 'lu']:
            veh_phi_rad_delta = (veh_v / (CROSSROAD_SIZE/2+0.5*LANE_WIDTH)) * self.tau if -CROSSROAD_SIZE/2 < veh_x < CROSSROAD_SIZE/2 \
                                                             and -CROSSROAD_SIZE/2 < veh_y < CROSSROAD_SIZE/2 else 0
        elif mode in ['dr', 'ru', 'ul', 'ld']:
            veh_phi_rad_delta = -(veh_v / (CROSSROAD_SIZE/2-2.5*LANE_WIDTH)) * self.tau if -CROSSROAD_SIZE/2 < veh_x < CROSSROAD_SIZE/2 \
                                                                and -CROSSROAD_SIZE/2 < veh_y < CROSSROAD_SIZE/2 else 0
        else:
            veh_phi_rad_delta = 0
        next_veh_x, next_veh_y, next_veh_v, next_veh_phi_rad = \
            veh_x + veh_x_delta, veh_y + veh_y_delta, veh_v, veh_phis_rad + veh_phi_rad_delta
        next_veh_phi = next_veh_phi_rad * 180 / np.pi
        next_veh_phi = deal_with_phi(next_veh_phi)

        return [next_veh_x, next_veh_y, next_veh_v, next_veh_phi]

    def f_xu(self, x, u):
        next_ego = self.vd.f_xu(x, u, self.tau)
        next_tracking = self.tracking_error_pred(next_ego)
        # next_tracking = self.tracking_error_pred(x, u)

        return next_ego + next_tracking

    def g_x(self, x):
        ego_x, ego_y, ego_phi = x[3], x[4], x[5]
        g_list = []
        ego_lws = (L - W) / 2.
        coeff = 1.
        rho_ego = W / 2. * coeff
        ego_front_points = ego_x + ego_lws * cos(ego_phi * np.pi / 180.), \
                           ego_y + ego_lws * sin(ego_phi * np.pi / 180.)
        ego_rear_points = ego_x - ego_lws * cos(ego_phi * np.pi / 180.), \
                          ego_y - ego_lws * sin(ego_phi * np.pi / 180.)
        # for vehs_index in range(len(self.veh_mode_list)):
        #     veh = self.vehs[vehs_index * self.per_veh_info_dim:(vehs_index + 1) * self.per_veh_info_dim]
        #     veh_x, veh_y = veh[0], veh[1]
        #     rela_phis_rad = atan2(veh_y - ego_y, veh_x - ego_x)
        #     ego_phis_rad = ego_phi * np.pi / 180.
        #     cos_value, sin_value = cos(rela_phis_rad - ego_phis_rad), sin(rela_phis_rad - ego_phis_rad)
        #     sin_value = if_else(sin_value<0, -sin_value, sin_value)
        #     dist = sqrt(power(veh_x - ego_x, 2) + power(veh_y - ego_y, 2))
        #     g_list.append(if_else(logic_and(
        #         logic_and(cos_value > 0., dist * sin_value < (4.8) / 2), dist < 5),
        #         dist-5, 1))
        for vehs_index in range(len(self.veh_mode_list)):
            veh = self.vehs[vehs_index * self.per_veh_info_dim:(vehs_index + 1) * self.per_veh_info_dim]
            veh_x, veh_y, veh_phi = veh[0], veh[1], veh[3]
            veh_lws = (L - W) / 2.
            rho_vehs = W / 2. * coeff
            veh_front_points = veh_x + veh_lws * math.cos(veh_phi * np.pi / 180.), \
                               veh_y + veh_lws * math.sin(veh_phi * np.pi / 180.)
            veh_rear_points = veh_x - veh_lws * math.cos(veh_phi * np.pi / 180.), \
                              veh_y - veh_lws * math.sin(veh_phi * np.pi / 180.)
            for ego_point in [ego_front_points, ego_rear_points]:
                for veh_point in [veh_front_points, veh_rear_points]:
                    veh2veh_dist = sqrt(
                        power(ego_point[0] - veh_point[0], 2) + power(ego_point[1] - veh_point[1], 2)) - 3.5
                    g_list.append(veh2veh_dist)

        # for ego_point in [ego_front_points]:
        #     g_list.append(if_else(logic_and(ego_point[1]<-18, ego_point[0]<1), ego_point[0]-1, 1))
        #     g_list.append(if_else(logic_and(ego_point[1]<-18, 3.75-ego_point[0]<1), 3.75-ego_point[0]-1, 1))
        #     g_list.append(if_else(logic_and(ego_point[0]>0, 0-ego_point[1]<0), 0-ego_point[1], 1))
        #     g_list.append(if_else(logic_and(ego_point[1]>-18, 3.75-ego_point[0]<1), 3.75-ego_point[0]-1, 1))
        #     g_list.append(if_else(logic_and(ego_point[0]<0, 7.5-ego_point[1]<1), 7.5-ego_point[1]-1, 1))
        #     g_list.append(if_else(logic_and(ego_point[0]<-18, ego_point[1]-0<1), ego_point[1]-0-1, 1))

        return g_list


class ModelPredictiveControl(object):
    def __init__(self, horizon, task, num_future_data, ref_index):
        self.horizon = horizon
        self.base_frequency = 10.
        self.num_future_data = num_future_data
        self.exp_v = 8.
        self.task = task
        self.ref_index = ref_index
        self.veh_mode_list = VEHICLE_MODE_LIST[self.task]
        self.DYNAMICS_DIM = 9
        self.ACTION_DIM = 2
        self.dynamics = None
        self._sol_dic = {'ipopt.print_level': 0,
                         # 'ipopt.max_iter': 10000,
                         'ipopt.sb': 'yes',
                         'print_time': 0}

    def mpc_solver(self, x_init, XO):
        self.dynamics = Dynamics(x_init, self.num_future_data, self.ref_index, self.task,
                                 self.exp_v, 1 / self.base_frequency, self.veh_mode_list)

        x = SX.sym('x', self.DYNAMICS_DIM)
        u = SX.sym('u', self.ACTION_DIM)

        # Create empty NLP
        w = []
        lbw = []
        ubw = []
        lbg = []
        ubg = []
        G = []
        J = 0

        # Initial conditions
        Xk = MX.sym('X0', self.DYNAMICS_DIM)
        w += [Xk]
        lbw += x_init[:9]
        ubw += x_init[:9]

        for k in range(1, self.horizon + 1):
            f = vertcat(*self.dynamics.f_xu(x, u))
            F = Function("F", [x, u], [f])
            g = vertcat(*self.dynamics.g_x(x))
            G_f = Function('Gf', [x], [g])

            # Local control
            Uname = 'U' + str(k - 1)
            Uk = MX.sym(Uname, self.ACTION_DIM)
            w += [Uk]
            lbw += [-1., -1.]
            ubw += [1., 1.]

            Fk = F(Xk, Uk)
            Gk = G_f(Xk)
            self.dynamics.vehs_pred()
            Xname = 'X' + str(k)
            Xk = MX.sym(Xname, self.DYNAMICS_DIM)

            # Dynamic Constraints
            G += [Fk - Xk]
            lbg += [0.0] * self.DYNAMICS_DIM
            ubg += [0.0] * self.DYNAMICS_DIM
            G += [Gk]
            lbg += [0.0] * (len(self.veh_mode_list) * 4)
            ubg += [inf] * (len(self.veh_mode_list) * 4)
            w += [Xk]
            lbw += [0.] + [-inf] * (self.DYNAMICS_DIM - 1)
            ubw += [10.] + [inf] * (self.DYNAMICS_DIM - 1)

            # Cost function
            F_cost = Function('F_cost', [x, u], [0.05 * power(x[8], 2)
                                                 + 0.8 * power(x[6], 2)
                                                 + 30 * power(x[7] * np.pi / 180., 2)
                                                 + 0.02 * power(x[2], 2)
                                                 + 5 * power(u[0], 2)
                                                 + 0.05 * power(u[1], 2)
                                                 ])
            J += F_cost(w[k * 2], w[k * 2 - 1])

        # Create NLP solver
        nlp = dict(f=J, g=vertcat(*G), x=vertcat(*w))
        S = nlpsol('S', 'ipopt', nlp, self._sol_dic)

        # Solve NLP
        r = S(lbx=vertcat(*lbw), ubx=vertcat(*ubw), x0=XO, lbg=vertcat(*lbg), ubg=vertcat(*ubg))
        state_all = np.array(r['x'])
        g_all = np.array(r['g'])
        state = np.zeros([self.horizon, self.DYNAMICS_DIM])
        control = np.zeros([self.horizon, self.ACTION_DIM])
        nt = self.DYNAMICS_DIM + self.ACTION_DIM  # total variable per step

        # save trajectories
        for i in range(self.horizon):
            state[i] = state_all[nt * i: nt * (i + 1) - self.ACTION_DIM].reshape(-1)
            control[i] = state_all[nt * (i + 1) - self.ACTION_DIM: nt * (i + 1)].reshape(-1)
        return state, control, state_all, g_all


def plot_mpc_rl(file_dir, mpc_name):
    data = np.load(file_dir, allow_pickle=True)
    iteration = np.array([i for i in range(len(data))])
    mpc_delta_v = np.array([trunk['mpc_obs'][0][0] - 20. for trunk in data])
    rl_delta_v = np.array([trunk['rl_obs'][0][0] - 20. for trunk in data])
    mpc_delta_y = np.array([trunk['mpc_obs'][0][3] for trunk in data])
    rl_delta_y = np.array([trunk['rl_obs'][0][3] for trunk in data])
    mpc_delta_phi = np.array([trunk['mpc_obs'][0][4] for trunk in data])
    rl_delta_phi = np.array([trunk['rl_obs'][0][4] for trunk in data])
    mpc_steer = np.array([0.4 * trunk['mpc_action'][0] for trunk in data])
    mpc_acc = np.array([3 * trunk['mpc_action'][1] for trunk in data])
    mpc_time = np.array([trunk['mpc_time'] for trunk in data])
    mpc_rew = np.array([trunk['mpc_rew'] for trunk in data])
    rl_steer = np.array([0.4 * trunk['rl_action'][0] for trunk in data])
    rl_acc = np.array([3 * trunk['rl_action'][1] for trunk in data])
    rl_steer_mpc = np.array([0.4 * trunk['rl_action_mpc'][0] for trunk in data])
    rl_acc_mpc = np.array([3 * trunk['rl_action_mpc'][1] for trunk in data])
    rl_time = np.array([trunk['rl_time'] for trunk in data])
    rl_rew = np.array([trunk['rl_rew'] for trunk in data])

    print("mean_mpc_time: {}, mean_rl_time: {}".format(np.mean(mpc_time), np.mean(rl_time)))
    print("var_mpc_time: {}, var_rl_time: {}".format(np.var(mpc_time), np.var(rl_time)))
    print("mpc_delta_y_mse: {}, rl_delta_y_mse: {}".format(np.sqrt(np.mean(np.square(mpc_delta_y))),
                                                           np.sqrt(np.mean(np.square(rl_delta_y)))))
    print("mpc_delta_v_mse: {}, rl_delta_v_mse: {}".format(np.sqrt(np.mean(np.square(mpc_delta_v))),
                                                           np.sqrt(np.mean(np.square(rl_delta_v)))))
    print("mpc_delta_phi_mse: {}, rl_delta_phi_mse: {}".format(np.sqrt(np.mean(np.square(mpc_delta_phi))),
                                                               np.sqrt(np.mean(np.square(rl_delta_phi)))))
    print("mpc_rew_sum: {}, rl_rew_sum: {}".format(np.sum(mpc_rew), np.sum(rl_rew)))

    df_mpc = pd.DataFrame({'algorithms': mpc_name,
                           'iteration': iteration,
                           'steer': mpc_steer,
                           'acc': mpc_acc,
                           'time': mpc_time,
                           'delta_v': mpc_delta_v,
                           'delta_y': mpc_delta_y,
                           'delta_phi': mpc_delta_phi,
                           'rew': mpc_rew})
    df_rl = pd.DataFrame({'algorithms': 'AMPC',
                          'iteration': iteration,
                          'steer': rl_steer,
                          'acc': rl_acc,
                          'time': rl_time,
                          'delta_v': rl_delta_v,
                          'delta_y': rl_delta_y,
                          'delta_phi': rl_delta_phi,
                          'rew': rl_rew})
    df_rl_same_obs_as_mpc = pd.DataFrame({'algorithms': 'AMPC_sameobs_mpc',
                                          'iteration': iteration,
                                          'steer': rl_steer_mpc,
                                          'acc': rl_acc_mpc,
                                          'time': rl_time,
                                          'delta_v': mpc_delta_v,
                                          'delta_y': mpc_delta_y,
                                          'delta_phi': mpc_delta_phi,
                                          'rew': mpc_rew})
    total_df = df_mpc.append([df_rl, df_rl_same_obs_as_mpc], ignore_index=True)
    f1 = plt.figure(1)
    ax1 = f1.add_axes([0.155, 0.12, 0.82, 0.86])
    sns.lineplot(x="iteration", y="steer", hue="algorithms", data=total_df, linewidth=2, palette="bright", )
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

    f4 = plt.figure(4)
    ax4 = f4.add_axes([0.155, 0.12, 0.82, 0.86])
    sns.lineplot(x="iteration", y="delta_v", hue="algorithms", data=total_df, linewidth=2, palette="bright")
    # ax3.set_ylabel('Average Q-value Estimation Bias', fontsize=15)
    # ax3.set_xlabel("Million iterations", fontsize=15)
    # plt.xlim(0, 3)
    # plt.ylim(-40, 80)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    f5 = plt.figure(5)
    ax5 = f5.add_axes([0.155, 0.12, 0.82, 0.86])
    sns.lineplot(x="iteration", y="delta_y", hue="algorithms", data=total_df, linewidth=2, palette="bright")
    # ax3.set_ylabel('Average Q-value Estimation Bias', fontsize=15)
    # ax3.set_xlabel("Million iterations", fontsize=15)
    # plt.xlim(0, 3)
    # plt.ylim(-40, 80)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    f6 = plt.figure(6)
    ax6 = f6.add_axes([0.155, 0.12, 0.82, 0.86])
    sns.lineplot(x="iteration", y="delta_phi", hue="algorithms", data=total_df, linewidth=2, palette="bright")
    # ax3.set_ylabel('Average Q-value Estimation Bias', fontsize=15)
    # ax3.set_xlabel("Million iterations", fontsize=15)
    # plt.xlim(0, 3)
    # plt.ylim(-40, 80)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    f7 = plt.figure(7)
    ax7 = f7.add_axes([0.155, 0.12, 0.82, 0.86])
    sns.lineplot(x="iteration", y="rew", hue="algorithms", data=total_df, linewidth=2, palette="bright")
    # ax3.set_ylabel('Average Q-value Estimation Bias', fontsize=15)
    # ax3.set_xlabel("Million iterations", fontsize=15)
    # plt.xlim(0, 3)
    # plt.ylim(-40, 80)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.show()


def run_mpc():
    horizon = 25
    task = 'right'
    num_future_data = 0
    num_simu = 1
    mpc_timer, rl_timer = TimerStat(), TimerStat()
    env4mpc = gym.make('CrossroadEnd2end-v0', training_task=task, num_future_data=num_future_data)
    # env4rl = gym.make('CrossroadEnd2end-v0', num_future_data=0)

    # rl_policy = LoadPolicy(rl_load_dir, rl_ite)

    def convert_vehs_to_abso(obs_rela):
        ego_infos, tracking_infos, veh_rela = obs_rela[:6], obs_rela[6:6+3*(1+num_future_data)], obs_rela[6+3*(1+num_future_data):]
        ego_vx, ego_vy, ego_r, ego_x, ego_y, ego_phi = ego_infos
        ego = np.array([ego_x, ego_y, 0, 0]*int(len(veh_rela)/4), dtype=np.float32)
        vehs_abso = veh_rela + ego
        out = np.concatenate((ego_infos, tracking_infos, vehs_abso), axis=0)
        return out

    for i in range(num_simu):
        data2plot = []
        obs = env4mpc.reset()
        ref_index = env4mpc.ref_path.ref_index
        # obs4rl = env4rl.reset(init_obs=obs, ref_index=ref_index)
        mpc = ModelPredictiveControl(horizon, task, num_future_data, ref_index)
        rew, rew4rl = 0., 0.
        state_all = np.array((list(obs[:6+3*(1+num_future_data)]) + [0, 0]) * horizon + list(obs[:6+3*(1+num_future_data)])).reshape((-1, 1))
        for _ in range(100):
            with mpc_timer:
                state, control, state_all, g_all = mpc.mpc_solver(list(convert_vehs_to_abso(obs)), state_all,)
            if any(g_all < -1):
                print('optimization fail')
                mpc_action = np.array([0., -1.])
                state_all = np.array((list(obs[:9]) + [0, 0]) * horizon + list(obs[:9])).reshape((-1, 1))
            else:
                state_all = np.array((list(obs[:9]) + [0, 0]) * horizon + list(obs[:9])).reshape((-1, 1))
                # np.zeros(shape=(11*horizon+9, 1))
                mpc_action = control[0]
            # with rl_timer:
            #     rl_action_mpc = rl_policy.run(obs).numpy()[0]
            #     rl_action = rl_policy.run(obs4rl).numpy()[0]

            data2plot.append(dict(mpc_obs=obs,
                                  # rl_obs=obs4rl,
                                  mpc_action=mpc_action,
                                  # rl_action=rl_action,
                                  # rl_action_mpc=rl_action_mpc,
                                  mpc_time=mpc_timer.mean,
                                  # rl_time=rl_timer.mean,
                                  mpc_rew=rew,
                                  # rl_rew=rew4rl[0],
                                  )
                             )

            mpc_action = mpc_action.astype(np.float32)
            obs, rew, _, _ = env4mpc.step(mpc_action)
            # obs4rl, rew4rl, _, _ = env4rl.step(np.array([rl_action]))
            env4mpc.render()
            plt.plot([state[i][3] for i in range(1, horizon - 1)], [state[i][4] for i in range(1, horizon - 1)],
                     'r*')
            plt.show()
            plt.pause(0.001)
        np.save('mpc_rl.npy', np.array(data2plot))


if __name__ == '__main__':
    # test = LoadPolicy('./mpc/rl_experiments/experiment-2020-10-20-14-52-58', 95000)
    run_mpc()
    # plot_mpc_rl('./mpc_rl.npy', 'IPOPT')
