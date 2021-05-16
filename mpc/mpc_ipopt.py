#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2021/02/24
# @Author  : Yang Guan, Yangang Ren (Tsinghua Univ.)
# @FileName: mpc_ipopt.py
# @Function: compare ADP and MPC
# =====================================

import math
import datetime
import matplotlib.pyplot as plt
from casadi import *

from endtoend import CrossroadEnd2end
from dynamics_and_models import ReferencePath, EnvironmentModel
from hierarchical_decision.multi_path_generator import StaticTrajectoryGenerator_origin
from endtoend_env_utils import CROSSROAD_SIZE, L, W, L_BIKE, W_BIKE, BIKE_MODE_LIST, PERSON_MODE_LIST, VEHICLE_MODE_LIST, BIKE_LANE_WIDTH, LANE_WIDTH, LANE_NUMBER, rotate_coordination
from mpc.main import TimerStat
from utils.load_policy import LoadPolicy
from utils.recorder import Recorder

EXP_V = 8.0


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
        steer, a_x = u[0], u[1]
        C_f = self.vehicle_params['C_f']
        C_r = self.vehicle_params['C_r']
        a = self.vehicle_params['a']
        b = self.vehicle_params['b']
        mass = self.vehicle_params['mass']
        I_z = self.vehicle_params['I_z']
        miu = self.vehicle_params['miu']
        g = self.vehicle_params['g']

        next_state = [v_x + tau * (a_x + v_y * r),
                      (mass * v_y * v_x + tau * (
                              a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * power(
                          v_x, 2) * r) / (mass * v_x - tau * (C_f + C_r)),
                      (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (
                              tau * (power(a, 2) * C_f + power(b, 2) * C_r) - I_z * v_x),
                      x + tau * (v_x * cos(phi) - v_y * sin(phi)),
                      y + tau * (v_x * sin(phi) + v_y * cos(phi)),
                      (phi + tau * r) * 180 / np.pi]

        return next_state


class Dynamics(object):
    def __init__(self, x_init, num_future_data, ref_index, task, exp_v, tau, bike_mode_list, person_mode_list,
                 veh_mode_list, per_bike_info_dim=4, per_person_info_dim=4, per_veh_info_dim=4):
        self.task = task
        self.exp_v = exp_v
        self.tau = tau
        self.per_bike_info_dim = per_bike_info_dim
        self.per_person_info_dim = per_person_info_dim
        self.per_veh_info_dim = per_veh_info_dim
        self.vd = VehicleDynamics()
        self.bike_mode_list = bike_mode_list
        self.person_mode_list = person_mode_list
        self.veh_mode_list = veh_mode_list
        self.bikes = x_init[6+3*(1+num_future_data):6+3*(1+num_future_data)+len(self.bike_mode_list)*self.per_bike_info_dim]
        self.persons = x_init[6+3*(1+num_future_data)+len(self.bike_mode_list)*self.per_bike_info_dim:6+3*(1+num_future_data)+len(self.bike_mode_list)*self.per_bike_info_dim+len(self.person_mode_list)*self.per_person_info_dim]
        self.vehs = x_init[6+3*(1+num_future_data)+len(self.bike_mode_list)*self.per_bike_info_dim+len(self.person_mode_list)*self.per_person_info_dim:]
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

    def bikes_pred(self):
            predictions = []
            for bikes_index in range(len(self.bike_mode_list)):
                predictions += \
                    self.predict_for_bike_mode(
                        self.bikes[bikes_index * self.per_bike_info_dim:(bikes_index + 1) * self.per_bike_info_dim],
                        self.bike_mode_list[bikes_index])
            self.bikes = predictions

    def persons_pred(self):
        predictions = []
        for persons_index in range(len(self.person_mode_list)):
            predictions += \
                self.predict_for_person_mode(
                    self.persons[persons_index * self.per_person_info_dim:(persons_index + 1) * self.per_person_info_dim],
                    self.person_mode_list[persons_index])
        self.persons = predictions

    def vehs_pred(self):
        predictions = []
        for vehs_index in range(len(self.veh_mode_list)):
            predictions += \
                self.predict_for_veh_mode(
                    self.vehs[vehs_index * self.per_veh_info_dim:(vehs_index + 1) * self.per_veh_info_dim],
                    self.veh_mode_list[vehs_index])
        self.vehs = predictions

    def predict_for_bike_mode(self, bikes, mode):
        bike_x, bike_y, bike_v, bike_phi = bikes[0], bikes[1], bikes[2], bikes[3]
        bike_phis_rad = bike_phi * np.pi / 180.
        bike_x_delta = bike_v * self.tau * math.cos(bike_phis_rad)
        bike_y_delta = bike_v * self.tau * math.sin(bike_phis_rad)

        if mode in ['dl_b', 'rd_b', 'ur_b', 'lu_b']:
            bike_phi_rad_delta = (bike_v / (CROSSROAD_SIZE/2+3*LANE_WIDTH + BIKE_LANE_WIDTH / 2)) * self.tau if -CROSSROAD_SIZE/2 < bike_x < CROSSROAD_SIZE/2 \
                                                             and -CROSSROAD_SIZE/2 < bike_y < CROSSROAD_SIZE/2 else 0
        elif mode in ['dr_b', 'ru_b', 'ul_b', 'ld_b']:
            bike_phi_rad_delta = -(bike_v / (CROSSROAD_SIZE/2-3.0*LANE_WIDTH - BIKE_LANE_WIDTH / 2)) * self.tau if -CROSSROAD_SIZE/2 < bike_x < CROSSROAD_SIZE/2 \
                                                                and -CROSSROAD_SIZE/2 < bike_y < CROSSROAD_SIZE/2 else 0
        else:
            bike_phi_rad_delta = 0
        next_bike_x, next_bike_y, next_bike_v, next_bike_phi_rad = \
            bike_x + bike_x_delta, bike_y + bike_y_delta, bike_v, bike_phis_rad + bike_phi_rad_delta
        next_bike_phi = next_bike_phi_rad * 180 / np.pi
        next_bike_phi = deal_with_phi(next_bike_phi)

        return [next_bike_x, next_bike_y, next_bike_v, next_bike_phi]

    def predict_for_person_mode(self, persons, mode):
        person_x, person_y, person_v, person_phi = persons[0], persons[1], persons[2], persons[3]
        person_phis_rad = person_phi * np.pi / 180.
        person_x_delta = person_v * self.tau * math.cos(person_phis_rad)
        person_y_delta = person_v * self.tau * math.sin(person_phis_rad)

        next_person_x, next_person_y, next_person_v, next_person_phi_rad = \
            person_x + person_x_delta, person_y + person_y_delta, person_v, person_phis_rad
        next_person_phi = next_person_phi_rad * 180 / np.pi
        next_person_phi = deal_with_phi(next_person_phi)

        return [next_person_x, next_person_y, next_person_v, next_person_phi]

    def predict_for_veh_mode(self, vehs, mode):
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
        next_ego = self.vd.f_xu(x, u, self.tau)           # Unit of heading angle is degree
        next_tracking = self.tracking_error_pred(next_ego)
        return next_ego + next_tracking

    def g_x(self, x):
        ego_x, ego_y, ego_phi = x[3], x[4], x[5]
        g_list = []
        ego_lws = (L - W) / 2.
        ego_front_points = ego_x + ego_lws * cos(ego_phi * np.pi / 180.), \
                           ego_y + ego_lws * sin(ego_phi * np.pi / 180.)
        ego_rear_points = ego_x - ego_lws * cos(ego_phi * np.pi / 180.), \
                          ego_y - ego_lws * sin(ego_phi * np.pi / 180.)

        for bikes_index in range(len(self.bike_mode_list)):
            bike = self.bikes[bikes_index * self.per_bike_info_dim:(bikes_index + 1) * self.per_bike_info_dim]
            bike_x, bike_y, bike_phi = bike[0], bike[1], bike[3]
            bike_lws = (L_BIKE - W_BIKE) / 2.
            bike_front_points = bike_x + bike_lws * math.cos(bike_phi * np.pi / 180.), \
                               bike_y + bike_lws * math.sin(bike_phi * np.pi / 180.)
            bike_rear_points = bike_x - bike_lws * math.cos(bike_phi * np.pi / 180.), \
                              bike_y - bike_lws * math.sin(bike_phi * np.pi / 180.)
            for ego_point in [ego_front_points, ego_rear_points]:
                for bike_point in [bike_front_points, bike_rear_points]:
                    veh2bike_dist = sqrt(power(ego_point[0] - bike_point[0], 2) + power(ego_point[1] - bike_point[1], 2)) - 3.5
                    g_list.append(veh2bike_dist)

        for persons_index in range(len(self.person_mode_list)):
            person = self.persons[persons_index * self.per_person_info_dim:(persons_index + 1) * self.per_person_info_dim]
            person_x, person_y, person_phi = person[0], person[1], person[3]
            person_point = person_x, person_y
            for ego_point in [ego_front_points, ego_rear_points]:
                veh2person_dist = sqrt(power(ego_point[0] - person_point[0], 2) + power(ego_point[1] - person_point[1], 2)) - 3.5
                g_list.append(veh2person_dist)

        for vehs_index in range(len(self.veh_mode_list)):
            veh = self.vehs[vehs_index * self.per_veh_info_dim:(vehs_index + 1) * self.per_veh_info_dim]
            veh_x, veh_y, veh_phi = veh[0], veh[1], veh[3]
            veh_lws = (L - W) / 2.
            veh_front_points = veh_x + veh_lws * math.cos(veh_phi * np.pi / 180.), \
                               veh_y + veh_lws * math.sin(veh_phi * np.pi / 180.)
            veh_rear_points = veh_x - veh_lws * math.cos(veh_phi * np.pi / 180.), \
                              veh_y - veh_lws * math.sin(veh_phi * np.pi / 180.)
            for ego_point in [ego_front_points, ego_rear_points]:
                for veh_point in [veh_front_points, veh_rear_points]:
                    veh2veh_dist = sqrt(power(ego_point[0] - veh_point[0], 2) + power(ego_point[1] - veh_point[1], 2)) - 3.5
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
        self.exp_v = EXP_V
        self.task = task
        self.ref_index = ref_index
        self.bike_mode_list = BIKE_MODE_LIST[self.task]
        self.person_mode_list = PERSON_MODE_LIST[self.task]
        self.veh_mode_list = VEHICLE_MODE_LIST[self.task]
        self.DYNAMICS_DIM = 9               # ego_info + track_error_dim
        self.ACTION_DIM = 2
        self.dynamics = None
        self._sol_dic = {'ipopt.print_level': 0,
                         'ipopt.sb': 'yes',
                         'print_time': 0}

    def mpc_solver(self, x_init, XO):
        self.dynamics = Dynamics(x_init, self.num_future_data, self.ref_index, self.task, self.exp_v,
                                 1 / self.base_frequency, self.bike_mode_list, self.person_mode_list, self.veh_mode_list)

        x = SX.sym('x', self.DYNAMICS_DIM)
        u = SX.sym('u', self.ACTION_DIM)

        # Create empty NLP
        w = []
        lbw = []                 # lower bound for state and action constraints
        ubw = []                 # upper bound for state and action constraints
        lbg = []                 # lower bound for distance constraint
        ubg = []                 # upper bound for distance constraint
        G = []                   # dynamic constraints
        J = 0                    # accumulated cost

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
            lbw += [-0.4, -3.]                    # todo: action constraints
            ubw += [0.4, 1.5]

            Fk = F(Xk, Uk)
            Gk = G_f(Xk)
            self.dynamics.bikes_pred()
            self.dynamics.persons_pred()
            self.dynamics.vehs_pred()
            Xname = 'X' + str(k)
            Xk = MX.sym(Xname, self.DYNAMICS_DIM)

            # Dynamic Constraints
            G += [Fk - Xk]                                         # ego vehicle dynamic constraints
            lbg += [0.0] * self.DYNAMICS_DIM
            ubg += [0.0] * self.DYNAMICS_DIM
            G += [Gk]                                              # surrounding bike constraints
            lbg += [0.0] * (len(self.bike_mode_list) * 4)
            ubg += [inf] * (len(self.bike_mode_list) * 4)
            #                                                      # surrounding person constraints
            lbg += [0.0] * (len(self.person_mode_list) * 2)
            ubg += [inf] * (len(self.person_mode_list) * 2)
            #                                                      # surrounding vehicle constraints
            lbg += [0.0] * (len(self.veh_mode_list) * 4)
            ubg += [inf] * (len(self.veh_mode_list) * 4)
            w += [Xk]
            lbw += [0.] + [-inf] * (self.DYNAMICS_DIM - 1)         # speed constraints
            ubw += [8.] + [inf] * (self.DYNAMICS_DIM - 1)

            # Cost function
            F_cost = Function('F_cost', [x, u], [0.5 * power(x[8], 2)
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

        # load constraints and solve NLP
        r = S(lbx=vertcat(*lbw), ubx=vertcat(*ubw), x0=XO, lbg=vertcat(*lbg), ubg=vertcat(*ubg))
        state_all = np.array(r['x'])
        g_all = np.array(r['g'])
        state = np.zeros([self.horizon, self.DYNAMICS_DIM])
        control = np.zeros([self.horizon, self.ACTION_DIM])
        nt = self.DYNAMICS_DIM + self.ACTION_DIM  # total variable per step
        cost = np.array(r['f']).squeeze(0)

        # save trajectories
        for i in range(self.horizon):
            state[i] = state_all[nt * i: nt * (i + 1) - self.ACTION_DIM].reshape(-1)
            control[i] = state_all[nt * (i + 1) - self.ACTION_DIM: nt * (i + 1)].reshape(-1)
        return state, control, state_all, g_all, cost


class HierarchicalMpc(object):
    def __init__(self, task, logdir):
        self.task = task
        # if self.task == 'left':
        #     self.policy = LoadPolicy('../utils/models/left/experiment-2021-03-15-16-39-00', 180000)
        # elif self.task == 'right':
        #     self.policy = LoadPolicy('G:\\env_build\\utils\\models\\right', 145000)
        # elif self.task == 'straight':
        #     self.policy = LoadPolicy('G:\\env_build\\utils\\models\\straight', 95000)

        self.logdir = logdir
        self.episode_counter = 0
        self.horizon = 25
        self.num_future_data = 0
        self.env = CrossroadEnd2end(training_task=self.task, num_future_data=self.num_future_data)
        self.model = EnvironmentModel(self.task)
        self.obs = self.env.reset()
        self.stg = StaticTrajectoryGenerator_origin(mode='static_traj')
        self.mpc_cal_timer = TimerStat()
        self.adp_cal_timer = TimerStat()
        self.recorder = Recorder()
        self.hist_posi = []

    def reset(self):
        self.obs = self.env.reset()
        self.stg = StaticTrajectoryGenerator_origin(mode='static_traj')
        self.recorder.reset()
        self.hist_posi = []

        if self.logdir is not None:
            os.makedirs(self.logdir + '/episode{}/figs'.format(self.episode_counter))
            self.recorder.save(self.logdir)
            self.episode_counter += 1
            # if self.episode_counter >= 1:
            #     self.recorder.plot_mpc_rl(self.episode_counter-1,
            #                               self.logdir + '/episode{}/figs'.format(self.episode_counter-1), isshow=False)
        return self.obs

    def convert_vehs_to_abso(self, obs_rela):
        ego_infos, tracking_infos, veh_rela = obs_rela[:6], \
                                              obs_rela[6:6 + 3 * (1 + self.num_future_data)],\
                                              obs_rela[6 + 3 * (1 + self.num_future_data):]
        ego_vx, ego_vy, ego_r, ego_x, ego_y, ego_phi = ego_infos
        ego = np.array([ego_x, ego_y, 0, 0] * int(len(veh_rela) / 4), dtype=np.float32)
        vehs_abso = veh_rela + ego
        out = np.concatenate((ego_infos, tracking_infos, vehs_abso), axis=0)
        return out

    def step(self):
        traj_list, _ = self.stg.generate_traj(self.task, self.obs)
        ADP_traj_return_value, MPC_traj_return_value = [], []
        action_total = []
        state_total = []

        with self.mpc_cal_timer:
            for ref_index, trajectory in enumerate(traj_list):
                mpc = ModelPredictiveControl(self.horizon, self.task, self.num_future_data, ref_index)
                state_all = np.array((list(self.obs[:6 + 3 * (1 + self.num_future_data)]) + [0, 0]) * self.horizon +
                                      list(self.obs[:6 + 3 * (1 + self.num_future_data)])).reshape((-1, 1))
                state, control, state_all, g_all, cost = mpc.mpc_solver(list(self.convert_vehs_to_abso(self.obs)), state_all)
                state_total.append(state)
                if any(g_all < -1):
                    print('optimization fail')
                    mpc_action = np.array([0., -1.])
                    state_all = np.array((list(self.obs[:9]) + [0, 0]) * self.horizon + list(self.obs[:9])).reshape(
                        (-1, 1))
                else:
                    state_all = np.array((list(self.obs[:9]) + [0, 0]) * self.horizon + list(self.obs[:9])).reshape(
                        (-1, 1))
                    mpc_action = control[0]

                MPC_traj_return_value.append(cost.squeeze().tolist())
                action_total.append(mpc_action)

            MPC_traj_return_value = np.array(MPC_traj_return_value, dtype=np.float32)
            MPC_path_index = np.argmin(MPC_traj_return_value)                           # todo: minimize
            MPC_action = action_total[MPC_path_index]

        self.obs, rew, done, _ = self.env.step(MPC_action)
        self.render(traj_list, MPC_traj_return_value, MPC_path_index, method='MPC')
        state = state_total[MPC_path_index]
        plt.plot([state[i][3] for i in range(1, self.horizon - 1)], [state[i][4] for i in range(1, self.horizon - 1)], 'r*')
        plt.pause(0.001)

        return done

    def render(self, traj_list, MPC_traj_return_value, MPC_path_index, method='MPC'):
        square_length = CROSSROAD_SIZE
        extension = 40
        lane_width = LANE_WIDTH
        light_line_width = 3
        dotted_line_style = '--'
        solid_line_style = '-'

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

        # ----------arrow--------------
        plt.arrow(lane_width / 2, -square_length / 2 - 10, 0, 5, color='b')
        plt.arrow(lane_width / 2, -square_length / 2 - 10 + 5, -0.5, 0, color='b', head_width=1)
        plt.arrow(lane_width * 1.5, -square_length / 2 - 10, 0, 4, color='b', head_width=1)
        plt.arrow(lane_width * 2.5, -square_length / 2 - 10, 0, 5, color='b')
        plt.arrow(lane_width * 2.5, -square_length / 2 - 10 + 5, 0.5, 0, color='b', head_width=1)

        # ----------horizon--------------

        plt.plot([-square_length / 2 - extension, -square_length / 2], [0.3, 0.3], color='orange')
        plt.plot([-square_length / 2 - extension, -square_length / 2], [-0.3, -0.3], color='orange')
        plt.plot([square_length / 2 + extension, square_length / 2], [0.3, 0.3], color='orange')
        plt.plot([square_length / 2 + extension, square_length / 2], [-0.3, -0.3], color='orange')

        #
        for i in range(1, LANE_NUMBER + 1):
            linestyle = dotted_line_style if i < LANE_NUMBER else solid_line_style
            linewidth = 1 if i < LANE_NUMBER else 2
            plt.plot([-square_length / 2 - extension, -square_length / 2], [i * lane_width, i * lane_width],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([square_length / 2 + extension, square_length / 2], [i * lane_width, i * lane_width],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([-square_length / 2 - extension, -square_length / 2], [-i * lane_width, -i * lane_width],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([square_length / 2 + extension, square_length / 2], [-i * lane_width, -i * lane_width],
                     linestyle=linestyle, color='black', linewidth=linewidth)

        for i in range(4, 5 + 1):
            linestyle = dotted_line_style if i < 5 else solid_line_style
            linewidth = 1 if i < 5 else 2
            plt.plot([-square_length / 2 - extension, -square_length / 2],
                     [3 * lane_width + (i - 3) * 2, 3 * lane_width + (i - 3) * 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([square_length / 2 + extension, square_length / 2],
                     [3 * lane_width + (i - 3) * 2, 3 * lane_width + (i - 3) * 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([-square_length / 2 - extension, -square_length / 2],
                     [-3 * lane_width - (i - 3) * 2, -3 * lane_width - (i - 3) * 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([square_length / 2 + extension, square_length / 2],
                     [-3 * lane_width - (i - 3) * 2, -3 * lane_width - (i - 3) * 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)

        # ----------vertical----------------
        plt.plot([0.3, 0.3], [-square_length / 2 - extension, -square_length / 2], color='orange')
        plt.plot([-0.3, -0.3], [-square_length / 2 - extension, -square_length / 2], color='orange')
        plt.plot([0.3, 0.3], [square_length / 2 + extension, square_length / 2], color='orange')
        plt.plot([-0.3, -0.3], [square_length / 2 + extension, square_length / 2], color='orange')

        #
        for i in range(1, LANE_NUMBER + 1):
            linestyle = dotted_line_style if i < LANE_NUMBER else solid_line_style
            linewidth = 1 if i < LANE_NUMBER else 2
            plt.plot([i * lane_width, i * lane_width], [-square_length / 2 - extension, -square_length / 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([i * lane_width, i * lane_width], [square_length / 2 + extension, square_length / 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([-i * lane_width, -i * lane_width], [-square_length / 2 - extension, -square_length / 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([-i * lane_width, -i * lane_width], [square_length / 2 + extension, square_length / 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)

        for i in range(4, 5 + 1):
            linestyle = dotted_line_style if i < 5 else solid_line_style
            linewidth = 1 if i < 5 else 2
            plt.plot([3 * lane_width + (i - 3) * 2, 3 * lane_width + (i - 3) * 2],
                     [-square_length / 2 - extension, -square_length / 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([3 * lane_width + (i - 3) * 2, 3 * lane_width + (i - 3) * 2],
                     [square_length / 2 + extension, square_length / 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([-3 * lane_width - (i - 3) * 2, -3 * lane_width - (i - 3) * 2],
                     [-square_length / 2 - extension, -square_length / 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)
            plt.plot([-3 * lane_width - (i - 3) * 2, -3 * lane_width - (i - 3) * 2],
                     [square_length / 2 + extension, square_length / 2],
                     linestyle=linestyle, color='black', linewidth=linewidth)

        v_light = self.env.v_light
        if v_light == 0 or v_light == 1:
            v_color, h_color = 'green', 'red'
        elif v_light == 2:
            v_color, h_color = 'orange', 'red'
        elif v_light == 3 or v_light == 4:
            v_color, h_color = 'red', 'green'
        else:
            v_color, h_color = 'red', 'orange'

        plt.plot([0, (LANE_NUMBER - 1) * lane_width], [-square_length / 2, -square_length / 2],
                 color=v_color, linewidth=light_line_width)
        plt.plot([(LANE_NUMBER - 1) * lane_width, LANE_NUMBER * lane_width], [-square_length / 2, -square_length / 2],
                 color='green', linewidth=light_line_width)

        plt.plot([-LANE_NUMBER * lane_width, -(LANE_NUMBER - 1) * lane_width], [square_length / 2, square_length / 2],
                 color='green', linewidth=light_line_width)
        plt.plot([-(LANE_NUMBER - 1) * lane_width, 0], [square_length / 2, square_length / 2],
                 color=v_color, linewidth=light_line_width)

        plt.plot([-square_length / 2, -square_length / 2], [0, -(LANE_NUMBER - 1) * lane_width],
                 color=h_color, linewidth=light_line_width)
        plt.plot([-square_length / 2, -square_length / 2], [-(LANE_NUMBER - 1) * lane_width, -LANE_NUMBER * lane_width],
                 color='green', linewidth=light_line_width)

        plt.plot([square_length / 2, square_length / 2], [(LANE_NUMBER - 1) * lane_width, 0],
                 color=h_color, linewidth=light_line_width)
        plt.plot([square_length / 2, square_length / 2], [LANE_NUMBER * lane_width, (LANE_NUMBER - 1) * lane_width],
                 color='green', linewidth=light_line_width)

        # ----------Oblique--------------

        plt.plot([LANE_NUMBER * lane_width + 4, square_length / 2],
                 [-square_length / 2, -LANE_NUMBER * lane_width - 4],
                 color='black', linewidth=2)
        plt.plot([LANE_NUMBER * lane_width + 4, square_length / 2],
                 [square_length / 2, LANE_NUMBER * lane_width + 4],
                 color='black', linewidth=2)
        plt.plot([-LANE_NUMBER * lane_width - 4, -square_length / 2],
                 [-square_length / 2, -LANE_NUMBER * lane_width - 4],
                 color='black', linewidth=2)
        plt.plot([-LANE_NUMBER * lane_width - 4, -square_length / 2],
                 [square_length / 2, LANE_NUMBER * lane_width + 4],
                 color='black', linewidth=2)

        # ----------人行横道--------------
        jj = 3.5
        for ii in range(23):
            if ii <= 3:
                continue
            ax.add_patch(plt.Rectangle((-square_length / 2 + jj + ii * 1.6, -square_length / 2 + 0.5), 0.8, 4,
                                       color='lightgray', alpha=0.5))
            ii += 1
        for ii in range(23):
            if ii <= 3:
                continue
            ax.add_patch(plt.Rectangle((-square_length / 2 + jj + ii * 1.6, square_length / 2 - 0.5 - 4), 0.8, 4,
                                       color='lightgray', alpha=0.5))
            ii += 1
        for ii in range(23):
            if ii <= 3:
                continue
            ax.add_patch(
                plt.Rectangle((-square_length / 2 + 0.5, square_length / 2 - jj - 0.8 - ii * 1.6), 4, 0.8,
                              color='lightgray',
                              alpha=0.5))
            ii += 1
        for ii in range(23):
            if ii <= 3:
                continue
            ax.add_patch(
                plt.Rectangle((square_length / 2 - 0.5 - 4, square_length / 2 - jj - 0.8 - ii * 1.6), 4, 0.8,
                              color='lightgray',
                              alpha=0.5))
            ii += 1

        def is_in_plot_area(x, y, tolerance=5):
            if -square_length / 2 - extension + tolerance < x < square_length / 2 + extension - tolerance and \
                    -square_length / 2 - extension + tolerance < y < square_length / 2 + extension - tolerance:
                return True
            else:
                return False

        def draw_rotate_rec(x, y, a, l, w, c):
            bottom_left_x, bottom_left_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
            ax.add_patch(plt.Rectangle((x + bottom_left_x, y + bottom_left_y), w, l, edgecolor=c,
                                       facecolor='white', angle=-(90 - a), zorder=50))

        def plot_phi_line(type, x, y, phi, color):
            # TODO:新增一个type项输入
            if type in ['bicycle_1', 'bicycle_2', 'bicycle_3']:
                line_length = 1.5
            elif type == 'DEFAULT_PEDTYPE':
                line_length = 1.5
            else:
                line_length = 3
            x_forw, y_forw = x + line_length * cos(phi*pi/180.),\
                             y + line_length * sin(phi*pi/180.)
            plt.plot([x, x_forw], [y, y_forw], color=color, linewidth=0.5)

        # plot cars
        for veh in self.env.all_vehicles:
            veh_x = veh['x']
            veh_y = veh['y']
            veh_phi = veh['phi']
            veh_l = veh['l']
            veh_w = veh['w']
            veh_type = veh['type']
            if veh_type in ['bicycle_1', 'bicycle_2', 'bicycle_3']:
                veh_color = 'navy'
            elif veh_type == 'DEFAULT_PEDTYPE':
                veh_color = 'purple'
            else:
                veh_color = 'black'
            if is_in_plot_area(veh_x, veh_y):
                plot_phi_line(veh_type, veh_x, veh_y, veh_phi, 'black')
                draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, 'black')

        # plot_interested vehs
        # for mode, num in self.veh_mode_dict.items():
        #     for i in range(num):
        #         veh = self.interested_vehs[mode][i]
        #         veh_x = veh['x']
        #         veh_y = veh['y']
        #         veh_phi = veh['phi']
        #         veh_l = veh['l']
        #         veh_w = veh['w']
        #         task2color = {'left': 'b', 'straight': 'c', 'right': 'm'}
        #
        #         if is_in_plot_area(veh_x, veh_y):
        #             plot_phi_line(veh_x, veh_y, veh_phi, 'black')
        #             task = MODE2TASK[mode]
        #             color = task2color[task]
        #             draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, color, linestyle=':')


        # plot_interested vehs
        for mode, num in self.veh_mode_dict.items():
            for i in range(num):
                veh = self.interested_vehs[mode][i]
                veh_x = veh['x']
                veh_y = veh['y']
                veh_phi = veh['phi']
                veh_l = veh['l']
                veh_w = veh['w']
                veh_type = veh['type']
                #TODO: 定义veh_type
                # print("车辆信息", veh)
                # veh_type = 'car_1'
                task2color = {'left': 'b', 'straight': 'c', 'right': 'm'}

                if is_in_plot_area(veh_x, veh_y):
                    plot_phi_line(veh_type, veh_x, veh_y, veh_phi, 'black')
                    task = MODE2TASK[mode]
                    color = task2color[task]
                    draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, color, linestyle=':')

        # plot_interested bicycle
        for mode, num in self.bicycle_mode_dict.items():
            for i in range(num):
                veh = self.interested_vehs[mode][i]
                veh_x = veh['x']
                veh_y = veh['y']
                veh_phi = veh['phi']
                veh_l = veh['l']
                veh_w = veh['w']
                veh_type = veh['type']
                # TODO: 定义veh_type
                # print("车辆信息", veh)
                # veh_type = 'bicycle_1'
                task2color = {'left': 'b', 'straight': 'c', 'right': 'm'}

                if is_in_plot_area(veh_x, veh_y):
                    plot_phi_line(veh_type, veh_x, veh_y, veh_phi, 'black')
                    task = MODE2TASK[mode]
                    color = task2color[task]
                    draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, color, linestyle=':')

        # plot_interested person
        for mode, num in self.person_mode_dict.items():
            for i in range(num):
                veh = self.interested_vehs[mode][i]
                veh_x = veh['x']
                veh_y = veh['y']
                veh_phi = veh['phi']
                veh_l = veh['l']
                veh_w = veh['w']
                veh_type = veh['type']
                # TODO: 定义veh_type
                # print("车辆信息", veh)
                # veh_type = 'bicycle_1'
                task2color = {'left': 'b', 'straight': 'c', 'right': 'm'}

                if is_in_plot_area(veh_x, veh_y):
                    plot_phi_line(veh_type, veh_x, veh_y, veh_phi, 'black')
                    task = MODE2TASK[mode]
                    color = task2color[task]
                    draw_rotate_rec(veh_x, veh_y, veh_phi, veh_l, veh_w, color, linestyle=':')

        ego_v_x = self.env.ego_dynamics['v_x']
        ego_v_y = self.env.ego_dynamics['v_y']
        ego_r = self.env.ego_dynamics['r']
        ego_x = self.env.ego_dynamics['x']
        ego_y = self.env.ego_dynamics['y']
        ego_phi = self.env.ego_dynamics['phi']
        ego_l = self.env.ego_dynamics['l']
        ego_w = self.env.ego_dynamics['w']
        ego_alpha_f = self.env.ego_dynamics['alpha_f']
        ego_alpha_r = self.env.ego_dynamics['alpha_r']
        alpha_f_bound = self.env.ego_dynamics['alpha_f_bound']
        alpha_r_bound = self.env.ego_dynamics['alpha_r_bound']
        r_bound = self.env.ego_dynamics['r_bound']

        plot_phi_line('self_car', ego_x, ego_y, ego_phi, 'fuchsia')
        draw_rotate_rec(ego_x, ego_y, ego_phi, ego_l, ego_w, 'fuchsia')
        self.hist_posi.append((ego_x, ego_y))

        # plot history
        xs = [pos[0] for pos in self.hist_posi]
        ys = [pos[1] for pos in self.hist_posi]
        plt.scatter(np.array(xs), np.array(ys), color='fuchsia', alpha=0.1)


        # plot future data
        tracking_info = self.obs[
                        self.env.ego_info_dim:self.env.ego_info_dim + self.env.per_tracking_info_dim * (self.env.num_future_data + 1)]
        future_path = tracking_info[self.env.per_tracking_info_dim:]
        for i in range(self.env.num_future_data):
            delta_x, delta_y, delta_phi = future_path[i * self.env.per_tracking_info_dim:
                                                      (i + 1) * self.env.per_tracking_info_dim]
            path_x, path_y, path_phi = ego_x + delta_x, ego_y + delta_y, ego_phi - delta_phi
            plt.plot(path_x, path_y, 'g.')
            plot_phi_line(path_x, path_y, path_phi, 'g')

        delta_, _, _ = tracking_info[:3]
        indexs, points = self.env.ref_path.find_closest_point(np.array([ego_x], np.float32), np.array([ego_y], np.float32))
        path_x, path_y, path_phi = points[0][0], points[1][0], points[2][0]
        # plt.plot(path_x, path_y, 'g.')
        delta_x, delta_y, delta_phi = ego_x - path_x, ego_y - path_y, ego_phi - path_phi

        # plot real time traj
        try:
            color = ['blue', 'coral', 'darkcyan']
            for i, item in enumerate(traj_list):
                if i == MPC_path_index:
                    plt.plot(item.path[0], item.path[1], color=color[i])
                else:
                    plt.plot(item.path[0], item.path[1], color=color[i], alpha=0.3)
                indexs, points = item.find_closest_point(np.array([ego_x], np.float32), np.array([ego_y], np.float32))
                path_x, path_y, path_phi = points[0][0], points[1][0], points[2][0]
                plt.plot(path_x, path_y, color=color[i])
        except Exception:
            pass

        # plot ego dynamics
        text_x, text_y_start = -120, 60
        ge = iter(range(0, 1000, 4))
        plt.text(text_x, text_y_start - next(ge), 'ego_x: {:.2f}m'.format(ego_x))
        plt.text(text_x, text_y_start - next(ge), 'ego_y: {:.2f}m'.format(ego_y))
        plt.text(text_x, text_y_start - next(ge), 'path_x: {:.2f}m'.format(path_x))
        plt.text(text_x, text_y_start - next(ge), 'path_y: {:.2f}m'.format(path_y))
        plt.text(text_x, text_y_start - next(ge), 'delta_: {:.2f}m'.format(delta_))
        plt.text(text_x, text_y_start - next(ge), 'delta_x: {:.2f}m'.format(delta_x))
        plt.text(text_x, text_y_start - next(ge), 'delta_y: {:.2f}m'.format(delta_y))
        plt.text(text_x, text_y_start - next(ge), r'ego_phi: ${:.2f}\degree$'.format(ego_phi))
        plt.text(text_x, text_y_start - next(ge), r'path_phi: ${:.2f}\degree$'.format(path_phi))
        plt.text(text_x, text_y_start - next(ge), r'delta_phi: ${:.2f}\degree$'.format(delta_phi))
        plt.text(text_x, text_y_start - next(ge), 'v_x: {:.2f}m/s'.format(ego_v_x))
        plt.text(text_x, text_y_start - next(ge), 'exp_v: {:.2f}m/s'.format(self.env.exp_v))
        plt.text(text_x, text_y_start - next(ge), 'v_y: {:.2f}m/s'.format(ego_v_y))
        plt.text(text_x, text_y_start - next(ge), 'yaw_rate: {:.2f}rad/s'.format(ego_r))
        plt.text(text_x, text_y_start - next(ge), 'yaw_rate bound: [{:.2f}, {:.2f}]'.format(-r_bound, r_bound))

        plt.text(text_x, text_y_start - next(ge), r'$\alpha_f$: {:.2f} rad'.format(ego_alpha_f))
        plt.text(text_x, text_y_start - next(ge), r'$\alpha_f$ bound: [{:.2f}, {:.2f}] '.format(-alpha_f_bound,
                                                                                                alpha_f_bound))
        plt.text(text_x, text_y_start - next(ge), r'$\alpha_r$: {:.2f} rad'.format(ego_alpha_r))
        plt.text(text_x, text_y_start - next(ge), r'$\alpha_r$ bound: [{:.2f}, {:.2f}] '.format(-alpha_r_bound,
                                                                                                alpha_r_bound))
        if self.env.action is not None:
            steer, a_x = self.env.action[0], self.env.action[1]
            plt.text(text_x, text_y_start - next(ge),
                     r'steer: {:.2f}rad (${:.2f}\degree$)'.format(steer, steer * 180 / np.pi))
            plt.text(text_x, text_y_start - next(ge), 'a_x: {:.2f}m/s^2'.format(a_x))

        text_x, text_y_start = 70, 60
        ge = iter(range(0, 1000, 4))

        # done info
        plt.text(text_x, text_y_start - next(ge), 'done info: {}'.format(self.env.done_type))

        # reward info
        if self.env.reward_info is not None:
            for key, val in self.env.reward_info.items():
                plt.text(text_x, text_y_start - next(ge), '{}: {:.4f}'.format(key, val))

        text_x, text_y_start = -36, -70
        ge = iter(range(0, 1000, 6))
        plt.text(text_x+10, text_y_start - next(ge), 'MPC', fontsize=14, color='r', fontstyle='italic')
        if MPC_traj_return_value is not None:
            for i, value in enumerate(MPC_traj_return_value):
                if i == MPC_path_index:
                    plt.text(text_x, text_y_start - next(ge), 'Path cost={:.4f}'.format(value), fontsize=14,
                             color=color[i], fontstyle='italic')
                else:
                    plt.text(text_x, text_y_start - next(ge), 'Path cost={:.4f}'.format(value), fontsize=12,
                             color=color[i], fontstyle='italic')
        plt.pause(0.001)


def main():
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = './results/{time}'.format(time=time_now)
    hier_decision = HierarchicalMpc('left', logdir)
    for i in range(15):
        done = 0
        for _ in range(150):
            done = hier_decision.step()
            if done:
                break
        # np.save('mpc.npy', np.array(hier_decision.data2plot))
        hier_decision.reset()


def plot_data(epi_num, logdir):
    recorder = Recorder()
    recorder.load(logdir)
    recorder.plot_mpc_rl(epi_num, logdir, sample=True)


if __name__ == '__main__':
    main()
    # plot_data(epi_num=6, logdir='./results/2021-03-17-22-33-09')  # 6 or 3
