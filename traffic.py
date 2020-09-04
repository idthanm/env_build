#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: traffic.py
# =====================================

import math
import os
import random
import sys
from collections import defaultdict
from math import fabs, cos, sin, pi

from endtoend_env_utils import shift_and_rotate_coordination

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumolib
from sumolib import checkBinary
import traci
from traci.exceptions import FatalTraCIError


def _convert_car_coord_to_sumo_coord(x_in_car_coord, y_in_car_coord, a_in_car_coord, car_length):  # a in deg
    x_in_sumo_coord = x_in_car_coord + car_length / 2 * math.cos(math.radians(a_in_car_coord))
    y_in_sumo_coord = y_in_car_coord + car_length / 2 * math.sin(math.radians(a_in_car_coord))
    a_in_sumo_coord = -a_in_car_coord + 90.
    return x_in_sumo_coord, y_in_sumo_coord, a_in_sumo_coord


def _convert_sumo_coord_to_car_coord(x_in_sumo_coord, y_in_sumo_coord, a_in_sumo_coord, car_length):
    a_in_car_coord = -a_in_sumo_coord + 90.
    x_in_car_coord = x_in_sumo_coord - (math.cos(a_in_car_coord / 180. * math.pi) * car_length / 2)
    y_in_car_coord = y_in_sumo_coord - (math.sin(a_in_car_coord / 180. * math.pi) * car_length / 2)
    return x_in_car_coord, y_in_car_coord, deal_with_phi(a_in_car_coord)


def deal_with_phi(phi):
    while phi > 180:
        phi -= 360
    while phi <= -180:
        phi += 360
    return phi


SUMO_BINARY = checkBinary('sumo')
SIM_PERIOD = 1.0 / 10


class Traffic(object):

    def __init__(self, step_length, mode, init_n_ego_dict, training_task='left'):  # mode 'display' or 'training'
        self.random_traffic = None
        self.sim_time = 0
        self.n_ego_vehicles = defaultdict(list)
        self.step_length = step_length
        self.step_time_str = str(float(step_length) / 1000)
        self.collision_flag = False
        self.n_ego_collision_flag = {}
        self.collision_ego_id = None
        self.v_light = None
        self.n_ego_dict = init_n_ego_dict
        # dict(DL1=dict(x=1.875, y=-30, v=3, a=90, l=4.8, w=2.2),
        #      UR1=dict(x=-1.875, y=30, v=3, a=-90, l=4.8, w=2.2),
        #      DR1=dict(x=5.625, y=-30, v=3, a=90, l=4.8, w=2.2),
        #      RU1=dict(x=5.625, y=-30, v=3, a=90, l=4.8, w=2.2))

        self.mode = mode
        self.training_light_phase = 0
        if training_task == 'right':
            if random.random() > 0.5:
                self.training_light_phase = 2

        # SUMO_BINARY = checkBinary('sumo-gui')
        dirname = os.path.dirname(__file__)
        try:
            traci.start(
                [SUMO_BINARY, "-c", dirname + "/sumo_files/cross.sumocfg",
                 "--step-length", self.step_time_str,
                 "--lateral-resolution", "1.25",
                 "--random",
                 # "--start",
                 # "--quit-on-end",
                 "--no-warnings",
                 "--no-step-log",
                 # '--seed', str(int(seed))
                 ], numRetries=5)  # '--seed', str(int(seed))
        except FatalTraCIError:
            print('Retry by other port')
            port = sumolib.miscutils.getFreeSocketPort()
            traci.start(
                [SUMO_BINARY, "-c", dirname + "/sumo_files/cross.sumocfg",
                 "--step-length", self.step_time_str,
                 "--lateral-resolution", "1.25",
                 "--random",
                 # "--start",
                 # "--quit-on-end",
                 "--no-warnings",
                 "--no-step-log",
                 # '--seed', str(int(seed))
                 ], port=port, numRetries=5)  # '--seed', str(int(seed))

        traci.vehicle.subscribeContext('collector',
                                       traci.constants.CMD_GET_VEHICLE_VARIABLE,
                                       999999, [traci.constants.VAR_POSITION,
                                                traci.constants.VAR_LENGTH,
                                                traci.constants.VAR_WIDTH,
                                                traci.constants.VAR_ANGLE,
                                                traci.constants.VAR_SIGNALS,
                                                traci.constants.VAR_SPEED,
                                                traci.constants.VAR_TYPE,
                                                traci.constants.VAR_EMERGENCY_DECEL,
                                                traci.constants.VAR_LANE_INDEX,
                                                traci.constants.VAR_LANEPOSITION,
                                                traci.constants.VAR_EDGES,
                                                traci.constants.VAR_ROUTE_INDEX],
                                       0, 2147483647)
        while traci.simulation.getTime() < 200:
            if self.mode == "training":
                traci.trafficlight.setPhase('0', self.training_light_phase)
            traci.simulationStep()

    def __del__(self):
        traci.close()

    def add_self_car(self, n_ego_dict):
        for egoID, ego_dict in n_ego_dict.items():
            ego_v_x = ego_dict['v_x']
            ego_v_y = ego_dict['v_y']
            ego_l = ego_dict['l']
            ego_x = ego_dict['x']
            ego_y = ego_dict['y']
            ego_phi = ego_dict['phi']

            traci.vehicle.addLegacy(vehID=egoID, routeID=ego_dict['routeID'],
                                    depart=0, pos=20, lane=1, speed=ego_dict['v_x'],
                                    typeID='self_car')
            traci.vehicle.setLength(egoID, ego_dict['l'])
            traci.vehicle.setWidth(egoID, ego_dict['w'])
            ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo = _convert_car_coord_to_sumo_coord(ego_x, ego_y, ego_phi,
                                                                                           ego_l)
            traci.vehicle.moveToXY(egoID, '0', 1, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo)
            traci.vehicle.setSpeed(egoID, math.sqrt(ego_v_x ** 2 + ego_v_y ** 2))
            traci.vehicle.subscribeContext(egoID,
                                           traci.constants.CMD_GET_VEHICLE_VARIABLE,
                                           999999, [traci.constants.VAR_POSITION,
                                                    traci.constants.VAR_LENGTH,
                                                    traci.constants.VAR_WIDTH,
                                                    traci.constants.VAR_ANGLE,
                                                    traci.constants.VAR_SIGNALS,
                                                    traci.constants.VAR_SPEED,
                                                    traci.constants.VAR_TYPE,
                                                    traci.constants.VAR_EMERGENCY_DECEL,
                                                    traci.constants.VAR_LANE_INDEX,
                                                    traci.constants.VAR_LANEPOSITION,
                                                    traci.constants.VAR_EDGES,
                                                    traci.constants.VAR_ROUTE_INDEX],
                                           0, 2147483647)

    def generate_random_traffic(self):
        """
        generate initial random traffic
        """
        # to delete ego car of the last episode
        random_traffic = traci.vehicle.getContextSubscriptionResults('collector')
        for ego_id in self.n_ego_dict.keys():
            if ego_id in random_traffic:
                traci.vehicle.remove(ego_id)
        traci.simulationStep()

        random_traffic = traci.vehicle.getContextSubscriptionResults('collector')

        for ego_id in self.n_ego_dict.keys():
            if ego_id in random_traffic:
                del random_traffic[ego_id]

        return random_traffic

    def init_traffic(self, init_n_ego_dict):
        self.sim_time = 0
        self.n_ego_vehicles = defaultdict(list)
        self.collision_flag = False
        self.n_ego_collision_flag = {}
        self.collision_ego_id = None
        self.v_light = None
        self.training_light_phase = 0
        self.n_ego_dict = init_n_ego_dict
        random_traffic = self.generate_random_traffic()

        self.add_self_car(init_n_ego_dict)

        # move ego to the given position and remove conflict cars
        for egoID, ego_dict in self.n_ego_dict.items():
            ego_x, ego_y, ego_v_x, ego_v_y, ego_phi, ego_l, ego_w = ego_dict['x'], ego_dict['y'], ego_dict['v_x'],\
                                                                    ego_dict['v_y'], ego_dict['phi'], ego_dict['l'], \
                                                                    ego_dict['w']
            veh_to_pop = []
            for veh in random_traffic:
                x_in_sumo, y_in_sumo = random_traffic[veh][traci.constants.VAR_POSITION]
                a_in_sumo = random_traffic[veh][traci.constants.VAR_ANGLE]
                veh_l = random_traffic[veh][traci.constants.VAR_LENGTH]
                veh_w = random_traffic[veh][traci.constants.VAR_WIDTH]
                veh_v = random_traffic[veh][traci.constants.VAR_SPEED]
                x, y, a = _convert_sumo_coord_to_car_coord(x_in_sumo, y_in_sumo, a_in_sumo, veh_l)
                x_in_ego_coord, y_in_ego_coord, a_in_ego_coord = shift_and_rotate_coordination(x, y, a, ego_x,
                                                                                               ego_y, ego_phi)
                ego_x_in_veh_coord, ego_y_in_veh_coord, ego_a_in_veh_coord = shift_and_rotate_coordination(0, 0, 0,
                                                                                                           x_in_ego_coord,
                                                                                                           y_in_ego_coord,
                                                                                                           a_in_ego_coord)
                if (-5 < x_in_ego_coord < 2 * (ego_v_x-veh_v) + ego_l/2. + veh_l/2. and abs(y_in_ego_coord) < 3) or \
                        (-5 < ego_x_in_veh_coord < 2 * (veh_v-ego_v_x) + ego_l/2. + veh_l/2. and abs(ego_y_in_veh_coord) <3):
                    traci.vehicle.remove(vehID=veh)
                    veh_to_pop.append(veh)
            for veh in veh_to_pop:
                random_traffic.pop(veh)

    def _get_vehicles(self):
        """Get other vehicles' information not including ego vehicle.

        Get other vehicles' information in car coordination not including ego vehicle.
        """
        self.n_ego_vehicles = defaultdict(list)
        for egoID in self.n_ego_dict.keys():
            veh_info_dict = traci.vehicle.getContextSubscriptionResults(egoID)
            for egosid in self.n_ego_dict.keys():
                assert egosid in veh_info_dict
            veh_info_dict.pop(egoID)
            for i, veh in enumerate(veh_info_dict):
                length = veh_info_dict[veh][traci.constants.VAR_LENGTH]
                width = veh_info_dict[veh][traci.constants.VAR_WIDTH]
                route = veh_info_dict[veh][traci.constants.VAR_EDGES]
                x_in_sumo, y_in_sumo = veh_info_dict[veh][traci.constants.VAR_POSITION]
                a_in_sumo = veh_info_dict[veh][traci.constants.VAR_ANGLE]
                # transfer x,y,a in car coord
                x, y, a = _convert_sumo_coord_to_car_coord(x_in_sumo, y_in_sumo, a_in_sumo, length)
                v = veh_info_dict[veh][traci.constants.VAR_SPEED]
                self.n_ego_vehicles[egoID].append(dict(x=x, y=y, v=v, phi=a, l=length,
                                                       w=width, route=route))

    def _get_traffic_light(self):
        self.v_light = traci.trafficlight.getPhase('0')

    def sim_step(self):
        self.sim_time += SIM_PERIOD
        if self.mode == 'training':
            traci.trafficlight.setPhase('0', self.training_light_phase)
        traci.simulationStep()
        self._get_vehicles()
        self._get_traffic_light()
        self.collision_check()
        for egoID, collision_flag in self.n_ego_collision_flag.items():
            if collision_flag:
                self.collision_flag = True
                self.collision_ego_id = egoID

    def set_own_car(self, n_ego_dict_):
        """Insert ego vehicle into sumo's traffic modal.

        Args:
            x: Ego vehicle's current x coordination of it's shape center, m.
            y: Ego vehicle's current y coordination of it's shape center, m.
            v: Ego vehicle's current velocity, m/s.
            a: Ego vehicle's current heading angle under car coordinate, deg.

        Raises:
        """
        assert len(self.n_ego_dict) == len(n_ego_dict_)
        for egoID in self.n_ego_dict.keys():
            self.n_ego_dict[egoID]['v_x'] = ego_v_x = n_ego_dict_[egoID]['v_x']
            self.n_ego_dict[egoID]['v_y'] = ego_v_y = n_ego_dict_[egoID]['v_y']
            self.n_ego_dict[egoID]['r'] = ego_r = n_ego_dict_[egoID]['r']
            self.n_ego_dict[egoID]['x'] = ego_x = n_ego_dict_[egoID]['x']
            self.n_ego_dict[egoID]['y'] = ego_y = n_ego_dict_[egoID]['y']
            self.n_ego_dict[egoID]['phi'] = ego_phi = n_ego_dict_[egoID]['phi']

            ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo = _convert_car_coord_to_sumo_coord(ego_x, ego_y, ego_phi,
                                                                                           self.n_ego_dict[egoID]['l'])
            traci.vehicle.moveToXY(egoID, '0', 1, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo)
            traci.vehicle.setSpeed(egoID, math.sqrt(ego_v_x**2+ego_v_y**2))

    def collision_check(self):  # True: collision
        flag_dict = dict()
        for egoID, list_of_veh_dict in self.n_ego_vehicles.items():
            ego_x = self.n_ego_dict[egoID]['x']
            ego_y = self.n_ego_dict[egoID]['y']
            ego_phi = self.n_ego_dict[egoID]['phi']
            ego_l = self.n_ego_dict[egoID]['l']
            ego_w = self.n_ego_dict[egoID]['w']
            ego_lw = (ego_l - ego_w) / 2
            ego_x0 = (ego_x + cos(ego_phi / 180 * pi) * ego_lw)
            ego_y0 = (ego_y + sin(ego_phi / 180 * pi) * ego_lw)
            ego_x1 = (ego_x - cos(ego_phi / 180 * pi) * ego_lw)
            ego_y1 = (ego_y - sin(ego_phi / 180 * pi) * ego_lw)
            flag_dict[egoID] = False

            for veh in list_of_veh_dict:
                if fabs(veh['x'] - ego_x) < 10 and fabs(veh['y'] - ego_y) < 10:
                    surrounding_lw = (veh['l'] - veh['w']) / 2
                    surrounding_x0 = (veh['x'] + cos(veh['phi'] / 180 * pi) * surrounding_lw)
                    surrounding_y0 = (veh['y'] + sin(veh['phi'] / 180 * pi) * surrounding_lw)
                    surrounding_x1 = (veh['x'] - cos(veh['phi'] / 180 * pi) * surrounding_lw)
                    surrounding_y1 = (veh['y'] - sin(veh['phi'] / 180 * pi) * surrounding_lw)
                    collision_check_dis = ((veh['w'] + ego_w) / 2 + 0.5) ** 2
                    if (ego_x0 - surrounding_x0) ** 2 + (ego_y0 - surrounding_y0) ** 2 < collision_check_dis:
                        flag_dict[egoID] = True
                    elif (ego_x0 - surrounding_x1) ** 2 + (ego_y0 - surrounding_y1) ** 2 < collision_check_dis:
                        flag_dict[egoID] = True
                    elif (ego_x1 - surrounding_x1) ** 2 + (ego_y1 - surrounding_y1) ** 2 < collision_check_dis:
                        flag_dict[egoID] = True
                    elif (ego_x1 - surrounding_x0) ** 2 + (ego_y1 - surrounding_y0) ** 2 < collision_check_dis:
                        flag_dict[egoID] = True

        self.n_ego_collision_flag = flag_dict


if __name__ == "__main__":
    pass
