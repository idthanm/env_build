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

from endtoend_env_utils import shift_and_rotate_coordination, rotate_and_shift_coordination

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
    return x_in_car_coord, y_in_car_coord, a_in_car_coord


SUMO_BINARY = checkBinary('sumo')
SIM_PERIOD = 1.0 / 10


class Traffic(object):

    def __init__(self, step_length, mode, training_task='left'):  # mode 'display' or 'training'
        self.traffic_change_flag = True
        self.random_traffic = None
        self.sim_time = 0
        self.n_ego_vehicles = defaultdict(list)
        self.n_ego_info = {}
        self.step_length = step_length
        self.step_time_str = str(float(step_length) / 1000)
        self.collision_flag = False
        self.n_ego_collision_flag = {}
        self.collision_ego_id = None
        self.v_light = None
        self.n_ego_dict = dict()
        # dict(DL1=dict(x=1.875, y=-30, v=3, a=90, l=4.8, w=2.2),
        #      UR1=dict(x=-1.875, y=30, v=3, a=-90, l=4.8, w=2.2),
        #      DR1=dict(x=5.625, y=-30, v=3, a=90, l=4.8, w=2.2),
        #      RU1=dict(x=5.625, y=-30, v=3, a=90, l=4.8, w=2.2))

        self.mode = mode
        self.training_light_phase = 0
        if training_task == 'right':
            if random.random() > 0.5:
                self.training_light_phase = 2

    def __del__(self):
        traci.close()

    def add_self_car(self, n_ego_dict):
        for egoID, ego_dict in n_ego_dict.items():
            traci.vehicle.addLegacy(vehID=egoID, routeID=ego_dict['routeID'],
                                    depart=0, pos=0, lane=1, speed=ego_dict['v'],
                                    typeID='self_car')
            traci.vehicle.setLength(egoID, ego_dict['l'])
            traci.vehicle.setWidth(egoID, ego_dict['w'])
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
        # wait for some time for cars to enter intersection
        self.add_self_car(dict(ego_init=dict(l=4.8, w=2.2, v=0, routeID='du')))
        random_start_time = random.randint(10, 30)

        while True:
            if traci.simulation.getTime() > random_start_time:
                random_traffic = traci.vehicle.getContextSubscriptionResults('ego_init')
                break
            # traci.vehicle.moveToXY('ego_init', '1o', 1, 1.875, 200, 0)
            if self.mode == "training":
                traci.trafficlight.setPhase('0', self.training_light_phase)
            traci.simulationStep()
        # delete ego car in getContextSubscriptionResults
        del random_traffic['ego_init']
        traci.vehicle.remove(vehID='ego_init')
        return random_traffic

    def init(self, init_n_ego_dict):
        self.sim_time = 0
        self.n_ego_dict = init_n_ego_dict
        # SUMO_BINARY = checkBinary('sumo-gui')
        seed = random.randint(30, 50)
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

        # insert ego car and random traffic

        self.random_traffic = self.generate_random_traffic()
        self.add_self_car(init_n_ego_dict)

        # move ego to the given position and remove conflict cars
        for egoID, ego_dict in self.n_ego_dict.items():
            ego_x, ego_y, ego_v, ego_a, ego_l, ego_w = ego_dict['x'], ego_dict['y'], ego_dict['v'], ego_dict['a'], \
                                                       ego_dict['l'], ego_dict['w']
            veh_to_pop = []
            for veh in self.random_traffic:
                x_in_sumo, y_in_sumo = self.random_traffic[veh][traci.constants.VAR_POSITION]
                a_in_sumo = self.random_traffic[veh][traci.constants.VAR_ANGLE]
                veh_length = self.random_traffic[veh][traci.constants.VAR_LENGTH]
                veh_width = self.random_traffic[veh][traci.constants.VAR_WIDTH]
                velocity = self.random_traffic[veh][traci.constants.VAR_SPEED]
                x, y, a = _convert_sumo_coord_to_car_coord(x_in_sumo, y_in_sumo, a_in_sumo, veh_length)
                x_in_ego_coord, y_in_ego_coord, a_in_ego_coord = shift_and_rotate_coordination(x, y, a, ego_x,
                                                                                               ego_y, ego_a)
                if abs(x_in_ego_coord) < 8 and abs(y_in_ego_coord) < 2:
                    traci.vehicle.remove(vehID=veh)
                    veh_to_pop.append(veh)
            for veh in veh_to_pop:
                self.random_traffic.pop(veh)
            ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo = _convert_car_coord_to_sumo_coord(ego_x, ego_y, ego_a, ego_l)
            traci.vehicle.moveToXY(egoID, '1o', 1, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo)
            traci.vehicle.setSpeed(egoID, ego_v)
        traci.simulationStep()
        self._get_vehicles()
        self._get_own_car()
        if self.mode == 'training':
            traci.trafficlight.setPhase('0', self.training_light_phase)
        else:
            traci.trafficlight.setPhase('0', traci.trafficlight.getPhase('0'))
        self._get_traffic_light()
        self.collision_check()
        for egoID, collision_flag in self.n_ego_collision_flag.items():
            if collision_flag:
                self.collision_flag = True
                self.collision_ego_id = egoID

    def _get_vehicles(self):
        """Get other vehicles' information not including ego vehicle.

        Get other vehicles' information in car coordination not including ego vehicle.
        """
        self.n_ego_vehicles = defaultdict(list)
        for egoID in self.n_ego_dict.keys():
            veh_info_dict = traci.vehicle.getContextSubscriptionResults(egoID)
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
                self.n_ego_vehicles[egoID].append(dict(x=x, y=y, v=v, heading=a, length=length,
                                                       width=width, route=route))

    def _get_traffic_light(self):
        self.v_light = traci.trafficlight.getPhase('0')

    def sim_step(self):
        self.sim_time += SIM_PERIOD
        if self.mode == 'training':
            traci.trafficlight.setPhase('0', self.training_light_phase)
        traci.simulationStep()
        self._get_vehicles()
        self._get_own_car()
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
            self.n_ego_dict[egoID]['x'] = ego_x = n_ego_dict_[egoID]['x']
            self.n_ego_dict[egoID]['y'] = ego_y = n_ego_dict_[egoID]['y']
            self.n_ego_dict[egoID]['a'] = ego_a = n_ego_dict_[egoID]['a']
            self.n_ego_dict[egoID]['v'] = ego_v = n_ego_dict_[egoID]['v']
            ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo = _convert_car_coord_to_sumo_coord(ego_x, ego_y, ego_a,
                                                                                           self.n_ego_dict[egoID]['l'])
            traci.vehicle.moveToXY(egoID, '1o', 1, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo)
            traci.vehicle.setSpeed(egoID, ego_v)

    def _get_own_car(self):
        self.n_ego_info = {}
        for egoID, ego_dict in self.n_ego_dict.items():
            self.n_ego_info[egoID] = dict(x=ego_dict['x'],
                                          y=ego_dict['y'],
                                          v=ego_dict['v'],
                                          heading=ego_dict['a'],
                                          Car_length=ego_dict['l'],
                                          Car_width=ego_dict['w'],
                                          Corner_point=self.cal_corner_point_of_ego_car(ego_dict))

    def cal_corner_point_of_ego_car(self, ego_dict):
        l = ego_dict['l']
        w = ego_dict['w']
        x = ego_dict['x']
        y = ego_dict['y']
        a = ego_dict['a']
        x0, y0, a0 = rotate_and_shift_coordination(l / 2, w / 2, 0, -x, -y, -a)
        x1, y1, a1 = rotate_and_shift_coordination(l / 2, -w / 2, 0, -x, -y, -a)
        x2, y2, a2 = rotate_and_shift_coordination(-l / 2, w / 2, 0, -x, -y, -a)
        x3, y3, a3 = rotate_and_shift_coordination(-l / 2, -w / 2, 0, -x, -y, -a)
        return (x0, y0), (x1, y1), (x2, y2), (x3, y3)

    def collision_check(self):  # True: collision
        flag_dict = dict()
        for egoID, list_of_veh_dict in self.n_ego_vehicles.items():
            ego_x = self.n_ego_dict[egoID]['x']
            ego_y = self.n_ego_dict[egoID]['y']
            ego_a = self.n_ego_dict[egoID]['a']
            ego_l = self.n_ego_dict[egoID]['l']
            ego_w = self.n_ego_dict[egoID]['w']
            ego_lw = (ego_l - ego_w) / 2
            ego_x0 = (ego_x + cos(ego_a / 180 * pi) * ego_lw)
            ego_y0 = (ego_y + sin(ego_a / 180 * pi) * ego_lw)
            ego_x1 = (ego_x - cos(ego_a / 180 * pi) * ego_lw)
            ego_y1 = (ego_y - sin(ego_a / 180 * pi) * ego_lw)
            flag_dict[egoID] = False

            for veh in list_of_veh_dict:
                if fabs(veh['x'] - ego_x) < 10 and fabs(veh['y'] - ego_y) < 2:
                    surrounding_lw = (veh['length'] - veh['width']) / 2
                    surrounding_x0 = (veh['x'] + cos(veh['heading'] / 180 * pi) * surrounding_lw)
                    surrounding_y0 = (veh['y'] + sin(veh['heading'] / 180 * pi) * surrounding_lw)
                    surrounding_x1 = (veh['x'] - cos(veh['heading'] / 180 * pi) * surrounding_lw)
                    surrounding_y1 = (veh['y'] - sin(veh['heading'] / 180 * pi) * surrounding_lw)
                    collision_check_dis = ((veh['width'] + ego_w) / 2 + 0.5) ** 2
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
