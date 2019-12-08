# coding=utf-8
import math
import optparse
import os
import sys
import copy
import random
from math import fabs, cos, sin, pi
from endtoend_env_utils import shift_and_rotate_coordination, rotate_and_shift_coordination
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import traci

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

    def __init__(self, step_length):  # 该部分可直接与gui相替换
        self.ego_x = 0.0  # m
        self.ego_y = 0.0  # m
        self.ego_v = 0.0  # m/s
        self.ego_a = 0.0  # car coordination, deg
        self.ego_length = 4.8
        self.ego_width = 2.2
        self.ego_lw = (self.ego_length - self.ego_width) / 2
        self.traffic_change_flag = True
        self.random_traffic = None
        self.sim_time = 0
        self.vehicles = []
        self.ego_info = None
        self.step_length = step_length
        self.step_time_str = str(float(step_length) / 1000)
        self.collision_flag = False

    def __del__(self):
        traci.close()

    def add_self_car(self):
        traci.vehicle.addLegacy(vehID='ego', routeID='self_route',
                                depart=0, pos=0, lane=-6, speed=0,
                                typeID='self_car')
        traci.vehicle.setLength('ego', self.ego_length)
        traci.vehicle.setWidth('ego', self.ego_width)
        traci.vehicle.subscribeContext('ego',
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
        random_start_time = 15  # random.randint(10, 70)
        while True:
            if traci.simulation.getTime() > random_start_time:
                random_traffic = traci.vehicle.getContextSubscriptionResults('ego')
                break
            traci.simulationStep()
        # delete ego car in getContextSubscriptionResults
        del random_traffic['ego']
        return random_traffic

    def init(self, ego_init_state):
        self.sim_time = 0

        # SUMO_BINARY = checkBinary('sumo-gui')
        seed = random.randint(10, 15)
        dirname = os.path.dirname(__file__)
        traci.start(
            [SUMO_BINARY, "-c", dirname+"/sumo_files/configuration.sumocfg",
             "--step-length", self.step_time_str,
             "--lateral-resolution", "1.25",
             "--random",
             # "--start",
             # "--quit-on-end",
             "--no-warnings",
             "--no-step-log",
             '--seed', str(int(seed))
             ], numRetries=5)  # '--seed', str(int(seed))

        # insert ego car and random traffic
        self.add_self_car()
        self.random_traffic = self.generate_random_traffic()

        # move ego to the given position and remove conflict cars
        self.ego_x, self.ego_y, self.ego_v, self.ego_a = ego_init_state
        for veh in self.random_traffic:
            x_in_sumo, y_in_sumo = self.random_traffic[veh][traci.constants.VAR_POSITION]
            a_in_sumo = self.random_traffic[veh][traci.constants.VAR_ANGLE]
            veh_length = self.random_traffic[veh][traci.constants.VAR_LENGTH]
            veh_width = self.random_traffic[veh][traci.constants.VAR_WIDTH]
            x, y, a = _convert_sumo_coord_to_car_coord(x_in_sumo, y_in_sumo, a_in_sumo, veh_length)
            x_in_ego_coord, y_in_ego_coord, a_in_ego_coord = shift_and_rotate_coordination(x, y, a, self.ego_x,
                                                                                           self.ego_y, self.ego_a)
            if abs(x_in_ego_coord) < 10 and abs(y_in_ego_coord) < 5:
                traci.vehicle.remove(vehID=veh)
                # traci.vehicle.moveToXY(vehID=veh,
                #                        edgeID='gneE32',
                #                        lane=0,
                #                        x=-3201.72,
                #                        y=-296.48,
                #                        angle=0,
                #                        keepRoute=2)
        ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo = _convert_car_coord_to_sumo_coord(self.ego_x, self.ego_y,
                                                                                       self.ego_a, self.ego_length)

        traci.vehicle.moveToXY('ego', 'gneE20', 0, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo, 0)
        traci.simulationStep()
        self._get_vehicles()
        self._get_own_car()

    def _get_vehicles(self):
        """Get other vehicles' information not including ego vehicle.

        Get other vehicles' information in car coordination not including ego vehicle.
        """
        self.vehicles = []
        # get info of all cars, including ego car
        veh_info_dict = traci.vehicle.getContextSubscriptionResults('ego')
        veh_info_dict.pop('ego')
        for i, veh in enumerate(veh_info_dict):
            length = veh_info_dict[veh][traci.constants.VAR_LENGTH]
            width = veh_info_dict[veh][traci.constants.VAR_WIDTH]
            route = veh_info_dict[veh][traci.constants.VAR_EDGES]
            edge_index = veh_info_dict[veh][traci.constants.VAR_ROUTE_INDEX]
            x_in_sumo, y_in_sumo = veh_info_dict[veh][traci.constants.VAR_POSITION]
            a_in_sumox = veh_info_dict[veh][traci.constants.VAR_ANGLE]
            # transfer x,y,a in car coord
            x, y, a = _convert_sumo_coord_to_car_coord(x_in_sumo, y_in_sumo, a_in_sumox, length)
            v = veh_info_dict[veh][traci.constants.VAR_SPEED]
            self.vehicles.append(dict(x=x, y=y, v=v, angle=a, length=length,
                                      width=width, route=route, edge_index=edge_index))

        return self.vehicles

    def sim_step(self):  # 该部分可直接与package相替换
        self.sim_time += SIM_PERIOD
        traci.simulationStep()
        self._get_vehicles()
        self._get_own_car()
        if not self.collision_check():
            self.collision_flag = True

    def set_own_car(self, x, y, v, a):
        """Insert ego vehicle into sumo's traffic modal.

        Args:
            x: Ego vehicle's current x coordination of it's shape center, m.
            y: Ego vehicle's current y coordination of it's shape center, m.
            v: Ego vehicle's current velocity, m/s.
            a: Ego vehicle's current heading angle under car coordinate, deg.

        Raises:
        """
        self.ego_x, self.ego_y, self.ego_v, self.ego_a = x, y, v, a
        ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo = _convert_car_coord_to_sumo_coord(self.ego_x, self.ego_y,
                                                                                       self.ego_a, self.ego_length)
        traci.vehicle.moveToXY('ego', 'gneE20', 0, ego_x_in_sumo,
                               ego_y_in_sumo, ego_a_in_sumo, 0)

    def _get_own_car(self):
        self.ego_info = {'x': self.ego_x, 'y': self.ego_y, 'v': self.ego_v, 'heading': self.ego_a,
                         'Car_length': self.ego_length, 'Car_width': self.ego_width,
                         'Corner_point': self.cal_corner_point_of_ego_car()}

    def cal_corner_point_of_ego_car(self):
        x0, y0, a0 = rotate_and_shift_coordination(self.ego_length/2, self.ego_width/2, 0, -self.ego_x, -self.ego_y, -self.ego_a)
        x1, y1, a1 = rotate_and_shift_coordination(self.ego_length/2, -self.ego_width/2, 0, -self.ego_x, -self.ego_y, -self.ego_a)
        x2, y2, a2 = rotate_and_shift_coordination(-self.ego_length/2, self.ego_width/2, 0, -self.ego_x, -self.ego_y, -self.ego_a)
        x3, y3, a3 = rotate_and_shift_coordination(-self.ego_length/2, -self.ego_width/2, 0, -self.ego_x, -self.ego_y, -self.ego_a)
        return (x0, y0), (x1, y1), (x2, y2), (x3, y3)

    def collision_check(self):
        for veh in self.vehicles:
            if (fabs(veh['x']-self.ego_x) < 10 and
               fabs(veh['y']-self.ego_y) < 2):
                ego_x0 = (self.ego_x + cos(self.ego_a/180*pi)*self.ego_lw)
                ego_y0 = (self.ego_y + sin(self.ego_a/180*pi)*self.ego_lw)
                ego_x1 = (self.ego_x - cos(self.ego_a/180*pi)*self.ego_lw)
                ego_y1 = (self.ego_y - sin(self.ego_a/180*pi)*self.ego_lw)
                surrounding_lw = (veh['length']-veh['width'])/2
                surrounding_x0 = (veh['x'] + cos(veh['angle'] / 180 * pi) * surrounding_lw)
                surrounding_y0 = (veh['y'] + sin(veh['angle'] / 180 * pi) * surrounding_lw)
                surrounding_x1 = (veh['x'] - cos(veh['angle'] / 180 * pi) * surrounding_lw)
                surrounding_y1 = (veh['y'] - sin(veh['angle'] / 180 * pi) * surrounding_lw)
                collision_check_dis = ((veh['width']+self.ego_width)/2+0.5)**2
                if (ego_x0 - surrounding_x0)**2 + (ego_y0 - surrounding_y0)**2 < collision_check_dis:
                    return False
                if (ego_x0 - surrounding_x1)**2 + (ego_y0 - surrounding_y1)**2 < collision_check_dis:
                    return False
                if (ego_x1 - surrounding_x1)**2 + (ego_y1 - surrounding_y1)**2 < collision_check_dis:
                    return False
                if (ego_x1 - surrounding_x0)**2 + (ego_y1 - surrounding_y0)**2 < collision_check_dis:
                    return False
        return True


if __name__ == "__main__":
    pass
