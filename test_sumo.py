#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: test_sumo.py
# =====================================

import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import sumolib
import traci

SUMO_BINARY = checkBinary('sumo-gui')
SIM_PERIOD = 1.0 / 10

dirname = os.path.dirname(__file__)
traci.start(
    [SUMO_BINARY, "-c", dirname + "/sumo_files/cross.sumocfg",
     "--step-length", str(SIM_PERIOD),
     "--lateral-resolution", "1.25",
     "--random",
     # "--start",
     # "--quit-on-end",
     "--no-warnings",
     "--no-step-log",
     # "--collision.check-junctions",
     # "--collision.action", "remove",
     # '--seed', str(int(seed))
     ], numRetries=5)  # '--seed', str(int(seed))
#
traci.vehicle.addLegacy(vehID='ego1', routeID='dl',
                        depart=0, pos=0, lane=1, speed=0,
                        typeID='self_car')
traci.vehicle.addLegacy(vehID='ego2', routeID='ud',
                        depart=0, pos=0, lane=1, speed=0,
                        typeID='self_car')
# traci.vehicle.addLegacy(vehID='car2', routeID='wn',
#                         depart=0, pos=0, lane=1, speed=0,
#                         typeID='car_1')
# traci.vehicle.addLegacy(vehID='car3', routeID='wn',
#                         depart=0, pos=0, lane=1, speed=0,
#                         typeID='car_1')
# traci.vehicle.addLegacy(vehID='car1', routeID='wn',
#                         depart=0, pos=0, lane=1, speed=0,
#                         typeID='car_1')
# traci.vehicle.moveToXY('ego', '1o', 1, 1.875, -40)
# traci.vehicle.moveToXY('car1', '1o', 1, 1.875, -50)
# traci.vehicle.moveToXY('car2', '1o', 0, 5.625, -40)
# traci.vehicle.moveToXY('car3', '1o', 1, 1.875, -30)

traci.vehicle.addLegacy(vehID='ego', routeID='dl',
                        depart=0, pos=0, lane=1, speed=0,
                        typeID='self_car')

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

traci.vehicle.subscribeContext('ego1',
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

def test_MSLCM_bug():
    # this is to debug
    # "sumo: ...MSLCM_SL2015.cpp:1913: virtual void MSLCM_SL2015::updateExpectedSublaneSpeeds(const MSLeaderDistanceInfo&, int, int): Assertion `preb.size() == lanes.size()' failed."
    # the problem happens the minute ego car get into the intersection, edgeID and lane should be tested
    for i in range(10000):
        if i <= 10000:
            traci.vehicle.moveToXY(vehID='ego', edgeID='1o', lane=1, x=1.5, y=-18.6, angle=0.)
            traci.vehicle.setSpeed('ego', 1.)
            traci.simulationStep()
            random_traffic = traci.vehicle.getContextSubscriptionResults('ego1')
            print(random_traffic['ego'])
            traci.vehicle.moveToXY(vehID='ego', edgeID='0', lane=0, x=1.5, y=-17, angle=0.)
            traci.vehicle.setSpeed('ego', 1.)
            traci.simulationStep()
            random_traffic = traci.vehicle.getContextSubscriptionResults('ego1')
            print(random_traffic['ego'])

            # traci.vehicle.moveToXY('ego', '1o', 1, x=-17, y=1.5, angle=-90)
            # # traci.vehicle.setSpeed('ego', 11.0)
            # traci.simulationStep()
            # traci.vehicle.moveToXY('ego', '1o', 1, x=-18.6, y=1.5, angle=-90)
            # # traci.vehicle.setSpeed('ego', 11.0)
            # traci.simulationStep()

            # traci.vehicle.moveToXY(vehID='ego', edgeID='1o', lane=0, x=1.5, y=17., angle=0.)
            # # traci.vehicle.setSpeed('ego', 11.0)
            # traci.simulationStep()
            # traci.vehicle.moveToXY(vehID='ego', edgeID='1o', lane=0, x=1.5, y=18.6, angle=0.)
            # # traci.vehicle.setSpeed('ego', 11.0)
            # traci.simulationStep()

            # traci.vehicle.moveToXY(vehID='ego', edgeID='1o', lane=0, x=17., y=-1.5, angle=0.)
            # # traci.vehicle.setSpeed('ego', 11.0)
            # traci.simulationStep()
            # traci.vehicle.moveToXY(vehID='ego', edgeID='1o', lane=0, x=18.6, y=-1.5, angle=0.)
            # # traci.vehicle.setSpeed('ego', 11.0)
            # traci.simulationStep()


def test_other_car_collision():
    # first of all, moveToXY, the lane matters (should be assigned to the right lane)!
    # the keeproute matters! better be 1
    # the collision related to veh speed, veh dist to ego, and whether rear veh recognize the ego
    from dynamics_and_models import ReferencePath
    from endtoend_env_utils import _convert_car_coord_to_sumo_coord
    from math import pi, cos, sin
    ref_path = ReferencePath('left')
    for i in range(10000):
        if 0 < i <= 10000:
            if i < 2:
                traci.vehicle.moveToXY(vehID='ego2', edgeID='4o', lane=1, x=-1.875, y=10, angle=-180)

            # traci.vehicle.setLength('ego', 5)
            # traci.vehicle.setWidth('ego', 2)
            traci.trafficlight.setPhase('0', 0)
            ego_x, ego_y, ego_phi = ref_path.indexs2points(1500)
            ego_x_sumo, ego_y_sumo, ego_phi_sumo = _convert_car_coord_to_sumo_coord(ego_x, ego_y, ego_phi, 4.8)

            traci.vehicle.setSpeed('ego', 0)

            traci.vehicle.moveToXY(vehID='ego', edgeID='0', lane=0, x=ego_x_sumo, y=ego_y_sumo, angle=ego_phi_sumo)
            # if i < 5:
            #    traci.vehicle.moveToXY(vehID='ego1', edgeID='1o', lane=1, x=1.875, y=min(-19, ego_y-8), angle=0)

            traci.simulationStep()
        else:
            traci.simulationStep()


if __name__ == '__main__':
    test_other_car_collision()
