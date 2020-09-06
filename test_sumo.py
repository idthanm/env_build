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
    [SUMO_BINARY, "-c", dirname + "/sumo_files/cross_test.sumocfg",
     "--step-length", str(SIM_PERIOD),
     "--lateral-resolution", "1.25",
     "--random",
     # "--start",
     # "--quit-on-end",
     "--no-warnings",
     "--no-step-log",
     # '--seed', str(int(seed))
     ], numRetries=5)  # '--seed', str(int(seed))
#
# traci.vehicle.addLegacy(vehID='ego', routeID='wn',
#                         depart=0, pos=0, lane=1, speed=0,
#                         typeID='self_car')
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

# this is to debug
# "sumo: ...MSLCM_SL2015.cpp:1913: virtual void MSLCM_SL2015::updateExpectedSublaneSpeeds(const MSLeaderDistanceInfo&, int, int): Assertion `preb.size() == lanes.size()' failed."
# the problem happens the minute ego car get into the intersection, edgeID and lane should be tested
for i in range(10000):
    if i <= 10000:
        traci.vehicle.moveToXY(vehID='ego', edgeID='1o', lane=0, x=1.5, y=-18.6, angle=0.)
        traci.vehicle.setSpeed('ego', 1.)
        traci.simulationStep()
        traci.vehicle.moveToXY(vehID='ego', edgeID='1o', lane=0, x=1.5, y=-17., angle=0.)
        traci.vehicle.setSpeed('ego', 1.)
        traci.simulationStep()

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
