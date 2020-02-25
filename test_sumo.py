import os, sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
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

for i in range(1000):
    if i <= 100:

        traci.vehicle.moveToXY('ego', '1o', 0, -2.241, -5.148, 90 - 129.2)
        traci.simulationStep()

    # traci.vehicle.moveToXY('car2', '1o', 0, 5.625, -40)
    # traci.vehicle.moveToXY('car3', '1o', 1, 1.875, -30)
    else:
        traci.simulationStep()
        # if i >= 150:
        #     random_traffic = traci.vehicle.getContextSubscriptionResults('ego')
