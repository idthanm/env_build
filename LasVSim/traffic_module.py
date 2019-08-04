# coding=utf-8
"""Traffic module of LasVSim

@Author: Xu Chenxiang
@Date: 2019.02.27
"""
import math
import optparse
import os
import sys
import copy
from LasVSim.data_structures import *
import _struct as struct


class TrafficData:
    """保存随机交通流初始状态的数据类"""

    def __init__(self):
        self.file = None  # 保存数据的二进制文件
        pass

    def __del__(self):
        self.file.close()

    def save_traffic(self, traffic, path):
        self.file = open(path+'/simulation_traffic_data.bin', 'wb')
        for veh in traffic:
            self.file.write(struct.pack('6f', *[traffic[veh][64],
                                                traffic[veh][66][0],
                                                traffic[veh][66][1],
                                                traffic[veh][67],
                                                traffic[veh][68],
                                                traffic[veh][77]]))
            # print(fmt),
            name_length = len(traffic[veh][79])
            fmt = 'i'
            self.file.write(struct.pack(fmt, *[name_length]))
            # print(fmt),
            fmt = str(name_length)+'s'
            self.file.write(struct.pack(fmt,
                                        *[traffic[veh][79].encode()]))
            # print(fmt),
            name_length = len(traffic[veh][87])
            fmt = 'i'
            self.file.write(struct.pack(fmt, *[name_length]))
            # print(name_length),
            for route in traffic[veh][87]:
                name_length = len(route)
                fmt = 'i'
                self.file.write(struct.pack(fmt, *[name_length]))
                # print(fmt),
                fmt = str(name_length) + 's'
                self.file.write(struct.pack(fmt, *[route.encode()]))
                # print(fmt),
        self.file.close()

    def load_traffic(self, path):
        if path is not None:
            traffic = {}
            with open(path+'/simulation_traffic_data.bin', 'rb') as traffic_data:
                fmt = '6f'
                buffer = traffic_data.read(struct.calcsize(fmt))
                # print(fmt),
                id = 0
                while len(buffer) > 0:
                    # 读取车辆位姿信息，float类型变量
                    v, x, y, heading, length, width = struct.unpack(fmt, buffer)

                    # 读取车辆类型，string类型变量
                    fmt = 'i'
                    name_length = struct.unpack(fmt, traffic_data.read(
                        struct.calcsize(fmt)))[0]  # 读取类型名长度
                    # print(fmt),
                    fmt = str(name_length)+'s'
                    type = struct.unpack(fmt, traffic_data.read(
                        struct.calcsize(fmt)))[0]
                    # print(fmt),

                    # 读取车辆路径，string类型变量
                    route = []
                    fmt = 'i'
                    name_length = struct.unpack(fmt, traffic_data.read(
                        struct.calcsize(fmt)))[0]  # 读取车辆路径长度
                    # print(name_length),
                    for i in range(name_length):
                        fmt = 'i'
                        route_length = struct.unpack(fmt, traffic_data.read(
                            struct.calcsize(fmt)))[0]  # 读取路径名长度
                        # print(fmt),
                        fmt = str(route_length)+'s'
                        route.append(struct.unpack(fmt, traffic_data.read(
                            struct.calcsize(fmt)))[0].decode())
                        # print(fmt),
                    traffic[str(id)] = {64: v, 66: (x, y), 67: heading, 68: length,
                                        77: width, 79: type.decode(), 87: route}
                    id += 1
                    fmt = '6f'
                    buffer = traffic_data.read(struct.calcsize(fmt))
                    # print(fmt),
            return traffic
        else:
            return None

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))

    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory "
        "of your sumo installation (it should contain folders 'bin', 'tools' "
        "and 'docs')")
import traci

SUMO_BINARY=checkBinary('sumo')
VEHICLE_COUNT = 501
VEHICLE_INDEX_START = 1
WINKER_PERIOD=0.5
TLS={'6': 'gneJ7', '7': 'gneJ4', '8': 'gneJ8', '11': 'gneJ0', '12': 'gneJ1', '13': 'gneJ2',
     '16':'gneJ6','17':'gneJ3','18':'gneJ9', '0':'gneJ9','1':'gneJ9',
     '2':'gneJ9','3':'gneJ9','4':'gneJ9','5':'gneJ9','9':'gneJ9','10':'gneJ9',
     '14':'gneJ9','15':'gneJ9', '19':'gneJ9','20':'gneJ9','21':'gneJ9',
     '22':'gneJ9','23':'gneJ9','24':'gneJ9'}
OWN_CAR_START_POS=(622.0+3.75/2,-400)
OWN_CAR_START_YAW=0.0
OWN_CAR_START_SPEED=0.0
OWN_CAR_START_LENGTH=4.5
OWN_CAR_START_WIDTH=1.8
SIM_PERIOD=1.0/10
TRAFFIC_LOADED_CHECK_TIME=100


def _getothercarInfo(othercar_dict, othercarname):  # 该部分可直接与gui相替换
    """Get all other vehicles' information in the simulation.

    The parameter othercarname should be the key list of parameter othercar_dict

    Args:
        othercar_dict: A dict containing all other vehicles' raw information.
        othercarname: A list containing all other vehicles' id

    Returns:
        A list containing all other vehicles' information in a special format.
        For example:

        [{'x':0.0, 'y': 0.0, 'v': 0.0, 'angle': 0.0, 'signals': 0, 'length':0.0,
        'width': 0.0, 'type': 0, 'centertohead', 0.0},
        {'x':0.0, 'y': 0.0, 'v': 0.0, 'angle': 0.0, 'signals': 0, 'length':0.0,
        'width': 0.0, 'type': 0, 'centertohead', 0.0},...]
    """
    othercarinfo = []
    for i in range(len(othercarname)):
        name = othercarname[i]
        # 发生这种情况是由于sumo丢失了车辆
        if name not in othercar_dict:
            othercarinfo.append({'x': 99999, 'y': 99999, 'v': 0, 'angle': 0,
                                 'signals': 0, 'length': 0, 'width': 0,
                                 'type': 0, 'centertohead': 0,
                                 'max_decel': 0, 'current_lane': 0})
            continue
        car = {}
        car['x'], car['y'] = othercar_dict[name][traci.constants.VAR_POSITION]
        car['v'] = othercar_dict[name][traci.constants.VAR_SPEED]
        car['angle'] = -othercar_dict[name][traci.constants.VAR_ANGLE] + 90
        car['signals'] = othercar_dict[name][traci.constants.VAR_SIGNALS]
        if car['signals'] == 1:
            car['signals'] = int(0x02)
        elif car['signals'] == 2:
            car['signals'] = int(0x01)
        else:
            car['signals'] = int(0x04)
        car['length'] = othercar_dict[name][68]
        car['width'] = othercar_dict[name][77]
        # sumo中车辆的位置由车辆车头中心表示，因此要计算根据sumo给的坐标换算
        # 车辆中心(形心)的坐标。
        car['x'] = car['x'] - (math.cos(car['angle'] / 180 * math.pi) *
                               car['length'] / 2)
        car['y'] = car['y'] - (math.sin(car['angle'] / 180 * math.pi) *
                               car['length'] / 2)
        car['type'] = othercar_dict[name][traci.constants.VAR_TYPE]
        car['centertohead'] = 2.0
        car['max_decel'] = othercar_dict[name][traci.constants.VAR_EMERGENCY_DECEL]
        car['current_lane'] = othercar_dict[name][traci.constants.VAR_LANE_INDEX]
        othercarinfo.append(car)
    return othercarinfo


# def _getcenterindex(x,y):  # 该部分可直接与gui相替换
#     """Get current intersection's id according to current position: (x,y).
#
#         For Urban Road Map only
#
#         Args:
#             x: Vehicle shape center's x coordination, float.
#             y: Vehicle shape center's y coordination, float.
#
#         Returns:
#             Intersection's id in Urban Road Map.
#
#         Raises:
#     """
#     if (x < -622 - 18 or (x > -622+18 and x < 0 - 18) or
#             (x > 0 + 18 and x < 622 - 18) or x > 622 + 18):  # horizontal
#         roll = (x + 1244) // 622
#         if y > 622:
#             index = 15 + roll
#         elif y < -622:
#             index = 6 + roll
#         elif 622 - 7.5 < y < 622:
#             index = 16 + roll
#         elif 0 < y < 7.5:
#             index = 10 + roll
#         elif -7.5 < y < 0:
#             index = 11 + roll
#         else:
#             index = 5 + roll
#     elif (y < -622 - 18 or (y > -622 + 18 and y < 0 - 18) or
#               (y > 18 and y < 622 - 18) or y > 622 + 18):  # vertical
#         roll = (y + 1244) // 622  # line
#         if x > 622:
#             index = 3 + (roll + 1) * 5
#         elif x < -622:
#             index = 1 + (roll) * 5
#         elif 622 - 7.5 < x < 622:
#             index = 3 + (roll) * 5
#         elif 0 < x < 7.5:
#             index = 2 + (roll + 1) * 5
#         elif -7.5 < x < 0:
#             index = 2 + (roll) * 5
#         else:
#             index = 1 + (roll + 1) * 5
#     else:
#         index = round((x+1244)/622)+round((y+1244)/622)*5
#     return index


class VehicleModels(object):  # 该部分可直接与gui相替换
    """Vehicle model class.

        Read vehicle model information file and load these information into
        simulation.

        Attributes:
            __info: A dict containing all vehicle models' information.
            __type_array: A list containing vehicle type's id.
    """
    __type_array = [0, 1, 2, 3, 7, 100, 1000, 200]

    def __init__(self, model_path):
        self.__info = dict()
        with open(model_path) as f:
            line = f.readline()
            while len(line)>0:
                data = line.split(',')
                type = int(data[1])
                if type not in self.__type_array:
                    line=f.readline()
                    continue
                length=float(data[7])
                width=float(data[8])
                height = float(data[5])
                img_path = []
                self.__info[type]=(length, width, height,img_path)
                line = f.readline()

    def get_types(self):
        return self.__type_array

    def get_vehicle(self,type):
        if type not in self.__info:
            type=0
        return self.__info[type]


HISTORY_TRAFFIC_SETTING = ['Traffic Type', 'Traffic Density',
                           'Map']  # 上次仿真的交通流配置
RANDOM_TRAFFIC = {}  # 随机交通流分布信息


class Traffic(object):
    """Traffic class.

        Traffic class to call sumo to model traffic.

        Attributes:
            __path: A string indicating the map used in simulation.
            random_traffic: A dict containing constant traffic initial
                state generated previously.
            vehicleName: A list containing all vehicles' id in simulation
                including ego vehicle's id 'ego' as the first element.
            vehicles: A list containing all vehicles' information in simulation
                including ego vehicle
            sim_time: A float variable for recording current simulation time.
            __own_x: Ego vehicle's current x coordination of it's shape center, m.
            __own_y: Ego vehicle's current y coordination of it's shape center, m.
            __own_v: Ego vehicle's current velocity, m/s.
            __own_a: Ego vehicle's current heading angle under base coordinate,
                deg.
    """
    def __init__(self, step_length, path=None, traffic_type=None,
                 traffic_density=None, init_traffic=None, seed=None):  # 该部分可直接与gui相替换
        self.seed = None
        if seed is not None:
            self.seed = seed
        self.__own_x = 0.0  # 自车x坐标，m
        self.__own_y = 0.0  # 自车y坐标，m
        self.__own_v = 0.0  # 自车速度标量，m/s
        self.__own_a = 0.0  # 自车偏航角，坐标系1，deg
        # self.__own_lane_pos = float(999.0)  # 自车距离当前车道停止线的距离,m
        # self.__own_lane_speed_limit = float(999.0)  # 自车当前车道限速, m/s
        self.traffic_change_flag = True

        self.__map_type = path
        parent_path = os.path.dirname(__file__)
        self.__path = parent_path + "Map/" + path + "/"
        self.type = traffic_type  # For example: Normal
        self.density = traffic_density  # For example: Middle
        self.step_length = str(float(step_length)/1000)

        global VEHICLE_COUNT, HISTORY_TRAFFIC_SETTING, RANDOM_TRAFFIC
        if traffic_type == 'No Traffic':
            VEHICLE_COUNT = 1
            self.vehicleName = ['ego']
            self.random_traffic = {}
        else:
            # 载入仿真项目时已有初始交通流分布数据
            if init_traffic is not None:
                self.random_traffic = init_traffic
                RANDOM_TRAFFIC = copy.deepcopy(self.random_traffic)
                HISTORY_TRAFFIC_SETTING = [traffic_type, traffic_density, path]
                self.traffic_change_flag = False
            elif (HISTORY_TRAFFIC_SETTING[0] != traffic_type or
                    HISTORY_TRAFFIC_SETTING[1] != traffic_density or
                    HISTORY_TRAFFIC_SETTING[2] != path):
                self.random_traffic = self.__generate_random_traffic()
                traffic_data = TrafficData()
                traffic_data.save_traffic(self.random_traffic, './Scenario/Highway_endtoend')
                RANDOM_TRAFFIC = copy.deepcopy(self.random_traffic)
                HISTORY_TRAFFIC_SETTING = [traffic_type, traffic_density, path]
                self.traffic_change_flag = True
            else:
                # 本次仿真的交通流配置与上次仿真一致则不用重新初始化随机交通流
                self.random_traffic = RANDOM_TRAFFIC
                self.traffic_change_flag = False
            # print(self.random_traffic.keys())
            self.vehicleName = ['ego'] + list(self.random_traffic.keys())
            VEHICLE_COUNT = len(self.vehicleName)

    def __del__(self):  # 该部分可直接与gui相替换
        traci.close()
        pass

    def init(self, source, egocar_length):
        """Initiate traffic.

        Initiate traffic.

        Args:
            source: Ego vehicle's current state.

        Raises:
        """
        self.sim_time = 0
        self.vehicles = [None] * VEHICLE_COUNT
        self.egocar_length = egocar_length

        # SUMO_BINARY = checkBinary('sumo-gui')
        if self.seed is not None:
            traci.start(
                [SUMO_BINARY, "-c", self.__path + "configuration.sumocfg",
                 "--step-length", self.step_length,
                 "--lateral-resolution", "1.25", "--seed", self.seed, "--start",
                 "--quit-on-end"])
        else:
            traci.start(
                [SUMO_BINARY, "-c", self.__path + "configuration.sumocfg",
                 "--step-length", self.step_length,
                 "--lateral-resolution", "1.25", "--random", "--start",
                 "--quit-on-end"])

        # 在sumo的交通流模型中插入自车
        x, y, v, a = source
        self.__own_x, self.__own_y, self.__own_v, self.__own_a = x, y, v, a

        # Sumo function to insert a vehicle.
        traci.vehicle.addLegacy(vehID='ego', routeID='self_route',
                                depart=0, pos=0, lane=-6, speed=0,
                                typeID='self_car')
        traci.vehicle.setLength('ego', 4.8)  # Sumo function
        traci.vehicle.setWidth('ego', 2.2)  # Sumo function
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
                                                traci.constants.VAR_LANEPOSITION],
                                       0, 2147483647)  # Sumo function to get

        # 初始化随机交通流分布
        self.__initiate_traffic()

        traci.vehicle.setLength('ego', 4.8)  # Sumo function
        traci.vehicle.setWidth('ego', 2.2)  # Sumo function
        traci.vehicle.moveToXY('ego', 'gneE20', 0, self.__own_x +
                               self.egocar_length / 2 *
                               math.cos(math.radians(self.__own_a)),
                               self.__own_y + self.egocar_length / 2 *
                               math.sin(math.radians(self.__own_a)), -self.__own_a+90, 0)  # 此处与gui不同
        traci.simulationStep()
        print('\nrandom traffic initialized')

    def get_vehicles(self):  # 该部分可直接与gui相替换
        """Get other vehicles' information not including ego vehicle.

        Get other vehicles' information not including ego vehicle.

        Returns:
            A list containing other vehicle's current state except ego vehicle.
            For example:

            [{'type':0, 'x':0.0, 'y': 0.0, 'v':0.0, 'angle': 0.0,
            'rotation': 0, 'winker': 0, 'winker_time': 0, 'render': False},
            {'type':0, 'x':0.0, 'y': 0.0, 'v':0.0, 'angle': 0.0,
            'rotation': 0, 'winker': 0, 'winker_time': 0, 'render': False},...]

        Raises:
        """

        # 获取仿真中所有车辆的信息，包括自车
        veh_info_dict = traci.vehicle.getContextSubscriptionResults('ego')

        # 更新自车所在位置信息（当前车道的限速，距离当前车道停止线的距离）
        # self.__own_lane_speed_limit = [5.56,
        #                                5.56,
        #                                16.67,
        #                                16.67][
        #     veh_info_dict['ego'][traci.constants.VAR_LANE_INDEX]]
        # if veh_info_dict['ego'][traci.constants.VAR_LANE_INDEX] in [2, 3]:
        #     self.__own_lane_pos = 588.0 - veh_info_dict['ego'][
        #         traci.constants.VAR_LANEPOSITION]
        # else:
        #     self.__own_lane_pos = 9999.9

        # 将周车列表转换为平台使用的数据格式
        other_veh_info = _getothercarInfo(veh_info_dict, self.vehicleName)
        ego_x, ego_y = veh_info_dict['ego'][traci.constants.VAR_POSITION]
        for i in range(VEHICLE_COUNT):
            c_a = other_veh_info[i]['angle']
            c_x = other_veh_info[i]['x']
            c_y = other_veh_info[i]['y']
            if other_veh_info[i]['type'] == 'car_1':
                c_t = 0
            elif other_veh_info[i]['type'] == 'car_2':
                c_t = 1
            elif other_veh_info[i]['type'] == 'car_3':
                c_t = 2
            elif other_veh_info[i]['type'] == 'truck_1':
                c_t = 100
            else:
                c_t = 200
            length = other_veh_info[i]['length']
            width = other_veh_info[i]['width']
            c_r = other_veh_info[i]['signals']
            c_v = other_veh_info[i]['v']
            r = c_r
            if self.vehicles[i] is not None:
                wt = self.vehicles[i]['winker_time']
                w = self.vehicles[i]['winker']
                if self.vehicles[i]['rotation'] != r:
                    w = 1
                    wt = self.sim_time
                else:
                    if (self.sim_time-self.vehicles[i]['winker_time'] >=
                            WINKER_PERIOD):
                        w = 1-w
                        wt = self.sim_time
            else:
                w = 1
                wt = self.sim_time
            # 超出自车视野的车辆不进行渲染
            if math.fabs(ego_x - c_x) > 200 or math.fabs(ego_y - c_y) > 200:
                render_flag = False
            else:
                render_flag = True
                # if math.fabs(ego_x-c_x) < 10 and math.fabs(ego_y-c_y) < 10:
                #     if i != 0:
                #         print(c_x, c_y)
                #         x1 = c_x + math.cos((c_a + 90) / 180 * math.pi) * (
                #         length / 2 - width / 2)
                #         x2 = c_x - math.cos((c_a + 90) / 180 * math.pi) * (
                #         length / 2 - width / 2)
                #         x11 = self.own_x + math.cos(self.own_a / 180 * math.pi) * (
                #         OWN_CAR_START_LENGTH / 2 - OWN_CAR_START_WIDTH / 2)
                #         x22 = self.own_x - math.cos(self.own_a / 180 * math.pi) * (
                #         OWN_CAR_START_LENGTH / 2 - OWN_CAR_START_WIDTH / 2)
                #         y1 = c_y + math.sin((c_a + 90) / 180 * math.pi) * (
                #         length / 2 - width / 2)
                #         y2 = c_y - math.sin((c_a + 90) / 180 * math.pi) * (
                #         length / 2 - width / 2)
                #         y11 = self.own_y + math.sin(self.own_a / 180 * math.pi) * (
                #         OWN_CAR_START_LENGTH / 2 - OWN_CAR_START_WIDTH / 2)
                #         y22 = self.own_y - math.sin(self.own_a / 180 * math.pi) * (
                #         OWN_CAR_START_LENGTH / 2 - OWN_CAR_START_WIDTH / 2)
                #         if math.sqrt((x1 - x11) ** 2 + (
                #             y1 - y11) ** 2) < width / 2 + OWN_CAR_START_WIDTH / 2:
                #             print(c_x, c_y)
                #             print('----------------')
                #             #return [1]
                #         if math.sqrt((x2 - x11) ** 2 + (
                #             y2 - y11) ** 2) < width / 2 + OWN_CAR_START_WIDTH / 2:
                #             print(c_x, c_y)
                #             print('----------------')
                #             #return [1]
                #         if math.sqrt((x1 - x22) ** 2 + (
                #             y1 - y22) ** 2) < width / 2 + OWN_CAR_START_WIDTH / 2:
                #             print(c_x, c_y)
                #             print('----------------')
                #             #return [1]
                #         if math.sqrt((x2 - x22) ** 2 + (
                #             y2 - y22) ** 2) < width / 2 + OWN_CAR_START_WIDTH / 2:
                #             print(c_x, c_y)
                #             print('----------------')
                #             #return [1]
            self.vehicles[i] = dict(type=c_t, x=c_x, y=c_y, v=c_v, angle=c_a,
                                    rotation=c_r, winker=w, winker_time=wt,
                                    render=render_flag, length=length,
                                    width=width,
                                    lane_index=other_veh_info[i][
                                        'current_lane'],
                                    max_decel=other_veh_info[i]['max_decel'])
            if self.vehicles[i]['type'] in [4, 5]:
                self.vehicles[i]['type'] = 0
        return self.vehicles[VEHICLE_INDEX_START:]  # 返回的x,y是车辆形心，自车x，y传入也被sumo当做形心

    def sim_step(self):  # 该部分可直接与package相替换
        self.sim_time += SIM_PERIOD
        traci.simulationStep()

    def set_own_car(self, x, y, v, a):
        """Insert ego vehicle into sumo's traffic modal.

        Args:
            x: Ego vehicle's current x coordination of it's shape center, m.
            y: Ego vehicle's current y coordination of it's shape center, m.
            v: Ego vehicle's current velocity, m/s.
            a: Ego vehicle's current heading angle under base coordinate, deg.

        Raises:
        """
        self.__own_x, self.__own_y, self.__own_v, self.__own_a = x, y, v, a  # 此处与package不同
        traci.vehicle.moveToXY('ego', 'gneE25', 0, self.__own_x +
                               self.egocar_length / 2 *
                               math.cos(math.radians(self.__own_a)),
                               self.__own_y + self.egocar_length / 2 *
                               math.sin(math.radians(self.__own_a)),
                               -self.__own_a + 90, 0)

    # def get_current_lane_speed_limit(self):  # 该部分可直接与package相替换
    #     return self.__own_lane_speed_limit

    # def get_current_distance_to_stopline(self):  # 该部分可直接与package相替换
    #     return self.__own_lane_pos

    def get_dis2center_line(self):  # 此处与gui不同 左正右负
        return traci.vehicle.getLateralLanePosition('ego')

    def get_egolane_index(self):  # 此处与gui不同 左正右负
        return traci.vehicle.getLaneIndex('ego')

    def get_road_related_info_of_ego(self):
        dis2center_line = self.get_dis2center_line()  # 左正右负
        egolane_index = self.get_egolane_index()
        return dict(dist2current_lane_center=dis2center_line,
                    egolane_index=egolane_index)

    def __generate_random_traffic(self):  # 该部分可直接与package相替换
        """生成仿真初始时刻的随机交通流

        --"""
        #  调用sumo
        # SUMO_BINARY = checkBinary('sumo-gui')
        if self.seed is not None:
            traci.start([SUMO_BINARY, "-c",
                         self.__path + "traffic_generation_" + self.type + "_" +
                         self.density + ".sumocfg",
                         "--step-length", "1",
                         "--seed", self.seed])
        else:
            traci.start([SUMO_BINARY, "-c",
                         self.__path+"traffic_generation_"+self.type+"_"+
                         self.density+".sumocfg",
                         "--step-length", "1",
                         "--random"])
        self.__add_self_car()

        vehicles = []
        global VEHICLE_COUNT
        if self.__map_type == MAPS[0]:
            if self.density == 'Dense':
                VEHICLE_COUNT = 501
            elif self.density == 'Middle':
                VEHICLE_COUNT = 201
            else:
                VEHICLE_COUNT = 41
        elif self.__map_type == MAPS[1] or MAPS[5]:
            if self.density == 'Dense':
                VEHICLE_COUNT = 401
            elif self.density == 'Middle':
                VEHICLE_COUNT = 241
            else:
                VEHICLE_COUNT = 161

        #  等待所有车辆都进入路网
        departed_vehicle = 0
        while departed_vehicle < VEHICLE_COUNT:
            traci.simulationStep()
            departed_vehicle = departed_vehicle + (traci.simulation.
                                                   getDepartedNumber())

        # 等待一段时间让交通流尽可能分布在整个路网中。
        while True:
            if traci.simulation.getTime() > 1600:
                random_traffic = traci.vehicle.getContextSubscriptionResults(
                    'ego')
                for veh in random_traffic:
                    # 无法通过getContextSubscriptionResults获取route信息，但需要
                    # 每辆车的route信息来初始化交通流，因此加入getRoute来获取每
                    # 辆车的route。
                    random_traffic[veh][87] = traci.vehicle.getRoute(vehID=veh)
                break
            traci.simulationStep()
        traci.close()
        # getContextSubscriptionResults返回的车辆同时包括自车，需要删去。
        del random_traffic['ego']
        print('\nrandom traffic generated')
        return random_traffic

    def __initiate_traffic(self):  # 该部分可直接与package相替换
        for veh in self.random_traffic:
            # Skip traffic vehicle which overlap with ego vehicle.
            if (math.fabs((self.random_traffic[veh]
                           [traci.constants.VAR_POSITION][0])
                          - self.__own_x) < 20
                and (math.fabs((self.random_traffic[veh]
                                [traci.constants.VAR_POSITION][1])
                               - self.__own_y) < 20)):
                continue
            traci.vehicle.addLegacy(vehID=veh,
                                    routeID='self_route',
                                    depart=2,
                                    pos=0,
                                    lane=-6,
                                    speed=self.random_traffic[veh]
                                            [traci.constants.VAR_SPEED],
                                    typeID=(self.random_traffic[veh]
                                            [traci.constants.VAR_TYPE]))
            traci.vehicle.setRoute(vehID=veh,
                                   edgeList=self.random_traffic[veh][87])
            traci.vehicle.moveToXY(vehID=veh,
                                   edgeID='gneE25',
                                   lane=0,
                                   x=(self.random_traffic[veh]
                                      [traci.constants.VAR_POSITION][0]),
                                   y=(self.random_traffic[veh]
                                      [traci.constants.VAR_POSITION][1]),
                                   angle=(self.random_traffic[veh]
                                          [traci.constants.VAR_ANGLE]),
                                   keepRoute=2)  # TODO

    def __add_self_car(self):  # 该部分可直接与package相替换
        traci.vehicle.addLegacy(vehID='ego', routeID='self_route',
                                depart=0, pos=0, lane=-6, speed=0,
                                typeID='self_car')
        traci.vehicle.subscribeContext('ego',
                                       traci.constants.CMD_GET_VEHICLE_VARIABLE,
                                       300000, [traci.constants.VAR_POSITION,
                                                traci.constants.VAR_ANGLE,
                                                traci.constants.VAR_TYPE,
                                                traci.constants.VAR_SPEED,
                                                traci.constants.VAR_LENGTH,
                                                traci.constants.VAR_WIDTH],
                                       0, 2147483647)


if __name__ == "__main__":
    sumoBinary = checkBinary('sumo-gui')
    print(__file__)
    traci.start([sumoBinary, "-c", "Map/Map3_Highway_v2/traffic_generation_Vehicle Only Traffic_Dense.sumocfg",
                 "--step-length", "1"])

    """add self car"""
    traci.vehicle.addLegacy(vehID='A', routeID='self_route',
                            depart=0, pos=0, lane=-6, speed=3,
                            typeID='self_car')
    traci.vehicle.setLength('A', 4.8)
    traci.vehicle.setWidth('A', 2.2)
    traci.vehicle.subscribeContext('A',
                                   traci.constants.CMD_GET_VEHICLE_VARIABLE,
                                   300000, [traci.constants.VAR_POSITION,
                                            traci.constants.VAR_ANGLE,
                                            traci.constants.VAR_TYPE,
                                            traci.constants.VAR_SPEED,
                                            traci.constants.VAR_LENGTH,
                                            traci.constants.VAR_WIDTH],
                                   0, 2147483647)
    while len(traci.vehicle.getRoadID('A')) == 0:
        traci.simulationStep()
    traci.vehicle.moveToXY('A', 'gneE25', 3, 300, 0, -90, 0)
    vehicleName = []
    while len(vehicleName) < 201:
        traci.vehicle.setSpeed('A', 3)
        traci.simulationStep()
        laneposition = traci.vehicle.getLateralLanePosition('A')
        laneindex = traci.vehicle.getLaneIndex('A')
        x, y = traci.vehicle.getPosition('A')
        print(laneposition, laneindex, y)
        Veh_dict = traci.vehicle.getContextSubscriptionResults('A')
        vehicleName = list(Veh_dict.keys())

    while True:
        """get random traffic"""
        if traci.simulation.getTime() > 500:
            random_traffic=traci.vehicle.getContextSubscriptionResults('A')
            for veh in random_traffic:
                random_traffic[veh][87]=traci.vehicle.getRoute(vehID=veh)
            print(random_traffic)
            break

        traci.simulationStep()
    input()