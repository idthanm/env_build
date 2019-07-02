# coding=utf-8

"""
Simulation Base Modal of Autonomous Car Simulation System
Author: Li Bing
Date: 2017-8-23
"""

import _struct as struct
import untangle
import os
from .traffic_module import *
from .agent_module import *
from xml.dom.minidom import Document
#import kdtree
#import StringIO
import time
from . import data_structures
from .default_value import *

class Data:
    """
    Simulation Data Manager
    """
    def __init__(self):
        self.data=[]
        self.file=open('tmp.bin','wb')

        self.vehicle_count = 0
        self.trajectory_count = 0
        self.max_path_points = 100
        

    def __del__(self):
        self.file.close()

    def append(self, self_status, self_info, vehicles, light_values,
               trajectory, dis=None, speed_limit=None):
        if self.file.closed:
            return
        t, x, y, v, a = self_status
        data_line = [t, x, y, v, a]
        data_line.extend(list(self_info))
        self.data.append(data_line)

        self.file.write(struct.pack('5f', *self_status))
        self.file.write(struct.pack('17f', *self_info))
        self.file.write(struct.pack('2f', *[dis, speed_limit]))
        self.vehicle_count = len(vehicles)
        for v in vehicles:
            self.file.write(struct.pack('4f3if', *[v['x'],
                                                   v['y'],
                                                   v['v'],
                                                   v['angle'],
                                                   v['type'],
                                                   v['rotation'],
                                                   v['lane_index'],
                                                   v['max_decel']]))
        self.trajectory_count = len(trajectory)
        self.file.write(struct.pack('2i', *light_values))
        for i in range(100):
            if i < self.trajectory_count:
                self.file.write(
                    struct.pack('5f', *trajectory[i]))
            else:
                self.file.write(
                    struct.pack('5f', *[0, 0, 0, 0, 0]))
        self.file.write(struct.pack('i', *[self.trajectory_count]))

    def __get_time(self):
        return [d[0] for d in self.data]

    def __get_speed(self):
        return [d[3]*3.6 for d in self.data]

    def __get_x(self):
        return [d[1] for d in self.data]

    def __get_y(self):
        return [d[2] for d in self.data]

    def __get_yaw(self):
        return [d[4] for d in self.data]

    def __get_accel(self):
        # if len(self.data)==0:
        #     return []
        # elif len(self.data)==1:
        #     return [0]
        # else:
        #     dt=self.data[1][0]-self.data[0][0]
        #     accel=[(self.data[i+1][3]-self.data[i][3])/dt for i in range(len(self.data)-1)]
        #     accel.insert(0,0)
        #     return accel
        return [d[11] for d in self.data]

    def get_data(self, type):
        if type =='Time':
            return self.__get_time()
        elif type =='Vehicle Speed':
            return self.__get_speed()
        elif type =='Position X':
            return self.__get_x()
        elif type =='Position Y':
            return self.__get_y()
        elif type =='Heading Angle':
            return self.__get_yaw()
        elif type =='Acceleration':
            return self.__get_accel()
        elif type =='Steering Wheel':
            return [d[5] for d in self.data]
        elif type =='Throttle':
            return [d[6] for d in self.data]
        elif type =='Brake Pressure':
            return [d[7] for d in self.data]
        elif type =='Gear':
            return [d[8] for d in self.data]
        elif type =='Engine Speed':
            return [d[9] for d in self.data]
        elif type =='Engine Torque':
            return [d[10] for d in self.data]
        elif type =='Side Slip':
            return [d[12] for d in self.data]
        elif type =='Yaw Rate':
            return [d[13] for d in self.data]
        elif type =='Lateral Velocity':
            return [d[14] for d in self.data]
        elif type =='Longitudinal Velocity':
            return [d[15] for d in self.data]
        elif type =='Front Wheel Angle':
            return [d[16] for d in self.data]
        elif type =='Steering Rate':
            return [d[17] for d in self.data]
        elif type =='Fuel Consumption':
            return [d[18] for d in self.data]
        elif type =='Longitudinal Acceleration':
            return [d[19] for d in self.data]
        elif type =='Lateral Acceleration':
            return [d[20] for d in self.data]
        elif type =='Fuel Rate':
            return [d[21] for d in self.data]
        else:
            return None

    def close_file(self):
        self.file.close()

    def export_csv(self, path):
        self.close_file()
        with open(path, 'w') as f:
            f.write('t(s),self_x(m),self_y(m),self_speed(m/s),self_yaw(degree)')
            f.write(',Steering Wheel(degree),Throttle(%),Brake Pressure(MPa),'
                    'Gear,Engine Speed(rpm)')
            f.write(',Engine Torque(N*m),Accelerate(m/s2),Side Slip(degree), '
                    'Yaw Rate(degree/s)')
            f.write(',Lateral Velocity(m/s),Longitudinal Velocity(m/s)')
            f.write(',Front Wheel Angle(deg),Steering Rate(deg/s)')
            f.write(',Fuel Consumption(L),Longitudinal Acceleration(m/s^2)')
            f.write(',Lateral Acceleration(m/s^2),Fuel Rate(L/s)')
            f.write(',Distance To Stop Line(m),Speed Limit(m/s)')
            for i in range(self.vehicle_count):
                f.write(',vehicle%d_x(m),vehicle%d_y(m),vehicle%d_speed(m/s),'
                        'vehicle%d_yaw(degree),vehicle%d_type,'
                        'vehicle%d_signals,vehicle%d_lane_index,'
                        'vehicle%d_max_decel(m/s^2)'
                        % (i+1, i+1, i+1, i+1, i+1, i+1, i+1, i+1))
            f.write(',light_horizontal,light_vertical')
            for i in range(100):
                f.write(',path point%d_t(m),path point%d_x(m),path point%d_y, '
                        'path point%d_v, path_point%d_heading'
                        % (i+1, i+1, i+1, i+1, i+1))
            f.write(',valid trajectory point number\n')

            with open('tmp.bin','rb') as fbin:
                fmt='5f'
                buffer=fbin.read(struct.calcsize(fmt))
                while len(buffer) > 0:
                    f.write('%.6f,%.2f,%.2f,%.2f,%.1f'
                            % struct.unpack(fmt, buffer))
                    fmt = '17f'
                    f.write(',%.0f,%.0f,%.1f,%.1f,%.0f,%.1f,%.2f,%.1f,%.2f,%.2f,'
                            '%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f'
                            % struct.unpack(fmt, fbin.read(struct.calcsize(fmt))))
                    fmt = '2f'
                    f.write(',%.2f,%.2f'
                            % struct.unpack(fmt, fbin.read(struct.calcsize(fmt))))
                    for i in range(self.vehicle_count):
                        fmt = '4f3if'
                        f.write(',%.2f,%.2f,%.2f,%.1f,%d,%d,%d,%.2f'
                                % struct.unpack(fmt, fbin.read(struct.calcsize(fmt))))
                    fmt = '2i'
                    f.write(',%d,%d'%struct.unpack(fmt, fbin.read(struct.calcsize(fmt))))
                    for i in range(100):
                        fmt='5f'
                        f.write(',%.2f,%.2f,%.2f,%.2f,%.2f'%struct.unpack(fmt, fbin.read(struct.calcsize(fmt))))
                    fmt = 'i'
                    f.write(',%d\n'%struct.unpack(fmt, fbin.read(struct.calcsize(fmt))))
                    fmt = '5f'
                    buffer=fbin.read(struct.calcsize(fmt))


class TrafficData:
    """保存随机交通流初始状态的数据类"""

    def __init__(self):
        self.file = None  # 保存数据的二进制文件
        pass

    def __del__(self):
        self.file.close()

    def save_traffic(self, traffic, path):
        self.file = open(path+'/traffic.bin', 'wb')
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
                                        *[traffic[veh][79]]))
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
                self.file.write(struct.pack(fmt, *[route]))
                # print(fmt),
        self.file.close()

    def load_traffic(self, path):
        traffic = {}
        with open(path+'/simulation traffic data.bin', 'rb') as traffic_data:
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
                        struct.calcsize(fmt)))[0])
                    # print(fmt),
                traffic[str(id)] = {64: v, 66: (x, y), 67: heading, 68: length,
                                    77: width, 79: type, 87: route}
                id += 1
                fmt = '6f'
                buffer = traffic_data.read(struct.calcsize(fmt))
                # print(fmt),
        return traffic


PLANNER_FREQUENCY = 2
CONTROLLER_FREQUENCY = 1
SENSOR_FREQUENCY = 1


class Simulation(object):
    """Simulation Class.

    Simulation class for one simulation.

    Attributes:
        tick_count: Simulation run time. Counted by simulation steps.
        sim_time: Simulation run time. Counted by simulation steps multiply step
            length.
        stopped: A bool variable as a flag indicating whether simulation is
            ended.
        traffic: A traffic module instance.
        agent: A Agent module instance.
        data: A data module instance.
        other_vehicles: A list containing all other vehicle's info at current
            simulation step from traffic module.
        light_status: A dic variable containing current intersection's traffic
            light state for each direction.



    """

    def __init__(self, setting_path=None):

        self.tick_count = 0  # Simulation run time. Counted by simulation steps.
        self.sim_time = 0.0  # Simulation run time. Counted by steps multiply stpe length.
        self.other_vehicles = None  # 仿真中所有他车状态信息
        self.light_status = None  # 当前十字路口信号灯状态
        self.stopped = False  # 仿真结束标志位
        self.simulation_loaded = False  # 仿真载入标志位
        self.traffic_data = TrafficData()  # 初始交通流数据对象
        self.settings = None  # 仿真设置对象

        self.external_control_flag = False  # 外部控制输入标识，若外部输入会覆盖内部控制器

        self.reset(settings=Settings(file_path=setting_path))
        # self.sim_step()

    def reset(self, settings=None, init_traffic=None):
        """Clear previous loaded module.

        Args:
            settings: LasVSim's setting class instance. Containing current
                simulation's configuring information.
        """
        if hasattr(self,'traffic'):
            del self.traffic
        if hasattr(self,'agent'):
            del self.agent
        if hasattr(self, 'data'):
            del self.data

        self.tick_count = 0
        self.settings = settings
        self.stopped = False
        self.data = Data()
        self.ego_history = {}

        """Load vehicle module library"""
        vehicle_models = VehicleModels('Library/vehicle_model_library.csv')

        """Load traffic module."""
        step_length = self.settings.step_length * self.settings.traffic_frequency
        self.traffic=Traffic(path=settings.map,
                             traffic_type=settings.traffic_type,
                             traffic_density=settings.traffic_lib,
                             step_length=step_length,
                             init_traffic=init_traffic)
        self.traffic.init(settings.points[0])
        self.other_vehicles = self.traffic.get_vehicles()
        self.light_status = self.traffic.get_light_status()

        """Load agent module."""
        self.agent=Agent(settings)
        self.agent.sensors.setVehicleModel(vehicle_models)

    def load_scenario(self, path):
        """Load an existing LasVSim simulation configuration file.

        Args:
            path:
        """
        if os.path.exists(path):
            settings = Settings()
            settings.load(path+'/simulation setting file.xml')
            #self.reset(settings)
            self.reset(settings, self.traffic_data.load_traffic(path))
            self.simulation_loaded = True
            return
        print('\033[31;0mSimulation loading failed: 找不到对应的根目录\033[0m')
        self.simulation_loaded = False

    def save_scenario(self, path):
        """Save current simulation configuration

        Args:
            path: Absolute path.
        """
        self.settings.save(path)

    def export_data(self, path):
        self.data.export_csv(path)

    def sim_step_internal(self, steps=None):
        if steps is None:
            steps = 1
        for step in range(steps):
            if self.stopped:
                print("Simulation Finished")
                return False

            # 传感器线程
            if self.tick_count % self.settings.sensor_frequency == 0:
                self.agent.update_info_from_sensor(self.other_vehicles)

            # 决策线程
            if self.tick_count % self.settings.router_frequency == 0:
                self.agent.update_plan_output(self.light_status)

            # 控制器线程
            if self.tick_count % self.settings.controller_frequency == 0:
                self.agent.update_control_input()

            # 动力学线程
            if self.tick_count % self.settings.dynamic_frequency == 0:
                self.agent.update_dynamic_state()

            # 交通流线程
            if self.tick_count % self.settings.traffic_frequency == 0:
                self.traffic.set_own_car(self.agent.dynamic.x,
                                         self.agent.dynamic.y,
                                         self.agent.dynamic.v,
                                         self.agent.dynamic.heading)
                self.traffic.sim_step()
                self.other_vehicles = self.traffic.get_vehicles()
                self.light_status = self.traffic.get_light_status()

            # 如果自车达到目的地则退出仿真，loop路径下仿真会一直循环不会结束
            if self.agent.mission.get_status() != MISSION_RUNNING:
                self.stop()

            # 保存当前步仿真数据
            self.data.append(
                self_status=[self.tick_count * float(self.settings.step_length),
                             self.agent.dynamic.x,
                             self.agent.dynamic.y,
                             self.agent.dynamic.v,
                             self.agent.dynamic.heading],
                self_info=self.agent.dynamic.get_info(),
                vehicles=self.other_vehicles,
                light_values=self.traffic.get_light_values(),
                trajectory=self.agent.route,
                dis=self.traffic.get_current_distance_to_stopline(),
                speed_limit=self.traffic.get_current_lane_speed_limit())
            self.tick_count += 1
        return True

    def sim_step(self, steps=None):
        if steps is None:
            steps = 1
        for step in range(steps):
            if self.stopped:
                print("Simulation Finished")
                return False

            # 传感器线程
            if self.tick_count % self.settings.sensor_frequency == 0:
                self.agent.update_info_from_sensor(self.other_vehicles)

            # 控制器线程
            if self.tick_count % self.settings.controller_frequency == 0:
                self.agent.update_control_input()

            # 动力学线程
            if self.tick_count % self.settings.dynamic_frequency == 0:
                self.agent.update_dynamic_state()

            # 交通流线程
            if self.tick_count % self.settings.traffic_frequency == 0:
                self.traffic.set_own_car(self.agent.dynamic.x,
                                         self.agent.dynamic.y,
                                         self.agent.dynamic.v,
                                         self.agent.dynamic.heading)
                self.traffic.sim_step()
                self.other_vehicles = self.traffic.get_vehicles()
                self.light_status = self.traffic.get_light_status()

            # 如果自车达到目的地则退出仿真，loop路径下仿真会一直循环不会结束
            if self.agent.mission.get_status() != MISSION_RUNNING:
                self.stop()

            # 保存当前步仿真数据
            self.data.append(
                self_status=[self.tick_count * float(self.settings.step_length),
                             self.agent.dynamic.x,
                             self.agent.dynamic.y,
                             self.agent.dynamic.v,
                             self.agent.dynamic.heading],
                self_info=self.agent.dynamic.get_info(),
                vehicles=self.other_vehicles,
                light_values=self.traffic.get_light_values(),
                trajectory=self.agent.route,
                dis=self.traffic.get_current_distance_to_stopline(),
                speed_limit=self.traffic.get_current_lane_speed_limit())
            self.tick_count += 1
        return True

    def get_all_objects(self):
        return self.other_vehicles

    def get_detected_objects(self):
       return self.agent.detected_objects

    def get_ego_position(self):
        return self.agent.dynamic.x, self.agent.dynamic.y

    def get_self_car_info(self):
        return self.agent.get_control_info()

    def get_time(self):
        return self.traffic.sim_time

    def get_pos(self):
        return self.mission.pos

    def get_controller_type(self):
        return self.agent.controller.model_type

    # def mission_update(self,pos):
    #     self.agent.mission.update(pos)

    def stop(self):
        self.stopped = True
        self.data.close_file()

    def update_evaluation_data(self):
        x, y, v, a = self.get_pos()
        speed = v
        plan_pos = self.agent.controller.get_plan_pos()
        if plan_pos is not None:
            plan_x, plan_y = plan_pos[1:3]
        else:
            plan_x, plan_y = x, y

        is_in_cross, is_change_lane, lane_x, lane_y, car2border = self.agent.get_drive_status()

        front_d=1000
        front_speed=-1
        front_x, front_y = plan_x, plan_y+1000
        if (not is_change_lane) and not (is_in_cross):
            status,lane_info=self.map.map_position(x,y)
            for vehicle in self.other_vehicles:
                vx,vy=vehicle['x'],vehicle['y']
                if get_distance((x,y),(vx,vy))>100:
                    continue
                vehicle_status,vehicle_lane_info=self.map.map_position(vx,vy)
                if vehicle_status is not MAP_IN_ROAD:
                    continue
                if lane_info!=vehicle_lane_info:
                    continue
                if lane_info['direction'] in 'NS':
                    ds=vy-y
                else:
                    ds=vx-x
                if lane_info['direction'] in 'SW':
                    ds=-ds
                if ds<0 or ds>front_d:
                    continue
                front_x,front_y=vx,vy
                front_d=ds
                front_speed=vehicle['v']

        steering_wheel,throttle, brake, gear, engine_speed, engine_torque, accl, \
        sideslip, yaw_rate, lateralvelocity, longitudinalvelocity, \
        frontwheel, steerrate, fuel, acc_lon, acc_lat, fuel_rate = self.agent.dynamic.get_info()

        evaluation_data=(x,y),(plan_x,plan_y),(lane_x, lane_y),(front_x,front_y),is_in_cross,is_change_lane, \
                        frontwheel, throttle/100, brake, longitudinalvelocity, lateralvelocity, accl, \
                        car2border, steerrate, fuel ,acc_lon, acc_lat, front_speed, \
                        speed, yaw_rate, steering_wheel, engine_speed, engine_torque, gear
        self.evaluator.update(evaluation_data)

    def get_current_task(self):
        return self.agent.mission.current_task

    def __collision_check(self):
        ego_point1 = [self.agent.dynamic.x + cos(self.agent.dynamic.heading / 180 * pi) * (self.agent.length / 2 - self.agent.width / 2),
                      self.agent.dynamic.y + sin(self.agent.dynamic.heading / 180 * pi) * (self.agent.length / 2 - self.agent.width / 2)]
        ego_point2 = [self.agent.dynamic.x - cos(self.agent.dynamic.heading / 180 * pi) * (self.agent.length / 2 - self.agent.width / 2),
                      self.agent.dynamic.y - sin(self.agent.dynamic.heading / 180 * pi) * (self.agent.length / 2 - self.agent.width / 2)]
        for veh in self.other_vehicles:
            if self.agent.dynamic.x - 10 < veh['x'] < self.agent.dynamic.x + 10:
                if self.agent.dynamic.y - 10 < veh['y'] < self.agent.dynamic.y + 10:
                    other_point1 = [veh['x'] + cos(veh['heading'] / 180 * pi) * (veh['length'] / 2 - veh['width'] / 2),
                                    veh['y'] + sin(veh['heading'] / 180 * pi) * (veh['length'] / 2 - veh['width'] / 2)]
                    other_point2 = [veh['x'] - cos(veh['heading'] / 180 * pi) * (veh['length'] / 2 - veh['width'] / 2),
                                    veh['y'] - sin(veh['heading'] / 180 * pi) * (veh['length'] / 2 - veh['width'] / 2)]
                    if (other_point1[0] - ego_point1[0]) ** 2 + (other_point1[1] - ego_point1[1]) ** 2 < (self.agent.width / 2 + veh['width'] / 2) ** 2:
                        return True
                    if (other_point2[0] - ego_point1[0]) ** 2 + (other_point2[1] - ego_point1[1]) ** 2 < (self.agent.width / 2 + veh['width'] / 2) ** 2:
                        return True
                    if (other_point2[0] - ego_point2[0]) ** 2 + (other_point2[1] - ego_point2[1]) ** 2 < (self.agent.width / 2 + veh['width'] / 2) ** 2:
                        return True
                    if (other_point1[0] - ego_point2[0]) ** 2 + (other_point1[1] - ego_point2[1]) ** 2 < (self.agent.width / 2 + veh['width'] / 2) ** 2:
                        return True
        return False


class Settings:  # 可以直接和package版本的Settings类替换,需要转换路径点的yaw坐标
    """
    Simulation Settings Class
    """

    def __init__(self, file_path=None):
        self.car_para = data_structures.CarParameter()  # 自车动力学模型参数
        self.load(file_path)

    def __del__(self):
        pass

    def load(self, filePath=None):
        if filePath is None:
            filePath = DEFAULT_SETTING_FILE
        self.__parse_xml(filePath)
        self.__load_step_length()
        self.__load_map()
        self.__load_self_car()
        self.__load_mission()
        self.__load_controller()
        self.__load_traffic()
        self.__load_sensors()
        self.__load_router()
        self.__load_dynamic()

    def __parse_xml(self, path):
        f = open(path)
        self.root = untangle.parse(f.read()).Simulation

    def __load_step_length(self):
        self.step_length = int(self.root.StepLength.cdata)

    def __load_map(self):
        self.map = str(self.root.Map.Type.cdata)

    def __load_mission(self):
        self.mission_type = str(self.root.Mission.Type.cdata)
        self.points = []
        for i in range(len(self.root.Mission.Point)):
            self.points.append([float(self.root.Mission.Point[i].X.cdata),
                                float(self.root.Mission.Point[i].Y.cdata),
                                float(self.root.Mission.Point[i].Speed.cdata),
                                float(self.root.Mission.Point[i].Yaw.cdata)])

    def __load_controller(self):
        self.controller_type = str(self.root.Controller.Type.cdata)
        self.controller_lib = str(self.root.Controller.Lib.cdata)
        self.controller_frequency = int(self.root.Controller.Frequency.cdata)
        self.controller_file_type = str(self.root.Controller.FileType.cdata)

    def __load_dynamic(self):
        self.dynamic_type=str(self.root.Dynamic.Type.cdata)
        self.dynamic_lib=str(self.root.Dynamic.Lib.cdata)
        self.dynamic_frequency = int(self.root.Dynamic.Frequency.cdata)

    def __load_traffic(self):
        self.traffic_type = str(self.root.Traffic.Type.cdata)
        self.traffic_lib = str(self.root.Traffic.Lib.cdata)
        self.traffic_frequency = int(self.root.Traffic.Frequency.cdata)

    def __load_self_car(self):
        self.car_length = float(self.root.SelfCar.Length.cdata)
        self.car_width = float(self.root.SelfCar.Width.cdata)
        self.car_weight = float(self.root.SelfCar.Weight.cdata)
        self.car_center2head = float(self.root.SelfCar.CenterToHead.cdata)
        self.car_faxle2center = float(self.root.SelfCar.FAxleToCenter.cdata)
        self.car_raxle2center = float(self.root.SelfCar.RAxleToCenter.cdata)
        self.car_para.LX_AXLE = self.car_faxle2center + self.car_raxle2center
        self.car_para.LX_CG_SU = self.car_faxle2center
        self.car_para.M_SU = float(self.root.SelfCar.M_SU.cdata)
        self.car_para.IZZ_SU = float(self.root.SelfCar.IZZ_SU.cdata)
        self.car_para.A = float(self.root.SelfCar.A.cdata)
        self.car_para.CFx = float(self.root.SelfCar.CFx.cdata)
        self.car_para.AV_ENGINE_IDLE = float(self.root.SelfCar.AV_ENGINE_IDLE.cdata)
        self.car_para.IENG = float(self.root.SelfCar.IENG.cdata)
        self.car_para.TAU = float(self.root.SelfCar.TAU.cdata)
        self.car_para.R_GEAR_TR1 = float(self.root.SelfCar.R_GEAR_TR1.cdata)
        self.car_para.R_GEAR_FD = float(self.root.SelfCar.R_GEAR_FD.cdata)
        self.car_para.BRAK_COEF = float(self.root.SelfCar.BRAK_COEF.cdata)
        self.car_para.Steer_FACTOR = float(self.root.SelfCar.Steer_FACTOR.cdata)
        self.car_para.M_US = float(self.root.SelfCar.M_US.cdata)
        self.car_para.RRE = float(self.root.SelfCar.RRE.cdata)
        self.car_para.CF = float(self.root.SelfCar.CF.cdata)
        self.car_para.CR = float(self.root.SelfCar.CR.cdata)
        self.car_para.ROLL_RESISTANCE = float(self.root.SelfCar.ROLL_RESISTANCE.cdata)

    def __load_sensors(self):
        self.sensor_model_type = str(self.root.Sensors.Type.cdata)
        self.sensor_model_lib = str(self.root.Sensors.Lib.cdata)
        self.sensor_frequency = int(self.root.Sensors.Frequency.cdata)
        sensor_array = SensorInfo * len(self.root.Sensors.Sensor)
        self.sensors = sensor_array()
        types = ['int', 'int', 'float', 'float', 'float', 'float', 'float',
                 'float', 'float', 'float', 'float', 'float', 'float', 'float']
        attrs = ['id',
                 'type',
                 'detection_angle',
                 'detection_range',
                 'installation_lateral_bias',
                 'installation_longitudinal_bias',
                 'installation_orientation_angle',
                 'accuracy_velocity',
                 'accuracy_location',
                 'accuracy_heading',
                 'accuracy_width',
                 'accuracy_length',
                 'accuracy_height',
                 'accuracy_radius']
        for i in range(len(self.sensors)):
            for j in range(len(attrs)):
                attr = attrs[j]
                dtype = types[j]
                exec(('self.sensors[i].%s=%s(self.root.Sensors.Sensor[i].%'
                      's.cdata)') % (attr, dtype, attr))

    def __load_router(self):
        self.router_output_type = str(self.root.Router.OutputType.cdata)
        self.router_type = str(self.root.Router.Type.cdata)
        self.router_lib = str(self.root.Router.Lib.cdata)
        self.router_frequency = int(self.root.Router.Frequency.cdata)
        self.router_file_type = str(self.root.Router.FileType.cdata)

    def save(self,path):
        file_name=re.split(r'[\\/]',str(path))[-1].split('.')[0]
        doc = Document();
        root = doc.createElement('Simulation')
        doc.appendChild(root)
        info_node=doc.createElement('Info')
        title_node=doc.createElement('Title')
        date_node=doc.createElement('Date')
        author_node=doc.createElement('Author')
        version_node=doc.createElement('Version')
        title_node.appendChild(doc.createTextNode(file_name))
        date_node.appendChild(doc.createTextNode(time.strftime('%Y-%m-%d')))
        author_node.appendChild(doc.createTextNode('Author Name'))
        version_node.appendChild(doc.createTextNode('1.0'))
        info_node.appendChild(title_node)
        info_node.appendChild(date_node)
        info_node.appendChild(author_node)
        info_node.appendChild(version_node)

        step_node = doc.createElement('StepLength')
        step_node.appendChild(doc.createTextNode(str(self.step_length)))

        """Save map info."""
        map_node = doc.createElement('Map')
        type_node = doc.createElement('Type')
        type_node.appendChild(doc.createTextNode(self.map))

        map_node.appendChild(type_node)

        """Save mission info."""
        mission_node = doc.createElement('Mission')
        type_node = doc.createElement('Type')
        type_node.appendChild(doc.createTextNode(self.mission_type))
        mission_node.appendChild(type_node)
        for i in range(len(self.points)):
            point_node = doc.createElement('Point')
            mission_node.appendChild(point_node)

            x_node = doc.createElement('X')
            x_node.appendChild(doc.createTextNode('%.3f' % self.points[i][0]))
            y_node = doc.createElement('Y')
            y_node.appendChild(doc.createTextNode('%.3f' % self.points[i][1]))
            speed_node = doc.createElement('Speed')
            speed_node.appendChild(doc.createTextNode('%.0f' % self.points[i][2]))
            yaw_node = doc.createElement('Yaw')
            yaw_node.appendChild(doc.createTextNode('%.0f' % (-self.points[i][3]+90)))
            point_node.appendChild(x_node)
            point_node.appendChild(y_node)
            point_node.appendChild(speed_node)
            point_node.appendChild(yaw_node)

        # 保存自车动力学模型参数
        selfcar_node = doc.createElement('SelfCar')

        length_node=doc.createElement('Length')  # 车长
        length_node.appendChild(doc.createTextNode('4.8'))
        selfcar_node.appendChild(length_node)

        width_node=doc.createElement('Width')  # 车宽
        width_node.appendChild(doc.createTextNode('1.8'))
        selfcar_node.appendChild(width_node)

        CenterToHead_node=doc.createElement('CenterToHead')  # 质心距车头距离
        CenterToHead_node.appendChild(doc.createTextNode('2.7'))
        selfcar_node.appendChild(CenterToHead_node)

        FAxleToCenter_node=doc.createElement('FAxleToCenter')  # 悬上质量质心至前轴距离，m
        FAxleToCenter_node.appendChild(doc.createTextNode(str(
            self.car_para.LX_CG_SU)))
        selfcar_node.appendChild(FAxleToCenter_node)

        RAxleToCenter_node=doc.createElement('RAxleToCenter')  # 悬上质量质心至后轴距离，m
        RAxleToCenter_node.appendChild(doc.createTextNode(str(
            self.car_para.LX_AXLE-self.car_para.LX_CG_SU)))
        selfcar_node.appendChild(RAxleToCenter_node)

        Weight_node = doc.createElement('Weight')  # 质量
        Weight_node.appendChild(doc.createTextNode(str(
            self.car_para.M_SU+self.car_para.M_US)))
        selfcar_node.appendChild(Weight_node)

        LX_AXLE = doc.createElement('LX_AXLE')  # 悬上质量，kg
        LX_AXLE.appendChild(doc.createTextNode(str(self.car_para.LX_AXLE)))
        selfcar_node.appendChild(LX_AXLE)

        LX_CG_SU = doc.createElement('LX_CG_SU')  # 悬上质量，kg
        LX_CG_SU.appendChild(doc.createTextNode(str(self.car_para.LX_CG_SU)))
        selfcar_node.appendChild(LX_CG_SU)

        M_SU = doc.createElement('M_SU')  # 悬上质量，kg
        M_SU.appendChild(doc.createTextNode(str(self.car_para.M_SU)))
        selfcar_node.appendChild(M_SU)

        IZZ_SU = doc.createElement('IZZ_SU')  # 转动惯量，kg*m^2
        IZZ_SU.appendChild(doc.createTextNode(str(self.car_para.IZZ_SU)))
        selfcar_node.appendChild(IZZ_SU)

        A_Wind = doc.createElement('A')  # 迎风面积，m^2
        A_Wind.appendChild(doc.createTextNode(str(self.car_para.A)))
        selfcar_node.appendChild(A_Wind)

        CFx = doc.createElement('CFx')  # 空气动力学侧偏角为零度时的纵向空气阻力系数
        CFx.appendChild(doc.createTextNode(str(self.car_para.CFx)))
        selfcar_node.appendChild(CFx)

        AV_ENGINE_IDLE = doc.createElement('AV_ENGINE_IDLE')  # 怠速转速，rpm
        AV_ENGINE_IDLE.appendChild(doc.createTextNode(str(
            self.car_para.AV_ENGINE_IDLE)))
        selfcar_node.appendChild(AV_ENGINE_IDLE)

        IENG = doc.createElement('IENG')  # 曲轴转动惯量，kg*m^2
        IENG.appendChild(doc.createTextNode(str(self.car_para.IENG)))
        selfcar_node.appendChild(IENG)

        TAU = doc.createElement('TAU')  # 发动机-变速箱输入轴 时间常数，s
        TAU.appendChild(doc.createTextNode(str(self.car_para.TAU)))
        selfcar_node.appendChild(TAU)

        R_GEAR_TR1 = doc.createElement('R_GEAR_TR1')  # 最低档变速箱传动比
        R_GEAR_TR1.appendChild(doc.createTextNode(str(self.car_para.R_GEAR_TR1)))
        selfcar_node.appendChild(R_GEAR_TR1)

        R_GEAR_FD = doc.createElement('R_GEAR_FD')  # 主减速器传动比
        R_GEAR_FD.appendChild(doc.createTextNode(str(self.car_para.R_GEAR_FD)))
        selfcar_node.appendChild(R_GEAR_FD)

        BRAK_COEF = doc.createElement('BRAK_COEF')  # 液压缸变矩系数,Nm/(MPa)
        BRAK_COEF.appendChild(doc.createTextNode(str(self.car_para.BRAK_COEF)))
        selfcar_node.appendChild(BRAK_COEF)

        Steer_FACTOR = doc.createElement('Steer_FACTOR')  # 转向传动比
        Steer_FACTOR.appendChild(doc.createTextNode(str(
            self.car_para.Steer_FACTOR)))
        selfcar_node.appendChild(Steer_FACTOR)

        M_US = doc.createElement('M_US')  # 簧下质量，kg
        M_US.appendChild(doc.createTextNode(str(self.car_para.M_US)))
        selfcar_node.appendChild(M_US)

        RRE = doc.createElement('RRE')  # 车轮有效滚动半径，m
        RRE.appendChild(doc.createTextNode(str(self.car_para.RRE)))
        selfcar_node.appendChild(RRE)

        CF = doc.createElement('CF')  # 前轮侧偏刚度，N/rad
        CF.appendChild(doc.createTextNode(str(self.car_para.CF)))
        selfcar_node.appendChild(CF)

        CR = doc.createElement('CR')  # 后轮侧偏刚度，N/rad
        CR.appendChild(doc.createTextNode(str(self.car_para.CR)))
        selfcar_node.appendChild(CR)

        ROLL_RESISTANCE = doc.createElement('ROLL_RESISTANCE')  # 滚动阻力系数
        ROLL_RESISTANCE.appendChild(doc.createTextNode(str(
            self.car_para.ROLL_RESISTANCE)))
        selfcar_node.appendChild(ROLL_RESISTANCE)

        """Save traffic setting info."""
        traffic_node = doc.createElement('Traffic')
        type_node=doc.createElement('Type')
        type_node.appendChild(doc.createTextNode(self.traffic_type))
        traffic_node.appendChild(type_node)
        lib_node = doc.createElement('Lib')
        lib_node.appendChild(doc.createTextNode(self.traffic_lib))
        traffic_node.appendChild(lib_node)
        fre_node = doc.createElement('Frequency')
        fre_node.appendChild(doc.createTextNode(str(self.traffic_frequency)))
        traffic_node.appendChild(fre_node)

        """Save controller setting info."""
        control_node = doc.createElement('Controller')
        type_node=doc.createElement('Type')
        type_node.appendChild(doc.createTextNode(self.controller_type))
        control_node.appendChild(type_node)
        lib_node = doc.createElement('Lib')
        lib_node.appendChild(doc.createTextNode(self.controller_lib))
        control_node.appendChild(lib_node)
        fre_node = doc.createElement('Frequency')
        fre_node.appendChild(doc.createTextNode(str(self.controller_frequency)))
        control_node.appendChild(fre_node)
        file_node = doc.createElement('FileType')
        file_node.appendChild(doc.createTextNode(self.controller_file_type))
        control_node.appendChild(file_node)

        """Save vehicle dynamic model parameters."""
        dynamic_node = doc.createElement('Dynamic')
        type_node = doc.createElement('Type')
        type_node.appendChild(doc.createTextNode(self.dynamic_type))
        dynamic_node.appendChild(type_node)
        lib_node = doc.createElement('Lib')
        lib_node.appendChild(doc.createTextNode(self.dynamic_lib))
        dynamic_node.appendChild(lib_node)
        fre_node = doc.createElement('Frequency')
        fre_node.appendChild(doc.createTextNode(str(self.dynamic_frequency)))
        dynamic_node.appendChild(fre_node)

        """Save decision module info."""
        router_node = doc.createElement('Router')
        output_type_node = doc.createElement('OutputType')
        output_type_node.appendChild(doc.createTextNode(self.router_output_type))
        router_node.appendChild(output_type_node)
        type_node=doc.createElement('Type')
        type_node.appendChild(doc.createTextNode(self.router_type))
        router_node.appendChild(type_node)
        lib_node = doc.createElement('Lib')
        lib_node.appendChild(doc.createTextNode(self.router_lib))
        router_node.appendChild(lib_node)
        fre_node = doc.createElement('Frequency')
        fre_node.appendChild(doc.createTextNode(str(self.router_frequency)))
        router_node.appendChild(fre_node)
        file_node = doc.createElement('FileType')
        file_node.appendChild(doc.createTextNode(self.router_file_type))
        router_node.appendChild(file_node)

        """Save sensor module info."""
        sensors_node = doc.createElement('Sensors')
        type_node=doc.createElement('Type')
        type_node.appendChild(doc.createTextNode(self.sensor_model_type))
        sensors_node.appendChild(type_node)
        lib_node = doc.createElement('Lib')
        lib_node.appendChild(doc.createTextNode(self.sensor_model_lib))
        sensors_node.appendChild(lib_node)
        fre_node = doc.createElement('Frequency')
        fre_node.appendChild(doc.createTextNode(str(self.sensor_frequency)))
        sensors_node.appendChild(fre_node)
        for i in range(len(self.sensors)):
            s = self.sensors[i]
            s_node = doc.createElement('Sensor')
            sensors_node.appendChild(s_node)

            ID_node = doc.createElement('id')
            ID_node.appendChild(doc.createTextNode('%d' % i))
            s_node.appendChild(ID_node)

            Type_node = doc.createElement('type')
            Type_node.appendChild(doc.createTextNode('%d' % s.type))
            s_node.appendChild(Type_node)

            Angle_node = doc.createElement('detection_angle')
            Angle_node.appendChild(
                doc.createTextNode('%.0f' % s.detection_angle))
            s_node.appendChild(Angle_node)

            Radius_node = doc.createElement('detection_range')
            Radius_node.appendChild(
                doc.createTextNode('%.1f' % s.detection_range))
            s_node.appendChild(Radius_node)

            Installation_lat_node=doc.createElement('installation_lateral_bias')
            Installation_lat_node.appendChild(
                doc.createTextNode('%.3f' % s.installation_lateral_bias))
            s_node.appendChild(Installation_lat_node)

            Installation_long_node = doc.createElement(
                'installation_longitudinal_bias')
            Installation_long_node.appendChild(
                doc.createTextNode('%.3f'% s.installation_longitudinal_bias))
            s_node.appendChild(Installation_long_node)

            Orientation_node=doc.createElement('installation_orientation_angle')
            Orientation_node.appendChild(
                doc.createTextNode('%.0f' % s.installation_orientation_angle))
            s_node.appendChild(Orientation_node)

            Accuracy_Vel_node=doc.createElement('accuracy_velocity')
            Accuracy_Vel_node.appendChild(
                doc.createTextNode('%.2f' % s.accuracy_velocity))
            s_node.appendChild(Accuracy_Vel_node)

            Accuracy_Location_node=doc.createElement('accuracy_location')
            Accuracy_Location_node.appendChild(
                doc.createTextNode('%.2f' % s.accuracy_location))
            s_node.appendChild(Accuracy_Location_node)

            Accuracy_Yaw_node=doc.createElement('accuracy_heading')
            Accuracy_Yaw_node.appendChild(
                doc.createTextNode('%.2f' % s.accuracy_heading))
            s_node.appendChild(Accuracy_Yaw_node)

            Accuracy_Width_node=doc.createElement('accuracy_width')
            Accuracy_Width_node.appendChild(
                doc.createTextNode('%.2f' % s.accuracy_width))
            s_node.appendChild(Accuracy_Width_node)

            Accuracy_Length_node=doc.createElement('accuracy_length')
            Accuracy_Length_node.appendChild(
                doc.createTextNode('%.2f' % s.accuracy_length))
            s_node.appendChild(Accuracy_Length_node)

            Detect_Turnlight=doc.createElement('accuracy_height')
            Detect_Turnlight.appendChild(
                doc.createTextNode('%.2f' % s.accuracy_height))
            s_node.appendChild(Detect_Turnlight)

            Detect_Vehicletype=doc.createElement('accuracy_radius')
            Detect_Vehicletype.appendChild(
                doc.createTextNode('%.2f' % s.accuracy_radius))
            s_node.appendChild(Detect_Vehicletype)

        root.appendChild(info_node)
        root.appendChild(step_node)
        root.appendChild(map_node)
        root.appendChild(mission_node)
        root.appendChild(selfcar_node)
        root.appendChild(traffic_node)
        root.appendChild(control_node)
        root.appendChild(dynamic_node)
        root.appendChild(router_node)
        root.appendChild(sensors_node)

        buffer = StringIO.StringIO()
        doc.writexml(buffer, addindent="\t", newl='\n', encoding='utf-8')
        txt = re.sub('\n\t+[^<^>]*\n\t+',
                     lambda x: re.sub('[\t\n]', '', x.group(0)),
                     buffer.getvalue())
        open(path, 'w').write(txt)


MAP_MIN_X=-933.0
MAP_MAX_X=933.0
MAP_MIN_Y=-933.0
MAP_MAX_Y=933.0
MAP_REGION_LEN=622.0
MAP_CROSS_WIDTH=36.0
MAP_ROAD_WIDTH=7.5
MAP_IN_FIELD=0
MAP_IN_ROAD=1
MAP_IN_CROSS=2

MISSION_GOTO_TARGET=0
MISSION_GOTO_CROSS=1
MISSION_TURNTO_ROAD=2

MISSION_START = -1
MISSION_RUNNING = 0
MISSION_COMPLETE = 1
MISSION_FAILED = 2

MISSION_LEFT_LANE = 'L'
MISSION_RIGHT_LANE = 'R'

