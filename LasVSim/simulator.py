# coding=utf-8

"""
Simulation Base Modal of Autonomous Car Simulation System
Author: Li Bing
Date: 2017-8-23
"""

import _struct as struct
import untangle
import os
from LasVSim.traffic_module import *
from LasVSim.agent_module import *
from xml.dom.minidom import Document
# import StringIO
import time
from LasVSim import data_structures
from LasVSim.traffic_module import TrafficData
from math import cos, sin, pi, fabs

# class Data:
#     """
#     Simulation Data Manager
#     """
#     def __init__(self):
#         self.data=[]
#         self.file=open('tmp.bin','wb')
#
#         self.vehicle_count = 0
#         self.trajectory_count = 0
#         self.max_path_points = 100
#
#
#     def __del__(self):
#         self.file.close()
#
#     def append(self, self_status, self_info, vehicles, light_values=None,
#                trajectory=None, dis=None, speed_limit=None):
#         if self.file.closed:
#             return
#         t, x, y, v, a = self_status
#         data_line = [t, x, y, v, a]
#         data_line.extend(list(self_info))
#         self.data.append(data_line)
#
#         self.file.write(struct.pack('5f', *self_status))
#         self.file.write(struct.pack('17f', *self_info))
#         self.file.write(struct.pack('2f', *[dis, speed_limit]))
#         self.vehicle_count = len(vehicles)
#         for v in vehicles:
#             self.file.write(struct.pack('4f3if', *[v['x'],
#                                                    v['y'],
#                                                    v['v'],
#                                                    v['angle'],
#                                                    v['type'],
#                                                    v['rotation'],
#                                                    v['lane_index'],
#                                                    v['max_decel']]))
#         self.trajectory_count = len(trajectory)
#         self.file.write(struct.pack('2i', *light_values))
#         for i in range(100):
#             if i < self.trajectory_count:
#                 self.file.write(
#                     struct.pack('5f', *trajectory[i]))
#             else:
#                 self.file.write(
#                     struct.pack('5f', *[0, 0, 0, 0, 0]))
#         self.file.write(struct.pack('i', *[self.trajectory_count]))
#
#     def __get_time(self):
#         return [d[0] for d in self.data]
#
#     def __get_speed(self):
#         return [d[3]*3.6 for d in self.data]
#
#     def __get_x(self):
#         return [d[1] for d in self.data]
#
#     def __get_y(self):
#         return [d[2] for d in self.data]
#
#     def __get_yaw(self):
#         return [d[4] for d in self.data]
#
#     def __get_accel(self):
#         # if len(self.data)==0:
#         #     return []
#         # elif len(self.data)==1:
#         #     return [0]
#         # else:
#         #     dt=self.data[1][0]-self.data[0][0]
#         #     accel=[(self.data[i+1][3]-self.data[i][3])/dt for i in range(len(self.data)-1)]
#         #     accel.insert(0,0)
#         #     return accel
#         return [d[11] for d in self.data]
#
#     def get_data(self, type):
#         if type =='Time':
#             return self.__get_time()
#         elif type =='Vehicle Speed':
#             return self.__get_speed()
#         elif type =='Position X':
#             return self.__get_x()
#         elif type =='Position Y':
#             return self.__get_y()
#         elif type =='Heading Angle':
#             return self.__get_yaw()
#         elif type =='Acceleration':
#             return self.__get_accel()
#         elif type =='Steering Wheel':
#             return [d[5] for d in self.data]
#         elif type =='Throttle':
#             return [d[6] for d in self.data]
#         elif type =='Brake Pressure':
#             return [d[7] for d in self.data]
#         elif type =='Gear':
#             return [d[8] for d in self.data]
#         elif type =='Engine Speed':
#             return [d[9] for d in self.data]
#         elif type =='Engine Torque':
#             return [d[10] for d in self.data]
#         elif type =='Side Slip':
#             return [d[12] for d in self.data]
#         elif type =='Yaw Rate':
#             return [d[13] for d in self.data]
#         elif type =='Lateral Velocity':
#             return [d[14] for d in self.data]
#         elif type =='Longitudinal Velocity':
#             return [d[15] for d in self.data]
#         elif type =='Front Wheel Angle':
#             return [d[16] for d in self.data]
#         elif type =='Steering Rate':
#             return [d[17] for d in self.data]
#         elif type =='Fuel Consumption':
#             return [d[18] for d in self.data]
#         elif type =='Longitudinal Acceleration':
#             return [d[19] for d in self.data]
#         elif type =='Lateral Acceleration':
#             return [d[20] for d in self.data]
#         elif type =='Fuel Rate':
#             return [d[21] for d in self.data]
#         else:
#             return None
#
#     def close_file(self):
#         self.file.close()
#
#     def export_csv(self, path):
#         self.close_file()
#         with open(path, 'w') as f:
#             f.write('t(s),self_x(m),self_y(m),self_speed(m/s),self_yaw(degree)')
#             f.write(',Steering Wheel(degree),Throttle(%),Brake Pressure(MPa),'
#                     'Gear,Engine Speed(rpm)')
#             f.write(',Engine Torque(N*m),Accelerate(m/s2),Side Slip(degree), '
#                     'Yaw Rate(degree/s)')
#             f.write(',Lateral Velocity(m/s),Longitudinal Velocity(m/s)')
#             f.write(',Front Wheel Angle(deg),Steering Rate(deg/s)')
#             f.write(',Fuel Consumption(L),Longitudinal Acceleration(m/s^2)')
#             f.write(',Lateral Acceleration(m/s^2),Fuel Rate(L/s)')
#             f.write(',Distance To Stop Line(m),Speed Limit(m/s)')
#             for i in range(self.vehicle_count):
#                 f.write(',vehicle%d_x(m),vehicle%d_y(m),vehicle%d_speed(m/s),'
#                         'vehicle%d_yaw(degree),vehicle%d_type,'
#                         'vehicle%d_signals,vehicle%d_lane_index,'
#                         'vehicle%d_max_decel(m/s^2)'
#                         % (i+1, i+1, i+1, i+1, i+1, i+1, i+1, i+1))
#             f.write(',light_horizontal,light_vertical')
#             for i in range(100):
#                 f.write(',path point%d_t(m),path point%d_x(m),path point%d_y, '
#                         'path point%d_v, path_point%d_heading'
#                         % (i+1, i+1, i+1, i+1, i+1))
#             f.write(',valid trajectory point number\n')
#
#             with open('tmp.bin','rb') as fbin:
#                 fmt='5f'
#                 buffer=fbin.read(struct.calcsize(fmt))
#                 while len(buffer) > 0:
#                     f.write('%.6f,%.2f,%.2f,%.2f,%.1f'
#                             % struct.unpack(fmt, buffer))
#                     fmt = '17f'
#                     f.write(',%.0f,%.0f,%.1f,%.1f,%.0f,%.1f,%.2f,%.1f,%.2f,%.2f,'
#                             '%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f'
#                             % struct.unpack(fmt, fbin.read(struct.calcsize(fmt))))
#                     fmt = '2f'
#                     f.write(',%.2f,%.2f'
#                             % struct.unpack(fmt, fbin.read(struct.calcsize(fmt))))
#                     for i in range(self.vehicle_count):
#                         fmt = '4f3if'
#                         f.write(',%.2f,%.2f,%.2f,%.1f,%d,%d,%d,%.2f'
#                                 % struct.unpack(fmt, fbin.read(struct.calcsize(fmt))))
#                     fmt = '2i'
#                     f.write(',%d,%d'%struct.unpack(fmt, fbin.read(struct.calcsize(fmt))))
#                     for i in range(100):
#                         fmt='5f'
#                         f.write(',%.2f,%.2f,%.2f,%.2f,%.2f'%struct.unpack(fmt, fbin.read(struct.calcsize(fmt))))
#                     fmt = 'i'
#                     f.write(',%d\n'%struct.unpack(fmt, fbin.read(struct.calcsize(fmt))))
#                     fmt = '5f'
#                     buffer=fbin.read(struct.calcsize(fmt))


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

    def __init__(self, default_setting_path=None):

        self.tick_count = 0  # Simulation run time. Counted by simulation steps.
        self.sim_time = 0.0  # Simulation run time. Counted by steps multiply stpe length.
        self.other_vehicles = None  # 仿真中所有他车状态信息
        self.light_status = None  # 当前十字路口信号灯状态
        self.stopped = False  # 仿真结束标志位
        self.simulation_loaded = False  # 仿真载入标志位
        self.traffic_data = TrafficData()  # 初始交通流数据对象
        self.settings = Settings(file_path=default_setting_path)  # 仿真设置对象
        self.step_length = self.settings.step_length
        self.external_control_flag = False  # 外部控制输入标识，若外部输入会覆盖内部控制器
        self.traffic = None
        self.agent = None
        self.ego_history = None
        self.data = None
        self.seed = None

        # self.reset(settings=self.settings, overwrite_settings=overwrite_settings, init_traffic_path=init_traffic_path)
        # self.sim_step()

    def set_seed(self, seed=None):  # call this just before training (usually only once)
        if seed is not None:
            self.seed = seed

    def reset(self, settings=None, overwrite_settings=None, init_traffic_path=None):
        """Clear previous loaded module.

        Args:
            settings: LasVSim's setting class instance. Containing current
                simulation's configuring information.
        """
        if hasattr(self, 'traffic'):
            del self.traffic
        if hasattr(self, 'agent'):
            del self.agent
        if hasattr(self, 'data'):
            del self.data

        self.tick_count = 0
        self.settings = settings
        if overwrite_settings is not None:
            self.settings.start_point = overwrite_settings['init_state']
        self.stopped = False
        # self.data = Data()
        self.ego_history = {}

        """Load traffic module."""
        step_length = self.settings.step_length * self.settings.traffic_frequency
        self.traffic = Traffic(path=settings.map,
                               traffic_type=settings.traffic_type,
                               traffic_density=settings.traffic_lib,
                               step_length=step_length,
                               init_traffic=self.traffic_data.load_traffic(init_traffic_path),
                               seed=self.seed)
        self.traffic.init(settings.start_point, settings.car_length)
        self.other_vehicles = self.traffic.get_vehicles()

        """Load agent module."""
        self.agent = Agent(settings)

    def load_scenario(self, path, overwrite_settings=None):
        """Load an existing LasVSim simulation configuration file.

        Args:
            path:
        """
        if os.path.exists(path):
            settings = Settings()
            settings.load(path+'/simulation_setting_file.xml')
            # self.reset(settings)
            self.reset(settings, overwrite_settings=overwrite_settings, init_traffic_path=path)
            self.simulation_loaded = True
            return
        print('\033[31;0mSimulation loading failed: 找不到对应的根目录\033[0m')
        self.simulation_loaded = False

    # def save_scenario(self, path):
    #     """Save current simulation configuration
    #
    #     Args:
    #         path: Absolute path.
    #     """
    #     self.settings.save(path)

    def export_data(self, path):
        self.data.export_csv(path)

    def sim_step(self, steps=None):
        if steps is None:
            steps = 1
        for step in range(steps):
            if self.stopped:
                print("Simulation Finished")
                return False

            # traffic
            if self.tick_count % self.settings.traffic_frequency == 0:
                self.traffic.set_own_car(self.agent.x,
                                         self.agent.y,
                                         self.agent.v,
                                         self.agent.heading)
                self.traffic.sim_step()
                self.other_vehicles = self.traffic.get_vehicles()

            if not self.__collision_check():
                self.stopped = True


            # 保存当前步仿真数据
            # self.data.append(
            #     self_status=[self.tick_count * float(self.settings.step_length),
            #                  self.agent.dynamic.x,
            #                  self.agent.dynamic.y,
            #                  self.agent.dynamic.v,
            #                  self.agent.dynamic.heading],
            #     self_info=self.agent.dynamic.get_info(),
            #     vehicles=self.other_vehicles
            #     # light_values=self.traffic.get_light_values(),
            #     # trajectory=self.agent.route
            #     # dis=self.traffic.get_current_distance_to_stopline(),
            #     # speed_limit=self.traffic.get_current_lane_speed_limit()
            #     )
            self.tick_count += 1
        return True

    def get_all_objects(self):
        return self.other_vehicles

    def get_ego_info(self):
        return self.agent.get_info()

    def get_ego_road_related_info(self):
        return self.traffic.get_road_related_info_of_ego()

    def get_time(self):
        return self.traffic.sim_time

    def __collision_check(self):
        for vehs in self.other_vehicles:
            if (fabs(vehs['x']-self.agent.x) < 10 and
               fabs(vehs['y']-self.agent.y) < 2):
                self.ego_x0 = (self.agent.x +
                               cos(self.agent.heading/180*pi)*self.agent.lw)
                self.ego_y0 = (self.agent.y +
                               sin(self.agent.heading/180*pi)*self.agent.lw)
                self.ego_x1 = (self.agent.x -
                               cos(self.agent.heading/180*pi)*self.agent.lw)
                self.ego_y1 = (self.agent.y -
                               sin(self.agent.heading/180*pi)*self.agent.lw)
                self.surrounding_lw = (vehs['length']-vehs['width'])/2
                self.surrounding_x0 = (
                    vehs['x'] + cos(
                        vehs['angle'] / 180 * pi) * self.surrounding_lw)
                self.surrounding_y0 = (
                    vehs['y'] + sin(
                        vehs['angle'] / 180 * pi) * self.surrounding_lw)
                self.surrounding_x1 = (
                    vehs['x'] - cos(
                        vehs['angle'] / 180 * pi) * self.surrounding_lw)
                self.surrounding_y1 = (
                    vehs['y'] - sin(
                        vehs['angle'] / 180 * pi) * self.surrounding_lw)
                self.collision_check_dis = ((vehs['width']+self.agent.width)/2+0.5)**2
                if ((self.ego_x0-self.surrounding_x0)**2 +
                    (self.ego_y0-self.surrounding_y0)**2
                        < self.collision_check_dis):
                    return False
                if ((self.ego_x0-self.surrounding_x1)**2 +
                    (self.ego_y0-self.surrounding_y1)**2
                        < self.collision_check_dis):
                    return False
                if ((self.ego_x1-self.surrounding_x1)**2 +
                    (self.ego_y1-self.surrounding_y1)**2
                        < self.collision_check_dis):
                    return False
                if ((self.ego_x1-self.surrounding_x0)**2 +
                    (self.ego_y1-self.surrounding_y0)**2
                        < self.collision_check_dis):
                    return False
        return True


class Settings:  # 可以直接和package版本的Settings类替换,需要转换路径点的yaw坐标
    """
    Simulation Settings Class
    """

    def __init__(self, file_path=None):
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
        self.__load_traffic()
        self.__load_start_point()

    def __parse_xml(self, path):
        f = open(path)
        self.root = untangle.parse(f.read()).Simulation

    def __load_step_length(self):
        self.step_length = int(self.root.StepLength.cdata)

    def __load_map(self):
        self.map = str(self.root.Map.Type.cdata)

    def __load_start_point(self):
        self.start_point = [float(self.root.Start_point.X.cdata),
                            float(self.root.Start_point.Y.cdata),
                            float(self.root.Start_point.Speed.cdata),
                            float(self.root.Start_point.Yaw.cdata)]


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

    # def save(self,path):
    #     file_name=re.split(r'[\\/]',str(path))[-1].split('.')[0]
    #     doc = Document();
    #     root = doc.createElement('Simulation')
    #     doc.appendChild(root)
    #     info_node=doc.createElement('Info')
    #     title_node=doc.createElement('Title')
    #     date_node=doc.createElement('Date')
    #     author_node=doc.createElement('Author')
    #     version_node=doc.createElement('Version')
    #     title_node.appendChild(doc.createTextNode(file_name))
    #     date_node.appendChild(doc.createTextNode(time.strftime('%Y-%m-%d')))
    #     author_node.appendChild(doc.createTextNode('Author Name'))
    #     version_node.appendChild(doc.createTextNode('1.0'))
    #     info_node.appendChild(title_node)
    #     info_node.appendChild(date_node)
    #     info_node.appendChild(author_node)
    #     info_node.appendChild(version_node)
    #
    #     step_node = doc.createElement('StepLength')
    #     step_node.appendChild(doc.createTextNode(str(self.step_length)))
    #
    #     """Save map info."""
    #     map_node = doc.createElement('Map')
    #     type_node = doc.createElement('Type')
    #     type_node.appendChild(doc.createTextNode(self.map))
    #
    #     map_node.appendChild(type_node)
    #
    #     """Save mission info."""
    #     mission_node = doc.createElement('Mission')
    #     type_node = doc.createElement('Type')
    #     type_node.appendChild(doc.createTextNode(self.mission_type))
    #     mission_node.appendChild(type_node)
    #     for i in range(len(self.points)):
    #         point_node = doc.createElement('Point')
    #         mission_node.appendChild(point_node)
    #
    #         x_node = doc.createElement('X')
    #         x_node.appendChild(doc.createTextNode('%.3f' % self.points[i][0]))
    #         y_node = doc.createElement('Y')
    #         y_node.appendChild(doc.createTextNode('%.3f' % self.points[i][1]))
    #         speed_node = doc.createElement('Speed')
    #         speed_node.appendChild(doc.createTextNode('%.0f' % self.points[i][2]))
    #         yaw_node = doc.createElement('Yaw')
    #         yaw_node.appendChild(doc.createTextNode('%.0f' % (-self.points[i][3]+90)))
    #         point_node.appendChild(x_node)
    #         point_node.appendChild(y_node)
    #         point_node.appendChild(speed_node)
    #         point_node.appendChild(yaw_node)
    #
    #     # 保存自车动力学模型参数
    #     selfcar_node = doc.createElement('SelfCar')
    #
    #     length_node=doc.createElement('Length')  # 车长
    #     length_node.appendChild(doc.createTextNode('4.8'))
    #     selfcar_node.appendChild(length_node)
    #
    #     width_node=doc.createElement('Width')  # 车宽
    #     width_node.appendChild(doc.createTextNode('1.8'))
    #     selfcar_node.appendChild(width_node)
    #
    #     CenterToHead_node=doc.createElement('CenterToHead')  # 质心距车头距离
    #     CenterToHead_node.appendChild(doc.createTextNode('2.7'))
    #     selfcar_node.appendChild(CenterToHead_node)
    #
    #     FAxleToCenter_node=doc.createElement('FAxleToCenter')  # 悬上质量质心至前轴距离，m
    #     FAxleToCenter_node.appendChild(doc.createTextNode(str(
    #         self.car_para.LX_CG_SU)))
    #     selfcar_node.appendChild(FAxleToCenter_node)
    #
    #     RAxleToCenter_node=doc.createElement('RAxleToCenter')  # 悬上质量质心至后轴距离，m
    #     RAxleToCenter_node.appendChild(doc.createTextNode(str(
    #         self.car_para.LX_AXLE-self.car_para.LX_CG_SU)))
    #     selfcar_node.appendChild(RAxleToCenter_node)
    #
    #     Weight_node = doc.createElement('Weight')  # 质量
    #     Weight_node.appendChild(doc.createTextNode(str(
    #         self.car_para.M_SU+self.car_para.M_US)))
    #     selfcar_node.appendChild(Weight_node)
    #
    #     LX_AXLE = doc.createElement('LX_AXLE')  # 悬上质量，kg
    #     LX_AXLE.appendChild(doc.createTextNode(str(self.car_para.LX_AXLE)))
    #     selfcar_node.appendChild(LX_AXLE)
    #
    #     LX_CG_SU = doc.createElement('LX_CG_SU')  # 悬上质量，kg
    #     LX_CG_SU.appendChild(doc.createTextNode(str(self.car_para.LX_CG_SU)))
    #     selfcar_node.appendChild(LX_CG_SU)
    #
    #     M_SU = doc.createElement('M_SU')  # 悬上质量，kg
    #     M_SU.appendChild(doc.createTextNode(str(self.car_para.M_SU)))
    #     selfcar_node.appendChild(M_SU)
    #
    #     IZZ_SU = doc.createElement('IZZ_SU')  # 转动惯量，kg*m^2
    #     IZZ_SU.appendChild(doc.createTextNode(str(self.car_para.IZZ_SU)))
    #     selfcar_node.appendChild(IZZ_SU)
    #
    #     A_Wind = doc.createElement('A')  # 迎风面积，m^2
    #     A_Wind.appendChild(doc.createTextNode(str(self.car_para.A)))
    #     selfcar_node.appendChild(A_Wind)
    #
    #     CFx = doc.createElement('CFx')  # 空气动力学侧偏角为零度时的纵向空气阻力系数
    #     CFx.appendChild(doc.createTextNode(str(self.car_para.CFx)))
    #     selfcar_node.appendChild(CFx)
    #
    #     AV_ENGINE_IDLE = doc.createElement('AV_ENGINE_IDLE')  # 怠速转速，rpm
    #     AV_ENGINE_IDLE.appendChild(doc.createTextNode(str(
    #         self.car_para.AV_ENGINE_IDLE)))
    #     selfcar_node.appendChild(AV_ENGINE_IDLE)
    #
    #     IENG = doc.createElement('IENG')  # 曲轴转动惯量，kg*m^2
    #     IENG.appendChild(doc.createTextNode(str(self.car_para.IENG)))
    #     selfcar_node.appendChild(IENG)
    #
    #     TAU = doc.createElement('TAU')  # 发动机-变速箱输入轴 时间常数，s
    #     TAU.appendChild(doc.createTextNode(str(self.car_para.TAU)))
    #     selfcar_node.appendChild(TAU)
    #
    #     R_GEAR_TR1 = doc.createElement('R_GEAR_TR1')  # 最低档变速箱传动比
    #     R_GEAR_TR1.appendChild(doc.createTextNode(str(self.car_para.R_GEAR_TR1)))
    #     selfcar_node.appendChild(R_GEAR_TR1)
    #
    #     R_GEAR_FD = doc.createElement('R_GEAR_FD')  # 主减速器传动比
    #     R_GEAR_FD.appendChild(doc.createTextNode(str(self.car_para.R_GEAR_FD)))
    #     selfcar_node.appendChild(R_GEAR_FD)
    #
    #     BRAK_COEF = doc.createElement('BRAK_COEF')  # 液压缸变矩系数,Nm/(MPa)
    #     BRAK_COEF.appendChild(doc.createTextNode(str(self.car_para.BRAK_COEF)))
    #     selfcar_node.appendChild(BRAK_COEF)
    #
    #     Steer_FACTOR = doc.createElement('Steer_FACTOR')  # 转向传动比
    #     Steer_FACTOR.appendChild(doc.createTextNode(str(
    #         self.car_para.Steer_FACTOR)))
    #     selfcar_node.appendChild(Steer_FACTOR)
    #
    #     M_US = doc.createElement('M_US')  # 簧下质量，kg
    #     M_US.appendChild(doc.createTextNode(str(self.car_para.M_US)))
    #     selfcar_node.appendChild(M_US)
    #
    #     RRE = doc.createElement('RRE')  # 车轮有效滚动半径，m
    #     RRE.appendChild(doc.createTextNode(str(self.car_para.RRE)))
    #     selfcar_node.appendChild(RRE)
    #
    #     CF = doc.createElement('CF')  # 前轮侧偏刚度，N/rad
    #     CF.appendChild(doc.createTextNode(str(self.car_para.CF)))
    #     selfcar_node.appendChild(CF)
    #
    #     CR = doc.createElement('CR')  # 后轮侧偏刚度，N/rad
    #     CR.appendChild(doc.createTextNode(str(self.car_para.CR)))
    #     selfcar_node.appendChild(CR)
    #
    #     ROLL_RESISTANCE = doc.createElement('ROLL_RESISTANCE')  # 滚动阻力系数
    #     ROLL_RESISTANCE.appendChild(doc.createTextNode(str(
    #         self.car_para.ROLL_RESISTANCE)))
    #     selfcar_node.appendChild(ROLL_RESISTANCE)
    #
    #     """Save traffic setting info."""
    #     traffic_node = doc.createElement('Traffic')
    #     type_node=doc.createElement('Type')
    #     type_node.appendChild(doc.createTextNode(self.traffic_type))
    #     traffic_node.appendChild(type_node)
    #     lib_node = doc.createElement('Lib')
    #     lib_node.appendChild(doc.createTextNode(self.traffic_lib))
    #     traffic_node.appendChild(lib_node)
    #     fre_node = doc.createElement('Frequency')
    #     fre_node.appendChild(doc.createTextNode(str(self.traffic_frequency)))
    #     traffic_node.appendChild(fre_node)
    #
    #     """Save controller setting info."""
    #     control_node = doc.createElement('Controller')
    #     type_node=doc.createElement('Type')
    #     type_node.appendChild(doc.createTextNode(self.controller_type))
    #     control_node.appendChild(type_node)
    #     lib_node = doc.createElement('Lib')
    #     lib_node.appendChild(doc.createTextNode(self.controller_lib))
    #     control_node.appendChild(lib_node)
    #     fre_node = doc.createElement('Frequency')
    #     fre_node.appendChild(doc.createTextNode(str(self.controller_frequency)))
    #     control_node.appendChild(fre_node)
    #     file_node = doc.createElement('FileType')
    #     file_node.appendChild(doc.createTextNode(self.controller_file_type))
    #     control_node.appendChild(file_node)
    #
    #     """Save vehicle dynamic model parameters."""
    #     dynamic_node = doc.createElement('Dynamic')
    #     type_node = doc.createElement('Type')
    #     type_node.appendChild(doc.createTextNode(self.dynamic_type))
    #     dynamic_node.appendChild(type_node)
    #     lib_node = doc.createElement('Lib')
    #     lib_node.appendChild(doc.createTextNode(self.dynamic_lib))
    #     dynamic_node.appendChild(lib_node)
    #     fre_node = doc.createElement('Frequency')
    #     fre_node.appendChild(doc.createTextNode(str(self.dynamic_frequency)))
    #     dynamic_node.appendChild(fre_node)
    #
    #     """Save decision module info."""
    #     router_node = doc.createElement('Router')
    #     output_type_node = doc.createElement('OutputType')
    #     output_type_node.appendChild(doc.createTextNode(self.router_output_type))
    #     router_node.appendChild(output_type_node)
    #     type_node=doc.createElement('Type')
    #     type_node.appendChild(doc.createTextNode(self.router_type))
    #     router_node.appendChild(type_node)
    #     lib_node = doc.createElement('Lib')
    #     lib_node.appendChild(doc.createTextNode(self.router_lib))
    #     router_node.appendChild(lib_node)
    #     fre_node = doc.createElement('Frequency')
    #     fre_node.appendChild(doc.createTextNode(str(self.router_frequency)))
    #     router_node.appendChild(fre_node)
    #     file_node = doc.createElement('FileType')
    #     file_node.appendChild(doc.createTextNode(self.router_file_type))
    #     router_node.appendChild(file_node)
    #
    #     """Save sensor module info."""
    #     sensors_node = doc.createElement('Sensors')
    #     type_node=doc.createElement('Type')
    #     type_node.appendChild(doc.createTextNode(self.sensor_model_type))
    #     sensors_node.appendChild(type_node)
    #     lib_node = doc.createElement('Lib')
    #     lib_node.appendChild(doc.createTextNode(self.sensor_model_lib))
    #     sensors_node.appendChild(lib_node)
    #     fre_node = doc.createElement('Frequency')
    #     fre_node.appendChild(doc.createTextNode(str(self.sensor_frequency)))
    #     sensors_node.appendChild(fre_node)
    #     for i in range(len(self.sensors)):
    #         s = self.sensors[i]
    #         s_node = doc.createElement('Sensor')
    #         sensors_node.appendChild(s_node)
    #
    #         ID_node = doc.createElement('id')
    #         ID_node.appendChild(doc.createTextNode('%d' % i))
    #         s_node.appendChild(ID_node)
    #
    #         Type_node = doc.createElement('type')
    #         Type_node.appendChild(doc.createTextNode('%d' % s.type))
    #         s_node.appendChild(Type_node)
    #
    #         Angle_node = doc.createElement('detection_angle')
    #         Angle_node.appendChild(
    #             doc.createTextNode('%.0f' % s.detection_angle))
    #         s_node.appendChild(Angle_node)
    #
    #         Radius_node = doc.createElement('detection_range')
    #         Radius_node.appendChild(
    #             doc.createTextNode('%.1f' % s.detection_range))
    #         s_node.appendChild(Radius_node)
    #
    #         Installation_lat_node=doc.createElement('installation_lateral_bias')
    #         Installation_lat_node.appendChild(
    #             doc.createTextNode('%.3f' % s.installation_lateral_bias))
    #         s_node.appendChild(Installation_lat_node)
    #
    #         Installation_long_node = doc.createElement(
    #             'installation_longitudinal_bias')
    #         Installation_long_node.appendChild(
    #             doc.createTextNode('%.3f'% s.installation_longitudinal_bias))
    #         s_node.appendChild(Installation_long_node)
    #
    #         Orientation_node=doc.createElement('installation_orientation_angle')
    #         Orientation_node.appendChild(
    #             doc.createTextNode('%.0f' % s.installation_orientation_angle))
    #         s_node.appendChild(Orientation_node)
    #
    #         Accuracy_Vel_node=doc.createElement('accuracy_velocity')
    #         Accuracy_Vel_node.appendChild(
    #             doc.createTextNode('%.2f' % s.accuracy_velocity))
    #         s_node.appendChild(Accuracy_Vel_node)
    #
    #         Accuracy_Location_node=doc.createElement('accuracy_location')
    #         Accuracy_Location_node.appendChild(
    #             doc.createTextNode('%.2f' % s.accuracy_location))
    #         s_node.appendChild(Accuracy_Location_node)
    #
    #         Accuracy_Yaw_node=doc.createElement('accuracy_heading')
    #         Accuracy_Yaw_node.appendChild(
    #             doc.createTextNode('%.2f' % s.accuracy_heading))
    #         s_node.appendChild(Accuracy_Yaw_node)
    #
    #         Accuracy_Width_node=doc.createElement('accuracy_width')
    #         Accuracy_Width_node.appendChild(
    #             doc.createTextNode('%.2f' % s.accuracy_width))
    #         s_node.appendChild(Accuracy_Width_node)
    #
    #         Accuracy_Length_node=doc.createElement('accuracy_length')
    #         Accuracy_Length_node.appendChild(
    #             doc.createTextNode('%.2f' % s.accuracy_length))
    #         s_node.appendChild(Accuracy_Length_node)
    #
    #         Detect_Turnlight=doc.createElement('accuracy_height')
    #         Detect_Turnlight.appendChild(
    #             doc.createTextNode('%.2f' % s.accuracy_height))
    #         s_node.appendChild(Detect_Turnlight)
    #
    #         Detect_Vehicletype=doc.createElement('accuracy_radius')
    #         Detect_Vehicletype.appendChild(
    #             doc.createTextNode('%.2f' % s.accuracy_radius))
    #         s_node.appendChild(Detect_Vehicletype)
    #
    #     root.appendChild(info_node)
    #     root.appendChild(step_node)
    #     root.appendChild(map_node)
    #     root.appendChild(mission_node)
    #     root.appendChild(selfcar_node)
    #     root.appendChild(traffic_node)
    #     root.appendChild(control_node)
    #     root.appendChild(dynamic_node)
    #     root.appendChild(router_node)
    #     root.appendChild(sensors_node)
    #
    #     buffer = StringIO.StringIO()
    #     doc.writexml(buffer, addindent="\t", newl='\n', encoding='utf-8')
    #     txt = re.sub('\n\t+[^<^>]*\n\t+',
    #                  lambda x: re.sub('[\t\n]', '', x.group(0)),
    #                  buffer.getvalue())
    #     open(path, 'w').write(txt)


