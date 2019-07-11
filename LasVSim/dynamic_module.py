# coding=utf-8
"""Dynamic module of LasVSim

@Author: Xu Chenxiang
@Date: 2019.02.28
"""
from math import pi
from LasVSim.data_structures import *
from _ctypes import FreeLibrary
import math


class VehicleDynamicModel(object):  # 可以直接与gui版本替换
    """Vehicle dynamic model for updating vehicle's state

    A interface with ctypes for the dynamic library compatible with both CVT
    and AMT controller model

    Attributes:
        path: Relative path to dynamic dll.
        type: Vehicle model type
        car_para: A dict containing vehicle model's parameter except parameters
            not accessible to vehicle's internal sensor.
        dll: A C/C++ dll instance.
        pos_time: Total simulation steps of LasVSim.
        engine_torque: Engine's output torque, N*m.
        brake_pressure: Brake system main pressure, Mpa.
        steer_wheel: Steering wheel angle, deg.
    """

    def __init__(self, x, y, a, v, step_length, car_parameter=None,
                 model_type=None):
        if model_type is None or model_type == 'CVT Car':
            self.path = CVT_MODEL_FILE_PATH
            self.type = 'CVT Car'
        elif model_type == 'AMT Car':
            self.path = AMT_MODEL_FILE_PATH
            self.type = model_type
        elif model_type == 'Truck':
            self.path = TRUCK_MODEL_FILE_PATH
            self.type = model_type
        self.step_length = float(step_length)/1000

        self.car_para = car_parameter
        self.x = x  # m
        self.y = y  # m
        self.heading = a  # deg,坐标系2
        self.v = v  # m/s
        self.acc = 0.0  # m/s^2
        self.engine_speed = self.car_para.AV_ENGINE_IDLE / 30 * pi \
            if self.v < 1 else self.v * self.car_para.R_GEAR_FD * self.car_para.R_GEAR_TR1 / self.car_para.RRE # rpm
        self.drive_ratio = self.car_para.R_GEAR_TR1
        self.engine_torque = 0.0  # N.m
        self.brake_pressure = 0.0  # Mpa
        self.steer_wheel = 0.0  # deg
        self.car_info = VehicleInfo()  # 自车信息结构体

        self.dll = CDLL(self.path)
        self.dll.init(c_float(self.x), c_float(self.y), c_float(self.heading), c_float(self.v),
                      c_float(self.step_length), byref(self.car_para))
        self.pos_time = 0

    def __del__(self):
        FreeLibrary(self.dll._handle)
        del self.dll

    def sim_step(self, EngTorque=None, BrakPressure=None, SteerWheel=None):
        if EngTorque is None:
            ET = c_float(self.engine_torque)
        else:
            ET = c_float(EngTorque)
            self.engine_torque = EngTorque
        if BrakPressure is None:
            BP = c_float(self.brake_pressure)
        else:
            BP = c_float(BrakPressure)
            self.brake_pressure = BrakPressure
        if SteerWheel is None:
            SW = c_float(self.steer_wheel)
        else:
            SW = c_float(SteerWheel)
            self.steer_wheel = SteerWheel
        x = c_float(0)
        y = c_float(0)
        yaw = c_float(0)
        acc = c_float(0)
        v = c_float(0)
        r = c_float(0)
        i = c_float(0)
        road_info = RoadParameter()
        road_info.slope = 0.0
        self.dll.sim(byref(road_info), byref(ET), byref(BP), byref(SW),
                     byref(x), byref(y), byref(yaw), byref(acc), byref(v),
                     byref(r), byref(i))

        (self.x, self.y, self.v, self.heading, self.acc, self.engine_speed,
         self.drive_ratio) = (x.value, y.value, v.value, yaw.value, acc.value,
                              r.value, i.value)

    def get_pos(self):
        return {'x': self.x,
                'y': self.y,
                'v': self.v,
                'heading': self.heading,  # (deg)
                'acceleration': self.acc,
                'engine_speed': self.engine_speed,  # 发动机转速(rad/s), # CVT range: [78.5, 680.5]
                'transmission_gear_ratio': self.drive_ratio}  # CVT range: [0.32, 2.25]

    def set_control_input(self, eng_torque, brake_pressure, steer_wheel):
        self.engine_torque = eng_torque
        self.brake_pressure = brake_pressure
        self.steer_wheel = steer_wheel

    def get_info(self):
        self.dll.get_info(byref(self.car_info))
        return {'Steer_wheel_angle': self.car_info.Steer_SW,  # 方向盘转角(deg)
                'Throttle': self.car_info.Throttle,  # 节气门开度 (0-100)
                'Bk_Pressure': self.car_info.Bk_Pressure,  # 制动压力(Mpa)
                'Transmission_gear_ratio': self.car_info.Rgear_Tr,  # 变速器ratio, CVT range: [0.32, 2.25]
                'Engine_crankshaft_spin': self.car_info.AV_Eng,  # 发动机转速(rpm), CVT range: [750, 6500]
                'Engine_output_torque': self.car_info.M_EngOut,  # 发动机输出转矩(N*m)
                'A': self.car_info.A,  # 车辆加速度(m^2/s)
                'beta_angle': self.car_info.Beta / pi * 180,  # 质心侧偏角(deg)
                'Yaw_rate': self.car_info.AV_Y / pi * 180,  # 横摆角速度(deg/s)
                'Lateral_speed': self.car_info.Vy,  # 横向速度(m/s)
                'Longitudinal_speed': self.car_info.Vx,  # 纵向速度(m/s)
                'Steer_L1': self.car_info.Steer_L1 / pi * 180,  # 自行车模型前轮转角(deg)
                'StrAV_SW': self.car_info.StrAV_SW,  # 方向盘角速度(deg/s）
                'Mass_of_fuel_consumed': self.car_info.Mfuel,  # 燃料消耗质量(g)
                'Longitudinal_acc': self.car_info.Ax,  # 纵向加速度(m^2/s)
                'Lateral_acc': self.car_info.Ay,  # 横向加速度(m^2/s)
                'Fuel_rate': self.car_info.Qfuel}  # 燃料消耗率(g/s)


if __name__ == "__main__":
    pass

