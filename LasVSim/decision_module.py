from ctypes import *
from _ctypes import dlclose
from math import pi
from LasVSim.default_value import *
DAGROUTER_MODEL_PATH='Modules/DecisionModule/Map1_XinLong.dll'
DAGROUTER_TASK_TURNLEFT=2
DAGROUTER_TASK_TURNRIGHT=3
DAGROUTER_TASK_GOSTRAIGHT=1
DAGROUTER_LIGHT_RED=0
DAGROUTER_LIGHT_GREEN=1
DAGROUTER_LIGHT_YELLOW=2


def get_prop_angle(a1,a2,k):
    """
    get angle between a1 and a2 with proportion k
    unit: degree
    """
    while a1-a2>180:
        a1-=360
    while a2-a1>180:
        a2-=360
    a=a1*(1-k)+a2*k
    a=a%360
    if a>180:
        a-=360
    return a


class DAGRouter(object):
    """
    DAG Route Plan Modal for Autonomous Car Simulation System
    a interface with ctypes for the dynamic library
    based on DAG roadnet model and a-star algorithm
    """
    def __init__(self, step_length, path=None):
        self.__step_length = step_length
        self.type = DECISION_TYPE[0]
        if path is None:
            self.dll = CDLL(DAGROUTER_MODEL_PATH)
        else:
            self.dll = CDLL(path)#CDLL(path)
        self.dll.init()

    def __del__(self):
        self.dll.delete_p()
        dlclose(self.dll._handle)
        del self.dll

    def set_task(self,cross_center,direction,task):
        self.center=cross_center
        self.direction=direction
        self.task=task

    def set_self_pos(self, pos):
        x, y, v, a = pos
        x, y, a = self.__global2local(x,y,a)
        self.dll.set_ego(c_float(x),c_float(y),c_float(v),c_float(a))

    def set_other_vehicles(self,vehicles):
        for id, x, y, v, a, w, h in vehicles:
            x1, y1, a1 = self.__global2local(x, y, a)
            self.dll.set_vel(c_int(id),
                             c_float(x1),
                             c_float(y1),
                             c_float(v),
                             c_float(h),
                             c_float(w),
                             c_float(a1))

    def __global2local(self, x0, y0, a0):
        #  Coordinates transferring changed comparing to older huawei's version.
        xc,yc=self.center
        if self.direction is 'E':
            x=x0-xc
            y=y0-yc
            a0 = -a0 + 90
            a=90.0-a0
        elif self.direction is 'W':
            x=xc-x0
            y=yc-y0
            a0 = -a0 + 90
            a=270.0-a0
        elif self.direction is 'N':
            x=y0-yc
            y=xc-x0
            a0 = -a0 + 90
            a=-a0
        else:
            x=yc-y0
            y=x0-xc
            a0 = -a0 + 90
            a=180.0-a0
        while a>180:
            a-=360.0
        while a<=-180:
            a+=360.0
        a=float(a)/180*pi
        return x,y,a

    def __local2global(self, x, y, a):
        #  Coordinates transferring changed comparing to older huawei's version.
        a = a/pi*180
        xc, yc = self.center
        if self.direction is 'E':
            x0 = x+xc
            y0 = y+yc
            a0 = -(90.0-a)+90
        elif self.direction is 'W':
            x0=xc-x
            y0=yc-y
            a0=-(270.0-a)+90
        elif self.direction is 'N':
            x0=xc-y
            y0=x+yc
            a0=a+90
        else:
            x0=y+xc
            y0=yc-x
            a0=-(180.0-a)+90
        while a0>180:
            a0-=360
        while a0<=-180:
            a0+=360
        return x0,y0,a0

    def plan(self, cross_center, direction, cross_task, source, other_vehicles,
             traffic_light, current_time):
        task=DAGROUTER_TASK_GOSTRAIGHT
        if cross_task is 'L':
            task=DAGROUTER_TASK_TURNLEFT
        elif cross_task is 'R':
            task=DAGROUTER_TASK_TURNRIGHT
        light_type=DAGROUTER_LIGHT_GREEN
        if traffic_light is 'red':
            light_type=DAGROUTER_LIGHT_RED
        self.current_time=current_time
        self.set_task(cross_center,direction,task)
        self.set_self_pos(source)
        self.dll.clear_vel_list()
        self.set_other_vehicles(other_vehicles)
        self.dll.set_task(c_int(task))
        self.dll.set_trafficLight(c_int(light_type))
        self.dll.trajectory_plan()
        track=self.get_track(source)
        # self.dll.delete_p()
        return track

    def get_track(self,source):
        c_n=c_int(0)
        self.dll.get_total_num(byref(c_n))
        if c_n.value<=0:
            return []

        arr=c_float*(c_n.value*4)
        data=arr()
        self.dll.get_optimal_path(byref(data))

        t = self.current_time
        x, y, v, a = source
        track = [(t, x, y, v, a)]
        dt = 0.1

        for i in range(c_n.value):
            x2, y2, v2, a2 = data[i*4], data[i*4+1], data[i*4+2], data[i*4+3]
            x2, y2, a2 = self.__local2global(x2, y2, a2)
            x1 = (x+x2)/2
            y1 = (y+y2)/2
            v1 = (v+v2)/2
            a1 = get_prop_angle(a, a2, 0.5)
            t+=dt
            track.append((t,x1,y1,v1,a1))
            t+=dt
            track.append((t, x2, y2, v2, a2))
            x,y,v,a=x2,y2,v2,a2
        return track