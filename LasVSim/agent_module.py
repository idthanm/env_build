# coding=utf-8
from LasVSim.sensor_module import *
from LasVSim.traffic_module import *
from LasVSim.default_value import *
import threading
from LasVSim.endtoend_env_utils import rotate_coordination



_WINKER_PERIOD=0.5


class Agent(object):
    """Agent Modal for Autonomous Car Simulation System.

    Attributes:
        simulation_settings: A Settings instance contains simulation settings'
             information.
        sensors: A Sensors instance
        ......
    """

    def __init__(self, settings):
        self.simulation_settings = settings
        self.route = []
        self.plan_output = []  # plan out put. Each member for example: [t,x,y,velocity,heading angle]
        self.plan_output_type = 1  # 0 for temporal-spatial trajectory, 1 for dynamic input(engine torque, brake pressure, steer wheel angle)

        #  TODO(Xu Chenxiang): Combined to one class
        # sub classes
        self.sensors = None  # sensor object

        # simulation parameter
        self.step_length = 0.0  # simulation step length, s
        self.time = int(0)  # simulation steps
        self.pos_time = 0  # simulation time from controller

        # ego vehicle's parameter
        self.traffic_lights = None  # current traffic light status
        self.length = settings.car_length  # car length, m
        self.width = settings.car_width  # car width, m
        self.lw = (self.length - self.width) / 2  # used for collision check

        self.x = settings.start_point[0]  # m
        self.y = settings.start_point[1]  # m
        self.v = settings.start_point[2] # m
        self.heading = settings.start_point[3] #  deg(In base coordinates)
        self.acceleration = 0.0  # m/s2
        self.surrounding_objects_numbers = int(0)  # number of objects detected by ego vehicle
        self.detected_objects = [] * 2000  # info list of objects detected by ego vehicle

        self.reset()

    def reset(self):
        """Agent resetting method"""
        if hasattr(self, 'sensors'):
            del self.sensors


        points = self.simulation_settings.start_point

        """Load sensor module."""
        step_length = (self.simulation_settings.step_length *
                       self.simulation_settings.sensor_frequency)
        self.sensors = Sensors(step_length=step_length,
                               sensor_info=self.simulation_settings.sensors)

    def update_info_from_sensor(self, traffic):
        """Get surrounding objects information from sensor module.
        """
        status = [self.x, self.y, self.v, self.heading]
        self.sensors.update(pos=status, vehicles=traffic)
        self.detected_objects = self.sensors.getVisibleVehicles()
        self.surrounding_objects_numbers = len(self.detected_objects)

    def is_light_green(self, direction):
        if direction in 'NS':
            return self.traffic_lights['v'] == 'green'
        else:
            return self.traffic_lights['h'] == 'green'

    def _cal_corner_point_coordination(self):
        x = self.x
        y = self.y
        heading = self.heading
        length = self.length
        width = self.width
        orig_front_left = [self.length / 2, self.width / 2]
        orig_front_right = [self.length / 2, -self.width / 2]
        orig_rear_left = [-self.length / 2, self.width / 2]
        orig_rear_right = [-self.length / 2, -self.width / 2]
        front_left_x, front_left_y, _ = rotate_coordination(orig_front_left[0], orig_front_left[1], 0, -heading)
        front_right_x, front_right_y, _ = rotate_coordination(orig_front_right[0], orig_front_right[1], 0, -heading)
        rear_left_x, rear_left_y, _ = rotate_coordination(orig_rear_left[0], orig_rear_left[1], 0, -heading)
        rear_right_x, rear_right_y, _ = rotate_coordination(orig_rear_right[0], orig_rear_right[1], 0, -heading)
        return [(front_left_x + x, front_left_y + y), (front_right_x + x, front_right_y + y),
                (rear_left_x + x, rear_left_y + y), (rear_right_x + x, rear_right_y + y)]

    def get_info(self):
        """
        get car data
        """
        car_info = {'x': self.x, 'y': self.y, 'v': self.v, 'heading': self.heading}
        car_info.update(dict(Car_length=self.simulation_settings.car_length,
                             Car_width=self.simulation_settings.car_width,
                             Corner_point=self._cal_corner_point_coordination()))
        return car_info


if __name__ == "__main__":
    f= open("data.txt",'r')
    a = f.readlines()
    print(a[0])

