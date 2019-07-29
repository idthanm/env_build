# coding=utf-8
from LasVSim.traffic_module import *
import threading
from LasVSim.endtoend_env_utils import rotate_coordination


_WINKER_PERIOD=0.5


class Agent(object):
    """Agent Modal for Autonomous Car Simulation System.

    Attributes:
        simulation_settings: A Settings instance contains simulation settings'
             information.
    """

    def __init__(self, settings):
        self.simulation_settings = settings

        #  TODO(Xu Chenxiang): Combined to one class

        # simulation parameter
        self.step_length = 0.0  # simulation step length, s
        self.time = int(0)  # simulation steps
        self.pos_time = 0  # simulation time from controller

        # ego vehicle's parameter
        self.length = settings.car_length  # car length, m
        self.width = settings.car_width  # car width, m
        self.lw = (self.length - self.width) / 2  # used for collision check
        self.front_length = settings.car_center2head  # distance from car center to front end, m
        self.back_length = self.length - self.front_length  # distance from car center to back end, m
        self.weight = settings.car_weight  # unladen weight, kg
        self.x = settings.start_point[0]  # m
        self.y = settings.start_point[1]  # m
        self.v = settings.start_point[2] # m
        self.heading = settings.start_point[3] #  deg(In base coordinates)
        self.maximum_acceleration = 2.0  # m/s2

        self.reset()

    def reset(self):
        pass

    def update_dynamic_state(self, x, y, v, heading):
        """Run ego's dynamic model according to given steps.

        Args:
            steps: Int variable bigger than 1.
        """
        self.x = x
        self.y = y
        self.v = v
        self.heading = heading
        self.time += 1

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
        return dict(x=self.x,
                    y=self.y,
                    v=self.v,
                    heading=self.heading,
                    length=self.length,
                    width=self.width), self._cal_corner_point_coordination()

if __name__ == "__main__":
    f= open("data.txt",'r')
    a = f.readlines()
    print(a[0])

