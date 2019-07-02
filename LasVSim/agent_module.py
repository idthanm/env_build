# coding=utf-8
from sensor_module import *
from controller_module import *
from dynamic_module import *
from traffic_module import *
from decision_module import *
from navigation_module import *
from default_value import *
import threading


_WINKER_PERIOD=0.5


def plan(agent):
    """Plan a new temporal-spatial route

    This function is for parallel computation.

    Args:
        agent: A agent class instance.
    """
    # rotation = 0
    # agent.pos_time = agent.dynamic.pos_time
    # x0, y0, v0, a0 = agent.mission.pos
    if agent.mission.status == MISSION_RUNNING:
        if agent.planner.type is 'LatticeRouter':
            pass  # TODO(Xu Chenxiang): Add lattice router later.
        else:
            task, task_data = agent.mission.current_task
            if task == MISSION_GOTO_TARGET:
                xt, yt, vt, at = task_data
                target_status, target_lane_info = agent.map.map_position(xt, yt)
                if target_status != MAP_IN_ROAD:
                    raise Exception('target not in road')
                direction = target_lane_info['direction']
                cross = target_lane_info['target']
                if cross is not None:
                    cross_center = agent.map.get_cross_center(cross)
                else:
                    cross_center = agent.map.get_out_cross_center(
                        target_lane_info['source'], direction)
                direction = agent.mission.current_lane['direction']
                turn = 'S'
                light = 'green' if agent.is_light_green(direction) else 'red'
                agent.route = agent.planner.plan(cross_center, direction, turn,
                                                 (agent.dynamic.x,
                                                  agent.dynamic.y,
                                                  agent.dynamic.v,
                                                  agent.dynamic.heading),
                                                agent.detected_objects,
                                                light, agent.pos_time)
            elif task == MISSION_GOTO_CROSS:
                cross, target_lane_info = task_data
                cross_center = agent.map.get_cross_center(cross)
                direction = agent.mission.current_lane['direction']
                turn = agent.get_turn(direction, agent.mission.next_direction)
                light = 'green' if agent.is_light_green(direction) else 'red'
                agent.route = agent.planner.plan(cross_center, direction, turn,
                                                 (agent.dynamic.x,
                                                  agent.dynamic.y,
                                                  agent.dynamic.v,
                                                  agent.dynamic.heading),
                                                agent.detected_objects, light,
                                                agent.pos_time)
            elif task == MISSION_TURNTO_ROAD:
                cross = task_data['source']
                cross_center = agent.map.get_cross_center(cross)
                direction = agent.mission.current_lane['direction']
                turn = agent.mission.cross_task
                light = 'green' if agent.is_light_green(direction) else 'red'
                agent.route = agent.planner.plan(cross_center, direction, turn,
                                                 (agent.dynamic.x,
                                                  agent.dynamic.y,
                                                  agent.dynamic.v,
                                                  agent.dynamic.heading),
                                                 agent.detected_objects,
                                                 light, agent.pos_time)
            agent.get_future_path()
            agent.controller.set_track(agent.route, 0)
    else:
        pass


def control(agent):
    pass


def perceiving(agent):
    pass


ENGINE_TORQUE_FLAG = 0
STEERING_ANGLE_FLAG = 1
BRAKE_PRESSURE_FLAG = 2
ACCELERATION_FLAG = 3
FRONT_WHEEL_ANGLE_FLAG = 4


class Agent(object):
    """Agent Modal for Autonomous Car Simulation System.

    Attributes:
        simulation_settings: A Settings instance contains simulation settings'
             information.
        mission: A Mission instance.
        map: A Map instance.
        controller: A Controller instance.
        planner: A DAGRouter instance.
        dynamic: A Dynamic instance.
        sensors: A Sensors instance
        ......
    """

    def __init__(self, settings):
        self.simulation_settings = settings
        self.route = []
        self.plan_output = []  # plan out put. Each member for example: [t,x,y,velocity,heading angle]
        self.plan_output_type = 0  # 0 for temporal-spatial trajectory, 1 for dynamic input(engine torque, brake pressure, steer wheel angle)

        #  TODO(Xu Chenxiang): Combined to one class
        # sub classes
        self.mission = None  # mission object
        self.map = None  # map object
        self.controller = None  # controller object
        self.planner = None  # router object
        self.dynamic = None  # dynamic object
        self.sensors = None  # sensor object
        self.flag = [0, 0, 0, 0, 0]  # indicating which decision output is given.

        # simulation parameter
        self.step_length = 0.0  # simulation step length, s
        self.time = int(0)  # simulation steps
        self.pos_time = 0  # simulation time from controller

        # ego vehicle's parameter
        self.traffic_lights = None  # current traffic light status
        self.future_path = None  # future geo track path to drive (2-D)
        self.rotation = 0  # turn status
        self.winker = 0  # winker flag for turn signal light
        self.winker_time = 0  # winker time for turn signal light
        self.length = settings.car_length  # car length, m
        self.width = settings.car_width  # car width, m
        self.front_length = settings.car_center2head  # distance from car center to front end, m
        self.back_length = self.length - self.front_length  # distance from car center to back end, m
        self.weight = settings.car_weight  # unladen weight, kg
        self.shape_center_x = settings.points[0][0]  # m
        self.shape_center_y = settings.points[0][1]  # m
        self.velocity = settings.points[0][2] # m
        self.heading = settings.points[0][3] #  deg(In base coordinates)
        self.acceleration = 0.0  # m/s2
        self.engine_speed = 0.0  # r/min
        self.gears_ratio = 1
        self.engine_torque = 0.0  # N*m
        self.steer_wheel = 0.0  # deg
        self.brake_pressure = 0.0  # Mpa
        self.front_wheel_angle = 0.0  # deg
        self.desired_acceleration = 0.0  # m/s^2
        self.minimum_turning_radius = 5.0  # m
        self.maximum_acceleration = 2.0  # m/s2
        self.surrounding_objects_numbers = int(0)  # number of objects detected by ego vehicle
        self.detected_objects = [] * 2000  # info list of objects detected by ego vehicle

        self.reset()

    def reset(self):
        """Agent resetting method"""
        if hasattr(self, 'controller'):
            del self.controller
        if hasattr(self, 'planner'):
            del self.planner
        if hasattr(self, 'sensors'):
            del self.sensors
        if hasattr(self, 'navigator'):
            del self.mission
        if hasattr(self, 'dynamic'):
            del self.dynamic

        points = self.simulation_settings.points

        """Load dynamic module."""
        step_length = (self.simulation_settings.step_length *
                       self.simulation_settings.dynamic_frequency)  # ms
        if self.simulation_settings.dynamic_type is None:
            pass # TODO 后期考虑加入其他车
        else:
            self.dynamic = VehicleDynamicModel(x=points[0][0],
                                          y=points[0][1],
                                          a=points[0][3],
                                          car_parameter=self.simulation_settings.car_para,
                                          step_length=step_length,
                                          model_type=self.simulation_settings.dynamic_type)

        """Load controller module."""
        step_length = (self.simulation_settings.step_length *
                       self.simulation_settings.controller_frequency)  # ms
        if self.simulation_settings.controller_type == CONTROLLER_TYPE[PID]:
            self.controller=CarControllerDLL(path=CONTROLLER_FILE_PATH,
                                             step_length=step_length,
                                             model_type=CONTROLLER_TYPE[PID],
                                             car_parameter=self.simulation_settings.car_para,
                                             input_type=self.simulation_settings.router_output_type,
                                             car_model=self.simulation_settings.dynamic_type)
        elif self.simulation_settings.controller_type == CONTROLLER_TYPE[EXTERNAL]:
            self.controller=CarControllerDLL(path=CONTROLLER_FILE_PATH,
                                             step_length=step_length,
                                             model_type=CONTROLLER_TYPE[PID],
                                             car_parameter=self.simulation_settings.car_para,
                                             input_type=self.simulation_settings.router_output_type,
                                             car_model=self.simulation_settings.dynamic_type)
        else:
            pass  # TODO 后期加入嵌入新算法后的选择功能

        """Load decision module."""
        step_length = (self.simulation_settings.step_length *
                       self.simulation_settings.router_frequency)
        if self.simulation_settings.router_type == 'No Planner':
            self.planner = None
        elif self.simulation_settings.router_type == '-':
            pass  # TODO(Xu Chenxiang):
        else:
                self.planner = DAGRouter(step_length=step_length,
                                         path=self.simulation_settings.router_lib)
        self.decision_thread = threading.Thread(target=plan, args=(self,))

        """Load sensor module."""
        step_length = (self.simulation_settings.step_length *
                       self.simulation_settings.sensor_frequency)
        self.sensors = Sensors(step_length=step_length,
                               sensor_info=self.simulation_settings.sensors)
        self.map = Map()  # TODO
        self.mission = Mission(self.map, points,
                               self.simulation_settings.mission_type)        # self.predictor = VehiclePredictor(self.map)

    def update_dynamic_state(self,):
        """Run ego's dynamic model according to given steps.

        Args:
            steps: Int variable bigger than 1.
        """
        self.dynamic.sim_step()
        self.time += 1

    def set_engine_torque(self, torque):
        self.engine_torque = torque
        self.flag[ENGINE_TORQUE_FLAG] = 1

    def set_brake_pressure(self, brake):
        self.brake_pressure = brake
        self.flag[BRAKE_PRESSURE_FLAG] = 1

    def set_steering_angle(self, steer):
        self.steer_wheel = steer
        self.flag[STEERING_ANGLE_FLAG] = 1

    def set_acceleration(self, acc):
        self.acceleration = acc
        self.flag[ACCELERATION_FLAG] = 1

    def set_front_wheel_angle(self, angle):
        self.front_wheel_angle = angle
        self.flag[FRONT_WHEEL_ANGLE_FLAG] = 1

    def update_control_input(self, torque=None, brake=None, steer=None):
        """Compute control input.

        If torque is None, brake is None and steer is not None, then choose
        internal longitudinal controller.
        If torque is not None, brake is not None and steer is None, then choose
        internal lateral controller.
        If torque is  None, brake is  None and steer is None, then choose no
        controller.
        Else choose both internal lateral and longitudinal controller.

        Args:
            torque: Engine's output torque, N*m
            brake: Braking system's main brake pressure, Mpa
            steer: Steering angle, deg
        """
        (engine_torque, brake_pressure,
         steer_wheel) = self.controller.sim_step(self.dynamic.x,
                                                 self.dynamic.y,
                                                 self.dynamic.heading,
                                                 self.dynamic.acc,
                                                 self.dynamic.v,
                                                 self.dynamic.engine_speed,
                                                 self.dynamic.drive_ratio)
        self.dynamic.set_control_input(eng_torque=engine_torque,
                                       brake_pressure=brake_pressure,
                                       steer_wheel=steer_wheel)

    def update_info_from_sensor(self, traffic):
        """Get surrounding objects information from sensor module.
        """
        status = [self.dynamic.x, self.dynamic.y, self.dynamic.v,
                  self.dynamic.heading]
        self.sensors.update(pos=status, vehicles=traffic)
        self.detected_objects = self.sensors.getVisibleVehicles()
        self.surrounding_objects_numbers = len(self.detected_objects)

    def update_plan_output(self, traffic_lights):
        """Ego vehicle plans, 内置的决策算法.

        According to current surrounding environment and driving tasks, returns
         a desired spatio-temporal trajectory

         Returns:
             A list containing state information of each point on the desired
             trajectory. For example:
             [[time:s, x:m, y:m, velocity:m/s, heading:deg],
             [0.0, 0.0, 0.0, 0.0,0.0]...]
        """
        # Update current plan state before planning.
        status = [self.dynamic.x, self.dynamic.y, self.dynamic.v,
                  self.dynamic.heading]
        self.mission.update(status)
        self.traffic_lights = traffic_lights
        self.__plan()

    # def get_lane_pos(self,lane_pos,direction,lane_end):
    #     """
    #     get position info of road lane
    #     """
    #     x=lane_pos['center']
    #     y1=lane_pos['source']
    #     y2=lane_pos['target']
    #
    #     if direction in ['N','E']:
    #         sgn=1
    #     else:
    #         sgn=-1
    #     if lane_end=='source':
    #         y=y1+sgn*self.back_length
    #     elif lane_end=='target':
    #         # y=y2-sgn*self.front_length
    #         y=y2+sgn*self.back_length
    #     else:
    #         y=y2-sgn*self.front_length
    #     if direction in ['E','W']:
    #         x,y=y,x
    #
    #     return (x,y)
    def get_future_path(self):
        """
        get geo track from temporal-spatial route
        """
        if self.route is not None:
            t,x,y,v,a=zip(*self.route)
            self.future_path=zip(x,y)
            f = open("data.txt",'a')
            f.write(str(self.future_path)+',')
            f.close()

    def is_light_green(self, direction):
        if direction in 'NS':
            return self.traffic_lights['v'] == 'green'
        else:
            return self.traffic_lights['h'] == 'green'

    def __plan(self):
        if self.decision_thread.is_alive():
            print('planning delay')
            self.decision_thread.join()

        self.decision_thread = threading.Thread(target=plan, args=(self,))
        self.decision_thread.start()
        self.decision_thread.join()

    def get_turn(self,d1,d2):
        """
        get turn status
        """
        dirs='NWSE'
        n1=dirs.find(d1)
        n2=dirs.find(d2)
        if n1<0 or n2<0:
            return None
        elif n1==n2:
            return 'S'
        elif abs(n1-n2) == 2:
            return 'U'
        elif n1+1 == n2 or n1-3 == n2:
            return 'L'
        else:
            return 'R'

    def get_control_info(self):
        """
        get car data from controller
        """
        return self.dynamic.get_info()[:4]

    # def plan_control_input(self):
    #     x, y, v, yaw=self.mission.pos
    #     acc,r,i=self.mission.engine_state
    #     self.mission.control_input=self.controller.sim_step(x,y,yaw,acc,v,r,i)


if __name__ == "__main__":
    f= open("data.txt",'r')
    a = f.readlines()
    print(a[0])

