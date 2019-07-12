import gym
from LasVSim import lasvsim
from gym.utils import seeding
import math
import numpy as np
from LasVSim.endtoend_env_utils import shift_coordination, rotate_coordination


# env_closer = closer.Closer()



class EndtoendEnv(gym.Env):
    r"""The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.

    The main API methods that users of this class need to know are:

        step
        reset
        render
        close
        seed

    And set the following attributes:

        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards

    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.

    The methods are accessed publicly as "step", "reset", etc.. The
    non-underscored versions are wrapper methods to which we may add
    functionality over time.
    """

    # Set this in SOME subclasses
    # metadata = {'render.modes': []}
    # reward_range = (-float('inf'), float('inf'))
    # spec = None

    # Set these in ALL subclasses

    def __init__(self, setting_path):
        self.setting_path = setting_path
        self.action_space = None
        self.observation_space = None
        self.detected_vehicles = None
        self.all_vehicles = None
        self.ego_dynamics = None
        self.ego_info = None
        self.road_related_info = None
        self.simulation = None
        self.init_state = []
        self.goal_state = []
        self.reference = Reference()

        self.seed()
        lasvsim.create_simulation(setting_path + 'simulation_setting_file.xml')
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):  # action is a np.array, [expected_acceleration, expected_steer]
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        I'm such a big pig that I didn't answer my baby's demand and I took her priority less significant.
        I'm praying and sighing and regretting.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        expected_acceleration, expected_steer = action
        lasvsim.set_steer_and_acc(expected_acceleration, expected_steer)
        lasvsim.sim_step()
        self.detected_vehicles = lasvsim.get_detected_objects()  # coordination 2
        self.all_vehicles = lasvsim.get_all_objects()  # coordination 2
        (self.ego_dynamics, self.ego_info), self.road_related_info = lasvsim.get_self_car_info()
        # ego_dynamics
        # {'x': self.x,
        #  'y': self.y,
        #  'v': self.v,
        #  'heading': self.heading,  # (deg)
        #  'acceleration': self.acc,
        #  'engine_speed': self.engine_speed,  # 发动机转速(rad/s), # CVT range: [78.5, 680.5]
        #  'transmission_gear_ratio': self.drive_ratio}  # CVT range: [0.32, 2.25]

        # ego_info
        # {'Steer_wheel_angle': self.car_info.Steer_SW,  # 方向盘转角(deg)
        #  'Throttle': self.car_info.Throttle,  # 节气门开度 (0-100)
        #  'Bk_Pressure': self.car_info.Bk_Pressure,  # 制动压力(Mpa)
        #  'Transmission_gear_ratio': self.car_info.Rgear_Tr,  # 变速器ratio, CVT range: [0.32, 2.25]
        #  'Engine_crankshaft_spin': self.car_info.AV_Eng,  # 发动机转速(rpm), CVT range: [750, 6500]
        #  'Engine_output_torque': self.car_info.M_EngOut,  # 发动机输出转矩(N*m)
        #  'A': self.car_info.A,  # 车辆加速度(m^2/s)
        #  'beta_angle': self.car_info.Beta / pi * 180,  # 质心侧偏角(deg)
        #  'Yaw_rate': self.car_info.AV_Y / pi * 180,  # 横摆角速度(deg/s)
        #  'Lateral_speed': self.car_info.Vy,  # 横向速度(m/s)
        #  'Longitudinal_speed': self.car_info.Vx,  # 纵向速度(m/s)
        #  'Steer_L1': self.car_info.Steer_L1 / pi * 180,  # 自行车模型前轮转角(deg)
        #  'StrAV_SW': self.car_info.StrAV_SW,  # 方向盘角速度(deg/s）
        #  'Mass_of_fuel_consumed': self.car_info.Mfuel,  # 燃料消耗质量(g)
        #  'Longitudinal_acc': self.car_info.Ax,  # 纵向加速度(m^2/s)
        #  'Lateral_acc': self.car_info.Ay,  # 横向加速度(m^2/s)
        #  'Fuel_rate': self.car_info.Qfuel, # 燃料消耗率(g/s)
        #  'Car_length' = self.simulation_settings.car_length,
        #  'Car_width' = self.simulation_settings.car_width,
        #  'Corner_point' = self._cal_corner_point_coordination()
        # }

        # road_related_info
        # {'dist2current_lane_center' = dis2center_line,
        #  'egolane_index' = egolane_index
        # }

        # all_vehicles
        # dict(type=c_t, x=c_x, y=c_y, v=c_v, angle=c_a,
        #      rotation=c_r, winker=w, winker_time=wt,
        #      render=render_flag, length=length,
        #      width=width,
        #      lane_index=other_veh_info[i][
        #          'current_lane'],
        #      max_decel=other_veh_info[i]['max_decel'])

        # detected_vehicles
        # {'id': id,
        #  'x': x,
        #  'y': y,
        #  'v': v,
        #  'angle': a,
        #  'width': w,
        #  'length': l}
        done = self._judge_done()  # 0 1 2 3 4
        rew = self.compute_reward(done)
        info = self.ego_info.update(self.road_related_info)
        obs = self.all_vehicles, self.detected_vehicles, self.ego_dynamics, self.ego_info,\
              self.road_related_info, self.goal_state
        return obs, rew, done is not 4, info

    def reset(self, **kwargs):
        """Resets the state of the environment and returns an initial observation.

        Returns:
            observation (object): the initial observation love my baby.
        """
        def initspeed2initgear(init_v):
            if 0 <= init_v < 10:
                init_gear = 2.25
            elif 10 <= init_v < 20:
                init_gear = 1.20
            else:
                init_gear = 0.80
            return init_gear

        if not kwargs:
            init_state, goal_state = self.generate_init_and_goal_state()  # output two list, [x, y, v, heading]
            self.init_state = init_state  # a list
            self.goal_state = goal_state
        else:
            self.init_state = kwargs['init_state']
            self.goal_state = kwargs['goal_state']
        self.init_gear = initspeed2initgear(self.init_state[2])
        lasvsim.reset_simulation(overwrite_settings={'init_gear': self.init_gear, 'init_state': self.init_state},
                                 init_traffic_path=self.setting_path)
        self.simulation = lasvsim.simulation
        self.all_vehicles = lasvsim.get_all_objects()
        self.detected_vehicles = lasvsim.get_detected_objects()
        (self.ego_dynamics, self.ego_info), self.road_related_info = lasvsim.get_self_car_info()
        obs = self.all_vehicles, self.detected_vehicles, self.ego_dynamics, self.ego_info, self.road_related_info, self.goal_state
        self.reference.reset_reference_path(self.init_state, self.goal_state)
        return obs

    def generate_init_and_goal_state(self):  # task and map dependent, hard coded TODO
        init_x = self.np_random.uniform(-900.0, 500.0)
        init_lane = self.np_random.randint(4)
        lane2y_dict = {0: -150 - 3.75 * 7.0 / 2, 1: -150 - 3.75 * 5.0 / 2, 2: -150 - 3.75 * 3.0 / 2,
                       3: -150 - 3.75 * 1.0 / 2}
        init_y = lane2y_dict[init_lane]
        init_v = self.np_random.uniform(0.0, 34.0)  # m/s
        init_heading = 0.0  # deg
        init_state = [init_x, init_y, init_v, init_heading]

        rela_x = self.np_random.uniform(5.0, 100.0)
        goal_x = init_x + rela_x
        goal_lane = self.np_random.randint(4)
        goal_y = lane2y_dict[goal_lane]
        goal_v = self.np_random.uniform(0.0, 34.0)
        goal_heading = 0.0
        goal_state = [goal_x, goal_y, goal_v, goal_heading]
        return init_state, goal_state

    # def render(self, mode='human'):
    #     """Renders the environment.
    #
    #     The set of supported modes varies per environment. (And some
    #     environments do not support rendering at all.) By convention,
    #     if mode is:
    #
    #     - human: render to the current display or terminal and
    #       return nothing. Usually for human consumption.
    #     - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
    #       representing RGB values for an x-by-y pixel image, suitable
    #       for turning into a video.
    #     - ansi: Return a string (str) or StringIO.StringIO containing a
    #       terminal-style text representation. The text can include newlines
    #       and ANSI escape sequences (e.g. for colors).
    #
    #     Note:
    #         Make sure that your class's metadata 'render.modes' key includes
    #           the list of supported modes. It's recommended to call super()
    #           in implementations to use the functionality of this method.
    #
    #     Args:
    #         mode (str): the mode to render with
    #
    #     Example:
    #
    #     class MyEnv(Env):
    #         metadata = {'render.modes': ['human', 'rgb_array']}
    #
    #         def render(self, mode='human'):
    #             if mode == 'rgb_array':
    #                 return np.array(...) # return RGB frame suitable for video
    #             elif mode == 'human':
    #                 ... # pop up a window and render
    #             else:
    #                 super(MyEnv, self).render(mode=mode) # just raise an exception
    #     """
    #     raise NotImplementedError
    #
    # def close(self):
    #     """Override close in your subclass to perform any necessary cleanup.
    #
    #     Environments will automatically close() themselves when
    #     garbage collected or when the program exits.
    #     """
    #     pass
    #
    # @property
    # def unwrapped(self):
    #     """Completely unwrap this env.
    #
    #     Returns:
    #         gym.Env: The base non-wrapped gym.Env instance
    #     """
    #     return self
    #
    # def __str__(self):
    #     if self.spec is None:
    #         return '<{} instance>'.format(type(self).__name__)
    #     else:
    #         return '<{}<{}>>'.format(type(self).__name__, self.spec.id)
    #
    # def __enter__(self):
    #     return self
    #
    # def __exit__(self, *args):
    #     self.close()
    #     # propagate exception
    #     return False

    def compute_reward(self, done):
        if done == 5:
            return -1
        elif done == 4:
            return 10
        else:
            return -10

    def _judge_done(self):
        '''
        :return:
         0: bad done: violate road constrain
         1: bad done: ego lose control
         2: bad done: violate traffic constrain (collision)
         3: bad done: task failed
         4: good done: task succeed
         5: not done
        '''
        if self._is_road_violation_or_lose_control():
            return 0
        elif self._is_lose_control():
            return 1
        elif self.simulation.stopped:
            return 2
        elif not self.reference.is_legit(self.ego_dynamics['x'], self.ego_dynamics['y']):
            return 3
        elif self.reference.is_pose_achieve_goal(self.ego_dynamics['x'], self.ego_dynamics['y'],
                                                 self.ego_dynamics['heading']):
            return 4
        else:
            return 5

    def _is_road_violation_or_lose_control(self):
        corner_points = self.ego_info['Corner_point']
        for corner_point in corner_points:
            if not judge_feasible(corner_point[0], corner_point[1]):
                return True  # violate road constrain
        return False

    def _is_lose_control(self):
        yaw_rate = self.ego_info['Yaw_rate']
        lateral_acc = self.ego_info['Lateral_acc']
        if yaw_rate > 18 or lateral_acc > 1.2:  # 正常120km/h测试最大为13deg/s和0.7m/s^2
            return True  # lose control
        else:
            return False


class Reference(object):
    def __init__(self):
        self.orig_init_x = None
        self.orig_init_y = None
        self.orig_init_v = None
        self.orig_init_heading = None
        self.orig_goal_x = None
        self.orig_goal_y = None
        self.orig_goal_v = None
        self.orig_goal_heading = None  # heading in deg
        self.goal_in_ref = None
        self.goalx_in_ref = None
        self.goaly_in_ref = None
        self.goalv_in_ref = None
        self.goalheading_in_ref = None
        self.index_mode = None
        self.reference_path = None
        self.reference_velocity = None
        # self.reset_reference_path(orig_init_state, orig_goal_state)

    def reset_reference_path(self, orig_init_state, orig_goal_state):
        self.orig_init_x, self.orig_init_y, self.orig_init_v, self.orig_init_heading = orig_init_state
        self.orig_goal_x, self.orig_goal_y, self.orig_goal_v, self.orig_goal_heading = orig_goal_state  # heading in deg
        self.goal_in_ref = self.orig2ref(self.orig_goal_x, self.orig_goal_y, self.orig_goal_v, self.orig_goal_heading)
        self.goalx_in_ref, self.goaly_in_ref, self.goalv_in_ref, self.goalheading_in_ref = self.goal_in_ref
        assert (self.goalx_in_ref > 0 and abs(self.goalheading_in_ref) < 180)
        if self.goaly_in_ref >= 0:
            assert (-90 < self.goalheading_in_ref < 180)
        else:
            assert (-180 < self.goalheading_in_ref < 90)
        self.goalheading_in_ref = self.goalheading_in_ref - 1 \
            if 90 - 0.1 < self.goalheading_in_ref < 90 + 0.1 else self.goalheading_in_ref
        self.goalheading_in_ref = self.goalheading_in_ref + 1 \
            if -90 - 0.1 < self.goalheading_in_ref < -90 + 0.1 else self.goalheading_in_ref
        self.index_mode = 'indexed_by_x' if abs(self.goalheading_in_ref) < 90 else 'indexed_by_y'
        self.reference_path = self.generate_reference_path()
        self.reference_velocity = self.generate_reference_velocity()

    def orig2ref(self, orig_x, orig_y, orig_v, orig_heading):
        orig_x, orig_y = shift_coordination(orig_x, orig_y, self.orig_init_x, self.orig_init_y)
        x_in_ref, y_in_ref, heading_in_ref = rotate_coordination(orig_x, orig_y, orig_heading, self.orig_init_heading)
        v_in_ref = orig_v
        return x_in_ref, y_in_ref, v_in_ref, heading_in_ref

    def is_pose_achieve_goal(self, orig_x, orig_y, orig_heading, distance_tolerance=1, heading_tolerance=15):
        x_in_ref, y_in_ref, _, heading_in_ref = self.orig2ref(orig_x, orig_y, 0, orig_heading)
        return True if abs(orig_x - self.goalx_in_ref) < distance_tolerance and \
                       abs(orig_y - self.goaly_in_ref) < distance_tolerance and \
                       abs(orig_heading - self.goalheading_in_ref) < heading_tolerance else False

    def is_legit(self, orig_x, orig_y):
        x_in_ref, y_in_ref, v_in_ref, heading_in_ref = self.orig2ref(orig_x, orig_y, 0, 0)
        return True if 0 < x_in_ref < self.goalx_in_ref or abs(y_in_ref) < abs(self.goaly_in_ref) else False

    def cal_bias(self, orig_x, orig_y, orig_v, orig_heading):
        assert self.index_mode == 'indexed_by_x' or self.index_mode == 'indexed_by_y'
        x_in_ref, y_in_ref, v_in_ref, heading_in_ref = self.orig2ref(orig_x, orig_y, orig_v, orig_heading)
        if self.index_mode == 'indexed_by_x':
            refer_point = self.access_path_point_indexed_by_x(x_in_ref) if x_in_ref < self.goalx_in_ref else self.goal_in_ref
        else:
            refer_point = self.access_path_point_indexed_by_y(y_in_ref) if y_in_ref < self.goaly_in_ref else self.goal_in_ref
        ref_x, ref_y, _, ref_heading = refer_point  # for now, we use goalv_in_ref instead ref_v
        position_bias = math.sqrt((ref_x - x_in_ref)**2 + (ref_y - y_in_ref)**2)
        velocity_bias = abs(self.goalv_in_ref - v_in_ref)
        heading_bias = abs(ref_heading - heading_in_ref)
        return position_bias, velocity_bias, heading_bias

    def generate_reference_path(self):  # for now, path is three order poly
        reference_path = []  # list of coefficient
        assert self.index_mode == 'indexed_by_x' or self.index_mode == 'indexed_by_y'
        if self.index_mode == 'indexed_by_x':
            a0 = 0
            a1 = 0
            slope = self._deg2slope(self.goalheading_in_ref)
            a3 = (slope - 2 * self.goaly_in_ref / self.goalx_in_ref) / self.goalx_in_ref**2
            a2 = self.goaly_in_ref / self.goalx_in_ref**2 - a3 * self.goalx_in_ref
            reference_path = [a0, a1, a2, a3]
        else:
            trans_x, trans_y, trans_d = rotate_coordination(self.goalx_in_ref, self.goaly_in_ref,
                                                            self.goalheading_in_ref, -90)
            a0 = 0
            a1 = -math.tan((90 - 1) * math.pi / 180) if trans_x <= 0 else math.tan((90 - 1) * math.pi / 180)
            a3 = (self._deg2slope(trans_d) - a1 - 2 * (trans_y - a1 * trans_x) / trans_x) / trans_x**2
            a2 = (trans_y - a1 * trans_x - a3 * trans_x ** 3) / trans_x**2
            reference_path = [a0, a1, a2, a3]
        return reference_path

    def generate_reference_velocity(self):  # for now, path is linear function
        reference_velocity = []  # list of weight (no bias)
        assert self.index_mode == 'indexed_by_x' or self.index_mode == 'indexed_by_y'
        if self.index_mode == 'indexed_by_x':
            reference_velocity.append(self.goalv_in_ref / self.goalx_in_ref)
        else:
            reference_velocity.append(self.goalv_in_ref / self.goalx_in_ref)
        return reference_velocity

    def access_path_point_indexed_by_x(self, x_in_ref):
        assert(self.index_mode == 'indexed_by_x')
        assert(0 < x_in_ref < self.goalx_in_ref)
        a0, a1, a2, a3 = self.reference_path
        w = self.reference_velocity[0]
        y_in_ref = a0 + a1 * x_in_ref + a2 * x_in_ref**2 + a3 * x_in_ref**3
        v = w * x_in_ref
        slope = a1 + 2 * a2 * x_in_ref + 3 * a3 * x_in_ref**2
        heading_in_ref = self._slope2deg(slope)
        return x_in_ref, y_in_ref, v, heading_in_ref

    def access_path_point_indexed_by_y(self, y_in_ref):
        assert (self.index_mode == 'indexed_by_y')
        a0, a1, a2, a3 = self.reference_path
        w = self.reference_velocity[0]
        x_in_ref = a0 + a1 * (-y_in_ref) + a2 * (-y_in_ref) ** 2 + a3 * (-y_in_ref) ** 3
        v = w * abs(y_in_ref)
        slope = a1 + 2 * a2 * (-y_in_ref) + 3 * a3 * (-y_in_ref) ** 2
        temp_heading = self._slope2deg(slope)
        if self.goaly_in_ref > 0:
            heading_in_ref = temp_heading + 90
        else:
            heading_in_ref = temp_heading - 90
        return x_in_ref, y_in_ref, v, heading_in_ref

    def _deg2slope(self, deg):
        return math.tan(deg * math.pi / 180)
    
    def _slope2deg(self, slope):
        return 180 * math.atan(slope) / math.pi


class Grid_3D(object):
    '''
    Consider coordination of ego car
    '''

    def __init__(self, back_dist, forward_dist, half_width, number_x, number_y, axis_z_type):
        self.back_dist = back_dist
        self.forward_dist = forward_dist
        self.half_width = half_width
        self.length = self.back_dist + self.forward_dist
        self.width = 2 * self.half_width
        self.number_x = number_x
        self.number_y = number_y
        self.axis_z_name_dict = self.get_axis_z_name_dict(axis_z_type)
        self.number_z = len(self.axis_z_name_dict)
        self.increment_x = self.length / self.number_x  # increment in x axis of car coordination
        self.increment_y = self.width / self.number_y
        self._encode_grid = np.zeros((self.number_z, self.number_x, self.number_y))
        self._encode_grid_flag = np.zeros((self.number_z, self.number_x, self.number_y), dtype=np.int)

    def get_axis_z_name_dict(self, axis_z_type):  # dict from name to index
        if axis_z_type == 'highway':
            return dict(position_x=0,
                        position_y=1,
                        v=2,
                        heading=3,
                        length=4,
                        width=5)

    def xyindex2range(self, index_x, index_y):  # index_x: [0, number_x - 1]
        index_x_in_car_coordi = index_y
        index_y_in_car_coordi = index_x
        left_upper_point_coordination_of_the_indexed_grid \
            = shift_coordination(index_x_in_car_coordi * self.increment_x, -index_y_in_car_coordi * self.increment_y,
                                 self.back_dist, -self.half_width)
        right_lower_point_coordination_of_the_indexed_grid \
            = shift_coordination((index_y + 1) * self.increment_y, -(index_x + 1) * self.increment_x,
                                 self.back_dist, -self.half_width)
        lower_x, upper_y = left_upper_point_coordination_of_the_indexed_grid
        upper_x, lower_y = right_lower_point_coordination_of_the_indexed_grid
        return lower_x, upper_x, lower_y, upper_y

    def xyindex2centerposition(self, index_x, index_y):  # index_x: [0, number_x - 1]
        lower_x, upper_x, lower_y, upper_y = self.xyindex2range(index_x, index_y)
        return 0.5 * (lower_x + upper_x), 0.5 * (lower_y + upper_y)

    def position2xyindex(self, x, y):
        x, y = shift_coordination(x, y, -self.back_dist, self.half_width)
        index_y = int(x//self.increment_y)
        index_x = int((-y)//self.increment_x)
        return index_x, index_y

    def is_in_2d_grid(self, x, y):
        if -self.back_dist < x < self.forward_dist and -self.half_width < y < self.half_width:
            return True
        else:
            return False

    def reset_grid(self):
        self._encode_grid = np.zeros((self.number_z, self.number_x, self.number_y))
        self._encode_grid_flag = np.zeros((self.number_z, self.number_x, self.number_y), dtype=np.int)

    def set_value(self, index_z, index_x, index_y, grid_value):
        self._encode_grid[index_z][index_x][index_y] = grid_value
        self._encode_grid_flag[index_z][index_x][index_y] = 1

    def set_xy_value_in_all_z(self, index_x, index_y, grid_value):
        for i in range(self.number_z):
            self.set_value(i, index_x, index_y, grid_value)

    def get_value(self, index_z, index_x, index_y):
        return self._encode_grid[index_z][index_x][index_y], \
               self._encode_grid_flag[index_z][index_x][index_y]

    def get_encode_grid_and_flag(self):
        return self._encode_grid, self._encode_grid_flag

# class GoalEnv(Env):
#     """A goal-based environment. It functions just as any regular OpenAI Gym environment but it
#     imposes a required structure on the observation_space. More concretely, the observation
#     space is required to contain at least three elements, namely `observation`, `desired_goal`, and
#     `achieved_goal`. Here, `desired_goal` specifies the goal that the agent should attempt to achieve.
#     `achieved_goal` is the goal that it currently achieved instead. `observation` contains the
#     actual observations of the environment as per usual.
#     """
#
#     def reset(self):
#         # Enforce that each GoalEnv uses a Goal-compatible observation space.
#         if not isinstance(self.observation_space, gym.spaces.Dict):
#             raise error.Error('GoalEnv requires an observation space of type gym.spaces.Dict')
#         for key in ['observation', 'achieved_goal', 'desired_goal']:
#             if key not in self.observation_space.spaces:
#                 raise error.Error('GoalEnv requires the "{}" key to be part of the observation dictionary.'.format(key))
#
#     def compute_reward(self, achieved_goal, desired_goal, info):
#         """Compute the step reward. This externalizes the reward function and makes
#         it dependent on an a desired goal and the one that was achieved. If you wish to include
#         additional rewards that are independent of the goal, you can include the necessary values
#         to derive it in info and compute it accordingly.
#
#         Args:
#             achieved_goal (object): the goal that was achieved during execution
#             desired_goal (object): the desired goal that we asked the agent to attempt to achieve
#             info (dict): an info dictionary with additional information
#
#         Returns:
#             float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
#             goal. Note that the following should always hold true:
#
#                 ob, reward, done, info = env.step()
#                 assert reward == env.compute_reward(ob['achieved_goal'], ob['goal'], info)
#         """
#         raise NotImplementedError
#
#
# class Wrapper(Env):
#     r"""Wraps the environment to allow a modular transformation.
#
#     This class is the base class for all wrappers. The subclass could override
#     some methods to change the behavior of the original environment without touching the
#     original code.
#
#     .. note::
#
#         Don't forget to call ``super().__init__(env)`` if the subclass overrides :meth:`__init__`.
#
#     """
#
#     def __init__(self, env):
#         self.env = env
#         self.action_space = self.env.action_space
#         self.observation_space = self.env.observation_space
#         self.reward_range = self.env.reward_range
#         self.metadata = self.env.metadata
#
#     def __getattr__(self, name):
#         if name.startswith('_'):
#             raise AttributeError("attempted to get missing private attribute '{}'".format(name))
#         return getattr(self.env, name)
#
#     @property
#     def spec(self):
#         return self.env.spec
#
#     @classmethod
#     def class_name(cls):
#         return cls.__name__
#
#     def step(self, action):
#         return self.env.step(action)
#
#     def reset(self, **kwargs):
#         return self.env.reset(**kwargs)
#
#     def render(self, mode='human', **kwargs):
#         return self.env.render(mode, **kwargs)
#
#     def close(self):
#         return self.env.close()
#
#     def seed(self, seed=None):
#         return self.env.seed(seed)
#
#     def compute_reward(self, achieved_goal, desired_goal, info):
#         return self.env.compute_reward(achieved_goal, desired_goal, info)
#
#     def __str__(self):
#         return '<{}{}>'.format(type(self).__name__, self.env)
#
#     def __repr__(self):
#         return str(self)
#
#     @property
#     def unwrapped(self):
#         return self.env.unwrapped


def judge_feasible(orig_x, orig_y):  # map dependant TODO
    return True if -900 < orig_x < 900 and -150 - 3.75 * 4 < orig_y < -150 else False


class ObservationWrapper(gym.Wrapper):
    def __init__(self, env, encoder_type=0, **kwargs):
        super().__init__(env)
        self.all_vehicles = None
        self.detected_vehicles = None
        self.ego_dynamics = None
        self.ego_info = None
        self.road_related_info = None
        self.goal_state = None
        self.encoder_type = encoder_type  # 0: one or more grids + one or more supplementary vectors;
                                          # 1: one or more vectors
        self.grid_setting_dict = dict(fill_type='single',  # single or cover
                                      size_list=[dict(back_dist=40, forward_dist=80, half_width=40)],
                                      number_x=240,
                                      number_y=160,
                                      axis_z_type='highway')

        if kwargs:
            self.grid_setting_dict = kwargs['grid_setting_dict']
        if self.encoder_type == 0:
            self.grid_3d = None
            self.grid_fill_type = self.grid_setting_dict['fill_type']
            self.grid_size_list = self.grid_setting_dict['size_list']
            self.grid_number_x = self.grid_setting_dict['number_x']
            self.grid_number_y = self.grid_setting_dict['number_y']
            self.gird_axis_z_type = self.grid_setting_dict['axis_z_type']

        self._FEASIBLE_VALUE = 100
        self._INFEASIBLE_VALUE = 300

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.all_vehicles, self.detected_vehicles, self.ego_dynamics, self.ego_info, \
        self.road_related_info, self.goal_state = observation
        return self.observation()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.all_vehicles, self.detected_vehicles, self.ego_dynamics, self.ego_info, \
        self.road_related_info, self.goal_state = observation
        return self.observation(), reward, done, info

    def observation(self):
        if self.encoder_type == 0:  # type 0: 3d grid + vector
            return self._3d_grid_v2x_no_noise_obs_encoder(), self._vector_supplement_for_grid_encoder()

    def _vector_supplement_for_grid_encoder(self):  # func for supplement vector of grid
        # encode road structure and task related info, hard coded for now TODO

        dist2left_road_border = -150 - self.ego_dynamics['y']
        dist2right_road_border = self.ego_dynamics['y'] - (-150 - 3.75 * 4)
        left_lane_number = 3 - self.road_related_info['egolane_index']
        right_lane_number = self.road_related_info['egolane_index']
        dist2current_lane_center = self.road_related_info['dist2current_lane_center']
        ego_v = self.ego_dynamics['v']
        ego_length = self.ego_info['Car_length']
        ego_width = self.ego_info['Car_width']
        # calculate relative goal
        goal_x, goal_y, goal_v, goal_heading = self.goal_state
        shift_x, shift_y = shift_coordination(goal_x, goal_y, self.ego_dynamics['x'], self.ego_dynamics['y'])
        rela_goal_x, rela_goal_y, rela_goal_heading \
            = rotate_coordination(shift_x, shift_y, goal_heading, self.ego_dynamics['heading'])
        # construct vector dict
        vector_dict = dict(dist2left_road_border=dist2left_road_border,
                           dist2right_road_border=dist2right_road_border,
                           left_lane_number=left_lane_number,
                           right_lane_number=right_lane_number,
                           dist2current_lane_center=dist2current_lane_center,
                           ego_v=ego_v,
                           ego_length=ego_length,
                           ego_width=ego_width,
                           rela_goal_x=rela_goal_x,
                           rela_goal_y=rela_goal_y,
                           goal_v=goal_v,
                           rela_goal_heading=rela_goal_heading)
        return vector_dict.values()

    def _3d_grid_v2x_no_noise_obs_encoder(self):  # func for grid encoder
        encoded_grid_list = []
        all_vehicles = self._v2x_unify_format_for_3dgrid()
        info_in_ego_coordination, recover_orig_position_fn = self._cal_info_in_transform_coordination(all_vehicles)
        for size_dict in self.grid_size_list:
            self.grid_3d = Grid_3D(size_dict['back_dist'], size_dict['forward_dist'], size_dict['half_width'],
                                   self.grid_number_x, self.grid_number_y, self.gird_axis_z_type)
            vehicles_in_grid = [veh for veh in info_in_ego_coordination if self.grid_3d.is_in_2d_grid(veh['trans_x'], veh['trans_y'])]
            self._add_vehicle_info_in_grid(vehicles_in_grid)
            self._add_feasible_area_info_in_grid(recover_orig_position_fn)
            encoded_grid_list.append(self.grid_3d.get_encode_grid_and_flag()[0])
        return encoded_grid_list

    def _3d_grid_sensors_with_noise_obs_encoder(self):  # func for grid encoder
        pass

    def _highway_v2x_no_noise_obs_encoder(self):  # func for vector encoder
        pass

    def _highway_sensors_with_noise_obs_encoder(self):  # func for vector encoder
        pass

    def _v2x_unify_format_for_3dgrid(self):  # unify output format
        results = []
        for veh in range(len(self.all_vehicles)):
            results.append({'x': self.all_vehicles[veh]['x'],
                            'y': self.all_vehicles[veh]['y'],
                            'v': self.all_vehicles[veh]['v'],
                            'heading': self.all_vehicles[veh]['angle'],
                            'width': self.all_vehicles[veh]['width'],
                            'length': self.all_vehicles[veh]['length']})
        return results

    def _sensors_unify_format_for_3dgrid(self):  # unify output format
        pass

    def _cal_info_in_transform_coordination(self, filtered_objects):
        results = []
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_v = self.ego_dynamics['v']
        ego_heading = self.ego_dynamics['heading']
        # egocar_length = self.ego_info['Car_length']
        # egocar_width = self.ego_info['Car_width']

        def recover_orig_position_fn(transformed_x, transformed_y):
            d = ego_heading * math.pi / 180
            transformed_x, transformed_y, _ = rotate_coordination(transformed_x, transformed_y, 0, -d)
            orig_x, orig_y = shift_coordination(transformed_x, transformed_y, -ego_x, -ego_y)
            return orig_x, orig_y

        for obj in filtered_objects:
            orig_x = obj['x']
            orig_y = obj['y']
            orig_v = obj['v']
            orig_heading = obj['heading']
            width = obj['width']
            length = obj['length']
            shifted_x, shifted_y = shift_coordination(orig_x, orig_y, ego_x, ego_y)
            trans_x, trans_y, trans_heading = rotate_coordination(shifted_x, shifted_y, orig_heading, ego_heading)
            trans_v = orig_v
            results.append({'trans_x': trans_x,
                            'trans_y': trans_y,
                            'trans_v': trans_v,
                            'trans_heading': trans_heading,
                            'width': width,
                            'length': length})
        return results, recover_orig_position_fn

    def _add_vehicle_info_in_grid(self, vehicles_in_grid):
        if self.grid_fill_type == 'single':
            for veh in vehicles_in_grid:
                x = veh['trans_x']
                y = veh['trans_y']
                v = veh['trans_v']
                heading = veh['trans_heading']
                length = veh['length']
                width = veh['width']
                index_x, index_y = self.grid_3d.position2xyindex(x, y)
                self.grid_3d.set_value(index_z=0, index_x=index_x, index_y=index_y, grid_value=x)
                self.grid_3d.set_value(index_z=1, index_x=index_x, index_y=index_y, grid_value=y)
                self.grid_3d.set_value(index_z=2, index_x=index_x, index_y=index_y, grid_value=v)
                self.grid_3d.set_value(index_z=3, index_x=index_x, index_y=index_y, grid_value=heading)
                self.grid_3d.set_value(index_z=4, index_x=index_x, index_y=index_y, grid_value=length)
                self.grid_3d.set_value(index_z=5, index_x=index_x, index_y=index_y, grid_value=width)

    def _add_feasible_area_info_in_grid(self, recover_orig_position_fn):
        number_x = self.grid_3d.number_x
        number_y = self.grid_3d.number_y
        for index_x in range(number_x):
            for index_y in range(number_y):
                x_in_egocar_coordi, y_in_egocar_coordi = self.grid_3d.xyindex2centerposition(index_x, index_y)
                orig_x, orig_y = recover_orig_position_fn(x_in_egocar_coordi, y_in_egocar_coordi)
                if not judge_feasible(orig_x, orig_y):
                    self.grid_3d.set_xy_value_in_all_z(index_x, index_y, self._INFEASIBLE_VALUE)
                elif not self.grid_3d.get_value(0, index_x, index_y):
                    self.grid_3d.set_xy_value_in_all_z(index_x, index_y, self._FEASIBLE_VALUE)


class RewardWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(observation, reward), done, info

    def reward(self, observation, reward):
        coeff_pos = -0.1
        coeff_vel = -0.1
        coeff_heading = -0.05
        _, _, ego_dynamics, _, _, _ = observation
        reference = Reference()
        reference = self.env.reference
        position_bias, velocity_bias, heading_bias = \
            reference.cal_bias(ego_dynamics['x'], ego_dynamics['y'], ego_dynamics['v'], ego_dynamics['heading'])
        reward += coeff_pos * position_bias + coeff_vel * velocity_bias + coeff_heading * heading_bias

# class ActionWrapper(Wrapper):
#     def reset(self, **kwargs):
#         return self.env.reset(**kwargs)
#
#     def step(self, action):
#         return self.env.step(self.action(action))
#
#     def action(self, action):
#         raise NotImplementedError
#
#     def reverse_action(self, action):
#         raise NotImplementedError
