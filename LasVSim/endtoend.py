import gym
from LasVSim import lasvsim
from gym.utils import seeding
import math
import numpy as np
from LasVSim.endtoend_env_utils import shift_coordination, rotate_coordination
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# env_closer = closer.Closer()


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = gym.spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        space = gym.spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class End2endEnv(gym.Env):  # cannot be used directly, cause observation space is not known
    # can it be map and task independent? and subclass should be task and map specific
    def __init__(self,
                 setting_path=None,
                 obs_type=2,  # 0:'vectors only', 1:'grids only', '2:grids_plus_vecs'
                 frameskip=4,
                 repeat_action_probability=0):
        metadata = {'render.modes': ['human']}
        if setting_path is not None:
            self.setting_path = setting_path
        else:
            import os
            dir = os.path.dirname(__file__)
            self.setting_path = dir + '/Scenario/Intersection_endtoend/'
        self._obs_type = obs_type
        self.frameskip = frameskip
        self.detected_vehicles = None
        self.all_vehicles = None
        self.ego_dynamics = None
        self.ego_info = None
        self.road_related_info = None
        self.simulation = None
        self.init_state = []
        self.goal_state = []
        self.task = None  # used to decide goal
        self.action_space = gym.spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)
        self.simulation = lasvsim.create_simulation(self.setting_path + 'simulation_setting_file.xml')
        self.seed()
        self.reset()
        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(self._action_transformation(action))
        self._set_observation_space(observation)
        plt.ion()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        lasvsim.seed(int(seed))
        return [seed]

    def step(self, action):  # action is a np.array, [expected_acceleration, delta_steer]
        action = self._action_transformation(action)
        reward = 0
        done = 0
        all_info = None
        for _ in range(self.frameskip):
            lasvsim.set_delta_steer_and_acc(action[0], action[1])
            lasvsim.sim_step()
            all_info = self._get_all_info()
            done_type, done = self._judge_done()
            reward += self.compute_reward(done_type)
            if done:
                self._print_done_info(done_type)
                break
        obs = self._get_obs()

        return obs, reward, done, all_info

    def reset(self, **kwargs):  # kwargs includes two keys: 'task': 0/1/2, 'init_state': [x, y, v, heading]
        if kwargs:
            self.init_state, self.task = kwargs['init_state'], kwargs['task']
        else:
            self.init_state, self.task = [3.75 / 2, -18 - 2.5, 0, 90], 0
        lasvsim.reset_simulation(overwrite_settings={'init_state': self.init_state},
                                 init_traffic_path=self.setting_path)
        # decide goal state at start
        self._generate_goal_state()
        self._get_all_info()
        obs = self._get_obs()
        return obs

    def _action_transformation(self, action):
        """assume action is in [-1, 1], and transform it to proper interval"""
        return action[0] * 3.75 - 1.25, action[1] * 30

    # observation related
    def _generate_goal_state(self):  # need to be override in subclass
        raise NotImplementedError

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def _get_all_info(self):  # after goal_state is generated, must be called every timestep before _get_obs
        # to fetch info
        self.all_vehicles = lasvsim.get_all_objects()  # coordination 2
        self.detected_vehicles = lasvsim.get_detected_objects()  # coordination 2
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
        # {'Steer_wheel_angle': self.car_info.Steer_SW,  # 方向盘转角(deg) [-705, 705]
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
        #  'egolane_index' = egolane_index,
        #  'current_lane_speed_limit'' = own_lane_speed_limit,
        #  'current_distance_to_stopline' = current_distance_to_stopline,
        #  'vertical_light_value' = vertical_light_value
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
        all_info = dict(all_vehicles=self.all_vehicles,
                        detected_vehicles=self.detected_vehicles,
                        ego_dynamics=self.ego_dynamics,
                        ego_info=self.ego_info,
                        road_related_info=self.road_related_info,
                        goal_state=self.goal_state)
        return all_info

    def _get_obs(self):
        """this func should be override in subclass to generate different types of observation"""
        raise NotImplementedError

    def render(self, mode='human'):
        if mode == 'human':

            # plot basic map
            square_length = 36
            extension = 10
            lane_width = 3.75
            dotted_line_style = '--'

            plt.cla()
            plt.title("Demo")
            ax = plt.axes(xlim=(-square_length / 2 - extension, square_length / 2 + extension),
                          ylim=(-square_length / 2 - extension, square_length / 2 + extension))
            plt.axis("equal")

            ax.add_patch(plt.Rectangle((-square_length / 2, -square_length / 2),
                                       square_length, square_length, edgecolor='black', facecolor='none'))
            ax.add_patch(plt.Rectangle((-square_length / 2 - extension, -square_length / 2 - extension),
                                       square_length + 2 * extension, square_length + 2 * extension, edgecolor='black',
                                       facecolor='none'))

            # ----------horizon--------------
            plt.plot([-square_length / 2 - extension, -square_length / 2], [0, 0], color='black')
            plt.plot([square_length / 2 + extension, square_length / 2], [0, 0], color='black')

            #
            plt.plot([-square_length / 2 - extension, -square_length / 2], [lane_width, lane_width],
                     linestyle=dotted_line_style, color='black')
            plt.plot([square_length / 2 + extension, square_length / 2], [lane_width, lane_width],
                     linestyle=dotted_line_style, color='black')

            plt.plot([-square_length / 2 - extension, -square_length / 2], [2 * lane_width, 2 * lane_width],
                     color='black')
            plt.plot([square_length / 2 + extension, square_length / 2], [2 * lane_width, 2 * lane_width],
                     color='black')
            #
            plt.plot([-square_length / 2 - extension, -square_length / 2], [-lane_width, -lane_width],
                     linestyle=dotted_line_style, color='black')
            plt.plot([square_length / 2 + extension, square_length / 2], [-lane_width, -lane_width],
                     linestyle=dotted_line_style, color='black')

            plt.plot([-square_length / 2 - extension, -square_length / 2], [-2 * lane_width, -2 * lane_width],
                     color='black')
            plt.plot([square_length / 2 + extension, square_length / 2], [-2 * lane_width, -2 * lane_width],
                     color='black')

            # ----------vertical----------------
            plt.plot([0, 0], [-square_length / 2 - extension, -square_length / 2], color='black')
            plt.plot([0, 0], [square_length / 2 + extension, square_length / 2], color='black')

            #
            plt.plot([lane_width, lane_width], [-square_length / 2 - extension, -square_length / 2],
                     linestyle=dotted_line_style, color='black')
            plt.plot([lane_width, lane_width], [square_length / 2 + extension, square_length / 2],
                     linestyle=dotted_line_style, color='black')

            plt.plot([2 * lane_width, 2 * lane_width], [-square_length / 2 - extension, -square_length / 2],
                     color='black')
            plt.plot([2 * lane_width, 2 * lane_width], [square_length / 2 + extension, square_length / 2],
                     color='black')

            #
            plt.plot([-lane_width, -lane_width], [-square_length / 2 - extension, -square_length / 2],
                     linestyle=dotted_line_style, color='black')
            plt.plot([-lane_width, -lane_width], [square_length / 2 + extension, square_length / 2],
                     linestyle=dotted_line_style, color='black')

            plt.plot([-2 * lane_width, -2 * lane_width], [-square_length / 2 - extension, -square_length / 2],
                     color='black')
            plt.plot([-2 * lane_width, -2 * lane_width], [square_length / 2 + extension, square_length / 2],
                     color='black')

            def is_in_plot_area(x, y, tolerance=5):
                if -square_length / 2 - extension+tolerance < x < square_length / 2 + extension-tolerance and \
                        -square_length / 2 - extension+tolerance < y < square_length / 2 + extension-tolerance:
                    return True
                else:
                    return False

            def draw_rotate_rec(x, y, a, l, w, color):
                RU_x, RU_y, _ = rotate_coordination(l / 2, w / 2, 0, -a)
                RD_x, RD_y, _ = rotate_coordination(l / 2, -w / 2, 0, -a)
                LU_x, LU_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
                LD_x, LD_y, _ = rotate_coordination(-l / 2, -w / 2, 0, -a)
                ax.plot([RU_x + x, RD_x + x], [RU_y + y, RD_y + y], color=color)
                ax.plot([RU_x + x, LU_x + x], [RU_y + y, LU_y + y], color=color)
                ax.plot([LD_x + x, RD_x + x], [LD_y + y, RD_y + y], color=color)
                ax.plot([LD_x + x, LU_x + x], [LD_y + y, LU_y + y], color=color)

            # plot cars
            for veh in self.all_vehicles:
                x = veh['x']
                y = veh['y']
                a = veh['angle']
                l = veh['length']
                w = veh['width']
                if is_in_plot_area(x, y):
                    draw_rotate_rec(x, y, a, l, w, 'black')

            # plot own car
            ego_x = self.ego_dynamics['x']
            ego_y = self.ego_dynamics['y']
            ego_a = self.ego_dynamics['heading']
            ego_l = self.ego_info['Car_length']
            ego_w = self.ego_info['Car_width']
            draw_rotate_rec(ego_x, ego_y, ego_a, ego_l, ego_w, 'red')
            plt.show()
            plt.pause(0.1)

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    # reward related
    def compute_reward(self, done):
        reward = 0
        if done == 5:
            reward += self._bias_reward()
            # reward += self._be_in_interested_area_reward()
            return reward
        elif done == 4:
            return 200
        else:
            if done == 0:
                return -500
            else:
                return -100

    def _calculate_heristic_bias(self):
        current_x, current_y, current_v, current_heading = self.ego_dynamics['x'], self.ego_dynamics['y'], \
                                                           self.ego_dynamics['v'], self.ego_dynamics['heading']
        goal_x, goal_y, goal_v, goal_heading = self.goal_state
        position_bias = math.sqrt((goal_x - current_x) ** 2 + (goal_y - current_y) ** 2)
        velocity_bias = math.fabs(goal_v - current_v)
        heading_bias = math.fabs(goal_heading - abs(current_heading))
        return position_bias, velocity_bias, heading_bias

    def _bias_reward(self):
        coeff_pos = -0.01
        coeff_vel = -0.2
        coeff_heading = 0
        position_bias, velocity_bias, heading_bias = self._calculate_heristic_bias()
        return coeff_pos * position_bias + coeff_vel * velocity_bias + coeff_heading * heading_bias

    # def _be_in_interested_area_reward(self):
    #     def judge_task0(x, y):
    #         return True if x > -18 and y > -18 and 18 < math.sqrt((x + 18) ** 2 + (y + 18) ** 2) < 18 + 3.75 else False
    #
    #     def judge_task1(x, y):
    #         return True if 0 < x < 18 + 3.75 and -18 < y < 18 else False
    #
    #     def judge_task2(x, y):
    #         return True if x < 18 and y > -18 and 18 - 3.75 * 2 < math.sqrt(
    #             (x - 18) ** 2 + (y + 18) ** 2) < 18 - 3.75 else False
    #
    #     def judge_01lane(x, y):
    #         return True if 0 < x < 3.75 and y < -18 else 0
    #
    #     def judge_2lane(x, y):
    #         return True if 3.75 < x < 2 * 3.75 and y < -18 else 0
    #
    #     x = self.ego_dynamics['x']
    #     y = self.ego_dynamics['y']
    #
    #     if self.task == 0:
    #         return 2 if judge_task0(x, y) or judge_01lane(x, y) else 0
    #
    #     if self.task == 1:
    #         return 2 if judge_task1(x, y) or judge_01lane(x, y) else 0
    #
    #     if self.task == 2:
    #         return 2 if judge_task2(x, y) or judge_2lane(x, y) else 0
    # # done related
    def _judge_done(self):
        """
        :return:
         0: bad done: violate road constrain
         1: bad done: ego lose control
         2: bad done: collision
         3: bad done: task failed
         4: good done: task succeed
         5: not done
        """
        if self._is_road_violation():
            return 0, 1
        elif self._is_lose_control():
            return 1, 1
        elif self.simulation.stopped:
            return 2, 1
        elif self._is_task_failed():
            return 3, 1
        elif self._is_achieve_goal(10, 90):
            return 4, 1
        else:
            return 5, 0

    def _print_done_info(self, done_type):
        done_info = ['road violation', 'lose control', 'collision', 'task failed', 'goal achieved!']
        print('done info: ' + done_info[done_type])

    def _is_task_failed(self):  # need to override in subclass
        raise NotImplementedError

    def _is_achieve_goal(self, distance_tolerance=1., heading_tolerance=30.):  # no need to override
        current_x, current_y, current_v, current_heading = self.ego_dynamics['x'], self.ego_dynamics['y'], \
                                                           self.ego_dynamics['v'], self.ego_dynamics['heading']
        goal_x, goal_y, goal_v, goal_heading = self.goal_state
        return True if abs(current_x - goal_x) < distance_tolerance and \
                       abs(current_y - goal_y) < distance_tolerance and \
                       abs(current_heading - goal_heading) < heading_tolerance else False

    def _is_road_violation(self):  # no need to override, just change judge_feasible func
        center_points = self.ego_dynamics['x'], self.ego_dynamics['y']
        if not judge_feasible(center_points[0], center_points[1]):
            return True  # violate road constrain
        return False

    def _is_lose_control(self):  # no need to override
        # # yaw_rate = self.ego_info['Yaw_rate']
        # lateral_acc = self.ego_info['Lateral_acc']
        # # if yaw_rate > 18 or lateral_acc > 1.2:  # 正常120km/h测试最大为13deg/s和0.7m/s^2
        # if lateral_acc > 10:
        #     return True  # lose control
        # else:
        #     return False
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

    def cal_path_maxx_maxy_miny(self):
        if self.index_mode == 'indexed_by_x':
            a0, a1, a2, a3 = self.reference_path
            x = np.linspace(0, self.goalx_in_ref, 100)
            y = a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3
            return self.goalx_in_ref, max(y), min(y)
        else:
            a0, a1, a2, a3 = self.reference_path
            y = np.linspace(0, -self.goaly_in_ref, 100)
            x = a0 + a1 * y + a2 * y ** 2 + a3 * y ** 3
            maxx = max(x)
            if self.goaly_in_ref >= 0:
                return maxx, self.goaly_in_ref, 0
            else:
                return maxx, 0, self.goaly_in_ref

    def orig2ref(self, orig_x, orig_y, orig_v, orig_heading):
        orig_x, orig_y = shift_coordination(orig_x, orig_y, self.orig_init_x, self.orig_init_y)
        x_in_ref, y_in_ref, heading_in_ref = rotate_coordination(orig_x, orig_y, orig_heading, self.orig_init_heading)
        v_in_ref = orig_v
        return x_in_ref, y_in_ref, v_in_ref, heading_in_ref

    def is_pose_achieve_goal(self, orig_x, orig_y, orig_heading, distance_tolerance=2.5, heading_tolerance=30):
        x_in_ref, y_in_ref, _, heading_in_ref = self.orig2ref(orig_x, orig_y, 0, orig_heading)
        return True if abs(orig_x - self.goalx_in_ref) < distance_tolerance and \
                       abs(orig_y - self.goaly_in_ref) < distance_tolerance and \
                       abs(orig_heading - self.goalheading_in_ref) < heading_tolerance else False

    def is_legit(self, orig_x, orig_y):
        maxx, maxy, miny = self.cal_path_maxx_maxy_miny()
        x_in_ref, y_in_ref, v_in_ref, heading_in_ref = self.orig2ref(orig_x, orig_y, 0, 0)
        return True if 0 < x_in_ref < maxx + 5 and miny - 5 < y_in_ref < maxy + 5 else False

    def cal_bias(self, orig_x, orig_y, orig_v, orig_heading):
        assert self.index_mode == 'indexed_by_x' or self.index_mode == 'indexed_by_y'
        x_in_ref, y_in_ref, v_in_ref, heading_in_ref = self.orig2ref(orig_x, orig_y, orig_v, orig_heading)
        if self.index_mode == 'indexed_by_x':
            refer_point = self.access_path_point_indexed_by_x(
                x_in_ref) if x_in_ref < self.goalx_in_ref else self.goal_in_ref
        else:
            refer_point = self.access_path_point_indexed_by_y(
                y_in_ref) if y_in_ref < self.goaly_in_ref else self.goal_in_ref
        ref_x, ref_y, _, ref_heading = refer_point  # for now, we use goalv_in_ref instead ref_v
        position_bias = math.sqrt((ref_x - x_in_ref) ** 2 + (ref_y - y_in_ref) ** 2)
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
            a3 = (slope - 2 * self.goaly_in_ref / self.goalx_in_ref) / self.goalx_in_ref ** 2
            a2 = self.goaly_in_ref / self.goalx_in_ref ** 2 - a3 * self.goalx_in_ref
            reference_path = [a0, a1, a2, a3]
        else:
            trans_x, trans_y, trans_d = rotate_coordination(self.goalx_in_ref, self.goaly_in_ref,
                                                            self.goalheading_in_ref, -90)
            a0 = 0
            a1 = -math.tan((90 - 1) * math.pi / 180) if trans_x <= 0 else math.tan((90 - 1) * math.pi / 180)
            a3 = (self._deg2slope(trans_d) - a1 - 2 * (trans_y - a1 * trans_x) / trans_x) / trans_x ** 2
            a2 = (trans_y - a1 * trans_x - a3 * trans_x ** 3) / trans_x ** 2
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
        assert (self.index_mode == 'indexed_by_x')
        assert (-1 < x_in_ref < self.goalx_in_ref)
        a0, a1, a2, a3 = self.reference_path
        w = self.reference_velocity[0]
        y_in_ref = a0 + a1 * x_in_ref + a2 * x_in_ref ** 2 + a3 * x_in_ref ** 3
        v = w * x_in_ref
        slope = a1 + 2 * a2 * x_in_ref + 3 * a3 * x_in_ref ** 2
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
    """
    Consider coordination of ego car
    """

    def __init__(self, back_dist, forward_dist, half_width, number_x, number_y, axis_z_type='highway'):
        self.back_dist = back_dist
        self.forward_dist = forward_dist
        self.half_width = half_width
        self.length = self.back_dist + self.forward_dist
        self.width = 2 * self.half_width
        self.number_x = number_x  # in car coordination ! not in matrix index
        self.number_y = number_y
        self.matrix_x = number_y
        self.matrix_y = number_x
        self.axis_z_name_dict = self.get_axis_z_name_dict(axis_z_type)
        self.number_z = len(self.axis_z_name_dict)
        self.increment_x = self.length / self.number_x  # increment in x axis of car coordination
        self.increment_y = self.width / self.number_y
        self._encode_grid = np.zeros((self.number_z, self.matrix_x, self.matrix_y))
        self._encode_grid_flag = np.zeros((self.number_z, self.matrix_x, self.matrix_y), dtype=np.int)

    def get_axis_z_name_dict(self, axis_z_type):  # dict from name to index
        if axis_z_type == 'highway':
            return dict(position_x=0,
                        position_y=1,
                        v=2,
                        heading=3,
                        length=4,
                        width=5)

    def xyindex2range(self, index_x, index_y):  # index_x: [0, matrix_x - 1]
        index_x_in_car_coordi = index_y
        index_y_in_car_coordi = index_x
        left_upper_point_coordination_of_the_indexed_grid \
            = shift_coordination(index_x_in_car_coordi * self.increment_x, -index_y_in_car_coordi * self.increment_y,
                                 self.back_dist, -self.half_width)
        right_lower_point_coordination_of_the_indexed_grid \
            = shift_coordination((index_x_in_car_coordi + 1) * self.increment_x,
                                 -(index_y_in_car_coordi + 1) * self.increment_y,
                                 self.back_dist, -self.half_width)
        lower_x, upper_y = left_upper_point_coordination_of_the_indexed_grid
        upper_x, lower_y = right_lower_point_coordination_of_the_indexed_grid
        return lower_x, upper_x, lower_y, upper_y

    def xyindex2centerposition(self, index_x, index_y):  # index_x: [0, matrix_x - 1]
        lower_x, upper_x, lower_y, upper_y = self.xyindex2range(index_x, index_y)
        return 0.5 * (lower_x + upper_x), 0.5 * (lower_y + upper_y)

    def position2xyindex(self, x, y):
        x, y = shift_coordination(x, y, -self.back_dist, self.half_width)
        index_x_in_car_coordi = int(x // self.increment_x)
        index_y_in_car_coordi = int((-y) // self.increment_y)
        index_x = index_y_in_car_coordi
        index_y = index_x_in_car_coordi
        return index_x, index_y

    def is_in_2d_grid(self, x, y):
        if -self.back_dist < x < self.forward_dist and -self.half_width < y < self.half_width:
            return True
        else:
            return False

    def reset_grid(self):
        self._encode_grid = np.zeros((self.number_z, self.matrix_x, self.matrix_y))
        self._encode_grid_flag = np.zeros((self.number_z, self.matrix_x, self.matrix_y), dtype=np.int)

    def set_value(self, index_z, index_x, index_y, grid_value):
        self._encode_grid[index_z][index_x][index_y] = grid_value
        self._encode_grid_flag[index_z][index_x][index_y] = 1

    def set_same_xy_value_in_all_z(self, index_x, index_y, grid_value):
        for i in range(self.number_z):
            self.set_value(i, index_x, index_y, grid_value)

    def set_xy_value_with_list(self, index_x, index_y, z_list):
        assert len(z_list) == self.number_z
        for i in range(self.number_z):
            self.set_value(i, index_x, index_y, z_list[i])

    def get_value(self, index_z, index_x, index_y):
        return self._encode_grid[index_z][index_x][index_y], \
               self._encode_grid_flag[index_z][index_x][index_y]

    def get_encode_grid_and_flag(self):
        return self._encode_grid, self._encode_grid_flag


def judge_feasible(orig_x, orig_y):  # map dependant TODO
    # return True if -900 < orig_x < 900 and -150 - 3.75 * 4 < orig_y < -150 else False
    def is_in_straight(orig_x, orig_y):
        return 0 < orig_x < 3.75 * 2

    def is_in_left(orig_x, orig_y):
        return 0 < orig_y < 3.75 * 2 and orig_x < -18

    def is_in_right(orig_x, orig_y):
        return -3.75 * 2 < orig_y < 0 and orig_x > 18

    def is_in_middle(orig_x, orig_y):
        return -18 < orig_y < 18 and -18 < orig_x < 18

    return True if is_in_straight(orig_x, orig_y) or is_in_left(orig_x, orig_y) or is_in_right(orig_x, orig_y) \
                   or is_in_middle(orig_x, orig_y) else False


class CrossroadEnd2end(End2endEnv):
    def __init__(self,
                 setting_path=None,
                 obs_type=2,  # 0:'vectors only', 1:'grids only', '2:grids_plus_vecs'
                 frameskip=4,
                 repeat_action_probability=0
                 ):
        self.grid_setting_dict = dict(fill_type='single',  # single or cover
                                      size_list=[dict(back_dist=10, forward_dist=30, half_width=20)],
                                      number_x=40,
                                      number_y=40,
                                      axis_z_type='highway')

        if obs_type == 2:
            self.grid_3d = None
            self.grid_fill_type = self.grid_setting_dict['fill_type']
            self.grid_size_list = self.grid_setting_dict['size_list']
            self.grid_number_x = self.grid_setting_dict['number_x']
            self.grid_number_y = self.grid_setting_dict['number_y']
            self.gird_axis_z_type = self.grid_setting_dict['axis_z_type']
            if self.gird_axis_z_type == 'highway':
                self._FEASIBLE_VALUE = 0
                self._INFEASIBLE_VALUE = -1

        super(CrossroadEnd2end, self).__init__(setting_path, obs_type, frameskip)

    def _get_obs(self):
        if self._obs_type == 2:  # type 2: 3d grid + vector
            return dict(grid=self._3d_grid_v2x_no_noise_obs_encoder()[0],
                        vector=self._vector_supplement_for_grid_encoder())

    def _vector_supplement_for_grid_encoder(self):  # func for supplement vector of grid
        # encode road structure and task related info, hard coded for now TODO

        # dist2left_road_border = -150 - self.ego_dynamics['y']
        # dist2right_road_border = self.ego_dynamics['y'] - (-150 - 3.75 * 4)
        # left_lane_number = 3 - self.road_related_info['egolane_index']
        # right_lane_number = self.road_related_info['egolane_index']
        # dist2current_lane_center = self.road_related_info['dist2current_lane_center']
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_v = self.ego_dynamics['v']
        ego_length = self.ego_info['Car_length']
        ego_width = self.ego_info['Car_width']
        # calculate relative goal
        goal_x, goal_y, goal_v, goal_heading = self.goal_state
        shift_x, shift_y = shift_coordination(goal_x, goal_y, self.ego_dynamics['x'], self.ego_dynamics['y'])
        rela_goal_x, rela_goal_y, rela_goal_heading \
            = rotate_coordination(shift_x, shift_y, goal_heading, self.ego_dynamics['heading'])
        # construct vector dict
        vector_dict = dict(ego_x=ego_x,
                           ego_y=ego_y,
                           ego_v=ego_v,
                           ego_length=ego_length,
                           ego_width=ego_width,
                           rela_goal_x=rela_goal_x,
                           rela_goal_y=rela_goal_y,
                           goal_v=goal_v,
                           rela_goal_heading=rela_goal_heading)
        return np.array(list(vector_dict.values()))

    def _3d_grid_v2x_no_noise_obs_encoder(self):  # func for grid encoder
        encoded_grid_list = []
        all_vehicles = self._v2x_unify_format_for_3dgrid()
        info_in_ego_coordination, recover_orig_position_fn = self._cal_info_in_transform_coordination(all_vehicles)
        for size_dict in self.grid_size_list:
            self.grid_3d = Grid_3D(size_dict['back_dist'], size_dict['forward_dist'], size_dict['half_width'],
                                   self.grid_number_x, self.grid_number_y, self.gird_axis_z_type)
            vehicles_in_grid = [veh for veh in info_in_ego_coordination if
                                self.grid_3d.is_in_2d_grid(veh['trans_x'], veh['trans_y'])]
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
            d = ego_heading  # TODO: check if correct
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
        number_x = self.grid_3d.number_x  # number_x is matrix_y
        number_y = self.grid_3d.number_y
        for index_x in range(number_y):
            for index_y in range(number_x):
                x_in_egocar_coordi, y_in_egocar_coordi = self.grid_3d.xyindex2centerposition(index_x, index_y)
                orig_x, orig_y = recover_orig_position_fn(x_in_egocar_coordi, y_in_egocar_coordi)
                if not self.grid_3d.get_value(0, index_x, index_y)[1]:
                    if not judge_feasible(orig_x, orig_y):
                        self.grid_3d.set_same_xy_value_in_all_z(index_x, index_y, self._INFEASIBLE_VALUE)
                    else:
                        self.grid_3d.set_same_xy_value_in_all_z(index_x, index_y, self._FEASIBLE_VALUE)

    def _generate_goal_state(self):
        if self.task == 0:
            self.goal_state = [-18 - 5 / 2, 3.75 / 2, 8, 180]
        elif self.task == 1:
            self.goal_state = [3.75 / 2, 18 + 5 / 2, 8, 90]
        else:
            self.goal_state = [-3.75 * 3 / 2, 18 + 5 / 2, 8, 0]

    def _is_task_failed(self):

        def is_in_initlane(orig_x, orig_y):
            return 0 < orig_x < 3.75 * 2 and -23 < orig_y < -18

        def is_in_straight(orig_x, orig_y):
            return 0 < orig_x < 3.75 * 2 and 18 < orig_y < 23

        def is_in_left(orig_x, orig_y):
            return 0 < orig_y < 3.75 * 2 and -23 < orig_x < -18

        def is_in_right(orig_x, orig_y):
            return -3.75 * 2 < orig_y < 0 and 18 < orig_x < 23

        def is_in_middle(orig_x, orig_y):
            return -18 < orig_y < 18 and -18 < orig_x < 18

        x = self.ego_dynamics['x']
        y = self.ego_dynamics['y']
        if self.task == 0:
            return True if not (is_in_initlane(x, y) or is_in_middle(x, y) or is_in_left(x, y)) else False
        elif self.task == 1:
            return True if not (is_in_initlane(x, y) or is_in_middle(x, y) or is_in_straight(x, y)) else False
        else:
            return True if not (is_in_initlane(x, y) or is_in_middle(x, y) or is_in_right(x, y)) else False


def test_grid3d():
    grid = Grid_3D(back_dist=20, forward_dist=40, half_width=20, number_x=120, number_y=40)
    index_x, index_y = grid.position2xyindex(0.56, 19.5)
    print('index_x=', str(index_x), 'index_y=', str(index_y))
    position_x, position_y = grid.xyindex2centerposition(0, 0)
    print('position_x=', str(position_x), 'position_y=', str(position_y))


if __name__ == '__main__':
    test_grid3d()
