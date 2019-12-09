import gym
from gym.utils import seeding
import math
import numpy as np
from endtoend_env_utils import shift_coordination, rotate_coordination, shift_and_rotate_coordination,\
    Path
from collections import OrderedDict, deque
import matplotlib.pyplot as plt
import bezier
from math import cos, sin, fabs, pi, sqrt
from traffic import Traffic
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
                 obs_type=2,  # 0:'vectors only', 1:'grids_plus_vecs', '2:fixed_grids_plus_vecs'
                 frameskip=1,
                 repeat_action_probability=0):
        metadata = {'render.modes': ['human']}
        self._obs_type = obs_type
        self.frameskip = frameskip
        self.detected_vehicles = None
        self.all_vehicles = None
        self.ego_dynamics = None
        self.road_related_info = None
        self.history_info = deque(maxlen=10)  # store infos of every step
        self.history_obs = deque(maxlen=10)  # store obs of every step
        self.init_state = []
        self.goal_state = []
        self.init_v = None
        self.goal_v = None
        self.path_info = None
        self.task = None  # used to decide goal
        self.action_number = 2
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.action_number,), dtype=np.float32)

        self.seed()
        self.step_length = 100  # ms
        self.traffic = Traffic(self.step_length)
        self.step_time = self.step_length / 1000.0
        self.reset()
        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(self._action_transformation_for_end2end(action))
        self._set_observation_space(observation)
        plt.ion()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):  # action is expected_acceleration
        acc, delta_phi = self._action_transformation_for_end2end(action)
        reward = 0
        done = 0
        all_info = None
        info_this_step = []
        for _ in range(self.frameskip):
            next_x, next_y, next_v, next_heading = self._get_next_position(acc, delta_phi)
            self.traffic.set_own_car(next_x, next_y, next_v, next_heading)
            self.traffic.sim_step()
            all_info = self._get_all_info()
            info_this_step.append(all_info)
            done_type, done = self._judge_done()
            self.history_obs.append(self._get_obs())
            reward += self.compute_reward(done_type)
            if done:
                self._print_done_info(done_type)
                break
        self.history_info.append(info_this_step)
        obs = self._process_obs()

        return obs, reward, done, all_info

    def _print_done_info(self, done_type):
        done_info = ['collision', 'break_road_constrain', 'good_done', 'not_done_yet']
        print(done_info[done_type-1], '\n')

    def reset(self, **kwargs):  # kwargs include three keys
        self.history_info.clear()
        self.history_obs.clear()
        self.goal_state = self._reset_goal_state()
        self.init_state = self._reset_init_state()
        self.traffic = Traffic(self.step_length)
        self.traffic.init(self.init_state)
        self._get_all_info()
        self.history_obs.append(self._get_obs())
        obs = self._process_obs()
        return obs

    def _process_obs(self):
        """this func should be override in subclass"""
        raise NotImplementedError

    def _reset_goal_state(self):  # decide center of goal area, [goal_x, goal_y, goal_a, goal_v]
        """this func should be override in subclass"""
        raise NotImplementedError

    def _reset_init_state(self):  # decide center of goal area, [init_x, init_y, init_a, init_v]
        """this func should be override in subclass"""
        raise NotImplementedError

    def _action_transformation_for_end2end(self, action):  # action = [acc, delta_phi]
        acc, delta_phi = action  # [0, 1]
        maximum_delta_phi = 3
        return acc * 4.5 - 3, (delta_phi - 0.5) * 2 * maximum_delta_phi

    def _get_next_position(self, acc, delta_phi):
        current_x = self.ego_dynamics['x']
        current_y = self.ego_dynamics['y']
        current_v = self.ego_dynamics['v']
        current_heading = self.ego_dynamics['heading']

        step_length = current_v * self.step_time
        next_v = np.clip(current_v + acc * self.step_time, 0, 15)
        next_x = current_x + step_length * cos(current_heading * pi / 180)
        next_y = current_y + step_length * sin(current_heading * pi / 180)
        delta_phi = 0 if step_length < 0.2 else delta_phi
        next_heading = current_heading + delta_phi
        return next_x, next_y, next_v, next_heading

    def _action_transformation_for_traj(self, action):  # action = [x, y(prop), x_a3, v(prop), v_a3]
        """assume action is in [0 ,1], and transform it to proper interval"""
        x = action[0]  # [0, 1]
        y_prop = action[1]  # [0, 1]
        x_a3 = action[2]  # [0, 1]
        v_prop = action[3]  # [0, 1]
        v_a3 = action[4]  # [0, 1]
        max_x = 10
        max_y = 5
        max_deltav = 1
        x = x * max_x  # [0, max_x]
        y_prop = (y_prop - 0.5) * 2  # [-1, 1]
        y = x * max_y / max_x * y_prop
        x_a3 = (x_a3 - 0.5) * 2 * 30  # range of x_a3 is not sure for now
        x_a2 = y / x**2 - x_a3 * x
        v_prop = (v_prop - 0.5) * 2  # [-1, 1]
        v = v_prop * max_deltav  # [-max_deltav, max_deltav]
        v_a3 = (v_a3 - 0.5) * 2 * 30  # range of x_a3 is not sure for now
        v_a2 = v / x**2 - v_a3 * x

        return [0, 0, x_a2, x_a3], [0, 0, v_a2, v_a3]

    def _generate_practical_trajectory(self, x_coeff, v_coeff):
        def x_formula(x, coeff=x_coeff):
            return coeff[0] + coeff[1] * x + coeff[2] * x**2 + coeff[3] * x**3

        def x_derivative_formula(x, coeff=x_coeff):
            return coeff[1] + coeff[2] * x + coeff[3] * x**2

        def v_formula(x, coeff=v_coeff):
            return coeff[0] + coeff[1] * x + coeff[2] * x**2 + coeff[3] * x**3

        current_v = self.ego_dynamics['v']

        def generate_next_point():
            pass

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def _get_all_info(self):  # used to update info, must be called every timestep before _get_obs
        # to fetch info
        self.all_vehicles = self.traffic.vehicles  # coordination 2
        self.ego_dynamics = self.traffic.ego_info  # coordination 2
        # ego_dynamics
        # {'x': self.x,
        #  'y': self.y,
        #  'v': self.v,
        #  'heading': self.heading,  # (deg)
        #  'Car_length',
        #  'Car_width',
        #  'Corner_point'
        #  }

        # all_vehicles
        # dict(x=x, y=y, v=v, angle=a, length=length,
        #      width=width, route=route, edge_index=edge_index)

        all_info = dict(all_vehicles=self.all_vehicles,
                        ego_dynamics=self.ego_dynamics,
                        goal_state=self.goal_state)
        return all_info

    def _get_obs(self):
        """this func should be override in subclass to generate different types of observation"""
        raise NotImplementedError

    def render(self, mode='human'):
        if mode == 'human':
            # plot basic map
            square_length = 36
            extension = 20
            lane_width = 3.75
            dotted_line_style = '--'

            plt.cla()
            plt.title("Demo")
            ax = plt.axes(xlim=(-square_length / 2 - extension, square_length / 2 + extension),
                          ylim=(-square_length / 2 - extension, square_length / 2 + extension))
            plt.axis("equal")

            # ax.add_patch(plt.Rectangle((-square_length / 2, -square_length / 2),
            #                            square_length, square_length, edgecolor='black', facecolor='none'))
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

            # ----------stop line--------------
            plt.plot([0, 2 * lane_width], [-square_length / 2, -square_length / 2],
                     color='black')
            plt.plot([-2 * lane_width, 0], [square_length / 2, square_length / 2],
                     color='black')
            plt.plot([-square_length / 2, -square_length / 2], [0, -2 * lane_width],
                     color='black')
            plt.plot([square_length / 2, square_length / 2], [2 * lane_width, 0],
                     color='black')

            # ----------Oblique--------------
            plt.plot([2 * lane_width, square_length / 2], [-square_length / 2, -2 * lane_width],
                     color='black')
            plt.plot([2 * lane_width, square_length / 2], [square_length / 2, 2 * lane_width],
                     color='black')
            plt.plot([-2 * lane_width, -square_length / 2], [-square_length / 2, -2 * lane_width],
                     color='black')
            plt.plot([-2 * lane_width, -square_length / 2], [square_length / 2, 2 * lane_width],
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
            ego_l = self.ego_dynamics['Car_length']
            ego_w = self.ego_dynamics['Car_width']
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
    def compute_reward(self, done_type):
        """this func should be override in subclass"""
        raise NotImplementedError

    def _judge_done(self):
        """
        :return:
         1: bad done: collision
         2: bad done: break_road_constrain
         3: good done: task succeed
         4: not done
        """
        if self.traffic.collision_flag:
            return 1, 1
        elif self._break_road_constrain():
            return 2, 1
        elif self._is_achieve_goal():
            return 3, 1
        else:
            return 4, 0

    def _break_road_constrain(self):
        """this func should be override in subclass"""
        raise NotImplementedError

    def _is_achieve_goal(self):  # decide goal area and if ego car is in it
        """this func should be override in subclass"""
        raise NotImplementedError


class Grid_3D(object):
    """
    Consider coordination of ego car
    """

    def __init__(self, back_dist, forward_dist, half_width, number_x, number_y, axis_z_type='single_layer'):
        self.back_dist = back_dist
        self.forward_dist = forward_dist
        self.half_width = half_width
        self.length = self.back_dist + self.forward_dist
        self.width = 2 * self.half_width
        self.number_x = number_x  # in car coordination ! not in matrix index
        self.number_y = number_y
        self.matrix_x = self.number_y
        self.matrix_y = self.number_x
        self.axis_z_name_dict = self.get_axis_z_name_dict(axis_z_type)
        self.number_z = len(self.axis_z_name_dict)
        self.increment_x = (back_dist+forward_dist)/self.number_x  # increment in x axis of car coordination
        self.increment_y = 2*half_width/self.number_y
        self._encode_grid = np.ones((self.number_z, self.matrix_x, self.matrix_y), dtype=np.float32) * 255.
        self._encode_grid_flag = np.zeros((self.number_z, self.matrix_x, self.matrix_y), dtype=np.int)

    def get_axis_z_name_dict(self, axis_z_type):  # dict from name to index
        if axis_z_type == 'multi_layers':
            return dict(position_x=0,
                        position_y=1,
                        v=2,
                        heading=3,
                        length=4,
                        width=5)
        elif axis_z_type == 'single_layer':
            return dict(feasible=0)

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

    def set_value(self, index_z, index_x, index_y, grid_value, range_number):
        min_index_x = max(0, index_x - range_number)
        max_index_x = min(self.matrix_x - 1, index_x + range_number)
        min_index_y = max(0, index_y - range_number)
        max_index_y = min(self.matrix_y - 1, index_y + range_number)
        for x in range(min_index_x, max_index_x+1):
            for y in range(min_index_y, max_index_y+1):
                self._encode_grid[index_z][x][y] = grid_value
                self._encode_grid_flag[index_z][x][y] = 1

    def set_same_xy_value_in_all_z(self, index_x, index_y, grid_value):
        for i in range(self.number_z):
            self.set_value(i, index_x, index_y, grid_value, 0)

    def set_xy_value_with_list(self, index_x, index_y, z_list):
        assert len(z_list) == self.number_z
        for i in range(self.number_z):
            self.set_value(i, index_x, index_y, z_list[i], 0)

    def get_value(self, index_z, index_x, index_y):
        return self._encode_grid[index_z][index_x][index_y], \
               self._encode_grid_flag[index_z][index_x][index_y]

    def get_encode_grid_and_flag(self):
        return self._encode_grid, self._encode_grid_flag


def judge_feasible_for_pre_grid(orig_x, orig_y):  # map dependant TODO
    # return True if -900 < orig_x < 900 and -150 - 3.75 * 4 < orig_y < -150 else False
    def is_in_straight(orig_x, orig_y):
        return -3.75 * 2 < orig_x < 3.75 * 2 or -3.75 * 2 < orig_y < 3.75 * 2

    def is_in_middle(orig_x, orig_y):
        return -18 < orig_y < 18 and -18 < orig_x < 18

    return True if is_in_straight(orig_x, orig_y) or is_in_middle(orig_x, orig_y) else False


def judge_feasible(orig_x, orig_y):  # map dependant TODO
    # return True if -900 < orig_x < 900 and -150 - 3.75 * 4 < orig_y < -150 else False
    def is_in_straight_before(orig_x, orig_y):
        return 0 < orig_x < 3.75 * 2 and orig_y <= -18

    def is_in_straight_after(orig_x, orig_y):
        return 0 < orig_x < 3.75 * 2 and orig_y >= 18

    def is_in_left(orig_x, orig_y):
        return 0 < orig_y < 3.75 * 2 and orig_x < -18

    def is_in_right(orig_x, orig_y):
        return -3.75 * 2 < orig_y < 0 and orig_x > 18

    def is_in_middle(orig_x, orig_y):
        if -18 < orig_y < 18 and -18 < orig_x < 18:
            if -3.75*2 < orig_x < 3.75*2:
                return True if -18 < orig_y < 18 else False
            elif orig_x > 3.75*2:
                return True if orig_x - (18+3.75*2) < orig_y < -orig_x + (18+3.75*2) else False
            else:
                return True if -orig_x - (18 + 3.75 * 2) < orig_y < orig_x + (18 + 3.75 * 2) else False
        else:
            return False

    # judge feasible for turn left
    return True if is_in_straight_before(orig_x, orig_y) or is_in_left(orig_x, orig_y) \
                   or is_in_middle(orig_x, orig_y) else False


class CrossroadEnd2end(End2endEnv):
    def __init__(self,
                 obs_type=1,  # 0:'vectors only', 1:'grids_plus_vecs', '2:fixed_grids_plus_vecs'
                 frameskip=4,
                 repeat_action_probability=0
                 ):
        self.history_number = 4
        self.history_frameskip = 1
        if obs_type == 1 or obs_type == 2:
            # self.grid_setting_dict = dict(fill_type='cover_but_no_different_layers',  # single or cover
            #                               size_list=[dict(back_dist=30, forward_dist=30, half_width=30)],
            #                               number_x=150,
            #                               number_y=150,
            #                               axis_z_type='single_layer')
            self.grid_setting_dict = dict(fill_type='cover_but_no_different_layers',  # single or cover
                                          size_list=[dict(back_dist=20, forward_dist=20, half_width=20)],
                                          number_x=160,
                                          number_y=160,
                                          axis_z_type='single_layer')

            self.grid_3d = None
            self.grid_fill_type = self.grid_setting_dict['fill_type']
            self.grid_size_list = self.grid_setting_dict['size_list']
            self.grid_number_x = self.grid_setting_dict['number_x']
            self.grid_number_y = self.grid_setting_dict['number_y']
            self.gird_axis_z_type = self.grid_setting_dict['axis_z_type']
            self._FEASIBLE_VALUE = 255
            self._INFEASIBLE_VALUE = 0
            self.grid_precision_x = (self.grid_size_list[0]['back_dist']+self.grid_size_list[0]['forward_dist'])/self.grid_number_x
            self.grid_precision_y = (2*self.grid_size_list[0]['half_width'])/self.grid_number_y

            self.pre_grid_list = []

            if obs_type == 2:
                self.prediction_time = 3  # unit: s
                self.prediction_frameskip = 4

            def make_pre_grid(grid, infeasible_value, feasible_value):
                for index_x in range(grid.matrix_x):
                    for index_y in range(grid.matrix_y):
                        x, y = grid.xyindex2centerposition(index_x, index_y)
                        if judge_feasible_for_pre_grid(x, y):
                            grid.set_value(0, index_x, index_y, feasible_value, 0)
                        else:
                            grid.set_value(0, index_x, index_y, infeasible_value, 0)

            for size_dict in self.grid_size_list:
                pre_grid_3d = Grid_3D(size_dict['back_dist'], size_dict['forward_dist'],
                                      size_dict['half_width'], self.grid_number_x,
                                      self.grid_number_y, self.gird_axis_z_type)
                if obs_type == 2:
                    make_pre_grid(pre_grid_3d, self._INFEASIBLE_VALUE, self._FEASIBLE_VALUE)
                self.pre_grid_list.append(pre_grid_3d)

        super(CrossroadEnd2end, self).__init__(obs_type, frameskip)

    def _get_obs(self):
        if self._obs_type == 1 or self._obs_type == 2:  # type 2: fixed grid + vector
            return dict(grid=self._3d_grid_v2x_no_noise_obs_encoder(self._obs_type),
                        vector=self._vector_supplement_for_grid_encoder())
        elif self._obs_type == 0:
            return self._vector_supplement_for_grid_encoder()

    def _process_obs(self):
        # history
        history_len = len(self.history_obs)
        history_obs_index = [0] + [-i*self.history_frameskip if i*self.history_frameskip < history_len
                             else -history_len+1 for i in range(1, self.history_number)]
        history_obs_index.reverse()
        history_obs_index = np.array(history_obs_index) - 1
        if self._obs_type == 1 or self._obs_type == 2:
            history_grids_list, history_vectors_list = [self.history_obs[i]['grid'] for i in history_obs_index],\
                                                       [self.history_obs[i]['vector'] for i in history_obs_index]
            history_grids = np.concatenate(history_grids_list, axis=0)
            ob_vector = np.concatenate(history_vectors_list, axis=0)
            if self._obs_type == 1:
                return dict(grid=history_grids,
                            vector=ob_vector)
            else:
                future_grids = self._3d_grid_v2x_no_noise_obs_encoder(self._obs_type, prediction=True)[1:]
                ob_grid = np.concatenate((history_grids, future_grids), axis=0)
                return dict(grid=ob_grid,
                            vector=ob_vector)
        elif self._obs_type == 0:
            history_vectors_list = [self.history_obs[i] for i in history_obs_index]
            ob_vector = np.concatenate(history_vectors_list, axis=0)
            return ob_vector

    def _break_traffic_rule(self):  # TODO: hard coded
        # judge traffic light breakage
        if len(self.history_info):
            history_vertical_light = self.history_info[-1][-1]['road_related_info']['vertical_light_value']
            history_y = self.history_info[-1][-1]['ego_dynamics']['y']
            current_vertical_light = self.road_related_info['vertical_light_value']
            current_y = self.ego_dynamics['y']
            traffic_light_breakage = history_vertical_light == 0 and current_vertical_light == 0 and \
                                     history_y < -18 < current_y
            # judge speed limit
            speed_limit = self.road_related_info['current_lane_speed_limit']
            v = self.ego_dynamics['v']
            speed_breakage = speed_limit < v
            return traffic_light_breakage or speed_breakage
        else:
            return False

    def _vector_supplement_for_grid_encoder(self):  # func for supplement vector of grid
        # encode road structure and task related info, hard coded for now TODO

        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_v = self.ego_dynamics['v']
        ego_heading = self.ego_dynamics['heading']
        ego_length = self.ego_dynamics['Car_length']
        ego_width = self.ego_dynamics['Car_width']
        # calculate relative goal
        goal_x, goal_y, goal_v, goal_a = self.goal_state
        rela_goal_x, rela_goal_y, rela_goal_a = shift_and_rotate_coordination(goal_x, goal_y, goal_a,
                                                                              ego_x, ego_y, ego_heading)
        # vehicle related
        vehs_vector = []
        all_vehicles = self._v2x_unify_format_for_3dgrid()
        info_in_ego_coordination = self._cal_info_in_transform_coordination(all_vehicles, ego_x, ego_y, ego_heading)
        tmp = [(idx, sqrt(veh['trans_x']**2+veh['trans_y']**2))
               for idx, veh in enumerate(info_in_ego_coordination)]
        tmp.sort(key=lambda x: x[1])
        n = 3
        len_veh = len(tmp)
        for i in range(n):
            if i < len_veh:
                veh = info_in_ego_coordination[tmp[i][0]]
                vehs_vector.extend([veh['trans_x'], veh['trans_y'], veh['trans_v'],
                                    veh['trans_heading']*pi/180])
            else:
                vehs_vector.extend([100, 100, 0, 0, 5, 2.5])

        vehs_vector = np.array(vehs_vector)

        # map related
        key_points_vector = []
        key_points = [(-7.5, 18), (0, 18), (7.5, 18),
                      (-7.5, -18), (0, -18), (7.5, -18),
                      (-18, -7.5), (-18, 0), (-18, 7.5),
                      (18, -7.5), (18, 0), (18, 7.5)]
        transfered_key_points = list(map(lambda x: shift_and_rotate_coordination(*x, 0, ego_x, ego_y, ego_heading)[0: 2],
                                         key_points))

        for key_point in transfered_key_points:
            key_points_vector.extend([key_point[0], key_point[1]])

        key_points_vector = np.array(key_points_vector)

        # construct vector dict
        vector_dict = dict(ego_x=ego_x,
                           ego_y=ego_y,
                           ego_v=ego_v,
                           ego_heading=ego_heading*pi/180,
                           ego_length=ego_length,
                           ego_width=ego_width,
                           rela_goal_x=rela_goal_x,
                           rela_goal_y=rela_goal_y,
                           rela_goal_a=rela_goal_a*pi/180,
                           goal_v=goal_v,
                           )
        _ = np.array(list(vector_dict.values()))
        vector = np.concatenate((_, vehs_vector, key_points_vector), axis=0)
        return vector

    def _3d_grid_v2x_no_noise_obs_encoder(self, obs_type, prediction=False):  # func for grid encoder
        import copy
        encoded_size_list = []
        all_vehicles = self._v2x_unify_format_for_3dgrid()
        if obs_type == 2:
            info_in_transformed_coordination = self._cal_info_in_transform_coordination(all_vehicles, 0, 0, 0)  # hard coded
            if prediction:
                for pre_grid in self.pre_grid_list:
                    self.grid_3d = copy.deepcopy(pre_grid)
                    encoded_time_list = []
                    vehicles_in_grid = [veh for veh in info_in_transformed_coordination if
                                        self.grid_3d.is_in_2d_grid(veh['trans_x'], veh['trans_y'])]
                    future_list_of_vehs = [self._prediction(veh, int(self.prediction_time/self.step_time))
                                           for veh in vehicles_in_grid]
                    for timestep in range(0, int(self.prediction_time/self.step_time), self.prediction_frameskip):
                        self.grid_3d = copy.deepcopy(pre_grid)
                        vehs_of_this_timestep = [veh_future[timestep] for veh_future in future_list_of_vehs]
                        vehicles_in_grid = [veh for veh in vehs_of_this_timestep if
                                            self.grid_3d.is_in_2d_grid(veh['trans_x'], veh['trans_y'])]
                        self._add_vehicle_info_in_grid(vehicles_in_grid)
                        encoded_time_list.append(self.grid_3d.get_encode_grid_and_flag()[0])
                    grid_this_size = np.concatenate(encoded_time_list, axis=0)
                    # self._add_feasible_area_info_in_grid(recover_orig_position_fn)
                    encoded_size_list.append(grid_this_size)
            else:
                for pre_grid in self.pre_grid_list:
                    self.grid_3d = copy.deepcopy(pre_grid)
                    vehicles_in_grid = [veh for veh in info_in_transformed_coordination if
                                        self.grid_3d.is_in_2d_grid(veh['trans_x'], veh['trans_y'])]
                    self._add_vehicle_info_in_grid(vehicles_in_grid)
                    encoded_size_list.append(self.grid_3d.get_encode_grid_and_flag()[0])
            return encoded_size_list[0]
        elif obs_type == 1:
            ego_x, ego_y, ego_a = self.ego_dynamics['x'], self.ego_dynamics['y'], self.ego_dynamics['heading']
            info_in_transformed_coordination = self._cal_info_in_transform_coordination(all_vehicles, ego_x, ego_y, ego_a)
            for pre_grid in self.pre_grid_list:
                self.grid_3d = copy.deepcopy(pre_grid)
                vehicles_in_grid = [veh for veh in info_in_transformed_coordination if
                                    self.grid_3d.is_in_2d_grid(veh['trans_x'], veh['trans_y'])]
                self._add_vehicle_info_in_grid(vehicles_in_grid)
                encoded_size_list.append(self.grid_3d.get_encode_grid_and_flag()[0])
            return encoded_size_list[0]


    def _route2behavior(self, route_list, edge_index):  # map dependent
        start = None
        end = None
        if route_list[edge_index] in ['-gneE0', 'gneE3', 'gneE1', 'gneE2']:
            start = route_list[edge_index-1]
            end = route_list[edge_index]
        else:
            start = route_list[edge_index]
            end = route_list[edge_index+1]
        if (start, end) in [('-gneE3', '-gneE0'), ('-gneE1', 'gneE3'), ('-gneE2', 'gneE1'), ('gneE12', 'gneE2')]:
            return 0, start, end  # turn left
        elif (start, end) in [('-gneE3', 'gneE2'), ('-gneE1', '-gneE0'), ('-gneE2', 'gneE3'), ('gneE12', 'gneE1')]:
            return 1, start, end  # go straight
        else:
            return 2, start, end  # turn right

    def _prediction(self, veh, timesteps):
        x = veh['trans_x']
        y = veh['trans_y']
        v = veh['trans_v']
        heading = veh['trans_heading']
        route_list = veh['route']
        edge_index = veh['edge_index']
        behavior, start, end = self._route2behavior(route_list, edge_index)
        start2rotation = {'-gneE3': 0, '-gneE1': 90, '-gneE2': 180, 'gneE12': -90}  # map dependent

        trans_x, trans_y, trans_heading = rotate_coordination(x, y, heading, start2rotation[start])
        if behavior == 0:  # left
            if trans_y < -18:
                future_traj = [(trans_x, trans_y + v * self.step_time * i, 90) for i in range(1, timesteps+1)]
            elif trans_x < -18:
                future_traj = [(trans_x - v * self.step_time * i, trans_y, 180) for i in range(1, timesteps + 1)]
            else:
                assert trans_x > -18 and trans_y > -18

                def turn_right_path_generator(startx, starty, starta, v, step_time, timesteps):
                    x = startx
                    y = starty
                    a = starta * math.pi / 180
                    step_length = v * step_time
                    for i in range(timesteps):
                        x = x + step_length * math.cos(a)
                        y = y + step_length * math.sin(a)
                        a = a + step_length / 19.875
                        a = a if 0 < a < math.pi else math.pi
                        yield (x, y, a * 180 / math.pi)
                future_traj = list(turn_right_path_generator(trans_x, trans_y, trans_heading, v, self.step_time, timesteps))
        elif behavior == 1:  # go straight
            future_traj = [(trans_x, trans_y + v * self.step_time * i, 90) for i in range(1, timesteps + 1)]
        else:
            if trans_y < -18:
                future_traj = [(trans_x, trans_y + v * self.step_time * i, 90) for i in range(1, timesteps+1)]
            elif trans_x > 18:
                future_traj = [(trans_x + v * self.step_time * i, trans_y, 0) for i in range(1, timesteps + 1)]
            else:
                assert trans_x < 18 and trans_y > -18

                def turn_left_path_generator(startx, starty, starta, v, step_time, timesteps):
                    x = startx
                    y = starty
                    a = starta * math.pi / 180
                    step_length = v * step_time
                    for i in range(timesteps):
                        x = x + step_length * math.cos(a)
                        y = y + step_length * math.sin(a)
                        a = a - step_length / 12.375
                        a = a if 0 < a < math.pi else 0
                        yield (x, y, a * 180 / math.pi)
                future_traj = list(turn_left_path_generator(trans_x, trans_y, trans_heading, v, self.step_time, timesteps))
        future_traj = list(map(lambda x: rotate_coordination(*x, -start2rotation[start]), future_traj))
        future_data = [{'trans_x': point[0], 'trans_y': point[1], 'trans_v': v, 'trans_heading': point[2],
                        'length': veh['length'], 'width': veh['width'], 'route': veh['route'],
                        'edge_index': veh['edge_index']} for point in future_traj]
        temp = [veh]
        temp.extend(future_data)
        return temp

    def _v2x_unify_format_for_3dgrid(self):  # unify output format
        results = []
        for veh in range(len(self.all_vehicles)):
            results.append({'x': self.all_vehicles[veh]['x'],
                            'y': self.all_vehicles[veh]['y'],
                            'v': self.all_vehicles[veh]['v'],
                            'heading': self.all_vehicles[veh]['angle'],
                            'width': self.all_vehicles[veh]['width'],
                            'length': self.all_vehicles[veh]['length'],
                            'route': self.all_vehicles[veh]['route'],
                            'edge_index': self.all_vehicles[veh]['edge_index']})
        return results

    def _cal_info_in_transform_coordination(self, filtered_objects, x, y, rotate_d):  # rotate_d is positive if anti
        results = []
        # ego_x = self.ego_dynamics['x']
        # ego_y = self.ego_dynamics['y']
        # ego_v = self.ego_dynamics['v']
        # ego_heading = self.ego_dynamics['heading']

        for obj in filtered_objects:
            orig_x = obj['x']
            orig_y = obj['y']
            orig_v = obj['v']
            orig_heading = obj['heading']
            width = obj['width']
            length = obj['length']
            route = obj['route']
            edge_index = obj['edge_index']
            shifted_x, shifted_y = shift_coordination(orig_x, orig_y, x, y)
            trans_x, trans_y, trans_heading = rotate_coordination(shifted_x, shifted_y, orig_heading, rotate_d)
            trans_v = orig_v
            results.append({'trans_x': trans_x,
                            'trans_y': trans_y,
                            'trans_v': trans_v,
                            'trans_heading': trans_heading,
                            'width': width,
                            'length': length,
                            'route': route,
                            'edge_index': edge_index})
        return results

    def recover_orig_position_fn(self, transformed_x, transformed_y, x, y, d):  # x, y, d are used to transform
                                                                                # coordination
        transformed_x, transformed_y, _ = rotate_coordination(transformed_x, transformed_y, 0, -d)
        orig_x, orig_y = shift_coordination(transformed_x, transformed_y, -x, -y)
        return orig_x, orig_y

    def _add_vehicle_info_in_grid(self, vehicles_in_grid):
        cover_range = 1
        if self.grid_fill_type == 'single':
            for veh in vehicles_in_grid:
                x = veh['trans_x']
                y = veh['trans_y']
                v = veh['trans_v']
                heading = veh['trans_heading']
                length = veh['length']
                width = veh['width']
                index_x, index_y = self.grid_3d.position2xyindex(x, y)
                self.grid_3d.set_value(index_z=0, index_x=index_x, index_y=index_y, grid_value=x, range=cover_range)
                self.grid_3d.set_value(index_z=1, index_x=index_x, index_y=index_y, grid_value=y, range=cover_range)
                self.grid_3d.set_value(index_z=2, index_x=index_x, index_y=index_y, grid_value=v, range=cover_range)
                self.grid_3d.set_value(index_z=3, index_x=index_x, index_y=index_y, grid_value=heading, range=cover_range)
                self.grid_3d.set_value(index_z=4, index_x=index_x, index_y=index_y, grid_value=length, range=cover_range)
                self.grid_3d.set_value(index_z=5, index_x=index_x, index_y=index_y, grid_value=width, range=cover_range)

        elif self.grid_fill_type == 'cover_but_no_different_layers':
            for veh in vehicles_in_grid:
                x = veh['trans_x']
                y = veh['trans_y']
                v = veh['trans_v']
                heading = veh['trans_heading']
                length = veh['length']
                width = veh['width']
                self.grid_of_veh = Grid_3D(length/2, length/2, width/2, int(length/self.grid_precision_x),
                                           int(width/self.grid_precision_x), self.gird_axis_z_type)
                for i in range(self.grid_of_veh.matrix_x):
                    for j in range(self.grid_of_veh.matrix_y):
                        grid_x_in_veh_coordi, grid_y_in_veh_coordi = self.grid_of_veh.xyindex2centerposition(i, j)
                        grid_x, grid_y = self.recover_orig_position_fn(grid_x_in_veh_coordi,
                                                                       grid_y_in_veh_coordi, x, y, heading)
                        index_x, index_y = self.grid_3d.position2xyindex(grid_x, grid_y)
                        self.grid_3d.set_value(index_z=0, index_x=index_x, index_y=index_y,
                                               grid_value=self._INFEASIBLE_VALUE, range_number=cover_range)

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

    def _break_road_constrain(self):
        results = list(map(lambda x: judge_feasible(*x), self.ego_dynamics['Corner_point']))
        return not all(results)

    def _is_achieve_goal(self):  # for now, only support turn left with the specific map
        goal_x, goal_y, goal_v, goal_a = self.goal_state
        x = self.ego_dynamics['x']
        y = self.ego_dynamics['y']
        return True if goal_x-2 < x < goal_x+2 and goal_y-3.75 < y < goal_y+3.75 else False

    def _reset_goal_state(self):  # decide center of goal area, [goal_x, goal_y, goal_a, goal_v]
        return [-18-4, 3.75, 8, 180]

    def _reset_init_state(self):
        nodes1 = np.asfortranarray([[3.75 / 2, 3.75 / 2, -18 + 10, -18],
                                    [-18 - 10, -18 + 18, 3.75 / 2, 3.75 / 2]])
        curve1 = bezier.Curve(nodes1, degree=3)
        nodes2 = np.asfortranarray([[3.75/2, 3.75/2, -18+10, -18],
                                    [-18-10, -18+18, 3.75*3/2, 3.75*3/2]])
        curve2 = bezier.Curve(nodes2, degree=3)
        start_point = None
        if np.random.random() > 0.5:
            start_point = curve1.evaluate(np.random.random())
        else:
            start_point = curve2.evaluate(np.random.random())
        x, y = start_point[0][0], start_point[1][0]
        if y < -18:
            a = 90.
        else:
            a = 90. + math.atan((y+18)/(x+18))*180/math.pi
        v = np.random.random()*7+5
        return [x, y, v, a]

    def _cal_collision_reward(self):  # can be override to do an analytic calculation
        return -100

    def _cal_achievegoal_reward(self):  # can be override to do an analytic calculation
        x, y, a, v = self.ego_dynamics['x'], self.ego_dynamics['y'], \
                     self.ego_dynamics['heading'], self.ego_dynamics['v']

        goal_x, goal_y, goal_v, goal_a = self.goal_state
        position_punishment = -5 * min(fabs(y - 3.75 / 2), fabs(y - 3.75 * 3 / 2))
        heading_punishment = -fabs(a - goal_a)
        return 100 + position_punishment + heading_punishment

    def _cal_step_reward(self):
        # data preparation
        x, y, a, v = self.ego_dynamics['x'], self.ego_dynamics['y'], self.ego_dynamics['heading'], self.ego_dynamics['v']
        goal_x, goal_y, goal_v, goal_a = self.goal_state
        dist_to_goal = math.sqrt((goal_x-x)**2+(goal_y-y)**2)
        v_difference = math.fabs(goal_v-v)
        nodes1 = np.asfortranarray([[3.75 / 2, 3.75 / 2, -18 + 10, -18],
                                    [-18 - 10, -18 + 18, 3.75 / 2, 3.75 / 2]])
        curve1 = bezier.Curve(nodes1, degree=3)
        nodes2 = np.asfortranarray([[3.75 / 2, 3.75 / 2, -18 + 10, -18],
                                    [-18 - 10, -18 + 18, 3.75 * 3 / 2, 3.75 * 3 / 2]])
        curve2 = bezier.Curve(nodes2, degree=3)
        s_vals = np.linspace(0, 1.0, 300)
        data1 = curve1.evaluate_multi(s_vals)
        data2 = curve2.evaluate_multi(s_vals)
        data1x, data1y = data1[0], data1[1]
        data2x, data2y = data2[0], data2[1]
        min_dist_to_curve1 = min(np.sqrt((data1x - x) ** 2 + (data1y - y) ** 2))
        min_dist_to_curve2 = min(np.sqrt((data2x - x) ** 2 + (data2y - y) ** 2))
        min_dist_to_curve = min(min_dist_to_curve1, min_dist_to_curve2)
        current_vector = None
        if self._obs_type == 0:
            current_vector = self.history_obs[-1]
        elif self._obs_type == 1 or self._obs_type == 2:
            current_vector = self.history_obs[-1]['vector']
        veh1x, veh1y = current_vector[10], current_vector[11]
        veh2x, veh2y = current_vector[16], current_vector[17]
        dist_to_veh1 = sqrt(veh1x**2 + veh1y**2)
        dist_to_veh2 = sqrt(veh2x**2 + veh2y**2)


        # step punishment
        reward = -1
        # goal position punishment
        reward -= dist_to_goal * 0.005
        # goal velocity punishment
        reward -= v_difference * 0.05
        # standard curve punishment
        reward -= min_dist_to_curve * 0.05
        # distance to other vehicle punishment
        reward -= 1/abs(dist_to_veh1-3) * 0.1
        # print('dist_to_goal', dist_to_goal, '   v_difference', v_difference,
        #       '   min_dist_to_curve', min_dist_to_curve, '   min_dist_to_veh', 1/abs(min_dist_to_veh-3))
        return reward

    def compute_reward(self, done_type):
        reward = 0
        if done_type == 4:
            return self._cal_step_reward()
        elif done_type == 3:
            return self._cal_achievegoal_reward()
        else:
            return self._cal_collision_reward()


def test_grid3d():
    grid = Grid_3D(back_dist=20, forward_dist=40, half_width=20, percision=0.3)
    index_x, index_y = grid.position2xyindex(0.56, 19.5)
    print('index_x=', str(index_x), 'index_y=', str(index_y))
    position_x, position_y = grid.xyindex2centerposition(0, 0)
    print('position_x=', str(position_x), 'position_y=', str(position_y))

def test_crossrode():
    env = CrossroadEnd2end()
    veh = {'trans_x': 1.875, 'trans_y': -17, 'trans_v': 7, 'trans_heading': 120,
           'route': ['-gneE3', '-gneE0'], 'edge_index': 0}
    veh = {'trans_x': 17, 'trans_y': 1.875, 'trans_v': 7, 'trans_heading': -160,
           'route': ['-gneE1', 'gneE3'], 'edge_index': 0}
    veh = {'trans_x': 17, 'trans_y': 1.875, 'trans_v': 7, 'trans_heading': -180,
           'route': ['-gneE1', '-gneE0'], 'edge_index': 0}
    veh = {'trans_x': 17, 'trans_y': 1.875*3, 'trans_v': 7, 'trans_heading': -180,
           'route': ['-gneE1', 'gneE2'], 'edge_index': 0}
    veh = {'trans_x': -1.875, 'trans_y': 19, 'trans_v': 7, 'trans_heading': -90,
           'route': ['-gneE2', 'gneE1'], 'edge_index': 0}
    veh = {'trans_x': -1.875, 'trans_y': 15, 'trans_v': 7, 'trans_heading': -70,
           'route': ['-gneE2', 'gneE1'], 'edge_index': 0}
    veh = {'trans_x': -1.875, 'trans_y': 15, 'trans_v': 7, 'trans_heading': -110,
           'route': ['-gneE2', '-gneE0'], 'edge_index': 0}
    veh = {'trans_x': -19, 'trans_y': -1.875, 'trans_v': 7, 'trans_heading': -10,
           'route': ['gneE12', 'gneE2'], 'edge_index': 0}
    veh = {'trans_x': -17, 'trans_y': -1.875, 'trans_v': 7, 'trans_heading': 20,
           'route': ['gneE12', 'gneE2'], 'edge_index': 0}
    veh = {'trans_x': -17, 'trans_y': -1.875, 'trans_v': 7, 'trans_heading': -10,
           'route': ['gneE12', 'gneE3'], 'edge_index': 0}
    future_traj = env._prediction(veh, 20)
    x = [a[0] for a in future_traj]
    y = [a[1] for a in future_traj]
    plt.plot(x, y)
    plt.show()
    plt.axis("equal")
    plt.pause(10)



if __name__ == '__main__':
    test_crossrode()
