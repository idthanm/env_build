import gym
from LasVSim import lasvsim
from gym.utils import seeding
import math
import numpy as np
from LasVSim.endtoend_env_utils import shift_coordination, rotate_coordination, shift_and_rotate_coordination,\
    Path
from collections import OrderedDict, deque
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
        self.road_related_info = None
        self.history_info = deque(maxlen=10)  # store infos of every step
        self.history_obs = deque(maxlen=10)  # store obs of every step
        self.simulation = None
        self.init_state = []
        self.init_v = None
        self.goal_v = None
        self.path_info = None
        self.task = None  # used to decide goal
        self.action_space = gym.spaces.Box(np.array([-1]), np.array([1]), dtype=np.float32)
        self.simulation = lasvsim.create_simulation(self.setting_path + 'simulation_setting_file.xml')
        self.seed()
        self.step_length = self.simulation.step_length
        self.path = Path(0.01)
        self.reset()
        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(self._action_transformation(action))
        self._set_observation_space(observation)
        plt.ion()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        lasvsim.seed(int(seed))
        return [seed]

    def step(self, action):  # action is expected_acceleration
        acc = self._action_transformation(action[0])
        reward = 0
        done = 0
        all_info = None
        info_this_step = []
        for _ in range(self.frameskip):
            next_x, next_y, next_v, next_heading = self._get_next_position(acc)
            lasvsim.set_ego_position(next_x, next_y, next_v, next_heading)
            lasvsim.sim_step()
            all_info = self._get_all_info()
            info_this_step.append(all_info)
            done_type, done = self._judge_done()
            reward += self.compute_reward(done_type)
            if done:
                self._print_done_info(done_type)
                break
        self.history_info.append(info_this_step)
        obs = self._get_obs()

        return obs, reward, done, all_info

    def _print_done_info(self, done_type):
        done_info = ['collision', 'break_traffic_rule', 'good_done', 'not_done_yet']
        print(done_info[done_type-1])


    def reset(self, **kwargs):  # kwargs include three keys
        self.history_info.clear()
        self.history_obs.clear()
        default_path = dict(dist_before_start_point=10,
                            start_point_info=(3.75/2, -18, 90),
                            end_point_info=(-18, 3.75/2, 180),
                            dist_after_end_point=5)
        if kwargs:
            self.path_info, self.init_v, self.goal_v = kwargs['path_info'], kwargs['init_v'], kwargs['goal_v']
        else:
            self.path_info, self.init_v, self.goal_v = default_path, 0, 30/3.6

        self.path.reset_path(**self.path_info)
        # reset simulation
        x, y, a = self.path.get_init_state()
        self.init_state = [x, y, self.init_v, a]
        lasvsim.reset_simulation(overwrite_settings={'init_state': self.init_state},
                                 init_traffic_path=self.setting_path)
        self._get_all_info()
        obs = self._get_obs()
        return obs

    def _action_transformation(self, action):
        """assume action is in [-1, 1], and transform it to proper interval"""
        return action * 3.75 - 1.25

    def _get_next_position(self, acc):
        delta_dist = np.clip(
            self.ego_dynamics['v'] * self.step_length / 1000 + 0.5 * acc * (self.step_length / 1000) ** 2, 0, 10)
        next_v = np.clip(self.ego_dynamics['v'] + acc * self.step_length / 1000, 0, 33)
        next_x, next_y, next_heading = self.path.get_next_info(delta_dist)
        return next_x, next_y, next_v, next_heading

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def _get_all_info(self):  # used to update info, must be called every timestep before _get_obs
        # to fetch info
        self.all_vehicles = lasvsim.get_all_objects()  # coordination 2
        self.detected_vehicles = lasvsim.get_detected_objects()  # coordination 2
        self.ego_dynamics, self.road_related_info = lasvsim.get_self_car_info()
        # ego_dynamics
        # {'x': self.x,
        #  'y': self.y,
        #  'v': self.v,
        #  'heading': self.heading,  # (deg)
        #  'Car_length',
        #  'Car_width',
        #  'Corner_point'}

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
                        road_related_info=self.road_related_info,
                        goal_state=self.path_info)
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
        reward = 0
        if done_type == 4:
            reward += -1
            reward += -0.05*abs(self.ego_dynamics['v']-self.goal_v)
            return reward
        elif done_type == 3:
            return 100
        else:
            return -100

    def _judge_done(self):
        """
        :return:
         1: bad done: collision
         2: bad done: break_traffic_rule
         3: good done: task succeed
         4: not done
        """
        if self.simulation.stopped:
            return 1, 1
        elif self._break_traffic_rule():
            return 2, 1
        elif self._is_achieve_goal():
            return 3, 1
        else:
            return 4, 0

    def _break_traffic_rule(self):
        """this func should be override in subclass"""
        raise NotImplementedError

    def _is_achieve_goal(self):  # no need to override
        return True if self.path.is_path_finished() else False


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

        # dist2left_road_border = -150 - self.ego_dynamics['y']
        # dist2right_road_border = self.ego_dynamics['y'] - (-150 - 3.75 * 4)
        # left_lane_number = 3 - self.road_related_info['egolane_index']
        # right_lane_number = self.road_related_info['egolane_index']
        # dist2current_lane_center = self.road_related_info['dist2current_lane_center']
        vertical_light_value = self.road_related_info['vertical_light_value']
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_v = self.ego_dynamics['v']
        ego_heading = self.ego_dynamics['heading']
        ego_length = self.ego_dynamics['Car_length']
        ego_width = self.ego_dynamics['Car_width']
        # calculate relative goal
        goal_v = self.goal_v
        start_x, start_y, start_a = self.path_info['start_point_info']
        end_x, end_y, end_a = self.path_info['end_point_info']
        rela_start_x, rela_start_y, rela_start_a = shift_and_rotate_coordination(start_x, start_y, start_a, ego_x, ego_y, ego_heading)
        rela_end_x, rela_end_y, rela_end_a = shift_and_rotate_coordination(end_x, end_y, end_a, ego_x, ego_y, ego_heading)

        # construct vector dict
        vector_dict = dict(ego_x=ego_x,
                           ego_y=ego_y,
                           ego_v=ego_v,
                           ego_length=ego_length,
                           ego_width=ego_width,
                           rela_start_x=rela_start_x,
                           rela_start_y=rela_start_y,
                           rela_start_a=rela_start_a,
                           rela_end_x=rela_end_x,
                           rela_end_y=rela_end_y,
                           rela_end_a=rela_end_a,
                           goal_v=goal_v,
                           vertical_light_value=vertical_light_value,)
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


def test_grid3d():
    grid = Grid_3D(back_dist=20, forward_dist=40, half_width=20, number_x=120, number_y=40)
    index_x, index_y = grid.position2xyindex(0.56, 19.5)
    print('index_x=', str(index_x), 'index_y=', str(index_y))
    position_x, position_y = grid.xyindex2centerposition(0, 0)
    print('position_x=', str(position_x), 'position_y=', str(position_y))


if __name__ == '__main__':
    test_grid3d()
