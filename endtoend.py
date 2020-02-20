import gym
from gym.utils import seeding
import math
import numpy as np
from endtoend_env_utils import shift_coordination, rotate_coordination, shift_and_rotate_coordination, \
    Path
from collections import OrderedDict, deque
import matplotlib.pyplot as plt
import bezier
from math import cos, sin, fabs, pi, sqrt, atan2
from traffic import Traffic
from collections import OrderedDict
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
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_number,), dtype=np.float32)

        self.seed()
        self.step_length = 100  # ms
        self.traffic = Traffic(self.step_length)
        self.step_time = self.step_length / 1000.0
        self.reset()
        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        self._set_observation_space(observation)
        self.planed_trj = None
        plt.ion()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        reward = 0
        done = 0
        all_info = None
        info_this_step = []
        for _ in range(self.frameskip):
            trans_action = self._action_transformation_for_end2end(action)
            next_x, next_y, next_v, next_heading = self._get_next_position(trans_action)
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
        print(done_info[done_type - 1], '\n')

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

    def _get_next_position(self, trans_action):
        acc, delta_phi = trans_action
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
                if -square_length / 2 - extension + tolerance < x < square_length / 2 + extension - tolerance and \
                        -square_length / 2 - extension + tolerance < y < square_length / 2 + extension - tolerance:
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

            # plot planed trj
            if self.planed_trj is not None:
                ax.plot(self.planed_trj[0], self.planed_trj[1], color='g')

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


def judge_feasible(orig_x, orig_y):  # map dependant TODO
    # return True if -900 < orig_x < 900 and -150 - 3.75 * 4 < orig_y < -150 else False
    def is_in_straight_before(orig_x, orig_y):
        return -1 < orig_x < 3.75 * 2 and orig_y <= -18

    def is_in_straight_after(orig_x, orig_y):
        return 0 < orig_x < 3.75 * 2 and orig_y >= 18

    def is_in_left(orig_x, orig_y):
        return 0 < orig_y < 3.75 * 2 and orig_x < -18

    def is_in_right(orig_x, orig_y):
        return -3.75 * 2 < orig_y < 0 and orig_x > 18

    def is_in_middle(orig_x, orig_y):
        if -18 < orig_y < 18 and -18 < orig_x < 18:
            if -3.75 * 2 < orig_x < 3.75 * 2:
                return True if -18 < orig_y < 18 else False
            elif orig_x > 3.75 * 2:
                return True if orig_x - (18 + 3.75 * 2) < orig_y < -orig_x + (18 + 3.75 * 2) else False
            else:
                return True if -orig_x - (18 + 3.75 * 2) < orig_y < orig_x + (18 + 3.75 * 2) else False
        else:
            return False

    # judge feasible for turn left
    return True if is_in_straight_before(orig_x, orig_y) or is_in_left(orig_x, orig_y) \
                   or is_in_middle(orig_x, orig_y) else False


class CrossroadEnd2end(End2endEnv):
    def __init__(self,
                 obs_type=0,
                 frameskip=1,
                 repeat_action_probability=0
                 ):
        self.history_number = 1
        self.history_frameskip = 1
        self.interested_vehs = None
        super(CrossroadEnd2end, self).__init__(obs_type, frameskip)

    def _get_obs(self):
        return self._vector_supplement_for_grid_encoder()

    def _process_obs(self):
        # history
        history_len = len(self.history_obs)
        history_obs_index = [0] + [-i * self.history_frameskip if i * self.history_frameskip < history_len
                                   else -history_len + 1 for i in range(1, self.history_number)]
        history_obs_index.reverse()
        history_obs_index = np.array(history_obs_index) - 1

        history_vectors_list = [self.history_obs[i] for i in history_obs_index]
        ob_vector = np.concatenate(history_vectors_list, axis=0)
        return ob_vector

    def _action_transformation_for_end2end(self, action):  # [-1, 1]
        prop, acc = action
        prop, acc = (prop + 1)/2, (acc + 1)/2  # [0, 1]
        current_x, current_y = self.ego_dynamics['x'], self.ego_dynamics['y']
        current_v = self.ego_dynamics['v']
        down_left = self.interested_vehs['down_left']
        down_up = self.interested_vehs['down_up']
        closest_down_left_dist = 99
        closest_down_up_dist = 99
        for veh in down_left:
            if veh['y'] > current_y and veh['x'] < 3.75:
                closest_down_left_dist = sqrt((current_x-veh['x'])**2+(current_y-veh['y'])**2)
        for veh in down_up:
            if veh['y'] > current_y and veh['x'] < 3.75:
                closest_down_up_dist = sqrt((current_x-veh['x'])**2+(current_y-veh['y'])**2)

        close_forward_dist = min(closest_down_left_dist, closest_down_up_dist)
        max_decel = min(current_v/3, 3)

        return 1.875 + 3.75 * prop, acc * (2+max_decel) - max_decel if close_forward_dist > 10 or current_y > -3 else -6

        # return 7.5 * prop, acc * 4.5 - 3 if close_forward_dist > 10 or current_y > -3 else -6

    def _get_next_position(self, trans_action):
        end_y, acc = trans_action
        current_x = self.ego_dynamics['x']
        current_y = self.ego_dynamics['y']
        current_v = self.ego_dynamics['v']
        current_heading = self.ego_dynamics['heading']
        step_length = current_v * self.step_time + 0.5 * acc * self.step_time**2
        step_length = step_length if step_length > 0 else 0
        u = current_x - (-20)
        u = u if u > 3 else 3

        control_point1 = (current_x, current_y) if current_y > -18 else (1.875, -18)
        control_point2 = (current_x + u * cos(current_heading * pi / 180), current_y + u * sin(
            current_heading * pi / 180)) if current_y > -18 else (1.875, -18 + u)
        control_point3 = -38 + min(10, 0.5*(current_x+38)), end_y  #-18 - 10 * (current_y+30)/40, end_y
        control_point4 = -38, end_y  #-28 - 10 * (current_y+30)/40, end_y

        node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                                  [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]])
        curve = bezier.Curve(node, degree=3)
        s_vals = np.linspace(0, 1.0, 500)
        trj_data = curve.evaluate_multi(s_vals)
        straight_line_x, straight_line_y = np.array([]), np.array([])
        if current_y < -18:
            straight_line_x = 1.875 * np.ones(shape=(100,))
            straight_line_y = np.linspace(current_y, -18, 100)
        self.planed_trj = np.append(straight_line_x, trj_data[0]), np.append(straight_line_y, trj_data[1])
        trj_x, trj_y = self.planed_trj[0], self.planed_trj[1]
        dist_sum = 0
        next_point_index = 0
        for i in range(1, len(trj_x)):
            dist_sum += sqrt((trj_x[i] - trj_x[i - 1]) ** 2 + (trj_y[i] - trj_y[i - 1]) ** 2)
            if dist_sum > step_length:
                next_point_index = i - 1
                break
        next_point = trj_x[next_point_index], trj_y[next_point_index]
        point_for_cal_heading = trj_x[next_point_index + 1], trj_y[next_point_index + 1]
        if next_point_index == 0:
            next_heading = current_heading
        else:
            if point_for_cal_heading[0] != next_point[0]:
                next_heading = atan2(point_for_cal_heading[1] - next_point[1],
                                     point_for_cal_heading[0] - next_point[0]) * 180 / pi
            else:
                next_heading = 90 if point_for_cal_heading[0] > next_point[0] else -90
        next_v = np.clip(current_v + acc * self.step_time, 0, 15)
        next_x, next_y = next_point
        return next_x, next_y, next_v, next_heading

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

        name_setting = dict(down_exit='-gneE3',
                            down_entra='gneE3',
                            right_exit='-gneE1',
                            right_entra='gneE1',
                            up_exit='-gneE2',
                            up_entra='gneE2',
                            left_exit='gneE12',
                            left_entra='-gneE0')

        def filter_interested_vehicles(vs):
            down_up = []
            down_left = []
            right_down = []
            right_left = []
            up_down = []
            up_left = []
            left_up = []
            left_right = []
            for v in vs:
                route_list = v['route']
                edge_index = v['edge_index']
                if route_list[edge_index] in ['-gneE0', 'gneE3', 'gneE1', 'gneE2']:
                    start = route_list[edge_index - 1]
                    end = route_list[edge_index]
                else:
                    start = route_list[edge_index]
                    end = route_list[edge_index + 1]
                if start == name_setting['down_exit'] and end == name_setting['up_entra']:
                    down_up.append(v)
                elif start == name_setting['down_exit'] and end == name_setting['left_entra']:
                    down_left.append(v)

                elif start == name_setting['right_exit'] and end == name_setting['down_entra']:
                    right_down.append(v)
                elif start == name_setting['right_exit'] and end == name_setting['left_entra']:
                    right_left.append(v)

                elif start == name_setting['up_exit'] and end == name_setting['down_entra']:
                    up_down.append(v)
                elif start == name_setting['up_exit'] and end == name_setting['left_entra']:
                    up_left.append(v)

                elif start == name_setting['left_exit'] and end == name_setting['up_entra']:
                    left_up.append(v)
                elif start == name_setting['left_exit'] and end == name_setting['right_entra']:
                    left_right.append(v)
            # fetch veh in range
            down_up = list(filter(lambda v: -32 < v['y'] < 18, down_up))
            down_left = list(filter(lambda v: v['x'] > -28 and v['y'] > -32, down_left))
            right_down = list(filter(lambda v: v['x'] < 28 and v['y'] > -18, right_down))
            right_left = list(filter(lambda v: -28 < v['x'] < 28, right_left))
            up_down = list(filter(lambda v: -18 < v['y'] < 28, up_down))
            up_left = list(filter(lambda v: v['x'] > -28 and v['y'] < 28, up_left))
            left_up = list(filter(lambda v: v['x'] > -28 and v['y'] < 18, left_up))
            left_right = list(filter(lambda v: -28 < v['x'] < 18, left_right))

            # sort
            down_up = sorted(down_up, key=lambda v: v['y'], reverse=True)
            down_left = sorted(down_left, key=lambda v: v['y'], reverse=True)
            right_down = sorted(right_down, key=lambda v: v['x'])
            right_left = sorted(right_left, key=lambda v: v['x'])
            up_down = sorted(up_down, key=lambda v: v['y'])
            up_left = sorted(up_left, key=lambda v: v['y'])
            left_up = sorted(left_up, key=lambda v: v['x'], reverse=True)
            left_right = sorted(left_right, key=lambda v: v['x'], reverse=True)

            # slice or fill to some number
            def slice_or_fill(sorted_list, fill_value, num):
                if len(sorted_list) >= num:
                    return sorted_list[:num]
                else:
                    while len(sorted_list) < num:
                        sorted_list.append(fill_value)
                    return sorted_list

            fill_value_for_down_up = dict(x=1.875, y=-35, v=0, heading=90, width=2.5, length=5, route=None,
                                          edge_index=None)
            fill_value_for_down_left = dict(x=1.875, y=-35, v=0, heading=90, width=2.5, length=5, route=None,
                                            edge_index=None)
            fill_value_for_right_down = dict(x=35, y=1.875, v=0, heading=180, width=2.5, length=5, route=None,
                                             edge_index=None)
            fill_value_for_right_left = dict(x=35, y=1.875, v=0, heading=180, width=2.5, length=5, route=None,
                                             edge_index=None)
            fill_value_for_up_down = dict(x=-1.875, y=35, v=0, heading=-90, width=2.5, length=5, route=None,
                                          edge_index=None)
            fill_value_for_up_left = dict(x=-5.625, y=35, v=0, heading=-90, width=2.5, length=5, route=None,
                                          edge_index=None)
            fill_value_for_left_up = dict(x=-35, y=-1.875, v=0, heading=0, width=2.5, length=5, route=None,
                                          edge_index=None)
            fill_value_for_left_right = dict(x=-35, y=-1.875, v=0, heading=0, width=2.5, length=5, route=None,
                                             edge_index=None)

            down_up = slice_or_fill(down_up, fill_value_for_down_up, 2)
            down_left = slice_or_fill(down_left, fill_value_for_down_left, 3)
            right_down = slice_or_fill(right_down, fill_value_for_right_down, 2)
            right_left = slice_or_fill(right_left, fill_value_for_right_left, 2)
            up_down = slice_or_fill(up_down, fill_value_for_up_down, 2)
            up_left = slice_or_fill(up_left, fill_value_for_up_left, 2)
            left_up = slice_or_fill(left_up, fill_value_for_left_up, 2)
            left_right = slice_or_fill(left_right, fill_value_for_left_right, 2)

            tmp = OrderedDict()
            tmp['down_left'] = down_left
            tmp['down_up'] = down_up
            tmp['right_down'] = []
            tmp['right_left'] = []
            tmp['up_down'] = up_down
            tmp['up_left'] = up_left
            tmp['left_up'] = []
            tmp['left_right'] = []
            return tmp

        list_of_interested_veh_dict = []
        self.interested_vehs = filter_interested_vehicles(all_vehicles)
        for part in list(self.interested_vehs.values()):
            list_of_interested_veh_dict.extend(part)

        list_of_interested_veh_dict_trans = self._cal_info_in_transform_coordination(list_of_interested_veh_dict, ego_x, ego_y,
                                                                                     ego_heading)
        for veh in list_of_interested_veh_dict_trans:
            vehs_vector.extend([veh['trans_x'], veh['trans_y'], veh['trans_v'],
                                veh['trans_heading'] * pi / 180])

        vehs_vector = np.array(vehs_vector)

        # map related
        key_points_vector = []
        key_points = [(-7.5, 18), (0, 18), (7.5, 18),
                      (-7.5, -18), (0, -18), (7.5, -18),
                      (-18, -7.5), (-18, 0), (-18, 7.5),
                      (18, -7.5), (18, 0), (18, 7.5)]
        transfered_key_points = list(
            map(lambda x: shift_and_rotate_coordination(*x, 0, ego_x, ego_y, ego_heading)[0: 2],
                key_points))

        for key_point in transfered_key_points:
            key_points_vector.extend([key_point[0], key_point[1]])

        key_points_vector = np.array(key_points_vector)

        # construct vector dict
        vector_dict = dict(ego_x=ego_x,
                           ego_y=ego_y,
                           ego_v=ego_v,
                           ego_heading=ego_heading * pi / 180,
                           ego_length=ego_length,
                           ego_width=ego_width,
                           rela_goal_x=rela_goal_x,
                           rela_goal_y=rela_goal_y,
                           rela_goal_a=rela_goal_a * pi / 180,
                           goal_v=goal_v,
                           )
        _ = np.array(list(vector_dict.values()))
        vector = np.concatenate((_, vehs_vector, key_points_vector), axis=0)
        return vector

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

    def _break_road_constrain(self):
        results = list(map(lambda x: judge_feasible(*x), self.ego_dynamics['Corner_point']))
        return not all(results)

    def _is_achieve_goal(self):  # for now, only support turn left with the specific map
        goal_x, goal_y, goal_v, goal_a = self.goal_state
        x = self.ego_dynamics['x']
        y = self.ego_dynamics['y']
        return True if goal_x - 2 < x < goal_x + 2 and goal_y - 3.75 < y < goal_y + 3.75 else False

    def _reset_goal_state(self):  # decide center of goal area, [goal_x, goal_y, goal_a, goal_v]
        return [-18 - 6, 3.75, 8, 180]

    def _reset_init_state(self):
        nodes1 = np.asfortranarray([[3.75 / 2, 3.75 / 2, -18 + 10, -18],
                                    [-18 - 15, -18 + 18, 3.75 / 2, 3.75 / 2]])
        curve1 = bezier.Curve(nodes1, degree=3)
        nodes2 = np.asfortranarray([[3.75 / 2, 3.75 / 2, -18 + 10, -18],
                                    [-18 - 15, -18 + 18, 3.75 * 3 / 2, 3.75 * 3 / 2]])
        curve2 = bezier.Curve(nodes2, degree=3)
        start_point = None
        if np.random.random() > 0.5:
            start_point = curve1.evaluate(0.2)
        else:
            start_point = curve2.evaluate(0.2)
        x, y = start_point[0][0], start_point[1][0]
        if y < -18:
            a = 90.
        else:
            a = 90. + math.atan((y + 18) / (x + 18)) * 180 / math.pi
        v = 3 * np.random.random() + 5
        return [x, y, v, a]
        # return [1.875, -28, 5, 90]

    def _cal_collision_reward(self):  # can be override to do an analytic calculation
        return -100

    def _cal_achievegoal_reward(self):  # can be override to do an analytic calculation
        x, y, a, v = self.ego_dynamics['x'], self.ego_dynamics['y'], \
                     self.ego_dynamics['heading'], self.ego_dynamics['v']

        goal_x, goal_y, goal_v, goal_a = self.goal_state
        position_punishment = -30 * min(fabs(y - 3.75 / 2), fabs(y - 3.75 * 3 / 2))
        heading_punishment = -fabs(a - (-180)) if a < 0 else -fabs(a - 180)
        return 100 + position_punishment + heading_punishment

    def _cal_step_reward(self):
        # data preparation
        x, y, a, v = self.ego_dynamics['x'], self.ego_dynamics['y'], self.ego_dynamics['heading'], self.ego_dynamics[
            'v']
        goal_x, goal_y, goal_v, goal_a = self.goal_state
        dist_to_goal = math.sqrt((goal_x - x) ** 2 + (goal_y - y) ** 2)
        v_difference = math.fabs(goal_v - v)

        # step punishment
        reward = -1
        # goal position punishment
        # reward -= dist_to_goal * 0.005
        # goal velocity punishment
        # reward -= v_difference * 0.05
        # standard curve punishment
        # reward -= min_dist_to_curve * 0.05
        return reward

    def compute_reward(self, done_type):
        reward = 0
        if done_type == 4:
            return self._cal_step_reward()
        elif done_type == 3:
            return self._cal_achievegoal_reward()
        else:
            return self._cal_collision_reward()


def test_crossrode():
    env = CrossroadEnd2end()
    veh = {'trans_x': 1.875, 'trans_y': -17, 'trans_v': 7, 'trans_heading': 120,
           'route': ['-gneE3', '-gneE0'], 'edge_index': 0}
    veh = {'trans_x': 17, 'trans_y': 1.875, 'trans_v': 7, 'trans_heading': -160,
           'route': ['-gneE1', 'gneE3'], 'edge_index': 0}
    veh = {'trans_x': 17, 'trans_y': 1.875, 'trans_v': 7, 'trans_heading': -180,
           'route': ['-gneE1', '-gneE0'], 'edge_index': 0}
    veh = {'trans_x': 17, 'trans_y': 1.875 * 3, 'trans_v': 7, 'trans_heading': -180,
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
