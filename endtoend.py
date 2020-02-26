import gym
from gym.utils import seeding
import math
import numpy as np
from endtoend_env_utils import shift_coordination, rotate_coordination, shift_and_rotate_coordination
from collections import OrderedDict, deque
import matplotlib.pyplot as plt
import bezier
from math import cos, sin, fabs, pi, sqrt, atan2
from traffic import Traffic
from collections import OrderedDict


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
                 training_task,  # 'left', 'straight', 'right'
                 frameskip=1,
                 display=False):
        metadata = {'render.modes': ['human']}
        self.training_task = training_task
        self.frameskip = frameskip
        self.detected_vehicles = None
        self.all_vehicles = None
        self.ego_dynamics = None
        self.road_related_info = None
        self.history_info = deque(maxlen=10)  # store infos of every step
        self.history_obs = deque(maxlen=10)  # store obs of every step
        self.init_state = {}
        self.goal_state = []
        self.action_number = 2
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_number,), dtype=np.float32)

        self.seed()
        self.v_light = None
        self.step_length = 100  # ms

        self.step_time = self.step_length / 1000.0
        self.goal_state = self._reset_goal_state()
        if not display:
            self.traffic = Traffic(self.step_length, mode='training')
            self.reset()
            action = self.action_space.sample()
            observation, _reward, done, _info = self.step(action)
            self._set_observation_space(observation)
            plt.ion()
        self.planed_trj = None

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
            self.traffic.set_own_car(dict(ego=dict(x=next_x,
                                                   y=next_y,
                                                   v=next_v,
                                                   a=next_heading)))
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
        self.traffic = Traffic(self.step_length, mode='training')
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
        self.all_vehicles = self.traffic.n_ego_vehicles['ego']  # coordination 2
        self.ego_dynamics = self.traffic.n_ego_info['ego']  # coordination 2
        self.v_light = self.traffic.v_light
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
        # dict(x=x, y=y, v=v, heading=a, length=length,
        #      width=width, route=route)

        all_info = dict(all_vehicles=self.all_vehicles,
                        ego_dynamics=self.ego_dynamics,
                        goal_state=self.goal_state,
                        v_light=self.v_light)
        return all_info

    def _get_obs(self):
        """this func should be override in subclass to generate different types of observation"""
        raise NotImplementedError

    def render(self, mode='human'):
        if mode == 'human':
            # plot basic map
            square_length = 36
            extension = 40
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
                a = veh['heading']
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


def judge_feasible(orig_x, orig_y, task):  # map dependant TODO
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
    if task == 'left':
        return True if is_in_straight_before(orig_x, orig_y) or is_in_left(orig_x, orig_y) \
                       or is_in_middle(orig_x, orig_y) else False
    elif task == 'straight':
        return True if is_in_straight_before(orig_x, orig_y) or is_in_straight_after(orig_x, orig_y) \
                       or is_in_middle(orig_x, orig_y) else False
    else:
        assert task == 'right'
        return True if is_in_straight_before(orig_x, orig_y) or is_in_right(orig_x, orig_y) \
                       or is_in_middle(orig_x, orig_y) else False


class CrossroadEnd2end(End2endEnv):
    def __init__(self,
                 training_task,  # 'left', 'straight', 'right'
                 frameskip=1,
                 display=False):
        self.history_number = 1
        self.history_frameskip = 1
        self.interested_vehs = None
        super(CrossroadEnd2end, self).__init__(training_task, frameskip, display=display)

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

    def _action_transformation_for_end2end_old(self, action):  # [-1, 1]
        prop, acc = action
        prop, acc = (prop + 1) / 2, (acc + 1) / 2  # [0, 1]
        current_x, current_y = self.ego_dynamics['x'], self.ego_dynamics['y']
        current_v = self.ego_dynamics['v']
        max_decel = min(current_v / 3, 3)

        if self.training_task == 'left' or self.training_task == 'straight':
            if self.v_light != 0 and -18 - 5 < current_y < -10:
                return 1.875 + 3.75 * prop, -6

            down_left = self.interested_vehs['dl']
            down_up = self.interested_vehs['du']
            closest_down_left_dist = 99
            closest_down_up_dist = 99
            for veh in down_left:
                if veh['y'] > current_y and veh['x'] < 3.75:
                    closest_down_left_dist = sqrt((current_x - veh['x']) ** 2 + (current_y - veh['y']) ** 2)
            for veh in down_up:
                if veh['y'] > current_y and veh['x'] < 3.75:
                    closest_down_up_dist = sqrt((current_x - veh['x']) ** 2 + (current_y - veh['y']) ** 2)

            close_forward_dist = min(closest_down_left_dist, closest_down_up_dist)
            return 1.875 + 3.75 * prop, acc * (
                        3 + max_decel) - max_decel if close_forward_dist > 10 or current_y > -10 else -6

        elif self.training_task == 'right':
            down_right = self.interested_vehs['dr']
            closest_down_right_dist = 99
            for veh in down_right:
                if veh['y'] > current_y and veh['x'] > 3.75:
                    closest_down_right_dist = sqrt((current_x - veh['x']) ** 2 + (current_y - veh['y']) ** 2)
            close_forward_dist = closest_down_right_dist
            return 1.875 + 3.75 * prop, acc * (
                        3 + max_decel) - max_decel if close_forward_dist > 10 or current_y > -10 else -6

    def _action_transformation_for_end2end(self, action):  # [-1, 1]
        prop, acc = action
        prop, acc = (prop + 1) / 2, (acc + 1) / 2  # [0, 1]
        current_x, current_y = self.ego_dynamics['x'], self.ego_dynamics['y']
        current_v = self.ego_dynamics['v']
        max_decel = min(current_v / 3, 3)

        if self.training_task == 'left' or self.training_task == 'straight':
            if self.v_light != 0 and -18 - 5 < current_y < -10:
                return 1.875 + 3.75 * prop, -6

            down_left = self.interested_vehs['dl']
            down_up = self.interested_vehs['du']
            closest_down_left_dist = sqrt((current_x - down_left[0]['x']) ** 2 + (current_y - down_left[0]['y']) ** 2)
            closest_down_up_dist = sqrt((current_x - down_up[0]['x']) ** 2 + (current_y - down_up[0]['y']) ** 2)
            closest_forward_dist = min(closest_down_left_dist, closest_down_up_dist)
            return 1.875 + 3.75 * prop, acc * (
                        3 + max_decel) - max_decel if closest_forward_dist > 10 or current_y > -10 else -6

        elif self.training_task == 'right':
            down_right = self.interested_vehs['dr']
            closest_down_right_dist = sqrt(
                (current_x - down_right[0]['x']) ** 2 + (current_y - down_right[0]['y']) ** 2)
            closest_forward_dist = closest_down_right_dist
            return 1.875 + 3.75 * prop, acc * (
                        3 + max_decel) - max_decel if closest_forward_dist > 10 or current_y > -10 else -6

    def _get_next_position(self, trans_action):
        end_offset, acc = trans_action
        current_x = self.ego_dynamics['x']
        current_y = self.ego_dynamics['y']
        current_v = self.ego_dynamics['v']
        current_heading = self.ego_dynamics['heading']
        step_length = current_v * self.step_time + 0.5 * acc * self.step_time ** 2
        step_length = step_length if step_length > 0 else 0
        if self.training_task == 'left':
            u1 = (current_x - (-38)) / 3
            u2 = (current_x - (-38)) / 2
            straight_x = 1.875

            control_point1 = (current_x, current_y) if current_y > -18 else (1.875, -18)
            control_point2 = (current_x + u1 * cos(current_heading * pi / 180), current_y + u1 * sin(
                current_heading * pi / 180)) if current_y > -18 else (1.875, -18 + u1)
            control_point3 = -38 + u2, end_offset
            control_point4 = -38, end_offset
        elif self.training_task == 'right':
            u1 = (38 - current_x) / 3
            u2 = (38 - current_x) / 2
            straight_x = 5.625

            control_point1 = (current_x, current_y) if current_y > -18 else (5.625, -18)
            control_point2 = (current_x + u1 * cos(current_heading * pi / 180), current_y + u1 * sin(
                current_heading * pi / 180)) if current_y > -18 else (5.625, -18 + u1)
            control_point3 = 38 - u2, -end_offset
            control_point4 = 38, -end_offset
        else:
            assert self.training_task == 'straight'
            u1 = (38 - current_y) / 3 if current_y > -18 else (38 - (-18)) / 3
            u2 = (38 - current_y) / 2 if current_y > -18 else (38 - (-18)) / 2
            straight_x = 1.875

            control_point1 = (current_x, current_y) if current_y > -18 else (1.875, -18)
            control_point2 = (current_x + u1 * cos(current_heading * pi / 180), current_y + u1 * sin(
                current_heading * pi / 180)) if current_y > -18 else (1.875, -18 + u1)
            control_point3 = end_offset, 38 - u2
            control_point4 = end_offset, 38

        node = np.asfortranarray([[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                                  [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]])
        curve = bezier.Curve(node, degree=3)
        s_vals = np.linspace(0, 1.0, 1000)
        trj_data = curve.evaluate_multi(s_vals)
        straight_line_x, straight_line_y = np.array([]), np.array([])
        if current_y < -18:
            straight_line_x = straight_x * np.ones(shape=(500,))
            straight_line_y = np.linspace(current_y, -18, 500)
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
                next_heading = 90 if point_for_cal_heading[1] > next_point[1] else -90
        next_v = np.clip(current_v + acc * self.step_time, 0, 10)
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

    def _vector_supplement_for_grid_encoder_old(self, exit='D'):  # func for supplement vector of grid
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

        name_settings = dict(D=dict(do='1o', di='1i', ro='2o', ri='2i', uo='3o', ui='3i', lo='4o', li='4i'),
                             R=dict(do='2o', di='2i', ro='3o', ri='3i', uo='4o', ui='4i', lo='1o', li='1i'),
                             U=dict(do='3o', di='3i', ro='4o', ri='4i', uo='1o', ui='1i', lo='2o', li='2i'),
                             L=dict(do='4o', di='4i', ro='1o', ri='1i', uo='2o', ui='2i', lo='3o', li='3i'))

        name_setting = name_settings[exit]

        def filter_interested_vehicles(vs, task):
            dl, du, dr, rd, rl, ru, ur, ud, ul, lu, lr, ld = [], [], [], [], [], [], [], [], [], [], [], []
            for v in vs:
                route_list = v['route']
                start = route_list[0]
                end = route_list[1]
                if start == name_setting['do'] and end == name_setting['li']:
                    dl.append(v)
                elif start == name_setting['do'] and end == name_setting['ui']:
                    du.append(v)
                elif start == name_setting['do'] and end == name_setting['ri']:
                    dr.append(v)

                elif start == name_setting['ro'] and end == name_setting['di']:
                    rd.append(v)
                elif start == name_setting['ro'] and end == name_setting['li']:
                    rl.append(v)
                elif start == name_setting['ro'] and end == name_setting['ui']:
                    ru.append(v)

                elif start == name_setting['uo'] and end == name_setting['ri']:
                    ur.append(v)
                elif start == name_setting['uo'] and end == name_setting['di']:
                    ud.append(v)
                elif start == name_setting['uo'] and end == name_setting['li']:
                    ul.append(v)

                elif start == name_setting['lo'] and end == name_setting['ui']:
                    lu.append(v)
                elif start == name_setting['lo'] and end == name_setting['ri']:
                    lr.append(v)
                elif start == name_setting['lo'] and end == name_setting['di']:
                    ld.append(v)
            # fetch veh in range
            dl = list(filter(lambda v: v['x'] > -28 and v['y'] > -32, dl))  # interest of left straight
            du = list(filter(lambda v: -32 < v['y'] < 18, du))  # interest of left straight
            dr = list(filter(lambda v: v['x'] < 28 and v['y'] > -32, dr))  # interest of right

            rd = rd  # not interest in case of traffic light
            rl = rl  # not interest in case of traffic light
            ru = list(filter(lambda v: v['x'] < 28 and v['y'] < 28, ru))  # interest of straight

            ur = list(filter(lambda v: v['x'] < 18 and v['y'] < 28, ur))  # interest of straight right
            ud = list(filter(lambda v: -18 < v['y'] < 28, ud))  # interest of left
            ul = list(filter(lambda v: v['x'] > -28 and v['y'] < 28, ul))  # interest of left

            lu = lu  # not interest in case of traffic light
            lr = lr  # not interest in case of traffic light
            ld = ld  # not interest in case of traffic light

            # sort
            dl = sorted(dl, key=lambda v: (v['y'], -v['x']), reverse=True)
            du = sorted(du, key=lambda v: v['y'], reverse=True)
            dr = sorted(dr, key=lambda v: (v['y'], v['x']), reverse=True)

            ru = sorted(ru, key=lambda v: (-v['x'], v['y']), reverse=True)

            ur = sorted(ur, key=lambda v: (-v['y'], v['x']), reverse=True)
            ud = sorted(ud, key=lambda v: -v['y'], reverse=True)
            ul = sorted(ul, key=lambda v: (-v['y'], -v['x']), reverse=True)

            # slice or fill to some number
            def slice_or_fill(sorted_list, fill_value, num):
                if len(sorted_list) >= num:
                    return sorted_list[:num]
                else:
                    while len(sorted_list) < num:
                        sorted_list.append(fill_value)
                    return sorted_list

            fill_value_for_dl = dict(x=1.875, y=-35, v=0, heading=90, width=2.5, length=5, route=None)
            fill_value_for_du = dict(x=1.875, y=-35, v=0, heading=90, width=2.5, length=5, route=None)
            fill_value_for_dr = dict(x=5.625, y=-35, v=0, heading=90, width=2.5, length=5, route=None)

            fill_value_for_ru = dict(x=35, y=5.625, v=0, heading=180, width=2.5, length=5, route=None)

            fill_value_for_ur = dict(x=-1.875, y=35, v=0, heading=-90, width=2.5, length=5, route=None)
            fill_value_for_ud = dict(x=-1.875, y=35, v=0, heading=-90, width=2.5, length=5, route=None)
            fill_value_for_ul = dict(x=-5.625, y=35, v=0, heading=-90, width=2.5, length=5, route=None)

            dl = slice_or_fill(dl, fill_value_for_dl, 2)
            du = slice_or_fill(du, fill_value_for_du, 2)
            dr = slice_or_fill(dr, fill_value_for_dr, 2)

            ru = slice_or_fill(ru, fill_value_for_ru, 2)

            ur = slice_or_fill(ur, fill_value_for_ur, 2)
            ud = slice_or_fill(ud, fill_value_for_ud, 2)
            ul = slice_or_fill(ul, fill_value_for_ul, 2)

            tmp = OrderedDict()
            if task == 'left':
                tmp['dl'] = dl
                tmp['du'] = du
                tmp['ud'] = ud
                tmp['ul'] = ul
            elif task == 'straight':
                tmp['dl'] = dl
                tmp['du'] = du
                tmp['ru'] = ru
                tmp['ur'] = ur
            elif task == 'right':
                tmp['dr'] = dr
                tmp['ur'] = ur

            return tmp

        list_of_interested_veh_dict = []
        self.interested_vehs = filter_interested_vehicles(self.all_vehicles, self.training_task)
        for part in list(self.interested_vehs.values()):
            list_of_interested_veh_dict.extend(part)

        for veh in list_of_interested_veh_dict:
            vehs_vector.extend([veh['x'], veh['y'], veh['v'],
                                veh['heading'] * pi / 180])

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
                           # ego_length=ego_length,
                           # ego_width=ego_width,
                           # rela_goal_x=rela_goal_x,
                           # rela_goal_y=rela_goal_y,
                           # rela_goal_a=rela_goal_a * pi / 180,
                           # goal_v=goal_v,
                           )
        _ = np.array(list(vector_dict.values()))
        # vector = np.concatenate((_, vehs_vector, key_points_vector), axis=0)
        vector = np.concatenate((_, vehs_vector), axis=0)

        return vector

    def _vector_supplement_for_grid_encoder(self, exit='D'):  # func for supplement vector of grid
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

        name_settings = dict(D=dict(do='1o', di='1i', ro='2o', ri='2i', uo='3o', ui='3i', lo='4o', li='4i'),
                             R=dict(do='2o', di='2i', ro='3o', ri='3i', uo='4o', ui='4i', lo='1o', li='1i'),
                             U=dict(do='3o', di='3i', ro='4o', ri='4i', uo='1o', ui='1i', lo='2o', li='2i'),
                             L=dict(do='4o', di='4i', ro='1o', ri='1i', uo='2o', ui='2i', lo='3o', li='3i'))

        name_setting = name_settings[exit]

        def filter_interested_vehicles(vs, task):
            dl, du, dr, rd, rl, ru, ur, ud, ul, lu, lr, ld = [], [], [], [], [], [], [], [], [], [], [], []
            for v in vs:
                route_list = v['route']
                start = route_list[0]
                end = route_list[1]
                if start == name_setting['do'] and end == name_setting['li']:
                    dl.append(v)
                elif start == name_setting['do'] and end == name_setting['ui']:
                    du.append(v)
                elif start == name_setting['do'] and end == name_setting['ri']:
                    dr.append(v)

                elif start == name_setting['ro'] and end == name_setting['di']:
                    rd.append(v)
                elif start == name_setting['ro'] and end == name_setting['li']:
                    rl.append(v)
                elif start == name_setting['ro'] and end == name_setting['ui']:
                    ru.append(v)

                elif start == name_setting['uo'] and end == name_setting['ri']:
                    ur.append(v)
                elif start == name_setting['uo'] and end == name_setting['di']:
                    ud.append(v)
                elif start == name_setting['uo'] and end == name_setting['li']:
                    ul.append(v)

                elif start == name_setting['lo'] and end == name_setting['ui']:
                    lu.append(v)
                elif start == name_setting['lo'] and end == name_setting['ri']:
                    lr.append(v)
                elif start == name_setting['lo'] and end == name_setting['di']:
                    ld.append(v)
            # fetch veh in range
            dl = list(filter(lambda v: v['x'] > -28 and v['y'] > ego_y, dl))  # interest of left straight
            du = list(filter(lambda v: ego_y < v['y'] < 28, du))  # interest of left straight
            dr = list(filter(lambda v: v['x'] < 28 and v['y'] > ego_y, dr))  # interest of right

            rd = rd  # not interest in case of traffic light
            rl = rl  # not interest in case of traffic light
            ru = list(filter(lambda v: v['x'] < 28 and v['y'] < 28, ru))  # interest of straight

            ur_straight = list(filter(lambda v: v['x'] < ego_x + 7 and ego_y < v['y'] < 28, ur))  # interest of straight
            ur_right = list(filter(lambda v: v['x'] < 28 and v['y'] < 18, ur))  # interest of right
            ud = list(filter(lambda v: ego_y < v['y'] < 28, ud))  # interest of left
            ul = list(filter(lambda v: v['x'] > -28 and v['y'] < 28, ul))  # interest of left

            lu = lu  # not interest in case of traffic light
            lr = lr  # not interest in case of traffic light
            ld = ld  # not interest in case of traffic light

            # sort
            dl = sorted(dl, key=lambda v: (v['y'], -v['x']))
            du = sorted(du, key=lambda v: v['y'])
            dr = sorted(dr, key=lambda v: (v['y'], v['x']))

            ru = sorted(ru, key=lambda v: (-v['x'], v['y']), reverse=True)

            ur_straight = sorted(ur_straight, key=lambda v: v['y'])
            ur_right = sorted(ur_right, key=lambda v: (-v['y'], v['x']), reverse=True)

            ud = sorted(ud, key=lambda v: v['y'])
            ul = sorted(ul, key=lambda v: (-v['y'], -v['x']), reverse=True)

            # slice or fill to some number
            def slice_or_fill(sorted_list, fill_value, num):
                if len(sorted_list) >= num:
                    return sorted_list[:num]
                else:
                    while len(sorted_list) < num:
                        sorted_list.append(fill_value)
                    return sorted_list

            fill_value_for_dl = dict(x=-40, y=1.875, v=0, heading=180, width=2.5, length=5, route=None)
            fill_value_for_du = dict(x=1.875, y=40, v=0, heading=90, width=2.5, length=5, route=None)
            fill_value_for_dr = dict(x=40, y=-5.625, v=0, heading=0, width=2.5, length=5, route=None)

            fill_value_for_ru = dict(x=35, y=5.625, v=0, heading=180, width=2.5, length=5, route=None)

            fill_value_for_ur_straight = dict(x=-1.875, y=40, v=0, heading=-90, width=2.5, length=5, route=None)
            fill_value_for_ur_right = dict(x=-1.875, y=40, v=0, heading=-90, width=2.5, length=5, route=None)

            fill_value_for_ud = dict(x=-1.875, y=40, v=0, heading=-90, width=2.5, length=5, route=None)
            fill_value_for_ul = dict(x=-5.625, y=40, v=0, heading=-90, width=2.5, length=5, route=None)

            tmp = OrderedDict()
            if task == 'left':
                tmp['dl'] = slice_or_fill(dl, fill_value_for_dl, 2)
                tmp['du'] = slice_or_fill(du, fill_value_for_du, 2)
                tmp['ud'] = slice_or_fill(ud, fill_value_for_ud, 3)
                tmp['ul'] = slice_or_fill(ul, fill_value_for_ul, 3)
            elif task == 'straight':
                tmp['dl'] = slice_or_fill(dl, fill_value_for_dl, 2)
                tmp['du'] = slice_or_fill(du, fill_value_for_du, 2)
                tmp['ru'] = slice_or_fill(ru, fill_value_for_ru, 3)
                tmp['ur'] = slice_or_fill(ur_straight, fill_value_for_ur_straight, 3)
            elif task == 'right':
                tmp['dr'] = slice_or_fill(dr, fill_value_for_dr, 2)
                tmp['ur'] = slice_or_fill(ur_right, fill_value_for_ur_right, 3)

            return tmp

        list_of_interested_veh_dict = []
        self.interested_vehs = filter_interested_vehicles(self.all_vehicles, self.training_task)
        for part in list(self.interested_vehs.values()):
            list_of_interested_veh_dict.extend(part)

        for veh in list_of_interested_veh_dict:
            vehs_vector.extend([veh['x'], veh['y'], veh['v'],
                                veh['heading'] * pi / 180])

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
                           # ego_length=ego_length,
                           # ego_width=ego_width,
                           # rela_goal_x=rela_goal_x,
                           # rela_goal_y=rela_goal_y,
                           # rela_goal_a=rela_goal_a * pi / 180,
                           # goal_v=goal_v,
                           )
        _ = np.array(list(vector_dict.values()))
        # vector = np.concatenate((_, vehs_vector, key_points_vector), axis=0)
        vector = np.concatenate((_, vehs_vector), axis=0)

        return vector

    def recover_orig_position_fn(self, transformed_x, transformed_y, x, y, d):  # x, y, d are used to transform
        # coordination
        transformed_x, transformed_y, _ = rotate_coordination(transformed_x, transformed_y, 0, -d)
        orig_x, orig_y = shift_coordination(transformed_x, transformed_y, -x, -y)
        return orig_x, orig_y

    def _break_road_constrain(self):
        results = list(map(lambda x: judge_feasible(*x, self.training_task), self.ego_dynamics['Corner_point']))
        return not all(results)

    def _is_achieve_goal(self):  # for now, only support turn left with the specific map
        goal_x, goal_y, goal_v, goal_a = self.goal_state
        x = self.ego_dynamics['x']
        y = self.ego_dynamics['y']
        if self.training_task == 'left' or self.training_task == 'right':
            return True if goal_x - 2 < x < goal_x and goal_y - 3.75 < y < goal_y + 3.75 else False
        else:
            assert self.training_task == 'straight'
            return True if goal_y < y < goal_y + 2 and goal_x - 3.75 < x < goal_x + 3.75 else False

    def _reset_goal_state(self):  # decide center of goal area, [goal_x, goal_y, goal_a, goal_v]
        if self.training_task == 'left':
            return [-18 - 6, 3.75, 8, 180]
        elif self.training_task == 'straight':
            return [3.75, 18 + 6, 8, 90]
        else:
            assert self.training_task == 'right'
            return [18 + 6, -3.75, 8, 0]

    def _reset_init_state(self):
        # nodes1 = np.asfortranarray([[3.75 / 2, 3.75 / 2, -18 + 10, -18],
        #                             [-18 - 15, -18 + 18, 3.75 / 2, 3.75 / 2]])
        # curve1 = bezier.Curve(nodes1, degree=3)
        # nodes2 = np.asfortranarray([[3.75 / 2, 3.75 / 2, -18 + 10, -18],
        #                             [-18 - 15, -18 + 18, 3.75 * 3 / 2, 3.75 * 3 / 2]])
        # curve2 = bezier.Curve(nodes2, degree=3)
        # start_point = None
        # if np.random.random() > 0.5:
        #     start_point = curve1.evaluate(0.1 * np.random.random())
        # else:
        #     start_point = curve2.evaluate(0.1 * np.random.random())
        # x, y = start_point[0][0], start_point[1][0]
        # if y < -18:
        #     a = 90.
        # else:
        #     a = 90. + math.atan((y + 18) / (x + 18)) * 180 / math.pi
        # v = 3
        # return dict(ego=dict(x=x, y=y, v=v, a=a, l=4.8, w=2.2))
        x = 5.625 if self.training_task == 'right' else 1.875
        y = -18 - 15 * np.random.random()
        v = 3 + np.random.random()
        a = 90
        if self.training_task == 'left':
            routeID = 'dl'
        elif self.training_task == 'straight':
            routeID = 'du'
        else:
            assert self.training_task == 'right'
            routeID = 'dr'
        return dict(ego=dict(x=x,
                             y=y,
                             v=v,
                             a=a,
                             l=4.8,
                             w=2.2,
                             routeID=routeID))

    def _cal_collision_reward(self):  # can be override to do an analytic calculation
        return -100

    def _cal_achievegoal_reward(self):  # can be override to do an analytic calculation
        x, y, a, v = self.ego_dynamics['x'], self.ego_dynamics['y'], \
                     self.ego_dynamics['heading'], self.ego_dynamics['v']

        goal_x, goal_y, goal_v, goal_a = self.goal_state
        if self.training_task == 'left':
            position_punishment = -30 * min(fabs(y - 3.75 / 2), fabs(y - 3.75 * 3 / 2))
            heading_punishment = -fabs(a - (-180)) if a < 0 else -fabs(a - 180)
        elif self.training_task == 'straight':
            position_punishment = -30 * min(fabs(x - 3.75 / 2), fabs(x - 3.75 * 3 / 2))
            heading_punishment = -fabs(a - 90)
        else:
            assert self.training_task == 'right'
            position_punishment = -30 * min(fabs(y - (-3.75 / 2)), fabs(y - (-3.75 * 3 / 2)))
            heading_punishment = -fabs(a)

        return 100 + position_punishment + heading_punishment

    def _cal_step_reward(self):
        # data preparation
        x, y, a, v = self.ego_dynamics['x'], self.ego_dynamics['y'], self.ego_dynamics['heading'], self.ego_dynamics[
            'v']
        goal_x, goal_y, goal_v, goal_a = self.goal_state
        dist_to_goal = math.sqrt((goal_x - x) ** 2 + (goal_y - y) ** 2)
        v_difference = math.fabs(goal_v - v)

        # step punishment
        reward = -0.2
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


if __name__ == '__main__':
    pass
