import gym
from LasVSim import lasvsim
from gym.utils import seeding
import math
import numpy as np
from LasVSim.endtoend_env_utils import shift_coordination, rotate_coordination
from collections import deque
from LasVSim.reference import Reference


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

    def __init__(self, setting_path, plan_horizon, history_len):
        self.horizon = plan_horizon
        self.setting_path = setting_path
        self.action_space = None
        self.observation_space = None
        self.detected_vehicles = None
        self.all_vehicles = None
        self.ego_dynamics = None
        self.ego_info = None
        self.ego_road_related_info = None
        self.simulation = None
        self.init_state = []
        self.final_goal_x = None
        self.history_len = history_len
        self.obs_deque = deque(maxlen=history_len)
        self.reference = Reference()

        self.seed()
        lasvsim.create_simulation(setting_path + 'simulation_setting_file.xml')
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):  # action is a np.array, [goal_x, goal_y, goal_v]
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
        goal_x, goal_y, goal_v = action
        state_on_begin_of_step = [self.ego_dynamics['x'], self.ego_dynamics['y'], self.ego_dynamics['v'], self.ego_dynamics['heading']]
        ego_goal_state = [goal_x, goal_y, goal_v, 0]
        self.reference.reset_reference_path(state_on_begin_of_step, ego_goal_state)
        reward = 0
        done = 0
        done_type = 3
        for _ in range(self.horizon):
            x, y, v, heading = self.reference.sim_step()
            lasvsim.set_ego(x, y, v, heading)
            lasvsim.sim_step()
            self.all_vehicles = lasvsim.get_all_objects()  # coordination 2
            self.ego_dynamics = lasvsim.get_ego_info()
            self.ego_road_related_info = lasvsim.get_ego_road_related_info()
            # ego_dynamics
            # dict(x=self.x,
            #      y=self.y,
            #      v=self.v,
            #      heading=self.heading,
            #      length=self.length,
            #      width=self.width)

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
            self.obs_deque.append([self.all_vehicles, self.ego_dynamics, self.ego_road_related_info])

            done, done_type = self._judge_done()
            reward += self.compute_done_reward(done_type)
            if done:
                break
        state_on_end_of_step = [self.ego_dynamics['x'], self.ego_dynamics['y'], self.ego_dynamics['v'], self.ego_dynamics['heading']]
        longitudinal_reward = 1 * state_on_end_of_step[0] - state_on_begin_of_step[0]  # forward distance reward TODO: add weight
        lateral_reward = -5 * abs(state_on_end_of_step[1] - ego_goal_state[1])  # lateral distance reward
        info = dict(done_rew=reward, long_rew=longitudinal_reward, lat_rew=lateral_reward, done_type=done_type)
        reward += (longitudinal_reward + lateral_reward)
        return self.obs_deque, reward, done, info

    def reset(self, **kwargs):  # must assign 'init_state'
        #  clear deque
        self.obs_deque.clear()
        self.init_state = kwargs['init_state']
        self.final_goal_x = self.init_state['x'] + 500
        lasvsim.reset_simulation(overwrite_settings={'init_state': self.init_state},
                                 init_traffic_path=self.setting_path)
        self.simulation = lasvsim.simulation
        self.all_vehicles = lasvsim.get_all_objects()
        self.ego_dynamics = lasvsim.get_ego_info()
        self.ego_road_related_info = lasvsim.get_ego_road_related_info()
        self.obs_deque.append([self.all_vehicles, self.ego_dynamics, self.ego_road_related_info])
        return self.obs_deque

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

    def compute_done_reward(self, done):
        if done == 3:  # not end, just step reward
            return -1
        elif done == 2:  # good end reward
            return 1000
        else:            # bad end reward
            return -1000

    def _judge_done(self):
        '''
        :return:
         flag, 0: bad done: violate road constrain
         flag, 1: bad done: collision
         flag, 2: good done: complete
         flag, 3: not done
        '''
        if self._is_road_violation():  # go outside rode
            return 1, 0
        elif self.simulation.stopped:  # collision
            return 1, 1
        elif self.ego_dynamics['x'] > self.final_goal_x:  # complete whole journey
            return 1, 2
        else:
            return 0, 3  # not done

    def _is_road_violation(self):
        corner_points = self.ego_info['Corner_point']
        for corner_point in corner_points:
            if not judge_feasible(corner_point[0], corner_point[1]):
                return True  # violate road constrain
        return False


def judge_feasible(orig_x, orig_y):  # map dependant TODO
    return True if -900 < orig_x < 900 and -150 - 3.75 * 4 < orig_y < -150 else False


class ObservationWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ObservationWrapper, self).__init__(env)
        self.interested_vehicles = []
        self.interested_rear_dist = 30
        self.interested_front_dist = 60
        self.history_len = self.env.history_len
        self.encode_vec_len = 53  # 53 = 6dim * 8veh + 5ego
        self.encoded_obs = np.zeros((self.history_len, self.encode_vec_len))

    def reset(self, **kwargs):  # must assign init_state
        self.encoded_obs = np.zeros((self.history_len, self.encode_vec_len))
        observation = self.env.reset(**kwargs)

        return self.observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        for infos in observation:
            all_vehicles, ego_dynamics, ego_road_related_info = infos
            ego_x, ego_y, ego_v, ego_heading, ego_length, ego_width = ego_dynamics
            dist2current_lane_center, egolane_index = ego_road_related_info
            self.interested_vehicles = [veh for veh in all_vehicles
                                        if self._is_in_interested_area(ego_x, veh['x'], veh['y'])
                                        and veh['lane_index'] in self._interested_lane_index(egolane_index)]




    def _is_in_interested_area(self, ego_x, pos_x, pos_y):
        return True if ego_x - self.interested_rear_dist < pos_x < ego_x + self.interested_front_dist and -150 - 3.75 * 4 < pos_y < -150 else False

    def _interested_lane_index(self, ego_lane_index):
        info_list = [[1, 0, None], [2, 1, 0], [3, 2, 1], [None, 3, 2]]  # left, middle, right
        return info_list[ego_lane_index]

    def _laneindex2centery(self, lane_index):
        center_y_list = [-150-7*3.75/2, -150-5*3.75/2, -150-3*3.75/2, -150-1*3.75/2]
        return center_y_list[lane_index]

    def _divide_6parts_and_encode(self, ego_x, ego_y, ego_v, ego_heading, ego_length, ego_width, dist2current_lane_center, egolane_index):
        if egolane_index != 3:
            center_y = self._laneindex2centery(self._interested_lane_index(egolane_index)[0])
            LEFT_FRONT_NO_CAR_ENCODED_VECTOR = [self.interested_front_dist, center_y-ego_y, ego_v, 0, ego_length, ego_width]  # delta_x, delta_y, v, heading(in coord2), length, width
            LEFT_REAR_NO_CAR_ENCODED_VECTOR = [self.interested_rear_dist, center_y-ego_y, 0, 0, ego_length, ego_width]
        if egolane_index != 0:
            center_y = self._laneindex2centery(self._interested_lane_index(egolane_index)[1])
            RIGHT_FRONT_NO_CAR_ENCODED_VECTOR = [self.interested_front_dist, center_y-ego_y, ego_v, 0, ego_length, ego_width]
            RIGHT_REAR_NO_CAR_ENCODED_VECTOR = [self.interested_rear_dist, center_y-ego_y, 0, 0, ego_length, ego_width]
        MIDDLE_FRONT_NO_CAR_ENCODED_VECTOR = [self.interested_front_dist, -dist2current_lane_center, ego_v, 0, ego_length, ego_width]
        MIDDLE_REAR_NO_CAR_ENCODED_VECTOR = [self.interested_rear_dist, -dist2current_lane_center, 0, 0, ego_length, ego_width]

        NO_ROAD_ENCODED_VECTOR = []
        left_front = []
        left_rear = []
        middle_front = []
        middle_rear = []
        right_front = []
        right_rear = []

        # for veh in self.interested_vehicles:







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
        return position_bias, velocity_bias, heading_bias, reward

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
