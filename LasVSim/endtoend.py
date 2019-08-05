import gym
from LasVSim import lasvsim
from gym.utils import seeding
import math
import numpy as np
from LasVSim.endtoend_env_utils import shift_coordination, rotate_coordination
from collections import deque
from LasVSim.reference import Reference
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from math import pi
from collections import OrderedDict

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
        self.goal_length = 500  # episode ends on running 500m
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
        self.simulation = lasvsim.create_simulation(setting_path + 'simulation_setting_file.xml')
        self.seed()  # call this for giving self.np_random
        self.reference = Reference(self.simulation.step_length, self.horizon)
        self.interested_rear_dist = 30
        self.interested_front_dist = 60  # if you change this, you should change process action too
        self.interested_vehicles_4lane_list = []
        self.ego_dynamics_list = []
        self.interested_4lane_vehicles = []
        self.all_vehicles_list = []



        # self.reset(init_state=[-800, -150-3.75*5/2, 5, 0])

    def seed(self, seed=None):  # call this before only before training
        self.np_random, seed = seeding.np_random(seed)
        lasvsim.seed(seed)
        return [seed]

    def step(self, action):  # action is a np.array, [behavior, goal_delta_x, acc]
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
        self.all_vehicles_list = []
        self.ego_dynamics_list = []
        behavior, goal_delta_x, acc = action
        goal_y = ObservationWrapper.laneindex2centery(self.ego_road_related_info['egolane_index']) - (behavior - 1) * 3.75
        goal_delta_v = self.simulation.step_length/1000 * self.horizon * acc
        state_on_begin_of_step = [self.ego_dynamics['x'], self.ego_dynamics['y'], self.ego_dynamics['v'], self.ego_dynamics['heading']]
        ego_goal_state = [self.ego_dynamics['x'] + goal_delta_x, goal_y, np.clip(self.ego_dynamics['v'] + goal_delta_v, 0, 33), 0]
        self.reference.reset_reference_path(state_on_begin_of_step, ego_goal_state)
        reward = 0
        done = 0
        done_type = 3
        for _ in range(self.horizon):
            x, y, v, heading = self.reference.sim_step()
            lasvsim.set_ego(x, y, v, heading)
            lasvsim.sim_step()
            self.all_vehicles = lasvsim.get_all_objects()  # coordination 2
            self.ego_dynamics, self.ego_info = lasvsim.get_ego_info()
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
            self.ego_dynamics_list.append(self.ego_dynamics)
            self.all_vehicles_list.append(self.all_vehicles)

            if done:
                break
        longitudinal_reward = 0
        lateral_reward = 0
        lane_change_reward = 0
        velocity_reward = 0
        state_on_end_of_step = state_on_begin_of_step
        if not done:
            state_on_end_of_step = [self.ego_dynamics['x'], self.ego_dynamics['y'], self.ego_dynamics['v'], self.ego_dynamics['heading']]
            longitudinal_reward = 1 * state_on_end_of_step[0] - state_on_begin_of_step[0]  # forward distance reward TODO: add weight
            lateral_reward = -5 * abs(state_on_end_of_step[1] - ego_goal_state[1])  # lateral distance reward
            velocity_reward = 0.5 * (state_on_end_of_step[2] + state_on_begin_of_step[2])
            lane_change_reward = 10 * int(behavior in [0, 2])

        info = dict(done_rew=reward,
                    long_rew=longitudinal_reward,
                    lat_rew=lateral_reward,
                    vel_rew=velocity_reward,
                    lane_change_reward=lane_change_reward,
                    done_type=done_type,
                    state_on_end_of_step=state_on_end_of_step)
        reward += (longitudinal_reward + lateral_reward + lane_change_reward + velocity_reward)
        return self.obs_deque, reward, done, info

    def reset(self, **kwargs):  # if not assign 'init_state', it will generate random init state
        #  clear deque
        def random_init_state(flag=True):
            init_state = [-800, -150 - 3.75 * 5 / 2, 5, 0]
            if flag:
                x = self.np_random.uniform(0, 1) * 1000 - 800
                lane = self.np_random.choice([0, 1, 2, 3])
                y_fn = lambda lane: \
                [-150 - 3.75 * 7 / 2, -150 - 3.75 * 5 / 2, -150 - 3.75 * 3 / 2, -150 - 3.75 * 1 / 2][lane]
                y = y_fn(lane)
                v = self.np_random.uniform(0, 1) * 25
                heading = 0
                init_state = [x, y, v, heading]
            return init_state
        self.obs_deque.clear()
        if 'init_state' in kwargs:
            self.init_state = kwargs['init_state']
        else:
            self.init_state = random_init_state()

        self.final_goal_x = self.init_state[0] + self.goal_length
        lasvsim.reset_simulation(overwrite_settings={'init_state': self.init_state},
                                 init_traffic_path=self.setting_path)
        self.simulation = lasvsim.simulation
        self.all_vehicles = lasvsim.get_all_objects()
        self.ego_dynamics, self.ego_info = lasvsim.get_ego_info()
        self.ego_road_related_info = lasvsim.get_ego_road_related_info()
        self.obs_deque.append([self.all_vehicles, self.ego_dynamics, self.ego_road_related_info])
        return self.obs_deque

    def is_in_interested_area(self, ego_x, pos_x, pos_y):
        return True if ego_x - self.interested_rear_dist < pos_x < ego_x + self.interested_front_dist and -150 - 3.75 * 4 < pos_y < -150 else False

    def render(self, mode='human', **kwargs):
        self.interested_vehicles_4lane_list = []
        for index, all_vehicles in enumerate(self.all_vehicles_list):
            self.interested_4lane_vehicles = [veh for veh in all_vehicles
                                              if self.is_in_interested_area(self.ego_dynamics_list[index]['x'], veh['x'], veh['y'])]
            self.interested_vehicles_4lane_list.append(self.interested_4lane_vehicles)
        assert len(self.ego_dynamics_list) == len(self.interested_vehicles_4lane_list)
        path_points = self.reference.horizon_path_points
        for index, interested_vehicles in enumerate(self.interested_vehicles_4lane_list):
            ego_x, ego_y, ego_v, ego_heading, \
            ego_length, ego_width = self.ego_dynamics_list[index]['x'], self.ego_dynamics_list[index]['y'],\
                                    self.ego_dynamics_list[index]['v'], self.ego_dynamics_list[index]['heading'],\
                                    self.ego_dynamics_list[index]['length'], self.ego_dynamics_list[index]['width']
            shifted_x, shifted_y = shift_coordination(ego_x, ego_y, ego_x, -150 - 3.75 * 2)
            ego_car = {'ego_y': shifted_y, 'ego_width': ego_length, 'ego_height': ego_width,
                       'ego_angle': ego_heading}
            vehicles, points = self._process(interested_vehicles, ego_x, path_points)
            self._render(ego_car, vehicles, self.interested_rear_dist, self.interested_front_dist, points)

    def _process(self, interested_vehicles, ego_x, path_points):
        vehicles = []
        points = []
        for veh in interested_vehicles:
            veh_x, veh_y, veh_heading, veh_length, veh_width = veh['x'], veh['y'], veh['angle'], veh['length'], veh[
                'width']
            shifted_x, shifted_y = shift_coordination(veh_x, veh_y, ego_x, -150 - 3.75 * 2)
            vehicles.append(dict(car_x=shifted_x,
                                 car_y=shifted_y,
                                 car_width=veh_length,
                                 car_height=veh_width,
                                 car_angle=veh_heading))
        for point in path_points:
            x, y, v, heading = point['x'], point['y'], point['v'], point['heading']
            shifted_x, shifted_y = shift_coordination(x, y, ego_x, -150 - 3.75 * 2)
            points.append(dict(x=shifted_x,
                               y=shifted_y))
        return vehicles, points

    def _render(self, ego_car, vehicles, left_boder, right_boder, points):
        """
        left_boder=25
        right_boder=25
        ego_car={'ego_y': -2.5,
                       'ego_width': 5,
                       'ego_height': 2,
                       'ego_angle': 0,}
        vehicles = [{'car_x': 30-left_boder,
                       'car_y': 3,
                       'car_width': 5,
                       'car_height': 2,
                      'car_angle': 1.7
                       }]
        points=[{'x':10,'y':4},{'x':-7,'y':4}]
        """
        plt.cla()
        plt.title("Render")
        # x_major_locator = MultipleLocator(10)  # 把x轴的刻度间隔设置为5，并存在变量里
        # y_major_locator = MultipleLocator(7)

        ax = plt.gca()  # ax为两条坐标轴的实例
        # ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为5的倍数
        # ax.yaxis.set_major_locator(y_major_locator)
        plt.xlim((-left_boder, right_boder))
        plt.ylim((-10, 60))
        plt.title("test render")
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['left'].set_position(('data', 0))
        # 画车道线
        x = np.arange(-left_boder, right_boder)
        y3 = 0 * x - 3.75
        y1 = 0 * x + 0
        y2 = 0 * x + 3.75
        plt.xlabel("x ")
        plt.ylabel("y ")
        plt.plot(x, y3, color='b', linewidth=1, linestyle='--')
        plt.plot(x, y1, color='b', linewidth=1, linestyle='--')
        plt.plot(x, y2, color='b', linewidth=1, linestyle='--')
        # 环境车道线外框矩形

        rect = plt.Rectangle((0 - left_boder, -7), left_boder + right_boder, 14, angle=0.0, linewidth=1, edgecolor='b',
                             facecolor='none')
        ax.add_patch(rect)

        ego_x = 0  # 固定
        ego_y = ego_car['ego_y']
        ego_width = ego_car['ego_width']
        ego_height = ego_car['ego_height']
        ego_angle = ego_car['ego_angle']
        rect = plt.Rectangle((ego_x - 1 / 2 * ego_width, ego_y - 1 / 2 * ego_height), ego_width, ego_height, ego_angle,
                             linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # 他车
        for veh in vehicles:
            car_x = veh['car_x']
            car_y = veh['car_y']
            car_width = veh['car_width']
            car_height = veh['car_height']
            car_angle = veh['car_angle']
            rect = plt.Rectangle((car_x - 1 / 2 * car_width, car_y - 1 / 2 * car_height), car_width, car_height,
                                 car_angle,
                                 linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
        # 轨迹
        for n in points:
            n_x = n['x']
            n_y = n['y']
            plt.plot(n_x, n_y, '.', c='r')
        plt.axis('off')
        # plt.axis('equal')
        plt.pause(0.1)
        # plt.show()

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    @property
    def unwrapped(self):
        """Completely unwrap this env.

        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    def __str__(self):
        if self.spec is None:
            return '<{} instance>'.format(type(self).__name__)
        else:
            return '<{}<{}>>'.format(type(self).__name__, self.spec.id)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        # propagate exception
        return False

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
        corner_points = self.ego_info
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

        self.interested_rear_dist = self.env.interested_rear_dist
        self.interested_front_dist = self.env.interested_front_dist
        self.history_len = self.env.history_len
        self.encode_vec_len = 56  # 53 = 6dim * 8veh + 8ego
        self.encoded_obs = np.zeros((self.history_len, self.encode_vec_len))
        self.reset(init_state=[-800, -150-3.75*5/2, 5, 0])

    def reset(self, **kwargs):  # if not assign 'init_state', it will generate random init state
        self.encoded_obs = np.zeros((self.history_len, self.encode_vec_len))
        observation = self.env.reset(**kwargs)

        return self.observation(observation)

    def step(self, action):

        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        for infos in observation:
            all_vehicles, ego_dynamics, ego_road_related_info = infos
            ego_x, ego_y, ego_v, ego_heading, ego_length, ego_width = ego_dynamics['x'], \
                                                                      ego_dynamics['y'], ego_dynamics['v'], ego_dynamics['heading'], ego_dynamics['length'], ego_dynamics['width']
            dist2current_lane_center, egolane_index = ego_road_related_info['dist2current_lane_center'],\
                                                      ego_road_related_info['egolane_index']
            self.interested_vehicles = [veh for veh in all_vehicles
                                              if self.is_in_interested_area(ego_x, veh['x'], veh['y'])
                                              and veh['lane_index'] in self._interested_lane_index(egolane_index)]
            current_timestep_info = self._divide_6parts_and_encode(ego_x, ego_y, ego_v, ego_heading, ego_length,
                                                                   ego_width, dist2current_lane_center, egolane_index)
            self.encoded_obs = self.encoded_obs[1:]
            self.encoded_obs = np.append(self.encoded_obs, current_timestep_info.reshape((1, self.encode_vec_len)), axis=0)
        return self.encoded_obs  # [time_step, 6*8 + 8]

    def _interested_lane_index(self, ego_lane_index):
        info_list = [[1, 0, None], [2, 1, 0], [3, 2, 1], [None, 3, 2]]  # left, middle, right
        return info_list[ego_lane_index]

    @staticmethod
    def laneindex2centery(lane_index):
        center_y_list = [-150-7*3.75/2, -150-5*3.75/2, -150-3*3.75/2, -150-1*3.75/2]
        return center_y_list[lane_index]

    @staticmethod
    def laneindex2disttoroadedgy(lane_index, dist2current_lane_center):  # dist2current_lane_center (left positive)
        lane_center2road_left = [3.75*3+3.75/2, 3.75*2+3.75/2, 3.75*1+3.75/2, 3.75*0+3.75/2]
        lane_center2road_right = [3.75*0+3.75/2, 3.75*1+3.75/2, 3.75*2+3.75/2, 3.75*3+3.75/2]
        return lane_center2road_left[lane_index] - dist2current_lane_center, \
               lane_center2road_right[lane_index] + dist2current_lane_center

    def _divide_6parts_and_encode(self, ego_x, ego_y, ego_v, ego_heading, ego_length, ego_width, dist2current_lane_center, egolane_index):
        dist2roadleft, dist2roadright = self.laneindex2disttoroadedgy(egolane_index, dist2current_lane_center)
        EGO_ENCODED_VECTOR = [ego_v, ego_heading, ego_length, ego_width,
                              dist2current_lane_center, egolane_index, dist2roadleft, dist2roadright]  # 8 dim
        if egolane_index != 3:
            center_y = self.laneindex2centery(self._interested_lane_index(egolane_index)[0])
            LEFT_FRONT_NO_CAR_ENCODED_VECTOR = [self.interested_front_dist, center_y-ego_y, ego_v, 0, ego_length, ego_width]  # delta_x, delta_y, v, heading(in coord2), length, width
            LEFT_REAR_NO_CAR_ENCODED_VECTOR = [-self.interested_rear_dist, center_y-ego_y, 0, 0, ego_length, ego_width]
        if egolane_index != 0:
            center_y = self.laneindex2centery(self._interested_lane_index(egolane_index)[1])
            RIGHT_FRONT_NO_CAR_ENCODED_VECTOR = [self.interested_front_dist, center_y-ego_y, ego_v, 0, ego_length, ego_width]
            RIGHT_REAR_NO_CAR_ENCODED_VECTOR = [-self.interested_rear_dist, center_y-ego_y, 0, 0, ego_length, ego_width]
        MIDDLE_FRONT_NO_CAR_ENCODED_VECTOR = [self.interested_front_dist, -dist2current_lane_center, ego_v, 0, ego_length, ego_width]
        MIDDLE_REAR_NO_CAR_ENCODED_VECTOR = [-self.interested_rear_dist, -dist2current_lane_center, 0, 0, ego_length, ego_width]

        NO_ROAD_ENCODED_VECTOR = [0, 0, ego_v, 0, ego_length, ego_width]
        left_front = []
        left_rear = []
        middle_front = []
        middle_rear = []
        right_front = []
        right_rear = []
        # divide 6 parts
        if egolane_index == 3:
            for veh in self.interested_vehicles:
                delta_x = veh['x'] - ego_x
                delta_y = veh['y'] - ego_y
                v = veh['v']
                heading = veh['angle']
                length = veh['length']
                width = veh['width']
                if veh['lane_index'] == 3 and veh['x'] > ego_x:
                    middle_front.append([delta_x, delta_y, v, heading, length, width])
                elif veh['lane_index'] == 3 and veh['x'] < ego_x:
                    middle_rear.append([delta_x, delta_y, v, heading, length, width])
                elif veh['lane_index'] == 2 and veh['x'] > ego_x:
                    right_front.append([delta_x, delta_y, v, heading, length, width])
                elif veh['lane_index'] == 2 and veh['x'] < ego_x:
                    right_rear.append([delta_x, delta_y, v, heading, length, width])
                else:
                    assert 0, 'interested vehicles error'
        elif egolane_index == 0:
            for veh in self.interested_vehicles:
                delta_x = veh['x'] - ego_x
                delta_y = veh['y'] - ego_y
                v = veh['v']
                heading = veh['angle']
                length = veh['length']
                width = veh['width']
                if veh['lane_index'] == 0 and veh['x'] > ego_x:
                    middle_front.append([delta_x, delta_y, v, heading, length, width])
                elif veh['lane_index'] == 0 and veh['x'] < ego_x:
                    middle_rear.append([delta_x, delta_y, v, heading, length, width])
                elif veh['lane_index'] == 1 and veh['x'] > ego_x:
                    left_front.append([delta_x, delta_y, v, heading, length, width])
                elif veh['lane_index'] == 1 and veh['x'] < ego_x:
                    left_rear.append([delta_x, delta_y, v, heading, length, width])
                else:
                    assert 0, 'interested vehicles error'
        else:  # ego car in 1 or 2 lane
            for veh in self.interested_vehicles:
                delta_x = veh['x'] - ego_x
                delta_y = veh['y'] - ego_y
                v = veh['v']
                heading = veh['angle']
                length = veh['length']
                width = veh['width']
                if veh['lane_index'] == self._interested_lane_index(egolane_index)[0] and veh['x'] > ego_x:
                    left_front.append([delta_x, delta_y, v, heading, length, width])
                elif veh['lane_index'] == self._interested_lane_index(egolane_index)[0] and veh['x'] < ego_x:
                    left_rear.append([delta_x, delta_y, v, heading, length, width])
                elif veh['lane_index'] == egolane_index and veh['x'] > ego_x:
                    middle_front.append([delta_x, delta_y, v, heading, length, width])
                elif veh['lane_index'] == egolane_index and veh['x'] < ego_x:
                    middle_rear.append([delta_x, delta_y, v, heading, length, width])
                elif veh['lane_index'] == self._interested_lane_index(egolane_index)[2] and veh['x'] > ego_x:
                    right_front.append([delta_x, delta_y, v, heading, length, width])
                elif veh['lane_index'] == self._interested_lane_index(egolane_index)[2] and veh['x'] < ego_x:
                    right_rear.append([delta_x, delta_y, v, heading, length, width])
                else:
                    assert 0, 'interested vehicles error'

        # sort 6 parts
        if left_front:
            left_front.sort(key=lambda y: y[0])
        if left_rear:
            left_rear.sort(key=lambda y: y[0], reverse=True)
        if middle_front:
            middle_front.sort(key=lambda y: y[0])
        if middle_rear:
            middle_rear.sort(key=lambda y: y[0], reverse=True)
        if right_front:
            right_front.sort(key=lambda y: y[0])
        if right_rear:
            right_rear.sort(key=lambda y: y[0], reverse=True)

        if egolane_index == 3:
            # encode left front
            encode_left_front = NO_ROAD_ENCODED_VECTOR + NO_ROAD_ENCODED_VECTOR

            # encode left rear
            encode_left_rear = NO_ROAD_ENCODED_VECTOR

            # encode middle front
            if not middle_front:
                encode_middle_front = MIDDLE_FRONT_NO_CAR_ENCODED_VECTOR
            else:
                encode_middle_front = middle_front[0]

            # encode middle rear
            if not middle_rear:
                encode_middle_rear = MIDDLE_REAR_NO_CAR_ENCODED_VECTOR
            else:
                encode_middle_rear = middle_rear[0]

            # encode right front
            if not right_front:
                encode_right_front = RIGHT_FRONT_NO_CAR_ENCODED_VECTOR + RIGHT_FRONT_NO_CAR_ENCODED_VECTOR
            elif len(right_front) == 1:
                encode_right_front = RIGHT_FRONT_NO_CAR_ENCODED_VECTOR + right_front[0]
            else:
                assert len(right_front) >= 2
                encode_right_front = right_front[1] + right_front[0]

            # encode right rear
            if not right_rear:
                encode_right_rear = RIGHT_REAR_NO_CAR_ENCODED_VECTOR
            else:
                encode_right_rear = right_rear[0]
        elif egolane_index == 0:
            # encode left front
            if not left_front:
                encode_left_front = LEFT_FRONT_NO_CAR_ENCODED_VECTOR + LEFT_FRONT_NO_CAR_ENCODED_VECTOR
            elif len(left_front) == 1:
                encode_left_front = LEFT_FRONT_NO_CAR_ENCODED_VECTOR + left_front[0]
            else:
                assert len(left_front) >= 2
                encode_left_front = left_front[1] + left_front[0]

            # encode left rear
            if not left_rear:
                encode_left_rear = LEFT_REAR_NO_CAR_ENCODED_VECTOR
            else:
                encode_left_rear = left_rear[0]

            # encode middle front
            if not middle_front:
                encode_middle_front = MIDDLE_FRONT_NO_CAR_ENCODED_VECTOR
            else:
                encode_middle_front = middle_front[0]

            # encode middle rear
            if not middle_rear:
                encode_middle_rear = MIDDLE_REAR_NO_CAR_ENCODED_VECTOR
            else:
                encode_middle_rear = middle_rear[0]

            # encode right front
            encode_right_front = NO_ROAD_ENCODED_VECTOR + NO_ROAD_ENCODED_VECTOR

            # encode right rear
            encode_right_rear = NO_ROAD_ENCODED_VECTOR

        else:
            # encode left front
            if not left_front:
                encode_left_front = LEFT_FRONT_NO_CAR_ENCODED_VECTOR + LEFT_FRONT_NO_CAR_ENCODED_VECTOR
            elif len(left_front) == 1:
                encode_left_front = LEFT_FRONT_NO_CAR_ENCODED_VECTOR + left_front[0]
            else:
                encode_left_front = left_front[1] + left_front[0]

            # encode left rear
            if not left_rear:
                encode_left_rear = LEFT_REAR_NO_CAR_ENCODED_VECTOR
            else:
                encode_left_rear = left_rear[0]

            # encode middle front
            if not middle_front:
                encode_middle_front = MIDDLE_FRONT_NO_CAR_ENCODED_VECTOR
            else:
                encode_middle_front = middle_front[0]

            # encode middle rear
            if not middle_rear:
                encode_middle_rear = MIDDLE_REAR_NO_CAR_ENCODED_VECTOR
            else:
                encode_middle_rear = middle_rear[0]

            # encode right front
            if not right_front:
                encode_right_front = RIGHT_FRONT_NO_CAR_ENCODED_VECTOR + RIGHT_FRONT_NO_CAR_ENCODED_VECTOR
            elif len(right_front) == 1:
                encode_right_front = RIGHT_FRONT_NO_CAR_ENCODED_VECTOR + right_front[0]
            else:
                encode_right_front = right_front[1] + right_front[0]

            # encode right rear
            if not right_rear:
                encode_right_rear = RIGHT_REAR_NO_CAR_ENCODED_VECTOR
            else:
                encode_right_rear = right_rear[0]

        combined = encode_left_front + encode_left_rear + encode_middle_front +\
                   encode_middle_rear + encode_right_front + encode_right_rear + EGO_ENCODED_VECTOR
        return np.array(combined)





