import gym
from LasVSim import lasvsim
from gym.utils import seeding

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
        self.action_space = None
        self.observation_space = None
        lasvsim.create_simulation(setting_path)
        self.seed()
        print(self.reset())

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
        self.detected_objects = lasvsim.get_detected_objects()
        self.ego_position = lasvsim.get_ego_position()
        self.ego_info = lasvsim.get_self_car_info()
        obs = self.obs_encoder()
        rew = self.rew_function()
        done = self.judge_done()
        info = self.ego_info
        return obs, rew, done, info

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns:
            observation (object): the initial observation love my baby.
        """
        init_state, goal_state, init_gear = self.generate_init_and_goal_state() # output two list, [x, y, v, heading]
                                                                                # and initial transmission ratio
        self.init_state = init_state  # a list
        self.goal_state = goal_state
        lasvsim.reset_simulation(overwrite_settings={'init_gear': init_gear, 'init_state': init_state})
        self.simulation = lasvsim.simulation
        self.detected_objects = lasvsim.get_detected_objects()
        self.ego_position = lasvsim.get_ego_position()
        self.ego_info = lasvsim.get_self_car_info()
        return self.obs_encoder()

    def generate_init_and_goal_state(self):
        init_x = self.np_random.uniform(-900.0, 500.0)
        init_lane = self.np_random.randint(4)
        lane2y_dict = {0: -150-3.75*7.0/2, 1: -150-3.75*5.0/2, 2: -150-3.75*3.0/2, 3: -150-3.75*1.0/2}
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

        def initspeed2initgear(init_v):
            if init_v >= 0 and init_v < 10:
                init_gear = 2.25
            elif init_v >= 10 and init_v < 20:
                init_gear = 1.20
            else:
                init_gear = 0.80
            return init_gear

        init_gear = initspeed2initgear(init_v)
        return init_state, goal_state, init_gear

    def render(self, mode='human'):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        raise NotImplementedError

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

    def obs_encoder(self):
        return self.detected_objects, self.ego_position, self.ego_info

    def rew_function(self):
        pass

    def judge_done(self):
        pass



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
#
#
# class ObservationWrapper(Wrapper):
#     def reset(self, **kwargs):
#         observation = self.env.reset(**kwargs)
#         return self.observation(observation)
#
#     def step(self, action):
#         observation, reward, done, info = self.env.step(action)
#         return self.observation(observation), reward, done, info
#
#     def observation(self, observation):
#         raise NotImplementedError
#
#
# class RewardWrapper(Wrapper):
#     def reset(self, **kwargs):
#         return self.env.reset(**kwargs)
#
#     def step(self, action):
#         observation, reward, done, info = self.env.step(action)
#         return observation, self.reward(reward), done, info
#
#     def reward(self, reward):
#         raise NotImplementedError
#
#
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
