#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/10/12
# @Author  : Yang Guan; Yangang Ren (Tsinghua Univ.)
# @FileName: hier_decision.py
# =====================================
import numpy as np
from endtoend import CrossroadEnd2end
from dynamics_and_models import EnvironmentModel
from hierarchical_decision.static_traj_generator import StaticTrajectoryGenerator
from utils.load_policy import LoadPolicy
import time
import tensorflow as tf
from utils.recorder import Recorder
from utils.misc import TimerStat


class HierarchicalDecision(object):
    def __init__(self, task):
        self.task = task
        if self.task == 'left':
            self.policy = LoadPolicy('C:\\Users\\Yangang REN\\Desktop\\env_build\\utils\\models\\left', 100000)
        elif self.task == 'right':
            self.policy = LoadPolicy('C:\\Users\\Yangang REN\\Desktop\\env_build\\utils\\models\\right', 145000)
        elif self.task == 'straight':
            self.policy = LoadPolicy('C:\\Users\\Yangang REN\\Desktop\\env_build\\utils\\models\\straight', 95000)
        self.env = CrossroadEnd2end(training_task=self.task)
        self.model = EnvironmentModel(self.task)
        self.obs = self.env.reset()
        self.stg = StaticTrajectoryGenerator(mode='static_traj')
        self.recorder = Recorder()
        self.cal_timer = TimerStat()
        self.total_time = []

    def reset(self):
        self.obs = self.env.reset()
        self.stg = StaticTrajectoryGenerator(mode='static_traj')  # mode: static_traj or dyna_traj
        self.recorder.reset()
        self.recorder.save()
        self.total_time = []
        return self.obs

    # @tf.function
    # def virtual_rollout(self, obs):
    #     tracking, collision = tf.constant(0., dtype=tf.float32), tf.constant(0., dtype=tf.float32)
    #     for step in range(25):
    #         action = self.policy.run(obs)
    #         obs, rewards, _, punishment, _, _ = self.model.rollout_out(action)
    #         tracking = tracking - rewards
    #         collision += punishment
    #     return tracking, collision

    def safe_shield(self, real_obs, traj):
        action_bound = 1.0
        # action_safe_set = ([[-action_bound, -action_bound]], [[-action_bound, action_bound]], [[action_bound, -action_bound]],
        #                    [[action_bound, action_bound]])
        action_safe_set = [[[0, -action_bound]]]
        real_obs = tf.convert_to_tensor(real_obs[np.newaxis, :])
        obs = real_obs
        self.model.add_traj(obs, traj, mode='selecting')
        total_punishment = 0.0

        for step in range(3):
            action = self.policy.run(obs)
            obs, veh2veh4real = self.model.safety_calculation(obs, action) # todo: this state should be a relative state!
            total_punishment += veh2veh4real
        if total_punishment != 0:
            print('original action will cause collision within three steps!!!')
            for safe_action in action_safe_set:
                obs = real_obs
                total_punishment = 0
                # for step in range(1):
                #     obs, veh2veh4real = self.model.safety_calculation(obs, safe_action)
                #     total_punishment += veh2veh4real
                #     if veh2veh4real != 0:   # collide
                #         break
                # if total_punishment == 0:
                #     print('found the safe action', safe_action)
                #     safe_action = np.array(safe_action)
                #     break
                # else:
                #     print('still collide')
                #     safe_action = self.policy.run(real_obs).numpy().squeeze(0)
                return np.array(safe_action).squeeze(0)
        else:
            safe_action = self.policy.run(real_obs).numpy()[0]
            return safe_action

    def step(self):
        with self.cal_timer:
            start_time = time.time()
            traj_list, feature_points = self.stg.generate_traj(self.task, self.obs)
            traj_return_value = []
            # end_time = time.time()
            # print('generator time:', end_time - start_time)

            # start_time = time.time()
            for trajectory in traj_list:
                self.env.set_traj(trajectory)
                # initial state
                obs = self.env._get_obs(func='selecting')[np.newaxis, :]
                traj_value = self.policy.values(obs)
                traj_return_value.append(traj_value.numpy().squeeze().tolist())
            # end_time = time.time()
            # print('rollout time:', end_time - start_time)

            traj_return_value = np.array(traj_return_value, dtype=np.float32)
            path_index = np.argmax(traj_return_value[:, 0])

            self.env.set_traj(traj_list[path_index])
            self.obs_real = self.env._get_obs(func='tracking')

            # obtain safe action
            # start_time = time.time()
            # safe_action = self.safe_shield(self.obs_real, traj_list[path_index])
            safe_action = self.policy.run(self.obs_real).numpy()
            end_time = time.time()
            # print('Time for choosing safe action:', end_time - start_time)
            print('ALL TIME:', end_time - start_time)
        if self.env.v_light != 0 and self.env.ego_dynamics['y'] < -25 and self.env.ego_dynamics['y'] > -35 \
                and self.env.training_task != 'right':
            scaled_steer = 0.
            if self.env.ego_dynamics['v_x'] == 0.0:
                scaled_a_x = 0.33
            else:
                scaled_a_x = np.random.uniform(-0.6, -0.4)
            safe_action = np.array([scaled_steer, scaled_a_x], dtype=np.float32)
        self.total_time.append(end_time - start_time)
        self.env.render(traj_list, traj_return_value, path_index, feature_points)
        self.recorder.record(self.obs_real, safe_action, self.cal_timer.mean, path_index)

        self.obs, r, done, info = self.env.step(safe_action)
        return done


def plot_data(i):
    recorder = Recorder()
    recorder.load()
    recorder.plot_ith_episode_curves(i)


def main():
    hier_decision = HierarchicalDecision('straight')
    for i in range(300):
        done = 0
        while not done:
            done = hier_decision.step()
            if done == 2:
                print('Episode {} is SUCCESS!'.format(i))
                print(np.mean(hier_decision.total_time))
                print(hier_decision.total_time)
        hier_decision.reset()


if __name__ == '__main__':
    # main()
    plot_data(3)


