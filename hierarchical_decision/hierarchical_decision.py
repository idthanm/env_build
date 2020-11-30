#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/10/12
# @Author  : Yang Guan; Yangang Ren (Tsinghua Univ.)
# @FileName: hierarchical_decision.py
# =====================================
import numpy as np
from endtoend import CrossroadEnd2end
from dynamics_and_models import EnvironmentModel
from static_traj_generator import StaticTrajectoryGenerator
from mpc.mpc_ipopt import LoadPolicy
import time
import tensorflow as tf
from collections import Counter


class HierarchicalDecision(object):
    def __init__(self, task):
        self.task = task
        self.policy = LoadPolicy('C:\\Users\\Yangang REN\\Desktop\\env_build\\mpc\\rl_experiments\\experiment-2020-10-20-14-52-58', 145000)
        self.env = CrossroadEnd2end(training_task=self.task)
        self.model = EnvironmentModel(self.task)
        self.obs = self.env.reset()
        self.stg = StaticTrajectoryGenerator(self.task, self.obs, mode='static_traj')  # mode: static_traj or dyna_traj

    def reset(self):
        self.obs = self.env.reset()
        return self.obs

    @tf.function
    def virtual_rollout(self, obs):
        tracking, collision = tf.constant(0., dtype=tf.float32), tf.constant(0., dtype=tf.float32)
        for step in range(25):
            action = self.policy.run(obs)
            obs, rewards, _, punishment, _, _ = self.model.rollout_out(action)
            tracking = tracking - rewards
            collision += punishment
        return tracking, collision

    @staticmethod
    def path_selection(traj_return):
        tracking_error = traj_return[:, 0].tolist()
        collision_risk = traj_return[:, 1].tolist()
        zero_num = collision_risk.count(0)
        if zero_num == 3:
            path_index = np.argmin(tracking_error)
        elif zero_num == 2:
            index = [x for x, y in list(enumerate(collision_risk)) if y == 0.0]
            path_index = np.argmin(tracking_error[index])
        elif zero_num == 1:
            path_index = collision_risk.index(0)
        else:
            path_index = np.argmin(collision_risk)
        return path_index

    def step(self,):
        start_time = time.time()
        traj_list, feature_points = self.stg.generate_traj(self.task, self.obs)
        end_time = time.time()
        # print('generate time:', end_time-start_time)
        traj_return = []
        for i, trajectory in enumerate(traj_list):
            self.env.set_traj(trajectory)
            # initial state
            obs = tf.convert_to_tensor(self.env._get_obs(func='selecting')[np.newaxis, :])
            self.model.add_traj(obs, trajectory, mode='selecting')
            start_time = time.time()
            tracking, collision = self.virtual_rollout(obs)
            end_time = time.time()
            print('rollout time:', end_time-start_time)
            traj_return.append([tracking.numpy().squeeze().tolist(), collision.numpy().squeeze().tolist()])

        traj_return = np.array(traj_return, dtype=np.float32)
        path_index = self.path_selection(traj_return)
        self.env.render(traj_list, traj_return, path_index, feature_points)
        self.env.set_traj(traj_list[path_index])
        self.obs_real = self.env._get_obs(func='tracking')

        action = self.policy.run(self.obs_real)
        self.obs, r, done, info = self.env.step(action)
        return done


if __name__ == '__main__':
    hier_decision = HierarchicalDecision('left')
    done = 0
    for i in range(30):
        done = 0
        while not done:
            done = hier_decision.step()
        hier_decision.reset()

