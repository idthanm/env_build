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


class HierarchicalDecision(object):
    def __init__(self, task):
        self.task = task
        self.policy = LoadPolicy('C:\\Users\\Yangang REN\\Desktop\\env_build\\mpc\\rl_experiments\\experiment-2020-10-20-14-52-58', 120000)
        self.env = CrossroadEnd2end(training_task=self.task)
        self.model = EnvironmentModel(self.task)
        self.obs = self.env.reset()
        self.stg = StaticTrajectoryGenerator(self.task, self.obs, mode='static_traj')  # mode: static_traj or dyna_traj

    def reset(self):
        self.obs = self.env.reset()
        # Reselect the feature point according to the inital state
        self.stg._future_point_choice(self.obs)
        return self.obs

    @tf.function
    def virtual_rollout(self, obs):
        tracking, collision = tf.constant(0., dtype=tf.float32), tf.constant(0., dtype=tf.float32)
        for step in range(25):
            action = self.policy.run(obs)
            obs, rewards, _, punishment = self.model.rollout_out(action)
            tf.print(rewards)
            tracking = tracking - rewards
            collision += punishment
        return tracking, collision

    def step(self,):
        traj_list = self.stg.generate_traj(self.task, self.obs)
        traj_return = []
        for i, trajectory in enumerate(traj_list):
            self.env.set_traj(trajectory)
            # initial state
            obs = tf.convert_to_tensor(self.env._get_obs()[np.newaxis, :])
            self.model.add_traj(obs, trajectory, mode='selecting')
            start_time = time.time()
            tracking, collision = self.virtual_rollout(obs)
            end_time = time.time()
            # print('rollout time:', end_time-start_time)
            traj_return.append([tracking.numpy().squeeze().tolist(), collision.numpy().squeeze().tolist()])

        for i, value in enumerate(traj_return):
            traj_return[i].append(sum(value))

        # tracking in real env
        if abs(traj_return[0][1]-traj_return[1][1])>0.01:
            index = np.argmin([traj_return[0][1], traj_return[1][1]])          # todo: the maximum index
        else:
            index = np.argmin([traj_return[0][0], traj_return[1][0]])

        self.env.render(traj_list, traj_return, index)
        self.env.set_traj(traj_list[index])
        self.obs_real = self.env._get_obs()
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

