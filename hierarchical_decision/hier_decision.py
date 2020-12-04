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


class HierarchicalDecision(object):
    def __init__(self, task):
        self.task = task
        if self.task == 'left':
            self.policy = LoadPolicy('C:\\Users\\Yangang REN\\Desktop\\env_build\\utils\\models\\left', 80000)
        elif self.task == 'right':
            self.policy = LoadPolicy('C:\\Users\\Yangang REN\\Desktop\\env_build\\utils\\models\\right', 145000)
        elif self.task == 'straight':
            self.policy = LoadPolicy('C:\\Users\\Yangang REN\\Desktop\\env_build\\utils\\models\\straight', 60000)
        self.env = CrossroadEnd2end(training_task=self.task)
        self.model = EnvironmentModel(self.task)
        self.obs = self.env.reset()
        self.stg = StaticTrajectoryGenerator(mode='static_traj', ref_index=self.env.ref_path.ref_index)

    def reset(self):
        self.obs = self.env.reset()
        self.stg = StaticTrajectoryGenerator(mode='static_traj', ref_index=self.env.ref_path.ref_index)  # mode: static_traj or dyna_traj
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

    # @staticmethod
    # def path_selection(traj_return):
    #     tracking_error = traj_return[:, 0].tolist()
    #     collision_risk = traj_return[:, 1].tolist()
    #     zero_num = collision_risk.count(0)
    #     if zero_num == 3:
    #         path_index = np.argmin(tracking_error)
    #     elif zero_num == 2:
    #         index = [x for x, y in list(enumerate(collision_risk)) if y == 0.0]
    #         path_index = np.argmin(tracking_error[index])
    #     elif zero_num == 1:
    #         path_index = collision_risk.index(0)
    #     else:
    #         path_index = np.argmin(collision_risk)
    #     return path_index

    def safe_shield(self, real_obs, traj, action):
        action_bound = 1.05
        action_safe_set = ([[-action_bound, -action_bound]], [[-action_bound, action_bound]], [[action_bound, -action_bound]],
                           [[action_bound, action_bound]])
        real_obs = tf.convert_to_tensor(real_obs[np.newaxis, :])
        obs = real_obs
        self.model.add_traj(obs, traj, mode='selecting')
        total_punishment = 0
        for step in range(1):
            action = self.policy.run(obs)
            obs, veh2veh4real = self.model.safety_calculation(obs, action) # todo: this state should be a relative state!
            total_punishment += veh2veh4real
        if total_punishment != 0:
            print('original action will cause collision within three steps!!!')
            for safe_action in action_safe_set:
                obs = real_obs
                total_punishment = 0
                for step in range(1):
                    obs, veh2veh4real = self.model.safety_calculation(obs, safe_action)
                    total_punishment += veh2veh4real
                    if veh2veh4real != 0:   # collide
                        break
                if total_punishment == 0:
                    print('found the safe action', safe_action)
                    safe_action = np.array([safe_action])
                    break
                else:
                    print('still collide')
                    safe_action = self.policy.run(real_obs).numpy().squeeze(0)
        else:
            safe_action = self.policy.run(real_obs).numpy().squeeze(0)
        return safe_action

    def step(self):
        start_time = time.time()
        traj_list, feature_points = self.stg.generate_traj(self.task, self.obs)
        end_time = time.time()
        # print('generate time:', end_time-start_time)
        traj_return_rollout = []
        traj_return_value = []
        for trajectory in traj_list:
            self.env.set_traj(trajectory)
            # initial state
            obs = tf.convert_to_tensor(self.env._get_obs(func='selecting')[np.newaxis, :])
            start_time = time.time()
            traj_value = self.policy.values(obs)
            end_time = time.time()
            # print('rollout time:', end_time-start_time)
            # self.model.add_traj(obs, trajectory, mode='selecting')
            # start_time = time.time()
            # tracking, collision = self.virtual_rollout(obs)
            # end_time = time.time()
            # # print('rollout time:', end_time-start_time)
            # traj_return_rollout.append([tracking.numpy().squeeze().tolist(), collision.numpy().squeeze().tolist()])
            traj_return_value.append(traj_value.numpy().squeeze().tolist())

        # traj_return_rollout = np.array(traj_return_rollout, dtype=np.float32)
        traj_return_value = np.array(traj_return_value, dtype=np.float32)
        # path_index = self.path_selection(traj_return_rollout)
        path_index = np.argmax(traj_return_value[:, 0])
        self.env.render(traj_list, traj_return_value, path_index, feature_points)
        self.env.set_traj(traj_list[path_index])
        self.obs_real = self.env._get_obs(func='tracking')

        action = self.policy.run(self.obs_real[np.newaxis, :])
        # action = tf.constant([[-0.0, 1.0]], dtype=tf.float32)
        # add the safe_action
        safe_action = self.safe_shield(self.obs_real, traj_list[path_index], action)
        self.obs, r, done, info = self.env.step(safe_action)
        return done


if __name__ == '__main__':
    hier_decision = HierarchicalDecision('left')
    done = 0
    for i in range(300):
        done = 0
        while not done:
            done = hier_decision.step()
        hier_decision.reset()

