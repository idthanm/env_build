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


class HierarchicalDecision(object):
    def __init__(self):
        self.policy = LoadPolicy('C:\\Users\\Yangang REN\\Desktop\\env_build\\mpc\\rl_experiments\\experiment-2020-10-20-14-52-58', 150000)
        self.env = CrossroadEnd2end(training_task='left')
        self.model = EnvironmentModel('left')
        self.stg = StaticTrajectoryGenerator(self.env.training_task, self.env.init_state['ego'], self.env.v_light, mode='static_traj') # mode: static_traj or dyna_traj
        self.obs = self.reset()

    def reset(self):
        self.obs = self.env.reset()
        # Reselect the feature point according to the inital state
        self.stg._future_point_choice(self.env.init_state['ego'])
        return self.obs

    def step(self,):
        start_time = time.time()
        traj_list = self.stg.generate_traj(self.env.training_task, self.env.ego_dynamics, self.env.v_light)
        end_time = time.time()
        # print('time of generating:', end_time-start_time)

        traj_return = []
        for i, trajectory in enumerate(traj_list):
            tracking, collision = 0, 0
            self.env.set_traj(trajectory)
            # initial state
            obs = self.env._get_obs()
            for step in range(10):
                start_time = time.time()
                action = self.policy.run(obs)
                end_time1 = time.time()
                self.model.reset(obs, self.env.training_task, trajectory, mode='selecting')
                end_time2 = time.time()
                obs, rewards, punishment = self.model.rollout_out(action)
                end_time3 = time.time()
                obs = np.squeeze(obs)
                tracking += -rewards
                collision += punishment

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
    hier_decision = HierarchicalDecision()
    done = 0
    for i in range(30):
        done = 0
        while not done:
            done = hier_decision.step()
        hier_decision.reset()

