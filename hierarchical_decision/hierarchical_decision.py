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


class HierarchicalDecision(object):
    def __init__(self):
        self.policy = LoadPolicy('C:\\Users\\Yangang REN\\Desktop\\env_build\\mpc\\rl_experiments\\experiment-2020-10-20-14-52-58', 90000)
        self.env = CrossroadEnd2end(training_task='left')
        self.model = EnvironmentModel('left')
        self.stg = StaticTrajectoryGenerator(self.env.training_task, self.env.init_state['ego'], self.env.v_light)
        self.obs = self.reset()

    def reset(self):
        self.obs = self.env.reset()
        # Reselect the feature point according to the inital state
        self.stg._future_point_choice(self.env.init_state['ego'])
        return self.obs

    def step(self,):
        traj_list = self.stg.generate_traj(self.env.training_task, self.env.ego_dynamics, self.env.v_light)
        traj_return = []
        for i, trajectory in enumerate(traj_list):
            return_ = 0
            self.env.set_traj(trajectory, mode='selecting')
            # initial state
            self.new_obs = self.env._get_obs()
            obs = self.new_obs
            print('selection begin')
            for step in range(10):
                action = self.policy.run(obs)
                self.model.reset(obs, self.env.training_task, trajectory, mode='selecting')
                obs, rewards, punishment = self.model.rollout_out(action)
                obs = np.squeeze(obs)
                return_ += rewards
            traj_return.append(return_.numpy().tolist())
            print('selection finished!')

        traj_return = np.array(traj_return).squeeze()
        print("-------------------------------------------------------------------")
        print('The calculated indicator:', traj_return)
        print("-------------------------------------------------------------------")
        self.env.render(traj_list, traj_return)

        # tracking in real env
        index = np.argmax(traj_return)           # 性能指标越大越好
        self.env.set_traj(traj_list[index], mode='tracking')
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

