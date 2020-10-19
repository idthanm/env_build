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


class HierarchicalDecision(object):
    def __init__(self):
        self.env = CrossroadEnd2end(training_task='left')
        self.model = EnvironmentModel('left')
        # self.policy = LoadPolicy()
        self.stg = StaticTrajectoryGenerator(self.env.training_task, self.env.init_state['ego'], self.env.v_light)
        self.obs = self.reset()

    def reset(self):
        self.env.reset()
        # Reselect the feature point according to the inital state
        self.stg._future_point_choice(self.env.init_state['ego'])

    def step(self,):
        ego_position = {'x': self.env.ego_dynamics['x'], 'y': self.env.ego_dynamics['y'], 'v_x': self.env.ego_dynamics['v_x'], 'phi': self.env.ego_dynamics['phi']}
        v_light = self.env.v_light
        traj_list = self.stg.generate_traj(self.env.training_task, ego_position, v_light)

        # self.env.set_traj(final_traj)
        # self.obs = self.env._get_obs()
        # action = self.policy(self.obs)
        self.obs, r, done, info = self.env.step([np.random.random()*2-1, 1])
        self.env.render(traj_list)
        return done


if __name__ == '__main__':
    hier_decision = HierarchicalDecision()
    done = 0
    for i in range(30):
        done = 0
        while not done:
            done = hier_decision.step()
        hier_decision.reset()

