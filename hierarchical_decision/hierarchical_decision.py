#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/10/12
# @Author  : Yang Guan; Yangang Ren (Tsinghua Univ.)
# @FileName: hierarchical_decision.py
# =====================================

from endtoend import CrossroadEnd2end
from dynamics_and_models import EnvironmentModel


class HierarchicalDecision(object):
    def __init__(self, task):
        self.task = task
        env = CrossroadEnd2end()
        model = EnvironmentModel()
        # policy = LoadPolicy()
        # stg = StaticTrajectoryGenerator(model, policy)
        self.obs = self.reset()

    def reset(self):
        return self.env.reset()

    def step(self,):
        # traj_list, final_traj = self.stg.generate_and_select_static_trajectories(self.task, self.obs)
        # self.env.set_traj(final_traj)
        # self.obs = self.env._get_obs()
        # action = self.policy(self.obs)
        # self.obs, r, done, info = self.env.step(action)
        # self.env.render()
        # return done
        pass


if __name__ == '__main__':
    hier_decision = HierarchicalDecision()
    done = 0
    for i in range(10):
        while not done:
            done = hier_decision.step()
        hier_decision.reset()

