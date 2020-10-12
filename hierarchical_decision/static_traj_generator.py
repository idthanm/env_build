#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/10/12
# @Author  : Yang Guan; Yangang Ren (Tsinghua Univ.)
# @FileName: hierarchical_decision.py
# =====================================


class StaticTrajectoryGenerator(object):
    def __init__(self, model, policy):
        self.model = model
        self.policy = policy

    def generate_and_select_traj(self, task, obs):
        pass

    def generate_traj(self):
        """"""
        pass

    def select_traj(self):
        pass
        # for i in traj_list():
        #   pass
