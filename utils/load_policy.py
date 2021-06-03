#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/30
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: load_policy.py
# =====================================
import argparse
import json

import tensorflow as tf
import numpy as np

from endtoend import CrossroadEnd2endPiIntegrate
from utils.policy import Policy4Toyota
from utils.preprocessor import Preprocessor


class LoadPolicy(object):
    def __init__(self, exp_dir, iter):
        model_dir = exp_dir + '/models'
        parser = argparse.ArgumentParser()
        params = json.loads(open(exp_dir + '/config.json').read())
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        self.args = parser.parse_args()
        env = CrossroadEnd2endPiIntegrate(num_future_data=self.args.env_kwargs_num_future_data)
        self.policy = Policy4Toyota(self.args)
        self.policy.load_weights(model_dir, iter)
        self.preprocessor = Preprocessor((self.args.obs_dim,), self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.reward_scale, self.args.reward_shift, args=self.args,
                                         gamma=self.args.gamma)
        # self.preprocessor.load_params(load_dir)
        init_obs = env.reset()[np.newaxis, :]
        init_obs_ego = init_obs[:, :env.ego_info_dim + env.per_tracking_info_dim * (env.num_future_data+1)+env.task_info_dim]
        init_obs_other = np.reshape(init_obs[:, env.ego_info_dim + env.per_tracking_info_dim *
                                            (env.num_future_data+1) + env.task_info_dim:], [-1, env.per_veh_info_dim])
        self.run_batch(init_obs_ego, init_obs_other, [env.veh_num])
        self.obj_value_batch(init_obs_ego, init_obs_other, [env.veh_num])

    # @tf.function
    # def run(self, obs):
    #     processed_obs = self.preprocessor.np_process_obses(obs)
    #     action, _ = self.policy.compute_action(processed_obs[np.newaxis, :])
    #     return action[0]
    #
    # @tf.function
    # def obj_value(self, obs):
    #     processed_obs = self.preprocessor.np_process_obses(obs)
    #     value = self.policy.compute_obj_v(processed_obs[np.newaxis, :])
    #     return value

    # @tf.function
    def run_batch(self, obses_ego, obses_other, start_veh_num):
        processed_obs_ego, processed_obs_other = self.preprocessor.tf_process_obses_PI(obses_ego, obses_other)
        processed_obs = self.get_states(processed_obs_ego, processed_obs_other, start_veh_num, grad=False)
        actions, _ = self.policy.compute_action(processed_obs)
        return actions

    # @tf.function
    def obj_value_batch(self, obses_ego, obses_other, start_veh_num):
        processed_obs_ego, processed_obs_other = self.preprocessor.tf_process_obses_PI(obses_ego, obses_other)
        processed_obs = self.get_states(processed_obs_ego, processed_obs_other, start_veh_num, grad=False)
        values = self.policy.compute_obj_v(processed_obs)
        return values

    def get_states(self, processed_obses_ego, processed_obses_other, start_veh_num, grad):
        PI_obses_other = self.policy.compute_PI(processed_obses_other)
        PI_obses_other_sum = []
        index = 0
        for i in range(len(start_veh_num)):
            PI_obses_other_sum.append(tf.math.reduce_sum(PI_obses_other[index: index+start_veh_num[i], :], keepdims=True, axis=0))
            index += start_veh_num[i]
        PI_obses_other_sum = tf.concat(PI_obses_other_sum, axis=0)
        if not grad:
            PI_obses_other_sum = tf.stop_gradient(PI_obses_other_sum)
        processed_obses = tf.concat((processed_obses_ego, PI_obses_other_sum), axis=1)
        return processed_obses

