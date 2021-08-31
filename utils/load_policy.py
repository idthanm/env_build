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

from endtoend import CrossroadEnd2endMixPiFix
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
        env = CrossroadEnd2endMixPiFix(training_task=self.args.env_kwargs_training_task,
                                       num_future_data=self.args.env_kwargs_num_future_data)
        self.policy = Policy4Toyota(self.args)
        self.policy.load_weights(model_dir, iter)
        self.preprocessor = Preprocessor((self.args.obs_dim,), self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.reward_scale, self.args.reward_shift, args=self.args,
                                         gamma=self.args.gamma)
        # self.preprocessor.load_params(load_dir)
        # init_obs = env.reset()[np.newaxis, :]
        # # extract infos for each kind of participants
        # start = 0; end = self.args.state_ego_dim + self.args.state_track_dim
        # obs_ego = init_obs[:, start:end]
        # start = end; end = start + self.args.state_bike_dim
        # obs_bike = init_obs[:, start:end]
        # start = end; end = start + self.args.state_person_dim
        # obs_person = init_obs[:, start:end]
        # start = end; end = start + self.args.state_veh_dim
        # obs_veh = init_obs[:, start:end]
        #
        # obs_bike = np.reshape(obs_bike, [-1, self.args.per_bike_dim])
        # obs_person = np.reshape(obs_person, [-1, self.args.per_person_dim])
        # obs_veh = np.reshape(obs_veh, [-1, self.args.per_veh_dim])
        #
        # self.run_batch(obs_ego, obs_bike, obs_person, obs_veh)
        # self.obj_value_batch(obs_ego, obs_bike, obs_person, obs_veh)

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
    def run_batch(self, obs_ego, obs_bike, obs_person, obs_veh):
        processed_obs_ego, processed_obs_bike, processed_obs_person, processed_obs_veh \
            = self.preprocessor.process_obs_PI(obs_ego, obs_bike, obs_person, obs_veh)

        processed_obs = self.get_states(processed_obs_ego, processed_obs_bike, processed_obs_person, processed_obs_veh, grad=False)
        actions, _ = self.policy.compute_action(processed_obs)
        return actions

    # @tf.function
    def obj_value_batch(self, obs_ego, obs_bike, obs_person, obs_veh):
        processed_obs_ego, processed_obs_bike, processed_obs_person, processed_obs_veh = self.preprocessor.tf_process_obses_PI(obs_ego, obs_bike, obs_person, obs_veh)
        processed_obs = self.get_states(processed_obs_ego, processed_obs_bike, processed_obs_person, processed_obs_veh, grad=False)
        values = self.policy.compute_obj_v(processed_obs)
        return values

    def get_states(self, processed_obses_ego, processed_obses_bike, processed_obses_person, processed_obses_veh, grad):
        PI_obses_bike = self.policy.compute_PI(processed_obses_bike)
        PI_obses_person = self.policy.compute_PI(processed_obses_person)
        PI_obses_veh = self.policy.compute_PI(processed_obses_veh)

        PI_obses_bike_sum, PI_obses_person_sum, PI_obses_veh_sum = [], [], []
        for i in range(len(processed_obses_ego)):
            PI_obses_bike_sum.append(tf.math.reduce_sum(PI_obses_bike[i * self.args.max_bike_num: (i+1) * self.args.max_bike_num, :],
                                                        keepdims=True, axis=0))
            PI_obses_person_sum.append(tf.math.reduce_sum(PI_obses_person[i * self.args.max_person_num: (i+1) * self.args.max_person_num, :],
                                                          keepdims=True, axis=0))
            PI_obses_veh_sum.append(tf.math.reduce_sum(PI_obses_veh[i * self.args.max_veh_num: (i+1) * self.args.max_veh_num, :],
                                                       keepdims=True, axis=0))

        PI_obses_bike_sum = tf.concat(PI_obses_bike_sum, axis=0)
        PI_obses_person_sum = tf.concat(PI_obses_person_sum, axis=0)
        PI_obses_veh_sum = tf.concat(PI_obses_veh_sum, axis=0)

        if self.args.per_bike_dim == self.args.per_person_dim == self.args.per_veh_dim:
            PI_obses_other_sum = PI_obses_bike_sum + PI_obses_person_sum + PI_obses_veh_sum
        else:
            PI_obses_other_sum = tf.concat([PI_obses_bike_sum, PI_obses_person_sum, PI_obses_veh_sum],axis=1)
        processed_obses = tf.concat((processed_obses_ego, PI_obses_other_sum), axis=1)
        return processed_obses

