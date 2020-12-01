#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: preprocessor.py
# =====================================

import numpy as np
import tensorflow as tf


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
        self.tf_mean = tf.Variable(tf.zeros(shape), dtype=tf.float32, trainable=False)
        self.tf_var = tf.Variable(tf.ones(shape), dtype=tf.float32, trainable=False)

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)
        self.tf_mean.assign(tf.constant(self.mean))
        self.tf_var.assign(tf.constant(self.var))

    def set_params(self, mean, var, count):
        self.mean = mean
        self.var = var
        self.count = count
        self.tf_mean.assign(tf.constant(self.mean))
        self.tf_var.assign(tf.constant(self.var))

    def get_params(self, ):
        return self.mean, self.var, self.count


class Preprocessor(object):
    def __init__(self, ob_shape, obs_ptype='normalize', rew_ptype='normalize', obs_scale=None,
                 rew_scale=None, rew_shift=None, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, **kwargs):
        self.obs_ptype = obs_ptype
        self.ob_rms = RunningMeanStd(shape=ob_shape) if self.obs_ptype == 'normalize' else None
        self.rew_ptype = rew_ptype
        self.ret_rms = RunningMeanStd(shape=()) if self.rew_ptype == 'normalize' else None
        self.obs_scale = np.array(obs_scale) if self.obs_ptype == 'scale' else None
        self.rew_scale = rew_scale if self.rew_ptype == 'scale' else None
        self.rew_shift = rew_shift if self.rew_ptype == 'scale' else None

        self.clipob = clipob
        self.cliprew = cliprew

        self.gamma = gamma
        self.epsilon = epsilon
        self.num_agent = None
        if 'num_agent' in kwargs.keys():
            self.ret = np.zeros(kwargs['num_agent'])
            self.num_agent = kwargs['num_agent']
        else:
            self.ret = 0

    def process_rew(self, rew, done):
        if self.rew_ptype == 'normalize':
            if self.num_agent is not None:
                self.ret = self.ret * self.gamma + rew
                self.ret_rms.update(self.ret)
                rew = np.clip(rew / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
                self.ret = np.where(done == 1, np.zeros(self.ret), self.ret)
            else:
                self.ret = self.ret * self.gamma + rew
                self.ret_rms.update(np.array([self.ret]))
                rew = np.clip(rew / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
                self.ret = 0 if done else self.ret
            return rew
        elif self.rew_ptype == 'scale':
            return (rew + self.rew_shift) * self.rew_scale
        else:
            return rew

    def process_obs(self, obs):
        if self.obs_ptype == 'normalize':
            if self.num_agent is not None:
                self.ob_rms.update(obs)
                obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
                return obs
            else:
                self.ob_rms.update(np.array([obs]))
                obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob,
                              self.clipob)
                return obs
        elif self.obs_ptype == 'scale':
            return obs * self.obs_scale
        else:
            return obs

    def np_process_obses(self, obses):
        if self.obs_ptype == 'normalize':
            obses = np.clip((obses - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obses
        elif self.obs_ptype == 'scale':
            return obses * self.obs_scale
        else:
            return obses

    def np_process_rewards(self, rewards):
        if self.rew_ptype == 'normalize':
            rewards = np.clip(rewards / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
            return rewards
        elif self.rew_ptype == 'scale':
            return (rewards + self.rew_shift) * self.rew_scale
        else:
            return rewards

    def tf_process_obses(self, obses):
        with tf.name_scope('obs_process') as scope:
            if self.obs_ptype == 'normalize':
                obses = tf.clip_by_value(
                    (obses - self.ob_rms.tf_mean) / tf.sqrt(self.ob_rms.tf_var + tf.constant(self.epsilon)),
                    -self.clipob,
                    self.clipob)
                return obses
            elif self.obs_ptype == 'scale':
                return obses * tf.convert_to_tensor(self.obs_scale, dtype=tf.float32)

            else:
                return tf.convert_to_tensor(obses, dtype=tf.float32)

    def tf_process_rewards(self, rewards):
        with tf.name_scope('reward_process') as scope:
            if self.rew_ptype == 'normalize':
                rewards = tf.clip_by_value(rewards / tf.sqrt(self.ret_rms.tf_var + tf.constant(self.epsilon)),
                                           -self.cliprew,
                                           self.cliprew)
                return rewards
            elif self.rew_ptype == 'scale':
                return (rewards+tf.convert_to_tensor(self.rew_shift, dtype=tf.float32)) \
                       * tf.convert_to_tensor(self.rew_scale, dtype=tf.float32)
            else:
                return tf.convert_to_tensor(rewards, dtype=tf.float32)

    def set_params(self, params):
        if self.ob_rms:
            self.ob_rms.set_params(*params['ob_rms'])
        if self.ret_rms:
            self.ret_rms.set_params(*params['ret_rms'])

    def get_params(self):
        tmp = {}
        if self.ob_rms:
            tmp.update({'ob_rms': self.ob_rms.get_params()})
        if self.ret_rms:
            tmp.update({'ret_rms': self.ret_rms.get_params()})

        return tmp

    def save_params(self, save_dir):
        np.save(save_dir + '/ppc_params.npy', self.get_params())

    def load_params(self, load_dir):
        params = np.load(load_dir + '/ppc_params.npy', allow_pickle=True)
        params = params.item()
        self.set_params(params)
