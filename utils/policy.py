#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: policy.py
# =====================================

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PolynomialDecay

from utils.model import MLPNet

NAME2MODELCLS = dict([('MLP', MLPNet),])


class Policy4Toyota(tf.Module):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, args):
        super().__init__()
        self.args = args
        obs_dim, act_dim = self.args.obs_dim, self.args.act_dim
        n_hiddens, n_units, hidden_activation = self.args.num_hidden_layers, self.args.num_hidden_units, self.args.hidden_activation
        value_model_cls, policy_model_cls = NAME2MODELCLS[self.args.value_model_cls], \
                                            NAME2MODELCLS[self.args.policy_model_cls]
        self.policy = policy_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, act_dim * 2, name='policy',
                                       output_activation=self.args.policy_out_activation)
        policy_lr_schedule = PolynomialDecay(*self.args.policy_lr_schedule)
        self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr_schedule, name='adam_opt')

        self.obj_v = value_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, 1, name='obj_v')

        obj_value_lr_schedule = PolynomialDecay(*self.args.value_lr_schedule)
        self.obj_value_optimizer = self.tf.keras.optimizers.Adam(obj_value_lr_schedule, name='objv_adam_opt')
        self.models = (self.obj_v, self.policy,)
        self.optimizers = (self.obj_value_optimizer, self.policy_optimizer)

    def save_weights(self, save_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.save(save_dir + '/ckpt_ite' + str(iteration))

    def load_weights(self, load_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.restore(load_dir + '/ckpt_ite' + str(iteration) + '-1')

    def get_weights(self):
        return [model.get_weights() for model in self.models]

    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            self.models[i].set_weights(weight)

    @tf.function
    def compute_mode(self, obs):
        logits = self.policy(obs)
        mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean

    def _logits2dist(self, logits):
        mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        act_dist = self.tfd.MultivariateNormalDiag(mean, self.tf.exp(log_std))
        if self.args.action_range is not None:
            act_dist = (
                self.tfp.distributions.TransformedDistribution(
                    distribution=act_dist,
                    bijector=self.tfb.Chain(
                        [self.tfb.Affine(scale_identity_multiplier=self.args.action_range),
                         self.tfb.Tanh()])
                ))
        return act_dist

    @tf.function
    def compute_action(self, obs):
        with self.tf.name_scope('compute_action') as scope:
            logits = self.policy(obs)
            if self.args.deterministic_policy:
                mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
                return self.args.action_range * self.tf.tanh(
                    mean) if self.args.action_range is not None else mean, 0.
            else:
                act_dist = self._logits2dist(logits)
                actions = act_dist.sample()
                logps = act_dist.log_prob(actions)
                return actions, logps

    @tf.function
    def compute_vs(self, obs):
        with self.tf.name_scope('compute_vs') as scope:
            return self.vs(obs)

class PolicyWithQs(tf.Module):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    tf.config.experimental.set_visible_devices([], 'GPU')

    def __init__(self, obs_dim, act_dim,
                 value_model_cls, value_num_hidden_layers, value_num_hidden_units,
                 value_hidden_activation, value_lr_schedule,
                 policy_model_cls, policy_num_hidden_layers, policy_num_hidden_units, policy_hidden_activation,
                 policy_out_activation, policy_lr_schedule,
                 alpha, alpha_lr_schedule,
                 policy_only, double_Q, target, tau, delay_update,
                 deterministic_policy, action_range, **kwargs):
        super().__init__()
        self.policy_only = policy_only
        self.double_Q = double_Q
        self.target = target
        self.tau = tau
        self.delay_update = delay_update
        self.deterministic_policy = deterministic_policy
        self.action_range = action_range
        self.alpha = alpha

        value_model_cls, policy_model_cls = NAME2MODELCLS[value_model_cls], \
                                            NAME2MODELCLS[policy_model_cls]
        self.policy = policy_model_cls(obs_dim, policy_num_hidden_layers, policy_num_hidden_units,
                                       policy_hidden_activation, act_dim * 2, name='policy',
                                       output_activation=policy_out_activation)
        self.policy_target = policy_model_cls(obs_dim, policy_num_hidden_layers, policy_num_hidden_units,
                                              policy_hidden_activation, act_dim * 2, name='policy_target',
                                              output_activation=policy_out_activation)
        policy_lr = PolynomialDecay(*policy_lr_schedule)
        self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr, name='policy_adam_opt')

        self.Q1 = value_model_cls(obs_dim + act_dim, value_num_hidden_layers, value_num_hidden_units,
                                  value_hidden_activation, 1, name='Q1')
        self.Q1_target = value_model_cls(obs_dim + act_dim, value_num_hidden_layers, value_num_hidden_units,
                                         value_hidden_activation, 1, name='Q1_target')
        self.Q1_target.set_weights(self.Q1.get_weights())
        value_lr = PolynomialDecay(*value_lr_schedule)
        self.Q1_optimizer = self.tf.keras.optimizers.Adam(value_lr, name='Q1_adam_opt')

        self.Q2 = value_model_cls(obs_dim + act_dim, value_num_hidden_layers, value_num_hidden_units,
                                  value_hidden_activation, 1, name='Q2')
        self.Q2_target = value_model_cls(obs_dim + act_dim, value_num_hidden_layers, value_num_hidden_units,
                                         value_hidden_activation, 1, name='Q2_target')
        self.Q2_target.set_weights(self.Q2.get_weights())
        self.Q2_optimizer = self.tf.keras.optimizers.Adam(value_lr, name='Q2_adam_opt')

        if self.policy_only:
            self.target_models = ()
            self.models = (self.policy,)
            self.optimizers = (self.policy_optimizer,)
        else:
            if self.double_Q:
                assert self.target
                self.target_models = (self.Q1_target, self.Q2_target, self.policy_target,)
                self.models = (self.Q1, self.Q2, self.policy,)
                self.optimizers = (self.Q1_optimizer, self.Q2_optimizer, self.policy_optimizer,)
            elif self.target:
                self.target_models = (self.Q1_target, self.policy_target,)
                self.models = (self.Q1, self.policy,)
                self.optimizers = (self.Q1_optimizer, self.policy_optimizer,)
            else:
                self.target_models = ()
                self.models = (self.Q1, self.policy,)
                self.optimizers = (self.Q1_optimizer, self.policy_optimizer,)

        if self.alpha == 'auto':
            self.alpha_model = AlphaModel(name='alpha')
            alpha_lr = self.tf.keras.optimizers.schedules.PolynomialDecay(*alpha_lr_schedule)
            self.alpha_optimizer = self.tf.keras.optimizers.Adam(alpha_lr, name='alpha_adam_opt')
            self.models += (self.alpha_model,)
            self.optimizers += (self.alpha_optimizer,)

    def save_weights(self, save_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        target_model_pairs = [(target_model.name, target_model) for target_model in self.target_models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + target_model_pairs + optimizer_pairs))
        ckpt.save(save_dir + '/ckpt_ite' + str(iteration))

    def load_weights(self, load_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        target_model_pairs = [(target_model.name, target_model) for target_model in self.target_models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + target_model_pairs + optimizer_pairs))
        ckpt.restore(load_dir + '/ckpt_ite' + str(iteration) + '-1')

    def get_weights(self):
        return [model.get_weights() for model in self.models] + \
               [model.get_weights() for model in self.target_models]

    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            if i < len(self.models):
                self.models[i].set_weights(weight)
            else:
                self.target_models[i-len(self.models)].set_weights(weight)

    @tf.function
    def apply_gradients(self, iteration, grads):
        if self.policy_only:
            policy_grad = grads
            self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))
        else:
            if self.double_Q:
                q_weights_len = len(self.Q1.trainable_weights)
                policy_weights_len = len(self.policy.trainable_weights)
                q1_grad, q2_grad, policy_grad = grads[:q_weights_len], grads[q_weights_len:2*q_weights_len], \
                                                grads[2*q_weights_len:2*q_weights_len+policy_weights_len]
                self.Q1_optimizer.apply_gradients(zip(q1_grad, self.Q1.trainable_weights))
                self.Q2_optimizer.apply_gradients(zip(q2_grad, self.Q2.trainable_weights))
                if iteration % self.delay_update == 0:
                    self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))
                    self.update_policy_target()
                    self.update_Q1_target()
                    self.update_Q2_target()
                    if self.alpha == 'auto':
                        alpha_grad = grads[-1:]
                        self.alpha_optimizer.apply_gradients(zip(alpha_grad, self.alpha_model.trainable_weights))
            else:
                q_weights_len = len(self.Q1.trainable_weights)
                policy_weights_len = len(self.policy.trainable_weights)
                q1_grad, policy_grad = grads[:q_weights_len], grads[q_weights_len:q_weights_len+policy_weights_len]
                self.Q1_optimizer.apply_gradients(zip(q1_grad, self.Q1.trainable_weights))
                if iteration % self.delay_update == 0:
                    self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))
                    if self.alpha == 'auto':
                        alpha_grad = grads[-1:]
                        self.alpha_optimizer.apply_gradients(zip(alpha_grad, self.alpha_model.trainable_weights))
                    if self.target:
                        self.update_policy_target()
                        self.update_Q1_target()

    def update_Q1_target(self):
        tau = self.tau
        for source, target in zip(self.Q1.trainable_weights, self.Q1_target.trainable_weights):
            target.assign(tau * source + (1.0 - tau) * target)

    def update_Q2_target(self):
        tau = self.tau
        for source, target in zip(self.Q2.trainable_weights, self.Q2_target.trainable_weights):
            target.assign(tau * source + (1.0 - tau) * target)

    def update_policy_target(self):
        tau = self.tau
        for source, target in zip(self.policy.trainable_weights, self.policy_target.trainable_weights):
            target.assign(tau * source + (1.0 - tau) * target)

    @tf.function
    def compute_mode(self, obs):
        logits = self.policy(obs)
        mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        return self.action_range * self.tf.tanh(mean) if self.action_range is not None else mean

    def _logits2dist(self, logits):
        mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        act_dist = self.tfd.MultivariateNormalDiag(mean, self.tf.exp(log_std))
        if self.action_range is not None:
            act_dist = (
                self.tfp.distributions.TransformedDistribution(
                    distribution=act_dist,
                    bijector=self.tfb.Chain(
                        [self.tfb.Affine(scale_identity_multiplier=self.action_range),
                         self.tfb.Tanh()])
                ))
        return act_dist

    @tf.function
    def compute_action(self, obs):
        with self.tf.name_scope('compute_action') as scope:
            logits = self.policy(obs)
            if self.deterministic_policy:
                mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
                return self.action_range * self.tf.tanh(mean) if self.action_range is not None else mean, 0.
            else:
                act_dist = self._logits2dist(logits)
                actions = act_dist.sample()
                logps = act_dist.log_prob(actions)
                return actions, logps

    @tf.function
    def compute_target_action(self, obs):
        with self.tf.name_scope('compute_target_action') as scope:
            logits = self.policy_target(obs)
            if self.deterministic_policy:
                mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
                return self.action_range * self.tf.tanh(mean) if self.action_range is not None else mean, 0.
            else:
                act_dist = self._logits2dist(logits)
                actions = act_dist.sample()
                logps = act_dist.log_prob(actions)
                return actions, logps

    @tf.function
    def compute_Q1(self, obs, act):
        with self.tf.name_scope('compute_Q1') as scope:
            Q_inputs = self.tf.concat([obs, act], axis=-1)
            return tf.squeeze(self.Q1(Q_inputs), axis=1)

    @tf.function
    def compute_Q2(self, obs, act):
        with self.tf.name_scope('compute_Q2') as scope:
            Q_inputs = self.tf.concat([obs, act], axis=-1)
            return tf.squeeze(self.Q2(Q_inputs), axis=1)

    @tf.function
    def compute_Q1_target(self, obs, act):
        with self.tf.name_scope('compute_Q1_target') as scope:
            Q_inputs = self.tf.concat([obs, act], axis=-1)
            return tf.squeeze(self.Q1_target(Q_inputs), axis=1)

    @tf.function
    def compute_Q2_target(self, obs, act):
        with self.tf.name_scope('compute_Q2_target') as scope:
            Q_inputs = self.tf.concat([obs, act], axis=-1)
            return tf.squeeze(self.Q2_target(Q_inputs), axis=1)

    @property
    def log_alpha(self):
        return self.alpha_model.log_alpha


if __name__ == '__main__':
    pass
