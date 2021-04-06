# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from framework.algorithm.model import Model as Model_Base


class Model(Model_Base):
    def __init__(self, with_random=True):
        super().__init__(with_random)

    def inference(self, feature):
        self.feature_float = tf.to_float(feature, name="obs_float")
        with tf.variable_scope("model"):
            self.h = self._cnn()
            self.pi_logits = self._create_policy_network()
            self.value = self._create_value_network()
            self.params = tf.trainable_variables()
        self.action = self._sample(self.pi_logits)
        self.neg_log_pi = self.neg_log_prob(self.action, "neg_log_pi_old", self.pi_logits)

    def neg_log_prob(self, action, name, pi_logits):
        one_hot_actions = tf.one_hot(action, self.action_dim)
        pi_logits_substract_max = self.pi_logits - tf.reduce_max(self.pi_logits, axis=-1, keep_dims=True)
        exp_pi_logits = tf.exp(pi_logits_substract_max)
        exp_pi_logits_sum = tf.reduce_sum(exp_pi_logits, axis=-1, keep_dims=True)
        self.softmax_pi = (1.0 * exp_pi_logits / exp_pi_logits_sum + \
                self.min_policy) / (1 + self.min_policy*self.action_dim)
        return -tf.reduce_sum(one_hot_actions*tf.log(self.softmax_pi), axis=-1)

    def _create_policy_network(self):
        pi_logits_all = []
        for i in range(self.action_dim):
            pi_logit_tmp = tf.layers.dense(self.h, 1, activation=None,\
                               kernel_initializer=Orthogonal(scale=0.01), name="logits_%d" % i)
            pi_logits_all.append(pi_logit_tmp)
        pi_logits_all_2 = tf.concat(pi_logits_all, axis=1, name="resuls_concat")
        return pi_logits_all_2

    def _create_value_network(self):
        value = tf.layers.dense(self.h, 1, activation=None,\
                                kernel_initializer=Orthogonal(), name="value")
        return value[:, 0]

    def _cnn(self):
        scaled_images = tf.cast(self.feature_float, tf.float32) / 255.
        h1 = tf.layers.conv2d(scaled_images,
                              name="conv1",
                              filters=32,
                              kernel_size=8,
                              kernel_initializer=Orthogonal(scale=np.sqrt(2)),
                              strides=4,
                              padding="valid",
                              activation=tf.nn.relu)
        h2 = tf.layers.conv2d(h1,
                              name="conv2",
                              filters=64,
                              kernel_size=4,
                              kernel_initializer=Orthogonal(scale=np.sqrt(2)),
                              strides=2,
                              padding="valid",
                              activation=tf.nn.relu)
        h3 = tf.layers.conv2d(h2,
                              name="conv3",
                              filters=64,
                              kernel_size=3,
                              kernel_initializer=Orthogonal(scale=np.sqrt(2)),
                              strides=1,
                              padding="valid",
                              activation=tf.nn.relu)
        nh = np.prod([v.value for v in h3.get_shape()[1:]])
        flat = tf.reshape(h3, [-1, nh])

        h_fc1 = tf.layers.dense(flat, 512,\
                activation=tf.nn.relu,\
                kernel_initializer=Orthogonal(scale=np.sqrt(2)),\
                name="fc_1")

        h_fc2 = tf.layers.dense(h_fc1, 256, activation=None, \
                kernel_initializer=Orthogonal(), name="fc_2")

        h = tf.layers.dense(h_fc2, 128, activation=None, \
                kernel_initializer=Orthogonal(), name="fc_3")
        return h

    def _sample(self, pi_logits):
        if self.with_random:
            softmax_pi_logits = tf.nn.softmax(pi_logits, axis=-1)
            softmax_pi_logits = tf.log(softmax_pi_logits) / self.T
            exp_pi_logits = tf.exp(softmax_pi_logits)
            new_pi_logits = exp_pi_logits / tf.reduce_sum(exp_pi_logits, axis=-1, keep_dims=True)
            return tf.reshape(tf.multinomial(tf.log(new_pi_logits), 1), [-1])
        else:
            return tf.argmax(pi_logits, axis=-1, name="action")


class Orthogonal(object):
    def __init__(self, scale=1.):
        self.scale = scale

    def __call__(self, shape, dtype=None, partition_info=None):
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (self.scale * q[:shape[0], :shape[1]]).astype(np.float32)

    def get_config(self):
        return {
            'scale': self.scale
        }
