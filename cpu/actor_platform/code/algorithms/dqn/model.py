# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from framework.algorithm.model import Model as Model_Base


class Model(Model_Base):
    def __init__(self, with_random=True):
        super().__init__(with_random)
        self.feature_shape = list(self.state_dim)
        self.feature_shape.insert(0, -1)

    def inference(self, feature):
        with tf.variable_scope("q_func"):
            self.feature_float = tf.to_float(feature, name="obs_float")
            self.feature_float = tf.reshape(self.feature_float, self.feature_shape)
            with tf.variable_scope("model"):
                self.h = self._cnn()
                self.q_value = self._create_value_network()
                self.value = tf.reduce_max(self.q_value, axis=-1)
                self.params = tf.trainable_variables()

    def _create_value_network(self):
        value_all = []
        for i in range(self.action_dim):
            value_tmp = tf.layers.dense(self.h, 1, activation=None, \
                    kernel_initializer=Orthogonal(scale=0.01), name="logits_%d" % i) 
            value_all.append(value_tmp)
        value_all_2 = tf.concat(value_all, axis=1, name="resuls_concat")
        return value_all_2

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
        h = tf.layers.dense(flat, 512,
                            activation=tf.nn.relu,
                            kernel_initializer=Orthogonal(scale=np.sqrt(2)),
                            name="hidden")
        return h


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
