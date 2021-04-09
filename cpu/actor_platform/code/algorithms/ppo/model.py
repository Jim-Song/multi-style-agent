# -*- coding: utf-8 -*-
import math
import tensorflow as tf
import numpy as np
from framework.algorithm.model import Model as Model_Base
from config.config import Config


class Model(Model_Base):
    def __init__(self, with_random=True):
        super().__init__(with_random)
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.state_dim = Config.INPUT_DIM

        self.long_term_style_interval = Config.LONG_TERM_STYLE_INTERVAL
        self.short_term_style_interval = Config.SHORT_TERM_STYLE_INTERVAL
        self.causal_cnn_receptive_field = Config.CAUSAL_CNN_RECEPTIVE_FIELD
        self.hidden_channel = Config.HIDDEN_CHANNEL
        self.long_term_causal_conv1d_layers = int(math.log2(self.long_term_style_interval) + 1 -
                                                  math.log2(self.causal_cnn_receptive_field))
        self.short_term_causal_conv1d_layers = int(math.log2(self.short_term_style_interval) + 1 -
                                                   math.log2(self.causal_cnn_receptive_field))

        assert self.long_term_causal_conv1d_layers == math.log2(self.long_term_style_interval) + 1 - math.log2(
            self.causal_cnn_receptive_field)
        assert self.short_term_causal_conv1d_layers == math.log2(self.short_term_style_interval) + 1 - math.log2(
            self.causal_cnn_receptive_field)

    def inference(self, feature, lstm_state, style_input):
        init_lstm_c, init_lstm_h = lstm_state
        with tf.variable_scope("model"):
            self.feature_float = tf.to_float(feature, name="obs_float")
            self.style_input = style_input
            self.h = self._cnn()
            self.h = tf.concat([self.h, self.style_input], axis=-1, name='style_concat')
            self.h = tf.layers.dense(self.h, 128, activation=None, name="style_fc")
            reshape_fc_public_result = tf.reshape(self.h, [-1, 128],
                                                  name="reshape_fc_public_result")
            lstm_c = tf.reshape(init_lstm_c, [-1, self.lstm_unit_size])
            lstm_h = tf.reshape(init_lstm_h, [-1, self.lstm_unit_size])
            lstm_initial_state = tf.nn.rnn_cell.LSTMStateTuple(lstm_c, lstm_h)
            with tf.variable_scope("public_lstm"):
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.lstm_unit_size, forget_bias=1.0)
                with tf.variable_scope("rnn"):
                    state = lstm_initial_state
                    lstm_output, self.lstm_state = lstm_cell(reshape_fc_public_result, state)
                self.reshape_lstm_output_result = tf.reshape(lstm_output, [-1, self.lstm_unit_size],
                                                              name="reshape_lstm_outputs_result")
            self.pi_logits = self._create_policy_network()
            self.value = self._create_value_network()
            self.params = tf.trainable_variables()
        self.action = self._sample(self.pi_logits)
        self.neg_log_pi = self.neg_log_prob(self.action, "neg_log_pi_old", self.pi_logits)

    def style_discriminator(self, feature, style_input):
        with tf.variable_scope("style_discriminator"):
            self.feature_float = tf.to_float(feature, name="obs_float")
            # self.feature_float = tf.reshape(self.feature_float, [-1, self.state_dim[0], self.state_dim[1], self.state_dim[2]])
            self.h = self._cnn()

            long_term_style = self.h
            long_term_style = tf.expand_dims(long_term_style, 0)
            for i in range(self.long_term_causal_conv1d_layers):
                dilation_rate = int(2 ** i * self.causal_cnn_receptive_field / 2)
                long_term_style = tf.compat.v1.layers.conv1d(long_term_style, self.hidden_channel, 2, strides=1,
                                                             dilation_rate=dilation_rate, padding='causal',
                                                             name='long_causal_conv1d{}'.format(i))
            long_term_style = tf.squeeze(long_term_style, 0)

            short_term_style = self.h
            short_term_style = tf.expand_dims(short_term_style, 0)
            for i in range(self.short_term_causal_conv1d_layers):
                dilation_rate = int(2 ** i * self.causal_cnn_receptive_field / 2)
                short_term_style = tf.compat.v1.layers.conv1d(short_term_style, self.hidden_channel, 2, strides=1,
                                                              dilation_rate=dilation_rate, padding='causal',
                                                              name='short_causal_conv1d{}'.format(i))
            short_term_style = tf.squeeze(short_term_style, 0)

            long_term_style = tf.layers.dense(long_term_style, 128, activation=None,
                                              kernel_initializer=Orthogonal(), name="style_long_dense_1")
            long_term_style = tf.layers.dense(long_term_style, 128, activation=None,
                                              kernel_initializer=Orthogonal(), name="style_long_dense_2")
            self.pred_long_term_style = tf.layers.dense(long_term_style, Config.NUM_HIDDEN_STYLES, activation=None,
                                                   kernel_initializer=Orthogonal(), name="style_long_dense_3")

            short_term_style = tf.layers.dense(short_term_style, 128, activation=None,
                                               kernel_initializer=Orthogonal(), name="style_short_dense_1")
            short_term_style = tf.layers.dense(short_term_style, 128, activation=None,
                                               kernel_initializer=Orthogonal(), name="style_short_dense_2")
            self.pred_short_term_style = tf.layers.dense(short_term_style, Config.NUM_HIDDEN_STYLES, activation=None,
                                                    kernel_initializer=Orthogonal(), name="style_short_dense_3")

            self.long_style_loss_orig = tf.nn.softmax_cross_entropy_with_logits(labels=style_input,
                                                                                logits=self.pred_long_term_style)
            self.short_style_loss_orig = tf.nn.softmax_cross_entropy_with_logits(labels=style_input,
                                                                                 logits=self.pred_short_term_style)
            self.long_style_loss = self.long_style_loss_orig[self.long_term_style_interval:]
            self.short_style_loss = self.short_style_loss_orig[self.short_term_style_interval:]

            return self.long_style_loss, self.short_style_loss, self.pred_long_term_style, self.pred_short_term_style


    def neg_log_prob(self, action, name, pi_logits):
        one_hot_actions = tf.one_hot(action, self.action_dim)
        pi_logits_substract_max = self.pi_logits - tf.reduce_max(self.pi_logits, axis=-1, keep_dims=True)
        exp_pi_logits = tf.exp(pi_logits_substract_max)
        exp_pi_logits_sum = tf.reduce_sum(exp_pi_logits, axis=-1, keep_dims=True)
        self.softmax_pi = (1.0 * exp_pi_logits / exp_pi_logits_sum + self.min_policy) / \
                (1 + self.min_policy*self.action_dim)
        return -tf.reduce_sum(one_hot_actions*tf.log(self.softmax_pi), axis=-1)

    def _create_policy_network(self):
        pi_logits_all = []
        for i in range(self.action_dim):
            pi_logit_tmp = tf.layers.dense(self.reshape_lstm_output_result, 1, activation=None, \
                    kernel_initializer=Orthogonal(scale=0.01), name="logits_%d" % i)
            pi_logits_all.append(pi_logit_tmp)
        pi_logits_all_2 = tf.concat(pi_logits_all, axis=1, name="resuls_concat")
        return pi_logits_all_2

    def _create_value_network(self):
        value = tf.layers.dense(self.reshape_lstm_output_result, 1, activation=None, \
                    kernel_initializer=Orthogonal(), name="value")
        return value[:, 0]

    def _cnn(self):
        scaled_images = tf.cast(self.feature_float, tf.float32) / 255.
        h1 = tf.layers.conv2d(scaled_images, name="conv1", filters=32, kernel_size=8, \
                kernel_initializer=Orthogonal(scale=np.sqrt(2)), strides=4, \
                padding="valid", activation=tf.nn.relu)
        h2 = tf.layers.conv2d(h1, name="conv2", filters=64, kernel_size=4, \
                kernel_initializer=Orthogonal(scale=np.sqrt(2)), strides=2, \
                padding="valid", activation=tf.nn.relu)
        h3 = tf.layers.conv2d(h2, name="conv3", filters=64, kernel_size=3, \
                kernel_initializer=Orthogonal(scale=np.sqrt(2)), strides=1, \
                padding="valid", activation=tf.nn.relu)
        nh = np.prod([v.value for v in h3.get_shape()[1:]])
        flat = tf.reshape(h3, [-1, nh])

        h_fc1 = tf.layers.dense(flat, 512, activation=tf.nn.relu, \
                kernel_initializer=Orthogonal(scale=np.sqrt(2)), name="fc_1")

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
