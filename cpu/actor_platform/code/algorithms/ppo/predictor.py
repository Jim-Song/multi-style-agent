# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from framework.algorithm.predictor import Predictor as Predictor_Base
from algorithms.ppo.model import Model
from config.config import Config


class Predictor(Predictor_Base):
    def __init__(self, with_random=True):
        super().__init__(with_random)
        self.model = Model(with_random)
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.num_hidden_styles = Config.NUM_HIDDEN_STYLES
        self.lstm_state_h = np.zeros([1, self.lstm_unit_size])
        self.lstm_state_c = np.zeros([1, self.lstm_unit_size])
        self.lstm_state = [self.lstm_state_c, self.lstm_state_h]
        self._init_sess()


    def get_value(self, feature):
        return self.sess.run(self.model.value,
                             feed_dict={self.feature: [feature['state']],
                                        self.lstm_state_c_ph: self.lstm_state[0],
                                        self.lstm_state_h_ph: self.lstm_state[1],
                                        self.style_input: feature['style_input'],
                                        })

    def process(self, feature):
        if feature['pred_style']:
            long_style_loss, short_style_loss = self.sess.run([self.model.long_style_loss, self.model.short_style_loss],
                                                              feed_dict={self.feature: feature['state'],
                                                                         self.style_input: feature['style_input'],
                                                                         })
            return long_style_loss, short_style_loss
        else:
            action, value, neg_log_pi, lstm_state = self.sess.run([self.model.action,
                                                                   self.model.value,
                                                                   self.model.neg_log_pi,
                                                                   self.model.lstm_state,
                                                                   ],
                                                                  feed_dict={self.feature: [feature['state']],
                                                                             self.lstm_state_c_ph: self.lstm_state[0],
                                                                             self.lstm_state_h_ph: self.lstm_state[1],
                                                                             self.style_input: feature['style_input'],
                                                                             })
            self.lstm_state = lstm_state
            return action, value, neg_log_pi, lstm_state

    def _init_sess(self):
        cpu_num = 1
        config = tf.ConfigProto(device_count={"CPU": cpu_num}, inter_op_parallelism_threads=cpu_num, \
                intra_op_parallelism_threads=cpu_num, log_device_placement=False)
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.feature = tf.placeholder(shape=(None, self.state_dim[0], self.state_dim[1], self.state_dim[2]), \
                    name="feature", dtype=np.uint8)
            self.lstm_state_c_ph = tf.placeholder(shape=(None, self.lstm_unit_size), name="lstm_state_c",
                                                  dtype=np.float32)
            self.lstm_state_h_ph = tf.placeholder(shape=(None, self.lstm_unit_size), name="lstm_state_h",
                                                  dtype=np.float32)
            self.style_input = tf.placeholder(dtype=tf.float32, shape=(None, self.num_hidden_styles),
                                              name="style_input")
            self.lstm_state_ph = [self.lstm_state_c_ph, self.lstm_state_h_ph]
            self.model.inference(self.feature, self.lstm_state_ph, self.style_input)
            self.model.style_discriminator(self.feature, self.style_input)
            self.init_saver = tf.train.Saver(tf.global_variables())
            self.sess = tf.Session(graph=self.graph, config=config)
            self.sess.run(tf.global_variables_initializer())
