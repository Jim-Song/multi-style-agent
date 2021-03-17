# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from framework.algorithm.predictor import Predictor as Predictor_Base
from algorithms.ppo.model import Model


class Predictor(Predictor_Base):
    def __init__(self, with_random=True):
        super().__init__(with_random)
        self.model = Model(with_random)
        self._init_sess()

    def get_value(self, feature):
        return self.sess.run(self.model.value,
                             feed_dict={self.feature: [feature]})

    def process(self, feature):
        return self.sess.run([self.model.action,
                              self.model.value,
                              self.model.neg_log_pi],
                             feed_dict={self.feature: [feature]})
    
    def _init_sess(self):
        cpu_num = 1
        config = tf.ConfigProto(device_count={"CPU": cpu_num}, inter_op_parallelism_threads=cpu_num, \
                intra_op_parallelism_threads=cpu_num, log_device_placement=False)
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.feature = tf.placeholder(shape=(None, self.state_dim[0], self.state_dim[1], self.state_dim[2]), \
                    name="feature", dtype=np.uint8)
            self.model.inference(self.feature)
            self.init_saver = tf.train.Saver(tf.global_variables())
            self.sess = tf.Session(graph=self.graph, config=config)
            self.sess.run(tf.global_variables_initializer())
