# -*- coding: utf-8 -*-
import random
import numpy as np
import tensorflow as tf
from framework.algorithm.predictor import Predictor as Predictor_Base
from algorithms.dqn.model import Model


class Predictor(Predictor_Base):
    def __init__(self, with_random=True):
        super().__init__(with_random)
        self.model = Model(with_random)
        self._init_sess()
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay_steps = 20000
        self.epsilons = np.linspace(self.epsilon_start,
                                    self.epsilon_end,
                                    self.epsilon_decay_steps)
        self.step = 0

    def get_value(self, feature):
        return self.sess.run(self.model.value,
                             feed_dict={self.feature: [feature]})

    def process(self, feature):
        q_value, value = self.sess.run([self.model.q_value, self.model.value], \
                feed_dict={self.feature: [feature]})
        return self._sample(q_value), value, None

    def _sample(self, q_value):
        self.step += 1
        epsilon = self.epsilons[min(self.step,
                                    self.epsilon_decay_steps - 1)]
        rand_float = float(random.uniform(0, self.epsilon_decay_steps)) / float(self.epsilon_decay_steps)
        if rand_float < epsilon and self.with_random:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(q_value, axis=1)
    
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
