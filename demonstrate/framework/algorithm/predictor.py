# -*- coding: utf-8 -*-
import os
from config.config import Config


class Predictor():
    def __init__(self, with_random=True):
        self.init_saver = None
        self.graph = None
        self.sess = None
        self.action_dim = int(Config.ACTION_DIM)
        self.state_dim = Config.INPUT_DIM
        self.with_random = with_random

    def get_value(self, feature):
        raise NotImplementedError("predictor_get_value: not implemented")

    def process(self, feature):
        raise NotImplementedError("predictor_process: not implemented")
 
    def init_model(self, model_path=None,
                   lastest_params=None,
                   config_id=None,
                   model_name=None,
                   mode_type=None):

        # model_path = "../model/model_" + str(config_id) + "/" + model_name + "/model.ckpt"
        model_path = 'checkpoints/model.ckpt'
        with self.graph.as_default():
            try:
                self.init_saver.restore(self.sess, save_path=model_path)
                print("load model:", model_path)
            except Exception as error:
                print('load model error')
    
    def _init_sess(self):
        raise NotImplementedError("predictor_init_sess: not implemented")
