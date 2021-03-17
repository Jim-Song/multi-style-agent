# -*- coding: utf-8 -*-
from config.config import Config


class Model():
    def __init__(self, with_random=True):
        self.state_dim = Config.INPUT_DIM
        self.action_dim = Config.ACTION_DIM
        self.with_random = with_random
        self.min_policy = Config.MIN_POLICY
        self.T = Config.T
        
    def inference(self, feature):
        raise NotImplementedError("build model: not implemented!")
