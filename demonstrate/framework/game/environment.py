# -*- coding: utf-8 -*-
class Environment:
    def __init__(self, name, config_id):
        raise NotImplementedError("env_init: not implemented")

    def step(self, action, is_test=False):
        raise NotImplementedError("env_step: not implemented!")

    def reset(self):
        raise NotImplementedError("env_reset: not implemented!")
