# -*- coding: utf-8 -*-
class AIProcess:
    def __init__(self, init_model=True):
        raise NotImplementedError("aiprocess_init: not implemented")

    def process(self, state):
        raise NotImplementedError("aiprocess_process: not implemented")

    def get_value(self, state):
        raise NotImplementedError("aiprocess_get_value: not implemented")
