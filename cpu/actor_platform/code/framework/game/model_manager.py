# -*- coding: utf-8 -*-
class ModelManager:
    def __init__(self):
        raise NotImplementedError("modelmanager_init: not implemented!")

    def init_predictor(self, predictor, init_path=None, update_path=None):
        raise NotImplementedError("modelmanager_init_predictor: not implemented!")

    def select_model(self):
        raise NotImplementedError("modelmanager_select_model: not implemented!")

    def process(self, feature):
        raise NotImplementedError("modelmanager_process: not implemented!")

    def get_value(self, feature):
        raise NotImplementedError("modelmanager_get_value: not implemented!")

    def init_model(self, lastest_params=None, config_id=None, \
                   model_name=None, mode_type=None):
        raise NotImplementedError("modelmanager_init_model: not implemented!")