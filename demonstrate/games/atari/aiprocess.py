#-*- coding: utf-8 -*-
from framework.game.aiprocess import AIProcess as AIProcess_Base
from games.atari.model_manager import ModelManager


class AIProcess(AIProcess_Base):
    _model_manager = ModelManager()

    def __init__(self, init_model=True):
        if init_model:
            self._model_manager.init_model()

    def process(self, state):
        return self._model_manager.process(state)

    def get_value(self, state):
        return self._model_manager.get_value(state)
