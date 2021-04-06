# -*- coding: utf-8 -*-
from framework.game.gamecore import Gamecore as Gamecore_Base

class Gamecore(Gamecore_Base):
    def __init__(self, env):
        super().__init__(env)
