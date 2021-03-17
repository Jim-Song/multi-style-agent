# -*- coding: utf-8 -*-
from framework.game.gamecore import Gamecore as Gamecore_Base
from framework.common.common_log import CommonLogger

LOG = CommonLogger.get_logger()


class Gamecore(Gamecore_Base):
    def __init__(self, env):
        super().__init__(env)
        LOG.info("Atari Gamecore init")
