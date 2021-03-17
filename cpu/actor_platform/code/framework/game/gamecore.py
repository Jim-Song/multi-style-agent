# -*- coding: utf-8 -*-
from framework.common.common_func import log_time


class Gamecore:
    def __init__(self, env):
        self.env = env
        self.state = self.env.reset()

    @log_time("gamecore_process")
    def process(self, action, is_test=False):
        next_state, reward, done, episode_info, was_real_done = self.env.step(action, is_test)
        self.state = next_state
        return next_state, reward, done, episode_info, was_real_done

    def get_state(self):
        return self.state

    def reset(self):
        self.state = self.env.reset()
        return self.state
