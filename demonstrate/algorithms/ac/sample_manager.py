# -*- coding: utf-8 -*-
from framework.common.sample_manager import SampleManager as SampleManager_Base


class SampleManager(SampleManager_Base):
    def __init__(self, game_id=None):
        super().__init__(game_id)

    def save_value(self, last_value):
        last_advantage = 0
        data_map_len = len(self.rl_data_map)
        map_keys = list(self.rl_data_map.keys())
        for i in reversed(range(data_map_len)):
            frame_no = map_keys[i]
            rl_data_info = self.rl_data_map[frame_no]
            mask = 1.0 - rl_data_info.done
            last_value = last_value * mask
            last_advantage = last_advantage * mask
            delta = rl_data_info.reward + self.gamma * last_value - rl_data_info.value
            last_advantage = delta + self.gamma * self.lamda * last_advantage
            self.rl_data_map[frame_no].advantage = last_advantage
            last_value = rl_data_info.value
