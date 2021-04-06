# -*- coding: utf-8 -*-
from config.config import Config
from framework.common.sample_manager import SampleManager as SampleManager_Base
IS_CHECK = Config.IS_CHECK


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
           
            if IS_CHECK:
                self._write_info(rl_data_info)

    def _write_info(self, rl_data_info):
        with open("../log/frame_{}.txt".format(rl_data_info.frame_no), 'w') as f_out:
            f_out.write("frame_no: " + str(rl_data_info.frame_no) + "\n")
            f_out.write("feature: " + str(rl_data_info.feature) + "\n")
            f_out.write("advantage: " + str(format(rl_data_info.advantage, '0.5f')) + "\n")
            f_out.write("action: " + str(rl_data_info.action_list) + "\n")
            f_out.write("neg_log_pi: " + str(rl_data_info.neg_log_pis) + "\n")
            f_out.write("value: " + str(rl_data_info.value) + "\n")
