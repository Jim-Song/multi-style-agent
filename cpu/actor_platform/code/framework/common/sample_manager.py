# -*- coding: utf-8 -*-
import lz4.block
import numpy as np
from framework.common.rl_data_info import RLDataInfo
from framework.proto.gym_rl_pb2 import OfflineRlInfo
from framework.common.common_log import CommonLogger
from config.config import Config

LOG = CommonLogger.get_logger()
IS_CHECK = Config.IS_CHECK
ACTION_DIM = Config.ACTION_DIM
INPUT_DIM = Config.INPUT_DIM


class SampleManager:
    def __init__(self, game_id=None):
        self.m_game_id = game_id
        self.rl_data_map = {}
        self.gamma = Config.GAMMA
        self.lamda = Config.LAMDA

    def save_sample(self,
                    frame_no,
                    state,
                    next_state,
                    action,
                    reward,
                    done,
                    info,
                    value,
                    neg_log_pis):
        rl_data_info = RLDataInfo()
        if frame_no in list(self.rl_data_map.keys()):
            LOG.error("frame_no already exists in rl_data_map")
            pass
        else:
            if not IS_CHECK:
                rl_data_info.frame_no = frame_no
                rl_data_info.feature = state.astype('f').tostring()
                rl_data_info.next_feature = next_state.astype('f').tostring()
                rl_data_info.action_list.append(action)
                rl_data_info.reward = reward
                rl_data_info.done = done
                rl_data_info.info = info
                rl_data_info.value = value
                if neg_log_pis is None:
                    neg_log_pis = 0
                rl_data_info.neg_log_pis = neg_log_pis
                self.rl_data_map[frame_no] = rl_data_info
            else:
                rl_data_info.frame_no = frame_no
                rl_data_info.feature = np.ones((INPUT_DIM[0], INPUT_DIM[1], INPUT_DIM[2])).astype('f').tostring()
                rl_data_info.next_feature = np.zeros((INPUT_DIM[0], INPUT_DIM[1], INPUT_DIM[2])).astype('f').tostring()
                rl_data_info.action_list.append(int(frame_no % ACTION_DIM))
                rl_data_info.reward = (frame_no % 10) * 0.05
                rl_data_info.done = 1 if (frame_no % 64) == 0 else 0
                rl_data_info.value = (frame_no % 10) * 0.1
                if neg_log_pis is None:
                    neg_log_pis = 0
                rl_data_info.neg_log_pis = (frame_no % 10) * 0.02
                self.rl_data_map[frame_no] = rl_data_info

    def save_value(self, value):
        raise NotImplementedError("save_value: not implemented!")

    def format_data(self, actor_type):
        sample = self._send_game_data(actor_type)
        return sample

    def _send_game_data(self, actor_type):
        sample = []
        len_data_map = len(self.rl_data_map)
        map_keys = list(self.rl_data_map.keys())
        for i in range(0, len_data_map):
            off_line_rl_info = OfflineRlInfo()
            frame_no = map_keys[i]
            rl_data_info = self.rl_data_map[frame_no]
            rl_data_info.struct_to_pb(off_line_rl_info)
            str_rl_info = off_line_rl_info.SerializeToString()
            if actor_type == "sync":
                sample.append(str_rl_info)
            elif actor_type == "async":
                sample.append(lz4.block.compress(str_rl_info, store_size=False))
            else:
                LOG.error("actor_type error")
        return sample
