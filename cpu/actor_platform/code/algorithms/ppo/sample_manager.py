# -*- coding: utf-8 -*-
import lz4.block
import numpy as np

from config.config import Config
from framework.common.sample_manager import SampleManager as SampleManager_Base
from framework.common.rl_data_info import RLDataInfo
from framework.proto.gym_rl_pb2 import OfflineRlInfo
from framework.proto.gym_rl_pb2 import RlFragmentInfo
from framework.common.common_log import CommonLogger

LOG = CommonLogger.get_logger()
IS_CHECK = Config.IS_CHECK
ACTION_DIM = Config.ACTION_DIM
INPUT_DIM = Config.INPUT_DIM


class SampleManager(SampleManager_Base):
    def __init__(self, game_id=None):
        super().__init__(game_id)

    def save_sample(self,
                    frame_no,
                    state,
                    next_state,
                    action,
                    reward,
                    done,
                    info,
                    value,
                    neg_log_pis,
                    lstm_state,
                    style,
                    ):
        rl_data_info = RLDataInfo()
        lstm_c, lstm_h = lstm_state
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
                rl_data_info.lstm_h = lstm_h.astype('f').tostring()
                rl_data_info.lstm_c = lstm_c.astype('f').tostring()
                if neg_log_pis is None:
                    neg_log_pis = 0
                rl_data_info.neg_log_pis = neg_log_pis
                rl_data_info.style = style
                self.rl_data_map[frame_no] = rl_data_info
            else:
                rl_data_info.frame_no = 1
                rl_data_info.feature = (2.0 * np.ones((INPUT_DIM[0], INPUT_DIM[1], INPUT_DIM[2]))).astype('f').tostring()
                rl_data_info.next_feature = (3.0 * np.zeros((INPUT_DIM[0], INPUT_DIM[1], INPUT_DIM[2]))).astype('f').tostring()
                rl_data_info.action_list.append(4)
                rl_data_info.reward = 5.0
                rl_data_info.done = 6
                rl_data_info.value = 7.0
                rl_data_info.lstm_h = (8.0 * np.ones(lstm_h.shape)).astype('f').tostring()
                rl_data_info.lstm_c = (9.0 * np.ones(lstm_h.shape)).astype('f').tostring()
                if neg_log_pis is None:
                    neg_log_pis = 0
                rl_data_info.neg_log_pis = 10.0
                self.rl_data_map[frame_no] = rl_data_info

    def save_samples(self, samples, long_style_loss, short_style_loss):
        assert len(samples) == len(long_style_loss) + Config.LONG_TERM_STYLE_INTERVAL
        for i in range(len(long_style_loss)):
            sample = samples[i]
            frame_no, state, next_state, action, reward, done, info, value, neg_log_pis, \
                lstm_state, style = sample
            reward += -1 * long_style_loss[i] * Config.LONG_TERM_STYLE_WEIGHT + short_style_loss[i] * Config.SHORT_TERM_STYLE_WEIGHT
            self.save_sample(frame_no,
                             state,
                             next_state,
                             action,
                             reward,
                             done,
                             info,
                             value,
                             neg_log_pis,
                             lstm_state,
                             style,
                             )

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

    def _send_game_data(self, actor_type):
        sample = []
        len_data_map = len(self.rl_data_map)
        map_keys = list(self.rl_data_map.keys())
        rl_fragment_info = RlFragmentInfo()
        tmp = []
        for i in range(0, len_data_map):
            off_line_rl_info = OfflineRlInfo()
            frame_no = map_keys[i]
            rl_data_info = self.rl_data_map[frame_no]
            rl_data_info.struct_to_pb(off_line_rl_info)
            tmp.append(off_line_rl_info)
            if len(tmp) == Config.LSTM_STEP:
                rl_fragment_info.samples.extend(tmp)
                tmp = []
                str_rl_info = rl_fragment_info.SerializeToString()
                if actor_type == "sync":
                    sample.append(str_rl_info)
                elif actor_type == "async":
                    sample.append(lz4.block.compress(str_rl_info, store_size=False))
                else:
                    LOG.error("actor_type error")
                rl_fragment_info = RlFragmentInfo()
        return sample