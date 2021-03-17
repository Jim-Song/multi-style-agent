import math
import numpy as np
from atari_ppo.gym_rl_pb2 import OfflineRlInfo
from atari_ppo.config import Config
from SailLearner.dataset.sample_generation.offline_rlinfo_adapter \
        import OfflineRlInfoAdapter as OfflineRlInfoAdapterBase


class OfflineRlInfoAdapter(OfflineRlInfoAdapterBase):
    def __init__(self):
        super().__init__()
        self.rl_info = OfflineRlInfo()

    def deserialization(self, receive_data):
        data = []
        self.rl_info.ParseFromString(receive_data)
        rl_info = self.rl_info

        feature = np.frombuffer(rl_info.feature, 'f4')
        action = np.array(rl_info.action_list).astype('float32')
        value = np.array([rl_info.value]).astype('float32')
        neg_log_pi = np.array([rl_info.neg_log_pis]).astype('float32')
        advantage = np.array([rl_info.advantage]).astype('float32')
        res = np.concatenate([feature, advantage, action, neg_log_pi, value])
        data.append(res)

        if np.isnan(res).sum() != 0 or math.isnan(advantage) or math.isnan(neg_log_pi) or math.isnan(value):
            print(np.isnan(res), advantage, neg_log_pi, value)
            print("data error!")

        return data
    
    @staticmethod
    def get_data_shapes():
        return Config.DATA_SHAPES
