import os
os.environ['dataop'] = '/data1/reinforcement_platform/rl_learner_platform/code/shm_lib/'
import sys
sys.path.append(os.getcwd() + "/code")
from atari_ppo.algorithm import Algorithm
from atari_ppo.model import Model
from atari_ppo.offline_rlinfo_adapter import OfflineRlInfoAdapter

from SailLearner import Trainer
from SailLearner.dataset.network_dataset.network_dataset_dataop import NetworkDataset

if __name__ == "__main__":
    config_path = os.getcwd() + "/code/common.conf"
    network = Algorithm(Model())
    atari_ppo_trainer = Trainer(network, NetworkDataset, OfflineRlInfoAdapter, config_path)
    atari_ppo_trainer.run()
