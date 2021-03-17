import os
import time
import subprocess
import importlib
from config.config import Config
from framework.common.common_log import CommonLogger

LOG = CommonLogger.get_logger()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Test:
    def __init__(self, local_ip):
        self.m_config_id = 1
        self.m_init_path = Config.INIT_PATH
        self.m_update_path = Config.UPDATE_PATH
        self.m_env_name = Config.ENV_NAME
        self.m_task_name = Config.TASK_NAME
        self.m_lib_gamecore = Config.LIB_GAMECORE
        self.m_lib_aiprocess = Config.LIB_AIPROCESS
        self.m_lib_enviroment = Config.LIB_ENVIRONMENT
        self.m_lib_predictor = Config.LIB_PREDICTOR
        self.m_algorithm = Config.ALGORITHM

        self.m_env = importlib.import_module(self.m_lib_enviroment).Environment(self.m_env_name, self.m_config_id)
        self.m_gamecore = importlib.import_module(self.m_lib_gamecore).Gamecore(self.m_env)
        self.m_predictor = importlib.import_module(self.m_lib_predictor).Predictor(with_random=False)
        self.m_aiprocess_lib = importlib.import_module(self.m_lib_aiprocess)
        self.m_aiprocess_lib.AIProcess._model_manager.init_predictor(self.m_predictor, self.m_init_path, \
                self.m_update_path)
        self.local_ip = local_ip.decode().strip()

    def _run_episode(self):
        aiprocess = self.m_aiprocess_lib.AIProcess(True)
        state = self.m_gamecore.get_state()
        frame_num = 0
        reward_sum = 0

        while True:
            frame_num += 1
            action, _, _ = aiprocess.process(state)

            if frame_num == 1:
                action = 0
            if frame_num == 2:
                action = 1
            if frame_num == 3:
                action = 2
            if frame_num == 4:
                action = 3
            next_state, reward, _, info, real_done = self.m_gamecore.process(action, True)
            reward_sum += reward

            if real_done:
                with open("../log/test_reward.txt", "a") as fin:
                    now_time = time.strftime("%m/%d/%Y-%H:%M:%S", time.localtime())
                    fin.writelines("%s %s %s %s %d %d %s\n"%(self.local_ip, "actor", self.m_task_name, \
                            self.m_aiprocess_lib.AIProcess._model_manager.model_version, reward_sum, \
                            frame_num, now_time))
                break
            state = next_state

if __name__ == "__main__":
    command = "hostname -i"
    local_ip = subprocess.check_output(command, shell=True)
    test = Test(local_ip)
    while True:
        test._run_episode()
        time.sleep(60 * 10)
