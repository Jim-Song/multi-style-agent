# -*- coding: utf-8 -*-
"""
    Atari Data production process
"""
import os
import sys
import time
import threading
import importlib
from collections import deque
import numpy as np
#sys.path.append("./third_party")
#import parl
from config.config import Config
from framework.common.sender import Sender
from framework.common.common_func import CommonFunc
from framework.common.common_log import CommonLogger
from framework.common.influxdb_tool import InfluxTool

IS_TRAIN = Config.IS_TRAIN
LOG = CommonLogger.get_logger()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Actor():
    def __init__(self, id, type):
        self.m_config_id = id
        self.m_action_type = type
        self.m_init_path = Config.INIT_PATH
        self.m_update_path = Config.UPDATE_PATH
        self.m_mem_pool_path = Config.MEM_POOL_PATH
        self.m_env_name = Config.ENV_NAME
        self.m_task_name = Config.TASK_NAME
        self.m_steps = Config.STEPS
        self.m_lib_gamecore = Config.LIB_GAMECORE
        self.m_lib_aiprocess = Config.LIB_AIPROCESS
        self.m_lib_enviroment = Config.LIB_ENVIRONMENT
        self.m_lib_predictor = Config.LIB_PREDICTOR
        self.m_lib_samplemanager = Config.LIB_SAMEPLEMANAGER
        self.m_algorithm = Config.ALGORITHM
        self.use_zmq = Config.use_zmq

        self.m_replay_buffer = deque()
        self.m_episode_info = deque(maxlen=100)
        self.m_print_info = False

        self.m_env = importlib.import_module(self.m_lib_enviroment).Environment(self.m_env_name, self.m_config_id)
        self.m_gamecore = importlib.import_module(self.m_lib_gamecore).Gamecore(self.m_env)
        self.m_predictor = importlib.import_module(self.m_lib_predictor).Predictor()
        self.m_aiprocess_lib = importlib.import_module(self.m_lib_aiprocess)

        #if self.m_action_type == "async":
        self.m_aiprocess_lib.AIProcess._model_manager.init_predictor(
            self.m_predictor,
            self.m_init_path,
            self.m_update_path)
        self.m_run_step = 0
        self.m_best_reward = 0
        self.m_ip = CommonFunc.get_local_ip()
        CommonLogger.set_config(self.m_config_id)
        self.m_influxdb = InfluxTool(self.m_config_id)
        #send data thread
        #if self.m_action_type == "async":
        self.m_sender = Sender(self.m_mem_pool_path, self.m_config_id, self.m_ip, self.use_zmq)

    def send_data(self, cond=None):
        while True:
            # try:
            cond.acquire()
            if len(self.m_replay_buffer) == 0:
                cond.wait()
            LOG.info("send_data: {:.2f}".format(len(self.m_replay_buffer)))
            self.m_sender.send_data(self.m_replay_buffer.popleft())
            cond.release()
            # except Exception as e:
                # LOG.error(e)

    def get_data(self, cond=None):
        while True:
            # try:
            game_id = CommonFunc.get_game_id()
            if cond:
                cond.acquire()
            samples = self._run_episode(game_id)
            #if self.m_action_type == "async":
            send_samples = CommonFunc.generate_data(*samples)
            self.m_replay_buffer.extend(send_samples)
            if cond:
                cond.notifyAll()
                cond.release()
            # except Exception as e:
            #     LOG.error(e)

    def _run_episode(self, game_id):
        #if self.m_action_type == "async":
        aiprocess = self.m_aiprocess_lib.AIProcess(True)
        sample_manager = importlib.import_module(self.m_lib_samplemanager).SampleManager(game_id)
        state = self.m_gamecore.get_state()
        frame_no = 0
        while frame_no < self.m_steps:
            frame_no += 1
            action, value, neg_log_pis, lstm_state = aiprocess.process(state)
            next_state, reward, done, info, _ = self.m_gamecore.process(action)
            if info:
                if info["reward"] > self.m_best_reward:
                    self.m_best_reward = info["reward"]
                self.m_episode_info.append(info)
                self.m_print_info = True
            sample_manager.save_sample(frame_no,
                                       state,
                                       next_state,
                                       action,
                                       reward,
                                       done,
                                       info,
                                       value,
                                       neg_log_pis,
                                       lstm_state,
                                       )
            state = next_state
        value = aiprocess.get_value(state)
        sample_manager.save_value(value[0])
        self._get_mean_episode_info()
        self.m_run_step += 1
        return sample_manager.format_data(self.m_action_type)

    def _get_mean_episode_info(self):
        if self.m_episode_info and self.m_print_info:
            now_time = time.strftime("%m/%d/%Y-%H:%M:%S", time.localtime())
            reward_mean = np.mean([info["reward"] for info in self.m_episode_info])
            length_mean = np.mean([info["length"] for info in self.m_episode_info])
            with open("../log/actor_%s.txt"%(self.m_config_id), "a") as f_info:
                info = "%s %s %s %d %.2f %.2f %.3f %s\n" \
                        % (str(self.m_ip), "actor", str(self.m_task_name), self.m_run_step,\
                        reward_mean, self.m_best_reward, length_mean, now_time)
                f_info.writelines(info)
            self.m_print_info = False
            info = {}
            info["reward_mean"] = reward_mean
            info["best_reward"] = self.m_best_reward
            info["reward"] = self.m_episode_info[-1]["reward"]
            self.m_influxdb.write('reward_info', info)
        else:
            LOG.info("reward_ino2 : {:4} {:4} {:.2f} {:.2f} {:.3f}" \
                     .format(self.m_config_id, self.m_run_step, -1, -1, -1))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        LOG.error("please used %s %s %s"%("actor.py", "type(sync/async)", "actor_id"))
        sys.exit(0)
    action_type = sys.argv[1]
    config_id = sys.argv[2]
    actor = Actor(config_id, action_type)
    if action_type == "async":
        if not IS_TRAIN:
            p = threading.Thread(target=actor.get_data,)
            p.start()
            p.join()
        else:
            cond = threading.Condition()
            p1 = threading.Thread(target=actor.send_data, args=(cond,))
            p2 = threading.Thread(target=actor.get_data, args=(cond,))
            p1.start()
            p2.start()
            p1.join()
            p2.join()
