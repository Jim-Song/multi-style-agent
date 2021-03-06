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

IS_TRAIN = Config.IS_TRAIN
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Actor():
    def __init__(self, num_hidden_styles, hidden_style):
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
        self.num_hidden_styles = num_hidden_styles
        self.hidden_style = hidden_style

        self.m_replay_buffer = deque()
        self.m_episode_info = deque(maxlen=100)
        self.m_print_info = False

        self.m_env = importlib.import_module(self.m_lib_enviroment).Environment(self.m_env_name)
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


    def _run_episode(self):
        # if self.m_action_type == "async":
        aiprocess = self.m_aiprocess_lib.AIProcess(True)
        state = self.m_gamecore.get_state()

        style_input = [np.eye(self.num_hidden_styles)[self.hidden_style]]
        state_dict = {'state': state, 'pred_style': False, 'style_input': style_input}
        frame_no = 0
        samples = []
        while frame_no < self.m_steps:
            frame_no += 1
            action, value, neg_log_pis, lstm_state = aiprocess.process(state_dict)
            next_state, reward, done, info, _ = self.m_gamecore.process(action)
            if info:
                if info["reward"] > self.m_best_reward:
                    self.m_best_reward = info["reward"]
                self.m_episode_info.append(info)
                self.m_print_info = True
            sample = [frame_no, state, next_state, action, reward, done, info,
                      value, neg_log_pis, lstm_state, hidden_style, ]
            samples.append(sample)
            state = next_state
            state_dict['state'] = next_state
            import time
            time.sleep(0.03)

        style_input = np.tile(style_input, (len(samples), 1))
        state_dict_s = {'state': [x[1] for x in samples], 'pred_style': True, 'style_input': style_input}
        long_style_loss, short_style_loss, style_long_pred, style_short_pred = aiprocess.process(state_dict_s)
        print('long_style_loss: ', long_style_loss)
        return None

    def _get_mean_episode_info(self):
        now_time = time.strftime("%m/%d/%Y-%H:%M:%S", time.localtime())
        reward_mean = np.mean([info["reward"] for info in self.m_episode_info])
        length_mean = np.mean([info["length"] for info in self.m_episode_info])
        with open("./actor.txt", "a") as f_info:
            info = "%s %s %d %.2f %.2f %.3f %s\n" \
                    % ("actor", str(self.m_task_name), self.m_run_step,\
                    reward_mean, self.m_best_reward, length_mean, now_time)
            f_info.writelines(info)
        self.m_print_info = False
        info = {}
        info["reward_mean"] = reward_mean
        info["best_reward"] = self.m_best_reward
        info["reward"] = self.m_episode_info[-1]["reward"]

if __name__ == "__main__":
    num_hidden_styles = int(sys.argv[1])
    hidden_style = int(sys.argv[2])
    actor = Actor(num_hidden_styles, hidden_style)
    while True:
        actor._run_episode()
