# -*- coding: utf-8 -*-
from collections import deque
import gym
import cv2
import numpy as np
from config.config import Config
from third_party.openai_baselines.common.atari_wrappers import (
        EpisodicLifeEnv, ClipRewardEnv)
from framework.game.environment import Environment as Environment_Base


class Environment(Environment_Base):
    def __init__(self, name, config_id, use_reward_clipping=False, use_episodic_life=False):
        self.env = gym.make(name)
        self.env.seed(47 + int(config_id))
        self.obs_2_max = np.zeros((2, Config.INPUT_DIM[0], Config.INPUT_DIM[1], 1), np.uint8)
        self.obs_4 = np.zeros((Config.INPUT_DIM[0], Config.INPUT_DIM[1], Config.INPUT_DIM[2]))
        self.rewards = []
        self.lives = 0
        self.was_real_done = True
        self.action_list_reward = None
        self.episode_idx = 0

        if use_episodic_life:
            self.env = EpisodicLifeEnv(self.env)
        if use_reward_clipping:
            self.env = ClipRewardEnv(self.env)

        # random state init
        self.action_list = []
        self.action_list_buffer = deque(maxlen=100)

    def step(self, action, test=False):
        reward = 0.
        done = None
        for i in range(4):
            obs, curr_reward, done, _ = self.env.step(action)
            self.was_real_done = done
            if i >= 2:
                self.obs_2_max[i % 2] = self._process_obs(obs)
            reward += curr_reward
            lives = self.env.unwrapped.ale.lives()
            if lives < self.lives and lives > 0:
                done = True
            self.lives = lives
            if done:
                break

        self.action_list.append(action)
        self.rewards.append(reward)
        if self.was_real_done:
            self.episode_idx += 1
            episode_info = {"reward": sum(self.rewards),
                            "length": len(self.rewards)}
            self.action_list_buffer.append(self.action_list)

            self.reset()
            if not test and self.episode_idx > 100:
                self.random_state_init()

            if not test and self.episode_idx <= 100:
                noop_num = np.random.randint(1, 30)
                for _ in range(noop_num):
                    self.env.step(0)

        else:
            episode_info = None
            obs = self.obs_2_max.max(axis=0)
            self.obs_4 = np.roll(self.obs_4, shift=-1, axis=-1)
            self.obs_4[..., -1:] = obs
        return self.obs_4, reward, done, episode_info, self.was_real_done

    def reset(self):
        obs = self.env.reset()
        obs = self._process_obs(obs)
        self.obs_4[..., 0:] = obs
        self.obs_4[..., 1:] = obs
        self.obs_4[..., 2:] = obs
        self.obs_4[..., 3:] = obs
        self.rewards = []
        self.action_list = []
        self.lives = self.env.unwrapped.ale.lives()
        return self.obs_4

    def random_state_init(self):
        selected_episode_index = np.random.choice(len(self.action_list_buffer), 1)
        selected_episode = self.action_list_buffer[selected_episode_index[0]]
        selected_frame_length = len(selected_episode)
        selected_frame_index = np.random.choice(selected_frame_length, 1)
        for i in range(selected_frame_index[0]):
            _ = self.step(selected_episode[i], 0)

    @staticmethod
    def _process_obs(obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (Config.INPUT_DIM[0], Config.INPUT_DIM[1]), interpolation=cv2.INTER_AREA)
        return obs[:, :, None]  # Shape (84, 84, 1)
