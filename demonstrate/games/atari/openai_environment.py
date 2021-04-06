# -*- coding: utf-8 -*-
import gym
import cv2
import numpy as np
from config.config import Config
from third_party.openai_baselines.common.atari_wrappers import (\
        NoopResetEnv, FireResetEnv, EpisodicLifeEnv, ClipRewardEnv)
from framework.game.environment import Environment as Environment_Base


class Environment(Environment_Base):
    def __init__(self, name, config_id, use_reward_clipping=False, use_episodic_life=False):
        self.env = gym.make(name)
        self.env = NoopResetEnv(self.env, noop_max=30)
        self.env.seed(47 + int(config_id))
        self.obs_2_max = np.zeros((2, Config.INPUT_DIM[0], Config.INPUT_DIM[1], 1), np.uint8)
        self.obs_4 = np.zeros((Config.INPUT_DIM[0], Config.INPUT_DIM[1], Config.INPUT_DIM[2]))
        self.rewards = []
        self.lives = 0
        self.was_real_done = True

        if use_episodic_life:
            self.env = EpisodicLifeEnv(self.env)
        if 'FIER' in self.env.unwrapped.get_action_meanings():
            self.env = FireResetEnv(self.env)
        if use_reward_clipping:
            self.env = ClipRewardEnv(self.env)

    def step(self, action, is_test=False):
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

        self.rewards.append(reward)
        if self.was_real_done:
            if is_test:
                return self.obs_4, reward, done, episode_info, self.was_real_done
            episode_info = {"reward": sum(self.rewards),
                            "length": len(self.rewards)}
            self.reset()
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
        self.lives = self.env.unwrapped.ale.lives()
        return self.obs_4

    @staticmethod
    def _process_obs(obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (Config.INPUT_DIM[0], Config.INPUT_DIM[1]), interpolation=cv2.INTER_AREA)
        #obs = obs/255.0
        return obs[:, :, None]  # Shape (84, 84, 1)
