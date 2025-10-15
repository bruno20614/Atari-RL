import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn 
import numpy as np
import pandas as pd
import gymnasium as gym
import ale_py
import argparse
import os
import random
import time
from distutils.util import strtobool
from ale_py import AtariEnv
from ale_py import ALEInterface
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from distutils.util import strtobool


def make_env(env_id,seed,idx,capture_video,run_name ):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode = "rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)

        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)

        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed) 

        return env
    
    return thunk