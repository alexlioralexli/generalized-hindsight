import os
import numpy as np
import torch
# from torch import nn
# from torch import distributions as pyd
# import torch.nn.functional as F
import gym
from collections import deque
import random
import math

import dmc2gym
from gym.spaces import Discrete, MultiBinary
from rlkit.envs.point_robot_new import PointEnv as PointEnv2
from rlkit.envs.point_reacher_env import PointReacherEnv
from rlkit.envs.updated_half_cheetah import HalfCheetahEnv
from rlkit.envs.wrappers import NormalizedBoxEnv, TimeLimit
from rlkit.envs.fetch_reach import FetchReachEnv
from rlkit.envs.updated_ant import AntEnv
from rlkit.envs.hopper import HopperEnv

def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == 'square':
        return SquareEnv(25, 100)

    if cfg.env in ['ball_in_cup_catch', 'point_mass_easy']:
        domain_name = '_'.join(cfg.env.split('_')[:-1])
        task_name = cfg.env.split('_')[-1]
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    gym_envList = ["Ant-v2", "HalfCheetahEnv", "PointEnv2", "PointReacherEnv"]
    rl_kitList = ["AntEnv"]
    if cfg.env in gym_envList:
        env = gym.make(cfg.env)
    elif cfg.env == "AntEnv": 
        print("Launched ANTENV")
        env = AntEnv()
    elif cfg.env == "HalfCheetahEnv":
        print("RLKIT HALF-CHEETAH IS USED")
        env = HalfCheetahEnv()
    elif cfg.env == "HopperEnv":
        env = HopperEnv()
    else:
    
        env = dmc2gym.make(domain_name=domain_name,
                        task_name=task_name,
                        seed=cfg.seed,
                        visualize_reward=True)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
