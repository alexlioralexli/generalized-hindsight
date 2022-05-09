#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils

import dmc2gym
import hydra
import argparse

# Import the environments

# envs
import gym
from gym.spaces import Discrete, MultiBinary
from rlkit.envs.point_robot_new import PointEnv as PointEnv2
from rlkit.envs.point_reacher_env import PointReacherEnv
from rlkit.envs.updated_half_cheetah import HalfCheetahEnv
from rlkit.envs.wrappers import NormalizedBoxEnv, TimeLimit
from rlkit.envs.fetch_reach import FetchReachEnv
# from rlkit.envs.updated_ant import AntEnv

NUM_GPUS_AVAILABLE = 4  # change this to the number of gpus on your system




def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    gym_envList = ["AntEnv", "HalfCheetahEnv", "PointEnv2", "PointReacherEnv"]

    if cfg.env in gym_envList:
        env = gym.make(domain_name=domain_name,
                        task_name=task_name,
                        seed=cfg.seed,
                        visualize_reward=True)
    else:
        env = dmc2gym.make(domain_name=domain_name,
                        task_name=task_name,
                        seed=cfg.seed,
                        visualize_reward=True)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class Workspace(object):
    def __init__(self, args):
        self.args = args
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        # self.cfg = cfg

        # self.logger = Logger(self.work_dir,
        #                      save_tb=args.log_save_tb,
        #                      log_frequency=cfg.log_frequency,
        #                      agent=cfg.agent.name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        utils.set_seed_everywhere(self.args.seed)
        self.device = torch.device(self.device)


        #THE ENVIRONMENT INFORMATION IS ADDED HERE IN THE PARAMETERS. 
        self.env = utils.make_env(self.args)

        # TODO(Mahi): Set up the skill space here.
        self.agent = torch.load(self.args.file)
        #DIAYN AGENT IS SETUP HERE.
        # print("New Information")
        # print(self.env.observation_space.shape[0])
        # print(self.env.action_space.shape[0])
        # args.agent.params.obs_dim = self.env.observation_space.shape[0]
        # args.agent.params.action_dim = self.env.action_space.shape[0]
        # args.agent.params.action_range = [
        #     float(self.env.action_space.low.min()),
        #     float(self.env.action_space.high.max())
        # ]
        # print("Env info: ")
        # print(f"Observation space: {self.env.observation_space.shape[0]}")


        # print(f"The observation shape in DIAYN is self.env.observation_space.shape[0], {self.env.observation_space.shape[0]}")

        # print(f"The replay buffer env shape self.env.observation_space.shape : {self.env.observation_space.shape}")
        # print(f"Replay buffer capacity: {cfg.replay_buffer_capacity}")



        # WE WANT TO LOAD THE AGENT.
        

        # TODO(Mahi): Set up the discriminator here


        # TODO(Mahi): Augment the replay buffer with the skill information
        # self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
        #                                   self.env.action_space.shape,
        #                                   (self.args.agent.params.skill_dim, ),
        #                                   int(self.args.replay_buffer_capacity),
        #                                   self.device)

        self.step = 0
    def skill_test(self):
        skill_set = [self.agent.skill_dist.sample() for _ in range(100)]
        
        print(skill_set)
    def run_trace(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import argparse
        import numpy as np
        filename = '{}_{}_{}_trace.png'.format(os.path.splitext(args.file)[0],
                                        args.dim_0, args.dim_1)

        #LOOP THROUGH THE SKILL. 
        #OBSERVE FROM IT
        # SAMPLE THE ACTIONS
        
        skill_array = [[1.0, 0.0, 0.0, 0.0], [0.0,1.0,0.0,0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        # Convert skill array to tensor
        skill_tensor = torch.tensor(skill_array).to("cuda")
        palette = sns.color_palette('hls', self.args.num_skills)

        for z in range(self.args.num_skills):
            current_skill = skill_tensor[z]
            for path_index in range(self.args.n_paths):
                obs = self.env.reset()
                if self.args.use_qpos:
                    qpos = self.env.wrapped_env.env.model.data.qpos[:, 0]
                    obs_vec = [qpos]
                else:
                    obs_vec = [obs]
                
                for t in range(self.args.max_path_length):
                    
                    action = self.agent.act(obs, current_skill,sample=True)
                    (next_obs, _, _, _) = self.env.step(action)
                    if args.use_qpos:
                        qpos = self.env.wrapped_env.env.model.data.qpos[:, 0]
                        obs_vec.append(qpos)
                    elif args.use_action:
                        obs_vec.append(action)
                    else:
                        obs_vec.append(next_obs)

                obs_vec = np.array(obs_vec)
                x = obs_vec[:, args.dim_0]
                y = obs_vec[:, args.dim_1]
                plt.plot(x, y, c=palette[z])

        plt.savefig(filename)
        plt.close()


        






#CFG comes from the configuration path. 
# @hydra.main(config_path='/home/yb1025/Research/GRAIL/HUSK/accelerate-skillDiscovery/library-algo/diayn-main/config/train.yaml', strict=True)
def main(args):
    print("Args received is")
    print(args)
    workspace = Workspace(args)
    workspace.run_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default=None, help='Path to the snapshot file.')
    parser.add_argument('--env', type=str, default="AntEnv", help='Environment on RLKIT')
    parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    parser.add_argument('--n_paths', type=int, default=1)
    parser.add_argument('--dim_0', type=int, default=0)
    parser.add_argument('--dim_1', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_skills', type=int, default=4)
    parser.add_argument('--use_qpos', type=bool, default=False)
    parser.add_argument('--use_action', type=bool, default=False)
    parser.add_argument('--deterministic', '-d', dest='deterministic',
                        action='store_true')
    parser.add_argument('--no-deterministic', '-nd', dest='deterministic',
                        action='store_false')

    parser.set_defaults(deterministic=True)

    args = parser.parse_args()
    main(args)