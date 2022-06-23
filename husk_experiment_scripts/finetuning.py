import os
import sys
import math
import time
import pickle as pkl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils
import hydra
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

NUM_GPUS_AVAILABLE = 1 # change this to the number of gpus on your system



class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)


        #THE ENVIRONMENT INFORMATION IS ADDED HERE IN THE PARAMETERS. 
        self.env = utils.make_env(self.args)

        # TODO(Mahi): Set up the skill space here.

        # TODO(Mahi): Set up the skill space here.

        #DIAYN AGENT IS SETUP HERE.
        # print("New Information")
        # print(self.env.observation_space.shape[0])
        # print(self.env.action_space.shape[0])
        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        # print("Env info: ")
        # print(f"Observation space: {self.env.observation_space.shape[0]}")


        # print(f"The observation shape in DIAYN is self.env.observation_space.shape[0], {self.env.observation_space.shape[0]}")

        # print(f"The replay buffer env shape self.env.observation_space.shape : {self.env.observation_space.shape}")
        # print(f"Replay buffer capacity: {cfg.replay_buffer_capacity}")
    
        self.sac_agent = hydra.utils.instantiate(cfg.agent)
        self.diayn_agent = torch.load(self.args.file)

        # TODO(Mahi): Set up the discriminator here


        # TODO(Mahi): Augment the replay buffer with the skill information
        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          (cfg.agent.params.skill_dim, ),
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0
    def get_best_skill(self, agent, env, num_skills, max_path_length):
        reward_list = []
        if self.args.type_skill == "DISCRETE":
            if self.args.num_skills == 4:
                skill_array = [[1.0, 0.0, 0.0, 0.0], [0.0,1.0,0.0,0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
            elif self.args.num_skills == 6:
                skill_array = [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
        for z in skill_array:
            new_paths = []
            new_paths = rollouts(env, self.agent, z,
                                 max_path_length, n_paths=2)
          
            """
                HIGHEST REWARDS NEED TO BE ADDED. 

                WE FIND THE Z WITH THE HIGHEST REWARD. 


                Z -> latent is just a number here. Not a vector.
            """
            total_returns = np.mean([path['rewards'].sum() for path in new_paths])
            # tf.logging.info('Reward for skill %d = %.3f', z, total_returns)
            reward_list.append(total_returns)

        best_z = np.argmax(reward_list)
        # tf.logging.info('Best skill found: z = %d, reward = %d', best_z,
        #                 reward_list[best_z])
        return best_z
    def run(self):
        recordArray = [25000, 50000, 1500000, 2500000, 5000000, 7500000]
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        # Get best skill elf, agent, env, num_skills, max_path_length
        best_skill = self.get_best_skill(self.agent, env, self.cfg.skill, cfg.max_path_length)
        

        self.sac_agent.set_actor(self.diayn_agent.actor)
        # QF, VF
        
        if self.cfg.pre_trained:
            # Use the same critics
            self.sac_agent.reset_critic_reset(self.diayn_agent.critic, self.diayn_agent.critic_target)

        else:
            # Use new critics
            # Do Nothing
            pass
        
        
        
        
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                self.sac_agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                # TODO(Mahi): Sample a skill here.

                #SAMPLE SKILL
                skill = utils.to_np(self.self.sac_agent.skill_dist.sample())
                print(f"Skill found is : {skill}")
                self.logger.log('train/episode', episode, self.step)


            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    # print(f"Shape of skill before in eval mode act is : {skill.shape}, obs shape is : {obs.shape}")
                    action = self.self.sac_agent.act(obs, skill, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                #HERE IS WHERE THE REPLAY BUFFER IS ADDED. 

                #ADDING INFORMATION TO THE UPDATE FUNCTION FOR DIAYN

                self.self.sac_agent.update(self.replay_buffer, self.logger, self.step)


            # print(f"The size of the action after act is: {action.size}")
            # print(f"The action is : {action}")

            next_obs, reward, done, _ = self.env.step(action)
            # print(f"The next_obs is : {next_obs}")

            # allow infinite bootstrap
            done = float(done)


            # DONENOMAX, the critic does not get updated in the last step?
            
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            # print(f"Values in DIAYN: Done: {done}, done_no_max: {done_no_max}")
            episode_reward += reward


            """

                ADDING THE INFORMATION ON THE REPLAY BUFFER. 

            """

            self.replay_buffer.add(obs, action, reward, next_obs, skill,
                                   done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1

            if self.step in recordArray:
                filePathSave = self.work_dir + "/" + str(self.step) + ".pkl"
                torch.save(self.agent,filePathSave)    


        # print("Final step is: ")
        # print(self.step)
        filePathSave = self.work_dir + "/finalModel.pkl"
        torch.save(self.agent,filePathSave)

        

        



#CFG comes from the configuration path. 
@hydra.main(config_path="/home/yb1025/Research/GRAIL/HUSK/library-algo/diayn-main/config/train5.yaml", strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default=None, help='Path to the snapshot file.')
    parser.add_argument('--env', type=str, default="AntEnv", help='Environment on RLKIT')
    parser.add_argument('--type_skill', type=str, default="DISCRETE", help='Environment on RLKIT')

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
