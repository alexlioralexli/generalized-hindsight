"""


launc_accel. 


Will be used for mulitple unsupverised skill discovery learning algorithms. 

Priority Order:
1. DIAYN
2. DADs
3. Disk

Only a launcher. Relabelers written in RLKIT. 




"""


"""
Launcher for experiments for Generalized Hindsight Experience Replay

"""

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
# import diayn-main.utils as diayn-utility

import dmc2gym
import hydra
from omegaconf import DictConfig, OmegaConf
# Import the environments

# envs
import gym

"""
Important files under consideration - Visit each one to find out how it fits into the picture. 

Then make the decisions. 

"""
import argparse
import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed, run_experiment
from rlkit.torch.sac.sac_gher import SACTrainer
from rlkit.torch.networks import LatentConditionedMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm, DIAYNBatchRLAlgorithm
from rlkit.data_management.task_relabeling_replay_buffer import MultiTaskReplayBuffer, DIAYNTaskReplayBuffer
from rlkit.samplers.data_collector.path_collector import TaskConditionedPathCollector, DIAYNTaskConditionedPathCollector
from rlkit.torch.sac.policies import MakeDeterministicLatentPolicy, LatentConditionedTanhGaussianPolicy, \
    TanhGaussianPolicy
from rlkit.util.hyperparameter import DeterministicHyperparameterSweeper

# relabelers
from rlkit.torch.multitask.pointmass_rewards import PointMassBestRandomRelabeler
from rlkit.torch.multitask.gym_relabelers import ReacherRelabelerWithGoalAndObs
from rlkit.torch.multitask.fetch_reach_relabelers import FetchReachRelabelerWithGoalAndObs
from rlkit.torch.multitask.half_cheetah_relabeler import HalfCheetahRelabelerMoreFeatures
from rlkit.torch.multitask.ant_direction_relabeler import AntDirectionRelabelerNewSparse


#DIAYN 
from rlkit.torch.diayn.agent.diayn_gher import DIAYNGHERAgent

#DIAYN
from rlkit.torch.diayn.diayn_relabelers.diayn_ant_relabeler import DIAYNAntDirectionRelabelerNewSparse
#train.py has the configurations for the networks. 
# from diayn-main.train import Workspace


"""

    ADD CONTROL FLOW FOR PATH HERE: RIGHT NOW, as you can see below the default path is DIAYN's

"""



#DIAYN Configuration Path: Networks and Training Setup



# envs
from gym.spaces import Discrete, MultiBinary
from rlkit.envs.point_robot_new import PointEnv as PointEnv2
from rlkit.envs.point_reacher_env import PointReacherEnv
from rlkit.envs.updated_half_cheetah import HalfCheetahEnv
from rlkit.envs.wrappers import NormalizedBoxEnv, TimeLimit
from rlkit.envs.fetch_reach import FetchReachEnv
from rlkit.envs.updated_ant import AntEnv

NUM_GPUS_AVAILABLE = 4  # change this to the number of gpus on your system


# NEED TO FIX THE PATH:


class Workspace(object):
    def __init__(self, cfg, variant):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        self.variant = variant

    def experiment(self):


        agent = hydra.utils.instantiate(self.cfg.agent)


        set_seed(int(variant['seed']))
        torch.manual_seed(int(args.seed))
        if variant['mode'] != 'ec2' and not variant['local_docker'] and torch.cuda.is_available():
            ptu.set_gpu_mode(True)

        if variant['env_name'] == 'pointmass2':
            print("pointmass")
            expl_env = NormalizedBoxEnv(PointEnv2(**variant['env_kwargs']))
            eval_env = NormalizedBoxEnv(PointEnv2(**variant['env_kwargs']))
            relabeler_cls = PointMassBestRandomRelabeler
        
        
        #Focus : AntEnv

        elif variant['env_name'] == "AntEnv":
            print(variant['env_name'])
            expl_env = NormalizedBoxEnv(AntEnv(**variant['env_kwargs']))
            eval_env = NormalizedBoxEnv(AntEnv(**variant['env_kwargs']))

            #Changing the relabeler to the DIAYN ant relabeler.
            relabeler_cls = DIAYNAntDirectionRelabelerNewSparse(agent)
        elif variant['env_name'] in {'halfcheetahhard'}:
            print("halfcheetah")
            expl_env = NormalizedBoxEnv(HalfCheetahEnv())
            eval_env = NormalizedBoxEnv(HalfCheetahEnv())
            relabeler_cls = HalfCheetahRelabelerMoreFeatures
        elif variant['env_name'] in {'pointreacherobs'}:
            print('pointreacher')
            expl_env = PointReacherEnv(**variant['env_kwargs'])
            eval_env = PointReacherEnv(**variant['env_kwargs'])
            relabeler_cls = ReacherRelabelerWithGoalAndObs
        elif variant['env_name'] in {'fetchreach'}:
            print('fetchreach')
            expl_env = TimeLimit(NormalizedBoxEnv(FetchReachEnv(**variant['env_kwargs'])),
                                max_episode_steps=variant['algo_kwargs']['max_path_length'],
                                insert_time=variant['insert_time'])
            eval_env = TimeLimit(NormalizedBoxEnv(FetchReachEnv(**variant['env_kwargs'])),
                                max_episode_steps=variant['algo_kwargs']['max_path_length'],
                                insert_time=variant['insert_time'])
            relabeler_cls = FetchReachRelabelerWithGoalAndObs
            variant['relabeler_kwargs']['fetchreach'] = variant['env_name'] == 'fetchreach'
        else:
            raise NotImplementedError



        

        #ALGORITHM: SETUP : ADD CASES AND CONTROL FLOW LATER ON

        """
            Algorithms:
            1. DIAYN
            2. DADS


        """

        #OBS DIM SETUP FROM GHER:
        #DIAYN SETUP FROM: DIAYN FOLDER
            # cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
            # cfg.agent.params.action_dim = self.env.action_space.shape[0]
            # cfg.agent.params.action_range = [
            #     float(self.env.action_space.low.min()),
            #     float(self.env.action_space.high.max())
            # ]

            # Dimensions taken from expl env mentioned above. Will these match the dimension setup from DIAYN folder.
        
            
            # obs_dim setup 
        if isinstance(expl_env.observation_space, Discrete) or isinstance(expl_env.observation_space, MultiBinary):
            obs_dim = expl_env.observation_space.n
            self.cfg.agent.params.obs_dim = expl_env.observation_space.n
        else:
            obs_dim = expl_env.observation_space.low.size
            self.cfg.agent.params.obs_dim = expl_env.observation_space.low.size
        action_dim = expl_env.action_space.low.size
        latent_dim = variant['replay_buffer_kwargs']['latent_dim']
        self.cfg.agent.params.action_dim = expl_env.action_space.low.size
        self.cfg.agent.params.action_range = [
            float(expl_env.action_space.low.size),
            float(expl_env.action_space.high.size)
        ]




            #DIAYN instantiate, then you don't need a trainer right?

        
        """
            NEED TO REPLACE THE REPLAY Buffer with GHER ReplayBuffer.

        """




        qf1, qf2 = agent.critic.qValueReturn()


        target_qf1, target_qf2 = agent.critic_target.qValueReturn()


        #Network creation -> Can be taken directly from DIAYN. NETWORKs instantiated using train.py
        # qf1 = LatentConditionedMlp(
        #     input_size=obs_dim + action_dim,
        #     latent_size=latent_dim,
        #     output_size=1,
        #     **variant['qf_kwargs']
        # )
        # qf2 = LatentConditionedMlp(
        #     input_size=obs_dim + action_dim,
        #     latent_size=latent_dim,
        #     output_size=1,
        #     **variant['qf_kwargs']
        # )
        

        #Policy is taken from here:  rlkit.torch.sac.policies


        # Policy is just the actor:  You need to replace this by the policy network for DIAYN

        # policy = LatentConditionedTanhGaussianPolicy(
        #     obs_dim=obs_dim,
        #     latent_dim=latent_dim,
        #     action_dim=action_dim,
        #     **variant['policy_kwargs']
        # )


        policy = agent.actor.returnPolicy()

        # Eval Policy :  rlkit.torch.sac.policies

        eval_policy = MakeDeterministicLatentPolicy(policy)


        """

            These policy: 

            1. Use RLKIT 
            2. DIAYN downstream task
            3. Implement DIAYN, 
            4. Turndown the relabeling. 
            5. Update the networks in the relabeler. There is no update in GHER. 


            SACTrainer needs the:

            1. 


        """
        #Taken from : from rlkit.torch.sac.sac_gher import SACTrainer

        #SAC TRAINER -> from policies. 

        

        # trainer = SACTrainer(
        #     env=eval_env,
        #     policy=policy,
        #     qf1=qf1,
        #     qf2=qf2,
        #     target_qf1=target_qf1,
        #     target_qf2=target_qf2,
        #     **variant['trainer_kwargs']
        # )
        expl_policy = policy



        #RELABELER ARGS, 

        """
            Also need to pass in the discriminator network in relabeler_cls

            Already taken care of, as you have shifted the network configuration through hydra.utils.instantiate(agent)

        """

        variant['relabeler_kwargs']['discount'] = variant['trainer_kwargs']['discount']
        relabeler = relabeler_cls(q1=qf1,
                                q2=qf2,
                                action_fn=eval_policy.wrapped_policy,
                                **variant['relabeler_kwargs'])
        eval_relabeler = relabeler_cls(q1=qf1,
                                    q2=qf2,
                                    action_fn=eval_policy.wrapped_policy,
                                    **variant['relabeler_kwargs'],
                                    is_eval=True)



        # MULTI TASK RELABELER: 

        """
            Should be the same as we are using RLKIT with DIAYN.

            WE KEEP THIS REPLAY BUFFER, DIAYN UPDATE FUNCTION. 
            
            Assuming, 

            CODE TRAIL DONE: for elabeler




            REPLAY BUFFER from DIAYN:

                self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                            self.env.action_space.shape,
                                            (cfg.agent.params.skill_dim, ),
                                            int(cfg.replay_buffer_capacity),
                                            self.device)

        """
        replay_buffer = DIAYNTaskReplayBuffer(
            env=expl_env,
            relabeler=relabeler,
            #Added from above, for the skill_dim, additional parameter so that the DIAYN algorithm can train.
            skill = cfg.agent.params.skill_dim, 
            **variant['replay_buffer_kwargs']
        )


        """

            EvalPathCollector.
            ExplPathCollector. 

            Do we still need these? Yes, we need these. Simple trajectory collectors.

            How do they fit into the algorithm?



            SHOULD BE UNTOUCHED.


            YOU WILL NEED TO CHANGE THE POLICY HERE, AND MAKE SURE THE RELABELER IS BEING USED CORRECTLY


        """
        eval_path_collector = DIAYNTaskConditionedPathCollector(
            eval_env,
            eval_policy,
            eval_relabeler,
            is_eval=True,  # variant['plot'],  # will attempt to plot if it's the pointmass
            **variant['path_collector_kwargs']
        )
        expl_path_collector = DIAYNTaskConditionedPathCollector(
            expl_env,
            expl_policy,
            relabeler,
            # calculate_rewards=False,
            **variant['path_collector_kwargs']
        )


        algorithm = TorchDIAYNBatchRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            cfg = cfg,
            **variant['algo_kwargs']
        )
        algorithm.to(ptu.device)
        algorithm.train()



global variant
@hydra.main(config_path='/home/yb1025/Research/GRAIL/relabeler-irl/accelerate-skillDiscovery/generalized-hindsight/rlkit/torch/diayn/config/train.yaml', strict=True)
def main(cfg):
    
    print(f"CFG num_seed_steps : {cfg.num_seed_steps}")    
    workspace = Workspace(cfg, variant)
    workspace.experiment()

if __name__ == '__main__':
    main()
  