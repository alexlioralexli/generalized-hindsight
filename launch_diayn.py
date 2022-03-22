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
    parser = argparse.ArgumentParser()

    #BEGIN ADD ARGUMENTS : SETUP
 
    parser.add_argument('--env', type=str, default='pointmass', help='name of env to run on')
    parser.add_argument('--alg', type=str, default='DIAYN', help='name of algorithm to run')
    parser.add_argument('--n_sampled_latents', type=int, default=5, help="number of latents to sample")
    parser.add_argument('--n_to_take', type=int, default=1,
                        help="number of latents to relabel with, should be less than n_sampled_latents")
    parser.add_argument('--relabel', action='store_true', help='whether to relabel')
    parser.add_argument('--use_advantages', '-use_adv', action='store_true', help='use_advantages for relabeling')
    parser.add_argument('--irl', action='store_true',
                        help='use approximate irl to choose relabeling latents')
    parser.add_argument('--plot', action='store_true', help='plot the trajectories')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--sparse', type=float, default=None)
    parser.add_argument('--ngradsteps', type=int, default=100)
    parser.add_argument('--nexpl', type=int, default=None)
    parser.add_argument('--horizon', type=int, default=None)
    parser.add_argument('--tau', type=float, default=5E-3)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--buffer_size', type=int, default=None)
    parser.add_argument('--discount', type=float, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ec2', '-ec2', action='store_true')
    parser.add_argument('--local_docker', '-local_docker', action='store_true')
    parser.add_argument('--reward_scale', type=float, default=None)
    parser.add_argument('--insert_time', action='store_true')
    parser.add_argument('--latent_shape_multiplier', type=int, default=1)
    parser.add_argument('--latent_to_all_layers', action='store_true')

    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--n_experiments', '-n', type=int, default=-1,
                        help="number of seeds to use. If not -1, overrides seed ")
    # experiment name
    parser.add_argument('--exp_name', '-name', type=str, default=None)
    parser.add_argument('--extra', '-x', type=str, default=None)
    parser.add_argument('--test', '-test', action='store_true')
    parser.add_argument('--epochs', type=int, default=50, help="number of latents to sample")
    parser.add_argument('--save_videos', action='store_true')

    # for reacher
    parser.add_argument('--safetyfn', '-safety', type=str, default='newlog')  # newlog, linear, inverse
    parser.add_argument('--energyfn', '-energy', type=str, default='velocity')  # work, kinetic, velocity
    parser.add_argument('--energyfactor', type=float, default=1.0, help="how much to multiply energy by")

    # for fetch reacher
    parser.add_argument('--truncate_obs', action='store_true', help='only return end_effector loc')

    # for ant
    parser.add_argument('--use_xy', action='store_true')
    parser.add_argument('--contact_forces', action='store_true')
    parser.add_argument('--directiontype', type=str, default='360')
    global args
    args = parser.parse_args()


    #FINISH ARGUMENTS

    if args.n_experiments != -1:
        seeds = list(range(10, 10 + 10 * args.n_experiments, 10))
    else:
        seeds = [args.seed]

    assert args.n_to_take <= args.n_sampled_latents


    #ARGUMENTS FOR ALGOs


    """

            MATCH THE HYPERPARAMETERS IN DIAYN. 


    """

    variant = dict(
        algorithm=args.alg,
        env_name=args.env,

        #MAKE SURE THAT THE ALGO_KWARDS ARE CORRECT
        algo_kwargs=dict(
            batch_size=256,
            num_epochs=args.epochs,
            num_eval_steps_per_epoch=5000,
            num_expl_steps_per_train_loop=75,
            num_trains_per_train_loop=args.ngradsteps,
            min_num_steps_before_training=1000,
            max_path_length=15,
        ),
        trainer_kwargs=dict(
            discount=0.90,  # 0.99
            soft_target_tau=args.tau,
            target_update_period=1,
            policy_lr=3E-3,  # 3e-4
            qf_lr=3E-3,  # 3e-4
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        replay_buffer_kwargs=dict(
            max_replay_buffer_size=100000,
            latent_dim=3,
            approx_irl=args.irl,
            plot=args.plot,
        ),
        relabeler_kwargs=dict(
            relabel=args.relabel,
            use_adv=args.use_advantages,
            n_sampled_latents=args.n_sampled_latents,
            n_to_take=args.n_to_take,
            cache=args.cache,
        ),
        qf_kwargs=dict(
            hidden_sizes=[300, 300, 300],
            latent_shape_multiplier=args.latent_shape_multiplier,
            latent_to_all_layers=args.latent_to_all_layers,
        ),
        policy_kwargs=dict(
            hidden_sizes=[300, 300, 300],
            latent_shape_multiplier=args.latent_shape_multiplier,
            latent_to_all_layers=args.latent_to_all_layers,
        ),
        path_collector_kwargs=dict(
            save_videos=args.save_videos
        ),
        use_advantages=args.use_advantages,
        proper_advantages=True,
        plot=args.plot,
        test=args.test,
        gpu=args.gpu,
        mode='ec2' if args.ec2 else 'here_no_doodad',
        local_docker=args.local_docker,
        insert_time=args.insert_time,
        latent_shape_multiplier=args.latent_shape_multiplier
    )

    logger_kwargs = dict(snapshot_mode='gap_and_last', snapshot_gap=min(50, args.epochs - 1))

    if args.env == 'pointmass2':
        variant['relabeler_kwargs']['power'] = 1
        variant['env_kwargs'] = dict(horizon=variant['algo_kwargs']['max_path_length'])
        exp_postfix = ''
        variant['algo_kwargs']['batch_size'] = 128
        variant['qf_kwargs']['hidden_sizes'] = [400, 300]
        variant['policy_kwargs']['hidden_sizes'] = [400, 300]

    #ANT environment

    elif args.env == "AntEnv":
        variant['replay_buffer_kwargs']['latent_dim'] = 1
        if args.env in {'antdirectionnewsparse'}:
            assert args.directiontype in {'90', '180', '360'}
            variant['relabeler_kwargs']['type'] = args.directiontype
        variant['algo_kwargs']['max_path_length'] = 1000
        variant['trainer_kwargs']['discount'] = 0.99
        variant['algo_kwargs']['num_expl_steps_per_train_loop'] = 1000
        variant['algo_kwargs']['num_train_loops_per_epoch'] = 1
        variant['algo_kwargs']['num_eval_steps_per_epoch'] = 25000
        variant['algo_kwargs']['min_num_steps_before_training'] = 1000
        variant['replay_buffer_kwargs']['max_replay_buffer_size'] = int(1E6)
        variant['qf_kwargs']['hidden_sizes'] = [256, 256]
        variant['policy_kwargs']['hidden_sizes'] = [256, 256]
        variant['env_kwargs'] = dict(use_xy=args.use_xy, contact_forces=args.contact_forces)
        exp_postfix = 'horizon{}'.format(variant['algo_kwargs']['max_path_length'])
    elif args.env in {'halfcheetahhard'}:
        variant['replay_buffer_kwargs']['latent_dim'] = 4
        variant['algo_kwargs']['max_path_length'] = 1000
        variant['trainer_kwargs']['discount'] = 0.99
        variant['algo_kwargs']['num_expl_steps_per_train_loop'] = 1000
        variant['algo_kwargs']['num_train_loops_per_epoch'] = 1
        variant['algo_kwargs']['num_eval_steps_per_epoch'] = 25000
        variant['algo_kwargs']['min_num_steps_before_training'] = 1000
        variant['replay_buffer_kwargs']['max_replay_buffer_size'] = int(1E6)
        variant['qf_kwargs']['hidden_sizes'] = [256, 256]
        variant['policy_kwargs']['hidden_sizes'] = [256, 256]
        exp_postfix = ''
    elif args.env in {'pointreacherobs'}:
        variant['algo_kwargs']['max_path_length'] = 20
        variant['trainer_kwargs']['discount'] = 0.97
        variant['algo_kwargs']['num_expl_steps_per_train_loop'] = 20
        variant['algo_kwargs']['num_train_loops_per_epoch'] = 5
        variant['algo_kwargs']['num_eval_steps_per_epoch'] = 1000
        variant['replay_buffer_kwargs']['max_replay_buffer_size'] = 2000
        variant['env_kwargs'] = dict(horizon=20)
        exp_postfix = 'horizon{}'.format(variant['algo_kwargs']['max_path_length'])
        if args.sparse:
            exp_postfix += 'sparse{}'.format(str(args.sparse))
        variant['replay_buffer_kwargs']['latent_dim'] = 6

        print('using sparse reward if specified')
        variant['relabeler_kwargs']['sparse_reward'] = args.sparse
        variant['relabeler_kwargs']['fixed_ratio'] = None
    elif args.env in {'fetchreach'}:
        variant['replay_buffer_kwargs']['latent_dim'] = 8
        variant['env_kwargs'] = dict(truncate_obs=args.truncate_obs)
        variant['algo_kwargs']['max_path_length'] = 50
        variant['trainer_kwargs']['discount'] = 0.98
        variant['algo_kwargs']['num_expl_steps_per_train_loop'] = 250
        variant['algo_kwargs']['num_train_loops_per_epoch'] = 1
        variant['algo_kwargs']['num_eval_steps_per_epoch'] = 25 * 50
        variant['replay_buffer_kwargs']['max_replay_buffer_size'] = 250000
        variant['qf_kwargs']['hidden_sizes'] = [256, 256]
        variant['policy_kwargs']['hidden_sizes'] = [256, 256]
        exp_postfix = 'horizon{}'.format(variant['algo_kwargs']['max_path_length'])
        variant['relabeler_kwargs']['sparse_reward'] = args.sparse
        if args.sparse:
            exp_postfix += 'sparse{}'.format(str(args.sparse))
    else:
        raise NotImplementedError

    # various command line argument changing
    if args.nexpl is not None:
        variant['algo_kwargs']['num_expl_steps_per_train_loop'] = args.nexpl
    if args.discount is not None:
        variant['trainer_kwargs']['discount'] = args.discount
    if args.lr is not None:
        variant['trainer_kwargs']['policy_lr'] = args.lr
        variant['trainer_kwargs']['qf_lr'] = args.lr
    if args.buffer_size is not None:
        variant['replay_buffer_kwargs']['max_replay_buffer_size'] = args.buffer_size
    if args.reward_scale is not None and args.reward_scale > 0:
        variant['trainer_kwargs']['reward_scale'] = args.reward_scale
        variant['trainer_kwargs']['use_automatic_entropy_tuning'] = False
    if args.exp_name is not None:
        exp_dir = args.exp_name
    else:
        exp_dir = 'gher-{}-{}-{}e-{}s-disc{}'.format(args.env,
                                                     variant['algorithm'],
                                                     str(args.epochs),
                                                     str(variant['algo_kwargs']['num_expl_steps_per_train_loop']),
                                                     str(variant['trainer_kwargs']['discount']))
        if len(exp_postfix) > 0:
            exp_dir += '-' + exp_postfix



    if args.extra is not None:
        exp_dir += '-' + args.extra
    if args.test:
        exp_dir += '-test'
    sweeper = DeterministicHyperparameterSweeper(dict(seed=seeds), variant)
    all_variants = sweeper.iterate_hyperparameters()
    for i, variant in enumerate(all_variants):
        variant['gpu_id'] = i % NUM_GPUS_AVAILABLE

    for variant in all_variants:
        if args.ec2:

            #from rlkit.launchers.launcher_util import setup_logger, set_seed, run_experiment
            run_experiment(experiment, mode='ec2', exp_prefix=exp_dir, variant=variant,
                           seed=variant['seed'], **logger_kwargs, use_gpu=False,
                           instance_type=None,
                           spot_price=None,
                           verbose=False,
                           region='us-west-1',
                           num_exps_per_instance=1)
        elif args.local_docker:
            run_experiment(experiment, mode='local_docker', exp_prefix=exp_dir, variant=variant,
                           seed=variant['seed'], **logger_kwargs, use_gpu=False,
                           instance_type=None,
                           spot_price=None,
                           verbose=False,
                           region='us-west-1',
                           num_exps_per_instance=1)
        else:
            setup_logger(exp_dir, variant=variant, seed=variant['seed'], **logger_kwargs)
            #experiment(variant)

    main()
  