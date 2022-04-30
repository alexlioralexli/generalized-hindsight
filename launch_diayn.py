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


# s# import pickle as pkl

# from video import VideoRecorder
from logger import Logger
# import diayn-main.utils as diayn-utility
import os
import hydra
# Import the environments

# envs

"""
Important files under consideration - Visit each one to find out how it fits into the picture. 

Then make the decisions. 

"""
import torch
import argparse
import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed, run_experiment
# from rlkit.torch.sac.sac_gher import SACTrainer
from rlkit.torch.networks import LatentConditionedMlp
from rlkit.torch.torch_rl_algorithm import TorchDIAYNBatchRLAlgorithm
from rlkit.data_management.task_relabeling_replay_buffer import MultiTaskReplayBuffer

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
#DIAYN
from diayn.diayn_relabelers.diayn_ant_relabeler import DIAYNAntDirectionRelabelerNewSparse
from diayn.diayn_relabelers.diayn_half_cheetah_relabeler import DIAYNHalfCheetahRelabelerMoreFeatures
#train.py has the configurations for the networks. 
# from diayn-main.train import Workspace

# from rlkit.torch.diayn.agent.agentFile import Agent

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

NUM_GPUS_AVAILABLE = 1 # change this to the number of gpus on your system


# NEED TO FIX THE PATH:


class Workspace(object):
    def __init__(self, cfg, variant):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        self.variant = variant

    def experiment(self):
        print(f"The agent recognised is: {self.cfg.agent}")
        
        self.logger = Logger(self.work_dir,
                            save_tb=self.cfg.log_save_tb,
                            log_frequency=self.cfg.log_frequency,
                            agent=self.cfg.agent.name)

       

        set_seed(int(self.variant['seed']))
        torch.manual_seed(int(self.cfg.seed))
        if self.variant['mode'] != 'ec2' and not self.variant['local_docker'] and torch.cuda.is_available():
            ptu.set_gpu_mode(True)

        if self.variant['env_name'] == 'pointmass2':
            print("pointmass")
            expl_env = NormalizedBoxEnv(PointEnv2(**self.variant['env_kwargs']))
            eval_env = NormalizedBoxEnv(PointEnv2(**self.variant['env_kwargs']))
            relabeler_cls = PointMassBestRandomRelabeler
        
        
        #Focus : AntEnv

        elif self.variant['env_name'] == "AntEnv":
            print(self.variant['env_name'])
            expl_env = NormalizedBoxEnv(AntEnv(**self.variant['env_kwargs']))
            eval_env = NormalizedBoxEnv(AntEnv(**self.variant['env_kwargs']))
            # expl_env = AntEnv()
            # eval_env = AntEnv()

            #Changing the relabeler to the DIAYN ant relabeler.
            relabeler_cls = DIAYNAntDirectionRelabelerNewSparse
        elif self.variant['env_name'] == "HalfCheetahEnv":
            print("halfcheetah")
            expl_env = NormalizedBoxEnv(HalfCheetahEnv())
            eval_env = NormalizedBoxEnv(HalfCheetahEnv())
            relabeler_cls = DIAYNHalfCheetahRelabelerMoreFeatures
        elif self.variant['env_name'] in {'pointreacherobs'}:
            print('pointreacher')
            expl_env = PointReacherEnv(**self.variant['env_kwargs'])
            eval_env = PointReacherEnv(**self.variant['env_kwargs'])
            relabeler_cls = ReacherRelabelerWithGoalAndObs
        elif self.variant['env_name'] in {'fetchreach'}:
            print('fetchreach')
            expl_env = TimeLimit(NormalizedBoxEnv(FetchReachEnv(**self.variant['env_kwargs'])),
                                max_episode_steps=self.variant['algo_kwargs']['max_path_length'],
                                insert_time=self.variant['insert_time'])
            eval_env = TimeLimit(NormalizedBoxEnv(FetchReachEnv(**self.variant['env_kwargs'])),
                                max_episode_steps=self.variant['algo_kwargs']['max_path_length'],
                                insert_time=self.variant['insert_time'])
            relabeler_cls = FetchReachRelabelerWithGoalAndObs
            self.variant['relabeler_kwargs']['fetchreach'] = self.variant['env_name'] == 'fetchreach'
        else:
            raise NotImplementedError


        if isinstance(expl_env.observation_space, Discrete) or isinstance(expl_env.observation_space, MultiBinary):
            obs_dim = expl_env.observation_space.n
            self.cfg.agent.params.obs_dim = expl_env.observation_space.n
        else:
            obs_dim = expl_env.observation_space.low.size
            self.cfg.agent.params.obs_dim = expl_env.observation_space.low.size
        action_dim = expl_env.action_space.low.size
        latent_dim = self.variant['replay_buffer_kwargs']['latent_dim']
        self.cfg.agent.params.action_dim = expl_env.action_space.low.size
        self.cfg.agent.params.action_range = [
            float(expl_env.action_space.low.size),
            float(expl_env.action_space.high.size)
        ]

        # print(f"Observation space with DIAYN print statement: {expl_env.observation_space.shape[0]}")


        # print(f"OBERSVATION SPACE:self.env.observation_space.shape {expl_env.observation_space.shape}")
        # print(f"OBERSVATION SPACE:expl_env.observation_space.low.size {expl_env.observation_space.low.size}")

        """
            INSTANTIATING AGENT OBJECT: TAKEN FROM DIAYN, LOOK AT HYDRA FILE
        """


        agent = hydra.utils.instantiate(self.cfg.agent)
        agent.setLogger(self.logger)

    

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
        policy = agent.actor.returnPolicy()
        # policy = LatentConditionedTanhGaussianPolicy(
        #     obs_dim=obs_dim,
        #     latent_dim=latent_dim,
        #     action_dim=action_dim,
        #     **variant['policy_kwargs']
        # )




        
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

        self.variant['relabeler_kwargs']['discount'] = self.variant['trainer_kwargs']['discount']
        if self.variant['env_name'] == "AntEnv":
            relabeler = relabeler_cls(q1=qf1,
                                    q2=qf2,
                                    agent=agent, 
                                    action_fn=eval_policy.wrapped_policy,
                                    **self.variant['relabeler_kwargs'])
            eval_relabeler = relabeler_cls(q1=qf1,
                                        q2=qf2,
                                        agent=agent,
                                        action_fn=eval_policy.wrapped_policy,
                                        **self.variant['relabeler_kwargs'],
                                        is_eval=True)

        elif self.variant['env_name'] == "HalfCheetahEnv":
            
            relabeler = relabeler_cls(q1=qf1,
                                    q2=qf2,
                                    action_fn=eval_policy.wrapped_policy,
                                    **self.variant['relabeler_kwargs'])
            eval_relabeler = relabeler_cls(q1=qf1,
                                        q2=qf2,
                                        action_fn=eval_policy.wrapped_policy,
                                        **self.variant['relabeler_kwargs'],
                                        is_eval=True)
            relabeler.agentSet(agent)
            eval_relabeler.agentSet(agent)


  

        """
            Add control flow for adding agent into the relabeler

        """
        # relabeler.agent = agent


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

        replay_buffer = MultiTaskReplayBuffer(
            agent=agent,
            env=expl_env,
            relabeler=relabeler,
            alg=self.cfg.alg,
            #Added from above, for the skill_dim, additional parameter so that the DIAYN algorithm can train.
            skill_dim = self.cfg.agent.params.skill_dim,
            cfg = self.cfg,  
            **self.variant['replay_buffer_kwargs']
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
            agent, 
            is_eval=True,  # variant['plot'],  # will attempt to plot if it's the pointmass
            **self.variant['path_collector_kwargs']
        )
        expl_path_collector = DIAYNTaskConditionedPathCollector(
            expl_env,
            expl_policy,
            relabeler,
            agent,
            # calculate_rewards=False,
            **self.variant['path_collector_kwargs']
        )


        algorithm = TorchDIAYNBatchRLAlgorithm(
            agent = agent,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            cfg = self.cfg,
            **self.variant['algo_kwargs']
        )
# /        print(f"DEVICE IS: {ptu.device}")
        algorithm.to(ptu.device)
        print("the number of cpu threads: {}".format(torch.get_num_threads()))
        algorithm.train()



global variant
@hydra.main(config_path='/home/yb1025/Research/GRAIL/HUSK/accelerate-skillDiscovery/generalized-hindsight/diayn-config/train.yaml', strict=True)
def main(cfg):

    if cfg.n_experiments != -1:
        seeds = list(range(10, 10 + 10 * cfg.n_experiments, 10))
    else:
        seeds = [cfg.seed]

    assert cfg.n_to_take <= cfg.n_sampled_latents


    #ARGUMENTS FOR ALGOs


    """

            MATCH THE HYPERPARAMETERS IN DIAYN. 


    """

    variant = dict(
        algorithm=cfg.alg,
        env_name=cfg.env,

        #MAKE SURE THAT THE ALGO_KWARDS ARE CORRECT
        algo_kwargs=dict(
            batch_size=128,
            num_epochs=cfg.epochs,
            num_eval_steps_per_epoch=5000,
            num_expl_steps_per_train_loop=75,
            num_trains_per_train_loop=cfg.ngradsteps,
            min_num_steps_before_training=1000,
            max_path_length=15,
        ),
        trainer_kwargs=dict(
            discount=0.90,  # 0.99
            soft_target_tau=cfg.tau,
            target_update_period=1,
            policy_lr=3E-3,  # 3e-4
            qf_lr=3E-3,  # 3e-4
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        replay_buffer_kwargs=dict(
            max_replay_buffer_size=100000,
            latent_dim=int(3),
            approx_irl=cfg.irl,
            plot=cfg.plot,
        ),
        relabeler_kwargs=dict(
            relabel=cfg.relabel,
            use_adv=cfg.use_advantages,
            alg=cfg.alg,
            n_sampled_latents=cfg.n_sampled_latents,
            n_to_take=cfg.n_to_take,
            cache=cfg.cache,

        ),
        qf_kwargs=dict(
            hidden_sizes=[300, 300, 300],
            latent_shape_multiplier=cfg.latent_shape_multiplier,
            latent_to_all_layers=cfg.latent_to_all_layers,
        ),
        policy_kwargs=dict(
            hidden_sizes=[300, 300, 300],
            latent_shape_multiplier=cfg.latent_shape_multiplier,
            latent_to_all_layers=cfg.latent_to_all_layers,
        ),
        path_collector_kwargs=dict(
            save_videos=cfg.save_videos
        ),
        use_advantages=cfg.use_advantages,
        proper_advantages=True,
        plot=cfg.plot,
        test=cfg.test,
        gpu=cfg.gpu,
        mode='ec2' if cfg.ec2 else 'here_no_doodad',
        local_docker=cfg.local_docker,
        insert_time=cfg.insert_time,
        latent_shape_multiplier=cfg.latent_shape_multiplier
    )

    logger_kwargs = dict(snapshot_mode='gap_and_last', snapshot_gap=min(50, cfg.epochs - 1))

    if cfg.env == 'pointmass2':
        variant['relabeler_kwargs']['power'] = 1
        variant['env_kwargs'] = dict(horizon=variant['algo_kwargs']['max_path_length'])
        exp_postfix = ''
        variant['algo_kwargs']['batch_size'] = 128
        variant['qf_kwargs']['hidden_sizes'] = [400, 300]
        variant['policy_kwargs']['hidden_sizes'] = [400, 300]

    #ANT environment

    elif cfg.env == "AntEnv":
        variant['replay_buffer_kwargs']['latent_dim'] = 1
        if cfg.env in {'antdirectionnewsparse'}:
            assert cfg.directiontype in {'90', '180', '360'}
            variant['relabeler_kwargs']['type'] = cfg.directiontype
        variant['algo_kwargs']['max_path_length'] = 1000
        variant['trainer_kwargs']['discount'] = 0.99
        variant['algo_kwargs']['num_expl_steps_per_train_loop'] = 1000
        variant['algo_kwargs']['num_train_loops_per_epoch'] = 1
        variant['algo_kwargs']['num_eval_steps_per_epoch'] = 0
        variant['algo_kwargs']['min_num_steps_before_training'] = 1000
        variant['replay_buffer_kwargs']['max_replay_buffer_size'] = cfg.max_replay_buffer_size
        variant['qf_kwargs']['hidden_sizes'] = [256, 256]
        variant['policy_kwargs']['hidden_sizes'] = [256, 256]
        variant['env_kwargs'] = dict(use_xy=True, contact_forces=cfg.contact_forces)
        exp_postfix = 'horizon{}'.format(variant['algo_kwargs']['max_path_length'])
    elif cfg.env == "HalfCheetahEnv":
        variant['replay_buffer_kwargs']['latent_dim'] = 4
        variant['algo_kwargs']['max_path_length'] = 1000
        variant['trainer_kwargs']['discount'] = 0.99
        variant['algo_kwargs']['num_expl_steps_per_train_loop'] = 1000
        variant['algo_kwargs']['num_train_loops_per_epoch'] = 1
        variant['algo_kwargs']['num_eval_steps_per_epoch'] = 0
        variant['algo_kwargs']['min_num_steps_before_training'] = 1000
        variant['replay_buffer_kwargs']['max_replay_buffer_size'] = cfg.max_replay_buffer_size
        variant['qf_kwargs']['hidden_sizes'] = [256, 256]
        variant['policy_kwargs']['hidden_sizes'] = [256, 256]
        exp_postfix = ''
    elif cfg.env in {'pointreacherobs'}:
        variant['algo_kwargs']['max_path_length'] = 20
        variant['trainer_kwargs']['discount'] = 0.97
        variant['algo_kwargs']['num_expl_steps_per_train_loop'] = 20
        variant['algo_kwargs']['num_train_loops_per_epoch'] = 5
        variant['algo_kwargs']['num_eval_steps_per_epoch'] = 1000
        variant['replay_buffer_kwargs']['max_replay_buffer_size'] = 2000
        variant['env_kwargs'] = dict(horizon=20)
        exp_postfix = 'horizon{}'.format(variant['algo_kwargs']['max_path_length'])
        if cfg.sparse:
            exp_postfix += 'sparse{}'.format(str(cfg.sparse))
        variant['replay_buffer_kwargs']['latent_dim'] = 6

        print('using sparse reward if specified')
        variant['relabeler_kwargs']['sparse_reward'] = cfg.sparse
        variant['relabeler_kwargs']['fixed_ratio'] = None
    elif cfg.env in {'fetchreach'}:
        variant['replay_buffer_kwargs']['latent_dim'] = 8
        variant['env_kwargs'] = dict(truncate_obs=cfg.truncate_obs)
        variant['algo_kwargs']['max_path_length'] = 50
        variant['trainer_kwargs']['discount'] = 0.98
        variant['algo_kwargs']['num_expl_steps_per_train_loop'] = 250
        variant['algo_kwargs']['num_train_loops_per_epoch'] = 1
        variant['algo_kwargs']['num_eval_steps_per_epoch'] = 25 * 50
        variant['replay_buffer_kwargs']['max_replay_buffer_size'] = 250000
        variant['qf_kwargs']['hidden_sizes'] = [256, 256]
        variant['policy_kwargs']['hidden_sizes'] = [256, 256]
        exp_postfix = 'horizon{}'.format(variant['algo_kwargs']['max_path_length'])
        variant['relabeler_kwargs']['sparse_reward'] = cfg.sparse
        if cfg.sparse:
            exp_postfix += 'sparse{}'.format(str(cfg.sparse))
    else:
        raise NotImplementedError

    # various command line argument changing
    if cfg.nexpl is not None:
        variant['algo_kwargs']['num_expl_steps_per_train_loop'] = cfg.nexpl
    if cfg.discount is not None:
        variant['trainer_kwargs']['discount'] = cfg.discount
    if cfg.lr is not None:
        variant['trainer_kwargs']['policy_lr'] = cfg.lr
        variant['trainer_kwargs']['qf_lr'] = cfg.lr
    if cfg.buffer_size is not None:
        variant['replay_buffer_kwargs']['max_replay_buffer_size'] = cfg.buffer_size
    if cfg.reward_scale is not None and cfg.reward_scale > 0:
        variant['trainer_kwargs']['reward_scale'] = cfg.reward_scale
        variant['trainer_kwargs']['use_automatic_entropy_tuning'] = False
    if cfg.exp_name is not None:
        exp_dir = cfg.exp_name
    else:
        exp_dir = 'gher-{}-{}-{}e-{}s-disc{}'.format(cfg.env,
                                                     variant['algorithm'],
                                                     str(cfg.epochs),
                                                     str(variant['algo_kwargs']['num_expl_steps_per_train_loop']),
                                                     str(variant['trainer_kwargs']['discount']))
        if len(exp_postfix) > 0:
            exp_dir += '-' + exp_postfix



    if cfg.extra is not None:
        exp_dir += '-' + cfg.extra
    if cfg.test:
        exp_dir += '-test'
    sweeper = DeterministicHyperparameterSweeper(dict(seed=seeds), variant)
    all_variants = sweeper.iterate_hyperparameters()
    for i, variant in enumerate(all_variants):
        variant['gpu_id'] = i % NUM_GPUS_AVAILABLE

    for variant in all_variants:
        print(f"Len of all variants: {len(all_variants)}")
        if cfg.ec2:

            #from rlkit.launchers.launcher_util import setup_logger, set_seed, run_experiment
            run_experiment(experiment, mode='ec2', exp_prefix=exp_dir, variant=variant,
                           seed=variant['seed'], **logger_kwargs, use_gpu=False,
                           instance_type=None,
                           spot_price=None,
                           verbose=False,
                           region='us-west-1',
                           num_exps_per_instance=1)
        elif cfg.local_docker:
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
            workspace = Workspace(cfg, variant)
            workspace.experiment()

if __name__ == '__main__':
    main()
  