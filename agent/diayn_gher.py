import os

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn


from logger import Logger


import numpy as np
import torch.nn.functional as F
import math
# from agent import Agent
import utils
import hydra

import abc
from collections import OrderedDict

from typing import Iterable

from rlkit.core.batch_rl_algorithm import BatchRLAlgorithm
from rlkit.core.online_rl_algorithm import OnlineRLAlgorithm
from rlkit.core.trainer import Trainer
from rlkit.torch.core import np_to_pytorch_batch

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer

# from .agentFile import Agent
from .agentFile import Agent

"""
1. Find similarities between SAC and sac_gher
2. SAC and DIAYN are extremely similar. 
3. Repeat pattern for DIAYN. 
4. Then trainer will be completed.
5. Make sure the networks are trained similarly. 


constructor

1. 



"""

class DIAYNGHERAgent(Agent):
    """DIAYN algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, skill_dim, obs_dim_weights, skill_type, 
                 device, critic_cfg, actor_cfg, discriminator_cfg, discount, 
                 init_temperature, alpha_lr, alpha_betas, actor_lr, actor_betas, 
                 actor_update_frequency, critic_lr, critic_betas, critic_tau, 
                 critic_target_update_frequency, discriminator_lr, 
                 discriminator_betas, discriminator_update_frequency, 
                 batch_size, learnable_temperature, log_frequency, log_save_tb, name_env, max_replay_buffer_size, plotter=None,render_eval_paths=False,use_automatic_entropy_tuning = True):
        super().__init__()
        
        

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        # self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic = hydra.utils.instantiate(critic_cfg)
        self.critic_target = hydra.utils.instantiate(critic_cfg)
        # self.critic_target = hydra.utils.instantiate(critic_cfg).to(
        #     self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)
        self.actor = hydra.utils.instantiate(actor_cfg)



        # NEED to make env from sac_her. We don't need it here. Since dimensions are set earlier. 



        self.obs_dim_weights = torch.tensor(obs_dim_weights).to(self.device)
        # self.obs_dim_weights = torch.tensor(obs_dim_weights)

        # self.discriminator = hydra.utils.instantiate(discriminator_cfg).to(self.device)
        self.discriminator = hydra.utils.instantiate(discriminator_cfg)

        self.skill_dim = skill_dim
        self.skill_type = skill_type
        if self.skill_type == 'discrete':
            # If the skill type is discrete, the shape of the skill gives us 
            # the number of different skills
            self.skill_dist = torch.distributions.OneHotCategorical(
                probs=torch.ones(self.skill_dim).to(self.device))
                # probs=torch.ones(self.skill_dim))

        else:
            # The skills are a contunious hypercube where every axis is 
            # between zero and one
            self.skill_dist = torch.distributions.Uniform(low=torch.zeros(self.skill_dim).to(self.device), 
                                                          high=torch.ones(self.skill_dim).to(self.device))
            # self.skill_dist = torch.distributions.Uniform(low=torch.zeros(self.skill_dim), 
            #                                               high=torch.ones(self.skill_dim))
                                                                                                       
        self.discriminator_update_frequency = discriminator_update_frequency

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        # self.log_alpha = torch.tensor(np.log(init_temperature))

        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                        lr=discriminator_lr,
                                                        betas=discriminator_betas)


        #FROM SAC_GHER:


        
        #HOW DO I INCORPORATE ENTROPY TUNING IN DIAYN?

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning



        #WHAT NEEDS TO BE DONE WITH ENTROPY TUNING
        
        # if self.use_automatic_entropy_tuning:
        #     if target_entropy:
        #         self.target_entropy = target_entropy
        #     else:
        #         self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
        #     self.log_alpha = ptu.zeros(1, requires_grad=True)
        #     self.alpha_optimizer = optimizer_class(
        #         [self.log_alpha],
        #         lr=policy_lr,
        #     )

        # END FROM SAC_GHER


        # FROM TRAIN.PY in main-diayn 

        # self.work_dir = os.getcwd()
        # # print(f'workspace: {self.work_dir}')

        # # #Logger Inputs cleaned
        # self.logger = Logger(self.work_dir,
        #                      save_tb=log_save_tb,
        #                      log_frequency=log_frequency,
        #                      agent=name_env)
        
        # FROM THE TRAINER IN sac_gher

        self._num_train_steps = 0
        self.env_name = name_env


        #RETURN FUNCTION FOR THE NETWORKS TAKEN FROM GHER"
        self.policy = self.actor.returnPolicy()
        self.qf1, self.qf2 = self.critic.qValueReturn()
        
        self.target_qf1, self.target_qf2 = self.critic_target.qValueReturn()
        self.logger = None

        self.trainParamSet()
        self.critic_target.train()



    #ORIGINAL
    # def train(self, training=True):
    #     self.training = training
    #     self.actor.train(training)
    #     self.critic.train(training)

    def trainParamSet(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
    #FROM GHER
    def setLogger(self, logger):
        self.logger = logger
    
    def train(self, np_batch, step, training = True):
        print("I AM INSIDE CORRECT TRAIN FUNCTION IN DIAYN GHER")
        # self._num_train_steps += 1
        batch = np_to_pytorch_batch(np_batch)

        """
            SHOULD THE NUM TRAIN STEPS BE DECIDING ANYTHING? 

            IS IT FOLLOWING THE CORRECT LOGIC?

        """
        self.update(batch, self.logger, step)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, skill, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        # obs = torch.FloatTensor(obs)

        obs = obs.unsqueeze(0)
        if not torch.is_tensor(skill):
            skill = torch.FloatTensor(skill).to(self.device)
            # skill = torch.FloatTensor(skill)

        skill = skill.unsqueeze(0)
        dist = self.actor(obs, skill)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, skill, not_done, logger,
                      step):
        print("I AM INSIDE: UPDATE CRITIC")

        dist = self.actor(next_obs, skill)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action, skill)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, skill)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_discriminator(self, obs, skill, next_obs, logger, step):
        print("I AM INSIDE: DISCRIMINATOR")

        next_obs = torch.narrow(next_obs, 1, 0, 2)
        skills_pred = self.discriminator(next_obs * self.obs_dim_weights)
        # TODO(Mahi): Figure out if this line is correct
        skill_actual = skill.argmax(axis=1)
        discriminator_loss = self.discriminator.criterion(skills_pred, skill_actual)
        
        logger.log('train_discriminator/loss', discriminator_loss, step)

        # optimize the discriminator
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        self.discriminator.log(logger, step)


    def update_actor_and_alpha(self, obs, skill, logger, step):
        print("I AM INSIDE: ACTOR AND ALPHA")

        dist = self.actor(obs, skill)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action, skill)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def compute_lone_diversity_reward(self, skill, next_obs):
        device = torch.device("cuda")

        #print(f"IN COMPUTE LONE DIVERSITY REWARD: {skill}")
        if (isinstance(skill, list)):
            skill = torch.as_tensor(skill, device = device).float()
            next_obs = torch.from_numpy(next_obs).float().to(device)
        import numpy as np
        # print(f"The type of next_obs: {type(next_obs)}")
        # next_obs = next_obs.cpu().detach().numpy()
        # print(f"Type of next obs after conversion: {type(self.obs_dim_weights)}")
        if isinstance(next_obs, (np.ndarray, np.generic)):
            next_obs = next_obs[:2]
            next_obs = torch.from_numpy(next_obs).float().to(device)
        else:
            next_obs = next_obs.cpu().detach().numpy()
            next_obs = next_obs[:2]
            next_obs = torch.from_numpy(next_obs).float().to(device)

        # print(f"NEXT OBS IS: {next_obs}")

        #next_obs = torch.narrow(next_obs, 1, 0, 2)
        skills_log_prob = self.skill_dist.log_prob(skill)
        state_prob_logits = self.discriminator(next_obs * self.obs_dim_weights)
        # print(f"state_prob_logits is: {state_prob_logits}, the shape is : {state_prob_logits.shape}")
        # print(f"skill is: {skill}, the shape is : {skill.shape}")

        log_x_i = torch.sum(state_prob_logits * skill, dim=0, keepdim=True)
        logsumexp_x = torch.logsumexp(state_prob_logits, dim=0, keepdim=True)
        discriminator_prob = log_x_i - logsumexp_x

        # print(f"discriminator_prob is: {discriminator_prob}, discriminator shape is : {discriminator_prob.shape}")
        # print(f"skills_log_prob is : {skills_log_prob}, skills_log_prob shape is : {skills_log_prob.shape}")
        return discriminator_prob - skills_log_prob, False 
    def compute_diversity_reward(self, skill, next_obs, singularSkill=False):
        device = torch.device("cuda")
        # print(f"SKILL RECEIVED BEFORE CONVERSION : {skill}")

        if (isinstance(skill, list)):
            print(f"SKILL IF INSTANCE")
            skill = torch.as_tensor(skill, device = device).float()
            next_obs = torch.from_numpy(next_obs).float().to(device)

        # print(f"SKILL RECEIVED AFTER CONVERSION : {skill}")

        if not singularSkill:
            skills_log_prob = self.skill_dist.log_prob(skill).unsqueeze_(1)
            print(f"I am not in singular skill")
            print(f"Skill log prob here look like {skills_log_prob}")
        else:
            print(f"I am in singularSkill")
            if isinstance(skill, (np.ndarray, np.generic)):
                # print(f"AGAIN INSIDE ISISNTANCE")
                skill = torch.from_numpy(skill).to(device)

            # print(f"THE TYPE OF SKILL IS: {type(skill)}")
            skills_log_prob = self.skill_dist.log_prob(skill)

        # print(f"The unsqueeze shape is : {skill.unsqueeze_(1)}")
        #print("The shape of next_obs is :  {}, the shape of self.obs_dim_weights is :{}".format(next_obs.shape, self.obs_dim_weights.shape))
        

        # self.obs_dim_weights was set to a default of 2. 
        
        #Since we are using Alex's RLkit , we can filter the first two
        if isinstance(next_obs, (np.ndarray, np.generic)):
            next_obs = torch.from_numpy(next_obs).float().to(device)

            
        if self.env_name == "AntEnv":
            next_obs = torch.narrow(next_obs, 1, 0, 2)
             # -> [1024, 29] -> [1024, 2]
        elif self.env_name == "HalfCheetahEnv":
            #print(f"The next obs in HalfCheetah Are: {next_obs.shape}, the type is : {type(next_obs)}")
            pass 
        elif self.env_name == "ReacherEnv":
            pass
        elif self.env_name == "HopperEnv":
            pass
        elif self.env_name == "FetchReachEnv":
            pass
            
        # print(f"I am in the agent's diversity reward, my skill is : {skill}")
        # print("The shape of next_obs is : {}".format(next_obs.size()))
        # #print("The shape of the array is: {}".format(new_next_obs.size()) )
        # print(self.obs_dim_weights.size())
        # #new_next_obs = new_next_obs[:, 0:2] 
        # print("The next_obs is : {}".format(next_obs))
        # #print("The new_next_obs is: {}".format(new_next_obs))
        # # new_next_obs.to(device="cuda")

        # # new_next_obs = next_obs.numpy()
        # # new_next_obs = new_next_obs[:2]
        # # print(new_next_obs)

        # THE OBSERVATION SPACE SHOULD NOT BE 111, BUT 2.
       
        state_prob_logits = self.discriminator(next_obs * self.obs_dim_weights)
        # This part will not work for continuous 
        # log_x_i = ((state_prob_logits * skill).sum(axis=1))
        log_x_i = torch.sum(state_prob_logits * skill, dim=1, keepdim=True)
        logsumexp_x = torch.logsumexp(state_prob_logits, dim=1, keepdim=True)
        discriminator_prob = log_x_i - logsumexp_x
        reward = discriminator_prob - skills_log_prob
        # print(f"Reward is : {discriminator_prob - skills_log_prob}, reward shape is : {reward.shape}")
        return discriminator_prob - skills_log_prob

    def update(self, batch, logger, step):

        #WHERE DOES THE REWARD ORIGINIATE? is it the correct reward from the discriminator?
        reward = batch['rewards']
        # terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        latents = batch['latents']
        # original_skill = batch['pureSkills']
        
        skill = batch['skill'] 
        print("I AM INSIDE: UPDATE")

        print(f"Skill received is: {skill}")

        pureSkill = batch["pureSkill"]
        
        not_dones_no_max =  batch['not_dones_no_max']


        """ 
        1. skill 
        2. not_done
        3. not_done_no_max


         self.replay_buffer.add(obs, action, reward, next_obs, skill,
                                   done, done_no_max)

        """


        #REST SHOULD BE THE SAME FROM HERE.


        # obs, action, reward, next_obs, skill, not_done, not_done_no_max = replay_buffer.sample(
        #     self.batch_size)

        # TODO(Mahi): Figure out the correct reward here.
        # Reward should be log q_phi(z | s_t+1) - log p(z)
        # DIVERSITY REWARD IS NOW BEING SAMPLED FROM THE REPLAY BUFFER

        diversity_reward = self.compute_diversity_reward(skill, next_obs)
        assert reward.shape == diversity_reward.shape



        logger.log('train/batch_reward', diversity_reward.mean(), step)

        self.update_critic(obs, actions, diversity_reward, next_obs, skill, not_dones_no_max,
                           logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, skill, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

        if step % self.discriminator_update_frequency == 0:
            # Discriminator is trained with pureSkill
            self.update_discriminator(obs, pureSkill, next_obs, logger, step)


    """
    In Trainer, it calls. 

    def train(self, np_batch):
        self._num_train_steps += 1
        batch = np_to_pytorch_batch(np_batch)
        self.train_from_torch(batch)


    """

    
    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
            self.discriminator
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
            discriminator=self.discriminator

       )


