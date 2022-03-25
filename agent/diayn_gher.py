from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn


import os
from logger import Logger

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer

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

# class Agent(object):
#     def reset(self):
#         """For state-full agents this function performs reseting at the beginning of each episode."""
#         pass

#     @abc.abstractmethod
#     def train(self, training=True):
#         """Sets the agent in either training or evaluation mode."""

#     @abc.abstractmethod
#     def update(self, replay_buffer, logger, step):
#         """Main function of the agent that performs learning."""

#     @abc.abstractmethod
#     def act(self, obs, sample=False):
#         """Issues an action given an observation."""

class DIAYNGHERAgent(Agent):
    """DIAYN algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, skill_dim, obs_dim_weights, skill_type, 
                 device, critic_cfg, actor_cfg, discriminator_cfg, discount, 
                 init_temperature, alpha_lr, alpha_betas, actor_lr, actor_betas, 
                 actor_update_frequency, critic_lr, critic_betas, critic_tau, 
                 critic_target_update_frequency, discriminator_lr, 
                 discriminator_betas, discriminator_update_frequency, 
                 batch_size, learnable_temperature, log_frequency, log_save_tb, name_env):
        super().__init__()
        

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)


        # NEED to make env from sac_her. We don't need it here. Since dimensions are set earlier. 



        self.obs_dim_weights = torch.tensor(obs_dim_weights).to(self.device)
        self.discriminator = hydra.utils.instantiate(discriminator_cfg).to(self.device)
        self.skill_dim = skill_dim
        self.skill_type = skill_type
        if self.skill_type == 'discrete':
            # If the skill type is discrete, the shape of the skill gives us 
            # the number of different skills
            self.skill_dist = torch.distributions.OneHotCategorical(
                probs=torch.ones(self.skill_dim).to(self.device))
        else:
            # The skills are a contunious hypercube where every axis is 
            # between zero and one
            self.skill_dist = torch.distributions.Uniform(low=torch.zeros(self.skill_dim).to(self.device), 
                                                          high=torch.ones(self.skill_dim).to(self.device))
        self.discriminator_update_frequency = discriminator_update_frequency

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
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
        
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        # END FROM SAC_GHER


        # FROM TRAIN.PY in main-diayn 

        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        #Logger Inputs cleaned
        self.logger = Logger(self.work_dir,
                             save_tb=log_save_tb,
                             log_frequency=log_frequency,
                             agent=name)
        
        # FROM THE TRAINER IN sac_gher

        self._num_train_steps = 0


        #RETURN FUNCTION FOR THE NETWORKS TAKEN FROM GHER"
        self.policy = self.critic.returnPolicy()
        self.qf1, self.qf2 = self.critic.qValueReturn()
        
        self.target_qf1, self.target_qf2 = self.critic_target.qValueReturn()


        self.train()
        self.critic_target.train()



    #ORIGINAL
    # def train(self, training=True):
    #     self.training = training
    #     self.actor.train(training)
    #     self.critic.train(training)

    #FROM GHER
    def train(self, np_batch, training = True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self._num_train_steps += 1
        batch = np_to_pytorch_batch(np_batch)
        self.update(batch, self.logger, self._num_train_steps)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, skill, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        if not torch.is_tensor(skill):
            skill = torch.FloatTensor(skill).to(self.device)
        skill = skill.unsqueeze(0)
        dist = self.actor(obs, skill)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, skill, not_done, logger,
                      step):
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

    def compute_diversity_reward(self, skill, next_obs):
        skills_log_prob = self.skill_dist.log_prob(skill).unsqueeze_(1)
        #print("The shape of next_obs is :  {}, the shape of self.obs_dim_weights is :{}".format(next_obs.shape, self.obs_dim_weights.shape))
        
        # self.obs_dim_weights was set to a default of 2. 
        
        #Since we are using Alex's RLkit , we can filter the first two
        next_obs = torch.narrow(next_obs, 1, 0, 2)

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
        return discriminator_prob - skills_log_prob

    def update(self, batch, logger, step):

        #WHERE DOES THE REWARD ORIGINIATE? is it the correct reward from the discriminator?
        reward = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        latents = batch['latents']
        skill = batch['skill']
        not_done = batch['not_done']
        not_dones_no_max =  batch['not_dones_no_max']


        #3 ARE MISSING:
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
        diversity_reward = self.compute_diversity_reward(skill, next_obs)
        assert reward.shape == diversity_reward.shape

        logger.log('train/batch_reward', diversity_reward.mean(), step)

        self.update_critic(obs, action, diversity_reward, next_obs, skill, not_done_no_max,
                           logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, skill, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

        if step % self.discriminator_update_frequency == 0:
            self.update_discriminator(obs, skill, next_obs, logger, step)


    """
    In Trainer, it calls. 

    def train(self, np_batch):
        self._num_train_steps += 1
        batch = np_to_pytorch_batch(np_batch)
        self.train_from_torch(batch)


    """
 

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        latents = batch['latents']

        # obs = torch.cat([obs, latents], dim=1)
        # next_obs = torch.cat([next_obs, latents], dim=1)

        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, latents, reparameterize=True, return_log_prob=True
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions, latents),
            self.qf2(obs, new_obs_actions, latents),
        )
        policy_loss = (alpha * log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions, latents)
        q2_pred = self.qf2(obs, actions, latents)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, latents, reparameterize=True, return_log_prob=True
        )
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions, latents),
            self.target_qf2(next_obs, new_next_actions, latents),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Update networks
        """
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1
    
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
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
        )


