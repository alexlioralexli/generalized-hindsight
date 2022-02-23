import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from agent import Agent
import utils

import hydra


class DIAYNAgent(Agent):
    """DIAYN algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, skill_dim, obs_dim_weights, skill_type, 
                 device, critic_cfg, actor_cfg, discriminator_cfg, discount, 
                 init_temperature, alpha_lr, alpha_betas, actor_lr, actor_betas, 
                 actor_update_frequency, critic_lr, critic_betas, critic_tau, 
                 critic_target_update_frequency, discriminator_lr, 
                 discriminator_betas, discriminator_update_frequency, 
                 batch_size, learnable_temperature):
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

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

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
        self.obs_dim_weights = next_obs
        state_prob_logits = self.discriminator(next_obs * self.obs_dim_weights)
        # This part will not work for continuous 
        # log_x_i = ((state_prob_logits * skill).sum(axis=1))
        log_x_i = torch.sum(state_prob_logits * skill, dim=1, keepdim=True)
        logsumexp_x = torch.logsumexp(state_prob_logits, dim=1, keepdim=True)
        discriminator_prob = log_x_i - logsumexp_x
        return discriminator_prob - skills_log_prob

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, skill, not_done, not_done_no_max = replay_buffer.sample(
            self.batch_size)

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
