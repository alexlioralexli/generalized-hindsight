import numpy as np
import torch
from torch import nn as nn

from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.core import eval_np
from rlkit.torch.distributions import TanhNormal
from rlkit.torch.networks import Mlp, LatentConditionedMlp


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class TanhGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )

class GHTanhGaussianPolicy(TanhGaussianPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            lat_dim,
            action_dim,
            latent_shape_multiplier=1,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        self.lat_dim = lat_dim
        self.latent_shape_multiplier = latent_shape_multiplier
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
            repeat_latent=True
    ):
        if repeat_latent:
            obs = obs[:,:-self.lat_dim]
            latent = obs[:,-self.lat_dim:]
            new_obs = torch.cat((obs, latent.repeat(1, self.latent_shape_multiplier)), dim=1)
            return super().forward(new_obs, reparameterize=reparameterize, deterministic=deterministic,
                            return_log_prob=return_log_prob)
        else:
            return super().forward(obs, reparameterize=reparameterize, deterministic=deterministic,
                            return_log_prob=return_log_prob)

class LatentConditionedTanhGaussianPolicy(LatentConditionedMlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            latent_dim,
            action_dim,
            latent_shape_multiplier=1,
            latent_to_all_layers=False,
            bilinear_integration=False,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            latent_size=latent_dim,
            output_size=action_dim,
            init_w=init_w,
            latent_shape_multiplier=latent_shape_multiplier,
            latent_to_all_layers=latent_to_all_layers,
            bilinear_integration=bilinear_integration,
            **kwargs
        )
        self.log_std = None
        self.std = std
        if std is None:
            if self.bilinear_integration:
                last_hidden_size = obs_dim * latent_dim * latent_shape_multiplier + obs_dim + latent_dim * latent_shape_multiplier
            else:
                last_hidden_size = obs_dim + latent_dim * latent_shape_multiplier
            if len(hidden_sizes) > 0:
                if self.latent_to_all_layers:
                    last_hidden_size = hidden_sizes[-1] + latent_dim * latent_shape_multiplier
                else:
                    last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, latent_np, deterministic=False):
        actions = self.get_actions(obs_np[None], latent_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, latent_np, deterministic=False):
        return eval_np(self, obs_np, latent_np, deterministic=deterministic)[0]

    def forward(
            self,
            *inputs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        latents = inputs[-1].repeat(1, self.latent_shape_multiplier)
        flat_inputs = torch.cat(inputs[:-1], dim=1)
        if self.bilinear_integration:
            h = torch.cat([torch.bmm(flat_inputs.unsqueeze(2), latents.unsqueeze(1)).flatten(start_dim=1),
                           flat_inputs, latents], dim=1)
        else:
            h = torch.cat((flat_inputs, latents), dim=1)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
            if self.latent_to_all_layers:
                h = torch.cat((h, latents), dim=1)
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )

class MakeDeterministic(Policy):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation, deterministic=True)

    def cuda(self):
        self.stochastic_policy.cuda()

    @property
    def wrapped_policy(self):
        return self.stochastic_policy


class MakeDeterministicLatentPolicy(Policy):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation, latent):
        return self.stochastic_policy.get_action(observation, latent, deterministic=True)

    def cuda(self):
        self.stochastic_policy.cuda()

    @property
    def wrapped_policy(self):
        return self.stochastic_policy
