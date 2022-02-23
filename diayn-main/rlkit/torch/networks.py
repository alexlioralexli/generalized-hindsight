"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import eval_np, np_ify
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm
from torch.distributions import Categorical, Normal
import rlkit.torch.pytorch_util as ptu
def identity(x):
    return x


class Mlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            input_activation=identity,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
            input_batch_norm=False
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.input_activation = input_activation
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        if input_batch_norm:
            # self.input_ln = LayerNorm(input_size)
            self.input_bn = nn.BatchNorm1d(input_size)
        else:
            self.input_bn = None
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        if self.input_bn:
            h = self.input_bn(h)
        h = self.input_activation(h)
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)

class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return eval_np(self, obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, output_activation=torch.tanh, **kwargs)

class GoalMlpPolicy(MlpPolicy):
    """
    A helper class for policies in an environment with a discrete action space.
    Uses softmax output activation
    """
    def __init__(self, preprocessor=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocessor = preprocessor

    def forward(self, obs, **kwargs):
        if self.preprocessor:
            obs = self.preprocessor.preprocess(obs)
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        # if len(obs) > 1:
        #     import ipdb; ipdb.set_trace()
        return super().forward(obs, **kwargs)

    def get_actions(self, obs):
        np_logits = eval_np(self, obs)
        logits = torch.tensor(np_logits) #.to(tpu.device)
        dist = Categorical(logits=logits)
        return np_ify(dist.sample())

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0], {}



class ModelMlp(Mlp):

    def __init__(self,
                 hidden_sizes,
                 output_dim,
                 input_size,
                 init_w=3e-3,
                 input_activation=identity,
                 hidden_activation=F.relu,
                 output_activation=identity,
                 hidden_init=ptu.fanin_init,
                 b_init_value=0.1,
                 layer_norm=False,
                 layer_norm_kwargs=None,
                 input_batch_norm=False):
        super().__init__(hidden_sizes, output_dim*2, input_size, init_w, input_activation, hidden_activation, output_activation,
                       hidden_init, b_init_value, layer_norm, layer_norm_kwargs, input_batch_norm)
        self.output_dim = output_dim

    def forward(self, x, **kwargs):
        return torch.split(super().forward(x, **kwargs), self.output_dim, dim=1)

    def predict_log_prob(self, obs, latent, next_obs):
        mu, log_std = self.forward(torch.cat((obs, latent), dim=1))
        dist = Normal(mu, log_std.exp())
        delta = next_obs - obs
        return dist.log_prob(delta)


class LatentConditionedMlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size, # without latent
            latent_size,
            latent_shape_multiplier=1,
            latent_to_all_layers=False,
            bilinear_integration=False,
            init_w=3e-3,
            input_activation=identity,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
            input_batch_norm=False
    ):
        super().__init__()
        self.latent_shape_multiplier = latent_shape_multiplier
        self.latent_to_all_layers = latent_to_all_layers
        self.bilinear_integration = bilinear_integration

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.input_activation = input_activation
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        if self.bilinear_integration:
            in_size = input_size * latent_size * latent_shape_multiplier + input_size + latent_size * latent_shape_multiplier
        else:
            in_size = input_size + latent_size * latent_shape_multiplier

        if input_batch_norm:
            # self.input_ln = LayerNorm(input_size)
            self.input_bn = nn.BatchNorm1d(input_size)
        else:
            self.input_bn = None
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            if self.latent_to_all_layers:
                in_size += self.latent_size * self.latent_shape_multiplier
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, *inputs, return_preactivations=False):
        # last element of inputs is the latent
        latents = inputs[-1].repeat(1, self.latent_shape_multiplier)
        flat_inputs = torch.cat(inputs[:-1], dim=1)
        if self.bilinear_integration:
            h = torch.cat([torch.bmm(flat_inputs.unsqueeze(2), latents.unsqueeze(1)).flatten(start_dim=1),
                           flat_inputs, latents], dim=1)
        else:
            h = torch.cat([flat_inputs, latents], dim=1)
        if self.input_bn:
            h = self.input_bn(h)
        h = self.input_activation(h)
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
            if self.latent_to_all_layers:
                h = torch.cat((h, latents), dim=1)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

























