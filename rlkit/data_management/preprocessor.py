import numpy as np
import torch
from rlkit.util.utils import int_to_onehot
import rlkit.torch.pytorch_util as ptu

class Preprocessor(object):
    def __init__(self, input_type, action_dim, goal_dim, obs_dim, goal_conditioned=False, network=None):
        self.input_type = input_type
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.obs_dim = obs_dim
        self.goal_conditioned = goal_conditioned
        self.network = network

    def preprocess(self, obs):
        # if self.expert:
        #     actions = torch.tensor(int_to_onehot(self.expert.get_actions(obs), action_dim), dtype=torch.float32)
        #     return torch.cat((actions, obs[:, obs_dim:]), dim=1)
        # return x


        if self.input_type == 'action':
            network_output = self.network.get_actions(obs)
            actions = torch.tensor(int_to_onehot(network_output, self.action_dim), dtype=torch.float32)
            return torch.cat((actions, obs[:, self.obs_dim:]), dim=1)
        elif self.input_type == 'obs':
            return obs
        elif self.input_type == 'obs-action':
            # action, obs, goal
            network_output = self.network.get_actions(obs)
            actions = torch.tensor(int_to_onehot(network_output, self.action_dim), dtype=torch.float32).to(ptu.device)
            return torch.cat((actions, obs), dim=1)  # not sure if this is right
        elif self.input_type == 'latent':
            # g_enc(obs), goal
            # import ipdb; ipdb.set_trace()
            if self.goal_conditioned:
                latent, _ = self.network.encode(obs)
            else:
                latent, _ = self.network.encode(obs[:, :self.obs_dim])
            return torch.cat((latent, obs[:, self.obs_dim:]), dim=1)
        elif self.input_type == 'obs-latent':
            # g_enc(obs), obs, goal
            if self.goal_conditioned:
                latent, _ = self.network.encode(obs)
            else:
                latent, _ = self.network.encode(obs[:, :self.obs_dim])
            return torch.cat((latent, obs), dim=1)
        elif self.input_type == 'latentencgoal':
            # obs, g_enc(obs), g_enc(goal)
            if self.goal_conditioned:
                latent, _ = self.network.encode(obs)
                goal_latent, _ = self.network.encode(torch.cat((obs[:, self.obs_dim], obs[:, self.obs_dim:]), dim=1))
            else:
                latent, _ = self.network.encode(obs[:, :self.obs_dim])
                goal_latent, _ = self.network.encode(obs[:, self.obs_dim:])

            return torch.cat((latent, goal_latent, obs[:, :self.obs_dim]), dim=1)
        else:
            raise NotImplementedError
