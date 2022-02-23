"""
Torch argmax policy
"""
import numpy as np
from torch import nn

import rlkit.torch.pytorch_util as ptu
from rlkit.policies.base import Policy


class ArgmaxDiscretePolicy(nn.Module, Policy):
    def __init__(self, qf):
        super().__init__()
        self.qf = qf

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = ptu.from_numpy(obs).float()
        q_values = self.qf(obs).squeeze(0)
        q_values_np = ptu.get_numpy(q_values)
        return q_values_np.argmax(), {}

    def get_actions(self, obs):
        return np.array([self.get_action(obs[i:i+1,:])[0] for i in range(len(obs))]).reshape(-1, 1)

class ArgmaxDiscretePolicyWithEncoder(ArgmaxDiscretePolicy):
    def __init__(self, qf, encoder):
        super().__init__(qf)
        self.encoder = encoder

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = ptu.from_numpy(obs).float()
        z = self.encoder(obs)
        q_values = self.qf(z).squeeze(0)
        q_values_np = ptu.get_numpy(q_values)
        return q_values_np.argmax(), {}