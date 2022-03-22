import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import utils


class Discriminator(nn.Module):
    """Discriminator network."""
    def __init__(self, obs_dim, action_dim, skill_dim, skill_type, hidden_dim, hidden_depth):
        super().__init__()
        # print(obs_dim)
        # print(skill_dim)
        # print(hidden_depth)
        # print(hidden_dim)

        #Needed to hard code this value, CHANGE This. It shouldn't be this way. 
        obs_dim = 2
        self.model = utils.mlp(obs_dim, hidden_dim, skill_dim, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

        if skill_type == 'discrete':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

    def forward(self, obs):
        logits = self.model(obs)
        self.outputs['logits'] = logits
        return logits

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_discriminator/{k}_hist', v, step)

        for i, m in enumerate(self.model):
            if type(m) is nn.Linear:
                logger.log_param(f'train_discriminator/model_fc{i}', m, step)
