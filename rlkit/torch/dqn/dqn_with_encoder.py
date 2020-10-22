from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class DQNTrainer(TorchTrainer):
    def __init__(
            self,
            qf,
            target_qf,
            encoder,
            target_encoder,
            learning_rate=1e-3,
            soft_target_tau=1e-3,
            target_update_period=1,
            qf_criterion=None,
            do_target_update=True,
            discount=0.99,
            reward_scale=1.0,
    ):
        super().__init__()
        self.qf = qf
        self.target_qf = target_qf
        self.encoder = encoder
        self.target_encoder = target_encoder
        self.learning_rate = learning_rate
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.qf_optimizer = optim.Adam(
            list(self.qf.parameters()) + list(self.encoder.parameters()),
            lr=self.learning_rate,
        )
        self.discount = discount
        self.reward_scale = reward_scale
        self.qf_criterion = qf_criterion or nn.MSELoss()
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.do_target_update = do_target_update

        # if not do_target_update:
        #     ptu.copy_model_params_from_to(self.qf, self.target_qf)

    def train_from_torch(self, batch):
        rewards = batch['rewards'] * self.reward_scale
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Compute loss
        """
        if not self.do_target_update:
            target_q_values = self.qf(self.encoder(next_obs)).detach().max(
                1, keepdim=True
            )[0]
        else:
            target_q_values = self.target_qf(self.target_encoder(next_obs)).detach().max(
                1, keepdim=True
            )[0]
        y_target = rewards + (1. - terminals) * self.discount * target_q_values
        y_target = y_target.detach()
        # actions is a one-hot vector
        y_pred = torch.sum(self.qf(self.encoder(obs)) * actions, dim=1, keepdim=True)
        qf_loss = self.qf_criterion(y_pred, y_target)

        """
        Soft target network updates
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0 and self.do_target_update:
            ptu.soft_update_from_to(
                self.qf, self.target_qf, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.encoder, self.target_encoder, self.soft_target_tau
            )


        """
        Save some statistics for eval using just one batch.
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Y Predictions',
                ptu.get_numpy(y_pred),
            ))

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.qf,
            self.target_qf,
            self.encoder,
            self.target_encoder
        ]

    def get_snapshot(self):
        return dict(
            qf=self.qf,
            target_qf=self.target_qf,
        )
