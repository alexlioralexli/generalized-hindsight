import numpy as np
from gym import spaces
from rlkit.torch.multitask.rewards import Relabeler, RandomRelabeler
import rlkit.torch.pytorch_util as ptu
import matplotlib
import os
import os.path as osp
from rlkit.core import logger
from itertools import product

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm
from rlkit.torch.multitask.gym_relabelers import ContinuousRelabeler
from rlkit.envs.hand_reach import HandReachEnv, FingerReachEnv

class HandRelabeler(ContinuousRelabeler):
    def __init__(self, test=False, sparse_reward=False, **kwargs):
        super().__init__(**kwargs)
        self.test = test
        self.sparse_reward = sparse_reward
        print("sparse reward:", self.sparse_reward)
        self.env = HandReachEnv()

    def sample_task(self):
        # 'robot0:S_fftip',
        # 'robot0:S_mftip',
        # 'robot0:S_rftip',
        # 'robot0:S_lftip',
        # 'robot0:S_thtip',
        # latent: [weight on pointer, weight on middle, weight on ring, weight on pinky, weight on thumb, weight on energy]
        # get a goal
        goal = self.env._sample_goal()
        # sample gaussian, normalize to norm 1
        sample = np.random.multivariate_normal(np.zeros(6), np.diag(np.ones(6)))
        while np.linalg.norm(sample) < 0.0001:
            sample = np.random.multivariate_normal(np.zeros(6), np.diag(np.ones(6)))
        return np.concatenate([np.abs(sample) / np.linalg.norm(sample), goal])

    def get_goal(self, latent):
        return latent[6:]

    def latent_to_coeffs(self, latent):
        return latent[:6]

    def interpret_latent(self, latent):
        return 'weight on each finger: {}, weight on energy: {:.2f}'.format(str(latent[1:6]), latent[0])

    def calculate_path_features(self, path, latent):
        raise NotImplementedError


    #todo: double check that this is right
    def reward_done(self, obs, action, latent, env_info=None):  #everything is 1d
        goal_pos = self.get_goal(latent)
        joint_diffs = (env_info['end_effector_loc'] - goal_pos).reshape([-1, 3])
        joint_dists = np.linalg.norm(joint_diffs, axis=1)
        # print(joint_dists)
        if self.sparse_reward <= 1.01:
            # reward_dist = 0.2 + 0.3 * (np.exp(-dist ** 2 / 4E-4) - 1)
            reward_dist = self.sparse_reward * (np.exp(-joint_dists ** 2 / 0.015**2))
            # reward_dist = np.array([self.sparse_reward * (np.exp(-dist ** 2 / 0.08**2)) for dist in joint_dists])
        elif self.sparse_reward <= 2.01:
            reward_dist = (joint_dists < 0.01).astype(np.float32)
            # reward_dist = np.array([(dist < 0.04).astype(np.float32) for dist in joint_dists])
        else:
            raise NotImplementedError
            reward_dist = 0.2 - dist
        # print(reward_dist)
        coeffs = self.latent_to_coeffs(latent)
        # print(reward_dist, env_info['reward_energy'])
        return reward_dist.dot(coeffs[:-1]) + env_info['reward_energy'] * coeffs[-1], False

    # todo: double check that this is right
    def calculate_path_reward(self, path, latent):
        env_infos = path['env_infos']
        goal_pos= self.get_goal(latent)
        end_effector_locs = np.array([env_info['end_effector_loc'] for env_info in env_infos])
        joint_diffs = (end_effector_locs - goal_pos).reshape([len(end_effector_locs), -1, 3])
        joint_dists = np.linalg.norm(joint_diffs, axis=2)
        if self.sparse_reward <= 1.01:
            # reward_dist = 0.2 + 0.3 * (np.exp(-dists ** 2 / 4E-4) - 1)
            reward_dist = self.sparse_reward * (np.exp(-joint_dists ** 2 / 0.08**2))
            # reward_dist = np.array([self.sparse_reward * (np.exp(-dist ** 2 / 0.08 ** 2)) for dist in joint_dists])
        elif self.sparse_reward <= 2.01:
            reward_dist = (joint_dists < 0.04).astype(np.float32)
            # reward_dist = np.array([(dist < 0.04).astype(np.float32) for dist in joint_dists])
        else:
            raise NotImplementedError
            reward_dist = 0.2 - dists
        # print(reward_dist)
        reward_energy = np.array([env_info['reward_energy'] for env_info in env_infos])
        coeffs = self.latent_to_coeffs(latent)
        return reward_dist.dot(coeffs[:-1]) + reward_energy * coeffs[-1]

    def get_reward_matrix(self, paths, latents):
        return np.array([[self.get_discounted_path_reward(path, latent) for latent in latents] for path in paths])


class FingerRelabeler(HandRelabeler):
    def __init__(self, test=False, sparse_reward=False, **kwargs):
        super().__init__(**kwargs)
        self.test = test
        self.sparse_reward = sparse_reward
        print("sparse reward:", self.sparse_reward)
        self.env = FingerReachEnv()

    def sample_task(self):
        # 'robot0:S_fftip',
        # 'robot0:S_mftip',
        # 'robot0:S_rftip',
        # 'robot0:S_lftip',
        # 'robot0:S_thtip',
        # latent: [weight on pointer, weight on middle, weight on ring, weight on pinky, weight on thumb, weight on energy]
        # get a goal
        # latent: [weight on pointer, weight on energy, target xyz position of pointer]
        goal = self.env._sample_goal()
        # sample gaussian, normalize to norm 1
        sample = np.random.multivariate_normal(np.zeros(2), np.diag(np.ones(2)))
        while np.linalg.norm(sample) < 0.0001:
            sample = np.random.multivariate_normal(np.zeros(2), np.diag(np.ones(2)))
        return np.concatenate([np.abs(sample) / np.linalg.norm(sample), goal])

    def get_goal(self, latent):
        return latent[2:]

    def latent_to_coeffs(self, latent):
        return latent[:2]

    def interpret_latent(self, latent):
        return 'weight on first finger: {:.2f}, weight on energy: {:.2f}, goal: {}'.format(latent[0], latent[1], str(latent[2:]))

    def calculate_path_features(self, path, latent):
        raise NotImplementedError


    #todo: double check that this is right
    def reward_done(self, obs, action, latent, env_info=None):  #everything is 1d
        goal_pos = self.get_goal(latent)
        joint_diff = env_info['end_effector_loc'][:3] - goal_pos
        joint_dists = np.linalg.norm(joint_diff)
        # print(joint_dists)
        if self.sparse_reward:
            reward_dist = (joint_dists < 0.01).astype(np.float32)
            # reward_dist = np.array([(dist < 0.04).astype(np.float32) for dist in joint_dists])
        else:
            raise NotImplementedError
            reward_dist = 0.2 - dist
        # print(reward_dist)
        coeffs = self.latent_to_coeffs(latent)
        # print(reward_dist, env_info['reward_energy'])
        return reward_dist * coeffs[0] + env_info['reward_energy'] * coeffs[1], False  #todo: fix the reward energy

    # todo: double check that this is right
    def calculate_path_reward(self, path, latent):
        env_infos = path['env_infos']
        goal_pos= self.get_goal(latent)
        end_effector_locs = np.array([env_info['end_effector_loc'] for env_info in env_infos])[:,:3]
        joint_diffs = (end_effector_locs - goal_pos)
        joint_dists = np.linalg.norm(joint_diffs, axis=1)
        if self.sparse_reward:
            reward_dist = (joint_dists < 0.01).astype(np.float32)
            # reward_dist = np.array([(dist < 0.04).astype(np.float32) for dist in joint_dists])
        else:
            raise NotImplementedError
            reward_dist = 0.2 - dists
        # print(reward_dist)
        reward_energy = np.array([env_info['reward_energy'] for env_info in env_infos])  #todo: fix by adding constant
        coeffs = self.latent_to_coeffs(latent)
        return reward_dist.dot(coeffs[0]) + reward_energy * coeffs[1]

if __name__ == '__main__':
    relabeler = HandRelabeler()
    env = FingerReachEnv()
    import ipdb; ipdb.set_trace()
    # test

