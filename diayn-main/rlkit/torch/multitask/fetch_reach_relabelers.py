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
from rlkit.envs.fetch_reach import FetchReachEnv
from rlkit.envs.point_reacher_env_3d import PointReacherEnv3D
from rlkit.torch.multitask.gym_relabelers import ReacherRelabeler
from rlkit.torch.multitask.reference_latents import FETCH_REACH_GOAL_OBS_LATENT

class FetchReachRelabelerWithGoalAndObs(ReacherRelabeler):
    def __init__(self, test=False, sparse_reward=False, fixed_ratio=None, fetchreach=False, **kwargs):
        super().__init__(**kwargs)
        self.test = test
        self.sparse_reward = sparse_reward
        print("sparse reward:", self.sparse_reward)
        assert sparse_reward
        self.fixed_ratio = fixed_ratio
        assert not self.fixed_ratio
        if fetchreach:
            self.env = FetchReachEnv()
        else:
            self.env = PointReacherEnv3D()
        if self.is_eval:
            self.eval_latents = FETCH_REACH_GOAL_OBS_LATENT.copy()
            self.curr_idx = 0

    def sample_task(self):
        # latent: [u, v, x_goal, y_goal, z_goal, x_obs, y_obs, z_obs]
        if self.is_eval:
            self.curr_idx = (self.curr_idx + 1) % len(self.eval_latents)
            return self.eval_latents[self.curr_idx].copy()
        goal_pos = self.env._sample_goal()
        obs_pos = self.env._sample_goal()
        weights = np.random.uniform(low=[0.0, 0.5], high=[0.25, 1.0], size=2)

        return np.concatenate([weights, goal_pos, obs_pos])

    def get_goal(self, latent):
        return latent[2:5]

    def get_obstacle(self, latent):
        return latent[5:]

    def calculate_path_features(self, path, latent):
        env_infos = path['env_infos']
        goal_pos, obs_pos = self.get_goal(latent), self.get_obstacle(latent)
        end_effector_locs = np.array([env_info['end_effector_loc'] for env_info in env_infos])  #todo: check end effector loc dimensionality
        dists = np.linalg.norm(end_effector_locs - goal_pos, axis=1)
        dists_obs = np.linalg.norm(end_effector_locs - obs_pos, axis=1)
        if self.sparse_reward <= 1.01:
            # reward_dist = 0.2 + 0.3 * (np.exp(-dists ** 2 / 4E-4) - 1)
            reward_dist = self.sparse_reward * (np.exp(-dists ** 2 / 0.08**2))
        elif self.sparse_reward <= 2.01:
            reward_dist = (dists < 0.04).astype(np.float32)
        else:
            reward_dist = 0.2 - dists
        # print(reward_dist)
        reward_safety = np.log10(dists_obs + 1e-2) / 5.0
        # reward_safety = 1 + np.log10(dists_obs + 1e-2) / 2.0
        reward_energy = np.array([env_info['reward_energy'] for env_info in env_infos])
        return np.concatenate([reward_dist[:, np.newaxis], reward_energy[:, np.newaxis], reward_safety[:, np.newaxis]], axis=1)


    def interpret_latent(self, latent):
        coeffs = self.latent_to_coeffs(latent)
        goal_pos = self.get_goal(latent)
        obs_pos = self.get_obstacle(latent)
        return "dist_weight:{:.2f}, energy_weight:{:.2f}, safety_weight:{:.2f}, goal pos:({:.2f}, {:.2f}, {:.2f}), obs pos:({:.2f}, {:.2f}, {:.2f})".format(
            coeffs[0], coeffs[1], coeffs[2], goal_pos[0], goal_pos[1], goal_pos[2], obs_pos[0], obs_pos[1], obs_pos[2]
        )

    def coords_to_latent(self, coords, goal_params=np.array([0.0, 0.3])):
        raise NotImplementedError

    #todo: double check that this is right
    def reward_done(self, obs, action, latent, env_info=None):
        goal_pos, obs_pos = self.get_goal(latent), self.get_obstacle(latent)
        dist = np.linalg.norm(env_info['end_effector_loc'] - goal_pos)
        dist_obs = np.linalg.norm(env_info['end_effector_loc'] - obs_pos)
        if self.sparse_reward <= 1.01:
            # reward_dist = 0.2 + 0.3 * (np.exp(-dist ** 2 / 4E-4) - 1)
            reward_dist = self.sparse_reward * (np.exp(-dist ** 2 / 0.08**2))
        elif self.sparse_reward <= 2.01:
            reward_dist = (dist < 0.04).astype(np.float32)
        else:
            reward_dist = 0.2 - dist
        # print(reward_dist)
        reward_safety = np.log10(dist_obs + 1e-2) / 5.0
        # reward_safety = 1 + np.log10(dist_obs + 1e-2)/2.0

        coeffs = self.latent_to_coeffs(latent)
        # print(reward_dist, env_info['reward_energy'], reward_safety)
        return reward_dist * coeffs[0] + \
               env_info['reward_energy'] * coeffs[1] + \
               reward_safety * coeffs[2], False

    # todo: double check that this is right
    def calculate_path_reward(self, path, latent):
        env_infos = path['env_infos']
        goal_pos, obs_pos = self.get_goal(latent), self.get_obstacle(latent)
        end_effector_locs = np.array([env_info['end_effector_loc'] for env_info in env_infos])
        dists = np.linalg.norm(end_effector_locs - goal_pos, axis=1)
        dists_obs = np.linalg.norm(end_effector_locs - obs_pos, axis=1)
        if self.sparse_reward <= 1.01:
            # reward_dist = 0.2 + 0.3 * (np.exp(-dists ** 2 / 4E-4) - 1)
            reward_dist = self.sparse_reward * (np.exp(-dists ** 2 / 0.08**2))
            # reward_dist = (dists < 0.05).astype(np.float32)
        elif self.sparse_reward <= 2.01:
            reward_dist = (dists < 0.04).astype(np.float32)
        else:
            reward_dist = 0.2 - dists
        # print(reward_dist)
        reward_safety = np.log10(dists_obs + 1e-2) / 5.0
        # reward_safety = 1 + np.log10(dists_obs + 1e-2) / 2.0
        reward_energy = np.array([env_info['reward_energy'] for env_info in env_infos])
        coeffs = self.latent_to_coeffs(latent)
        return reward_dist * coeffs[0] + reward_energy * coeffs[1] + reward_safety * coeffs[2]

    def get_reward_matrix(self, paths, latents):
        return np.array([[self.get_discounted_path_reward(path, latent) for latent in latents] for path in paths])

    def get_features(self, path, latent=None):
        return np.zeros(len(path))

class FetchReachRelabelerWithGoalSimple(ReacherRelabeler):
    def __init__(self, test=False, sparse_reward=False, fixed_ratio=None, **kwargs):
        super().__init__(**kwargs)
        self.test = test
        self.sparse_reward = sparse_reward
        print("sparse reward:", self.sparse_reward)
        assert sparse_reward
        self.fixed_ratio = fixed_ratio
        assert not self.fixed_ratio
        self.env = FetchReachEnv()

    def latent_to_coeffs(self, latent):
        return np.array([np.cos(latent[0]), np.sin(latent[1])])

    def sample_task(self):
        # latent: [alpha, x_goal, y_goal]
        goal_pos = self.env._sample_goal()
        weights = np.random.uniform(low=[0.0], high=[np.pi/2], size=1)

        return np.concatenate([weights, goal_pos])

    def get_goal(self, latent):
        return latent[1:]

    def calculate_path_features(self, path, latent):
        env_infos = path['env_infos']
        goal_pos = self.get_goal(latent)
        end_effector_locs = np.array([env_info['end_effector_loc'] for env_info in env_infos])  #todo: check end effector loc dimensionality
        dists = np.linalg.norm(end_effector_locs - goal_pos, axis=1)
        if self.sparse_reward <= 1.01:
            # reward_dist = 0.2 + 0.3 * (np.exp(-dists ** 2 / 4E-4) - 1)
            reward_dist = self.sparse_reward * (np.exp(-dists ** 2 / 0.04**2))
        elif self.sparse_reward <= 2.01:
            reward_dist = self.sparse_reward * (np.exp(-dists ** 2 / 0.04 ** 2))
        else:
            reward_dist = 0.2 - dists
        # reward_safety = 1 + np.log10(dists_obs + 1e-2) / 2.0
        reward_energy = np.array([env_info['reward_energy'] for env_info in env_infos]) * 5
        return np.concatenate([reward_dist[:, np.newaxis], reward_energy[:, np.newaxis]], axis=1)


    def interpret_latent(self, latent):
        coeffs = self.latent_to_coeffs(latent)
        goal_pos = self.get_goal(latent)
        obs_pos = self.get_obstacle(latent)
        return "dist_weight:{:.2f}, energy_weight:{:.2f}, safety_weight:{:.2f}, goal pos:({:.2f}, {:.2f}, {:.2f}), obs pos:({:.2f}, {:.2f}, {:.2f})".format(
            coeffs[0], coeffs[1], coeffs[2], goal_pos[0], goal_pos[1], goal_pos[2], obs_pos[0], obs_pos[1], obs_pos[2]
        )

    def coords_to_latent(self, coords, goal_params=np.array([0.0, 0.3])):
        raise NotImplementedError

    #todo: double check that this is right
    def reward_done(self, obs, action, latent, env_info=None):
        goal_pos = self.get_goal(latent)
        dist = np.linalg.norm(env_info['end_effector_loc'] - goal_pos)
        if self.sparse_reward:
            # reward_dist = 0.2 + 0.3 * (np.exp(-dist ** 2 / 4E-4) - 1)
            reward_dist = self.sparse_reward * (np.exp(-dist ** 2 / 0.08**2))
        else:
            reward_dist = 0.2 - dist

        coeffs = self.latent_to_coeffs(latent)
        # print(reward_dist, env_info['reward_energy'], reward_safety)
        return reward_dist * coeffs[0] + env_info['reward_energy'] * coeffs[1] * 5, False

    # todo: double check that this is right
    def calculate_path_reward(self, path, latent):
        env_infos = path['env_infos']
        goal_pos = self.get_goal(latent)
        end_effector_locs = np.array([env_info['end_effector_loc'] for env_info in env_infos])
        dists = np.linalg.norm(end_effector_locs - goal_pos, axis=1)
        if self.sparse_reward:
            # reward_dist = 0.2 + 0.3 * (np.exp(-dists ** 2 / 4E-4) - 1)
            reward_dist = self.sparse_reward * (np.exp(-dists ** 2 / 0.08**2))
        else:
            reward_dist = 0.2 - dists
        # reward_safety = np.log10(dists_obs + 1e-2) / 5.0
        # reward_safety = 1 + np.log10(dists_obs + 1e-2) / 2.0
        reward_energy = np.array([env_info['reward_energy'] for env_info in env_infos])
        coeffs = self.latent_to_coeffs(latent)
        return reward_dist * coeffs[0] + 5 * reward_energy * coeffs[1]

    def get_reward_matrix(self, paths, latents):
        return np.array([[self.get_discounted_path_reward(path, latent) for latent in latents] for path in paths])

class HERFetchReacherRelabeler(FetchReachRelabelerWithGoalAndObs):
    def get_latents_and_rewards(self, path):
        # sample n_to_take places along the trajectory and change that about the latent
        env_infos = path['env_infos']
        observations = [p['end_effector_loc'] for p in env_infos]
        indices = np.random.choice(len(observations), self.n_to_take)
        new_locs = [observations[i] for i in indices]

        # latent: [u, v, x_goal, y_goal, z_goal, x_obs, y_obs, z_obs]
        orig_latent = path['latents'][0]
        new_latents = [np.concatenate([orig_latent[:2],
                                       new_locs[i],
                                       orig_latent[-3:]]) for i in range(self.n_to_take)]
        return new_latents, [self.calculate_path_reward(path, latent) for latent in new_latents], []

class OutOfDistFetchReachRelabelerWithGoalAndObs(FetchReachRelabelerWithGoalAndObs):
    def sample_task(self):
        # latent: [u, v, x_goal, y_goal, z_goal, x_obs, y_obs, z_obs]
        z = super().sample_task()
        coeffs = self.latent_to_coeffs(z)
        while 0.3 < coeffs[1] < 0.4 or 0.8 < coeffs[1] < 0.9:
            z = super().sample_task()
            coeffs = self.latent_to_coeffs(z)
        return z

class OutOfDistHERFetchReacherRelabeler(HERFetchReacherRelabeler):
    def sample_task(self):
        # latent: [u, v, x_goal, y_goal, z_goal, x_obs, y_obs, z_obs]
        z = super().sample_task()
        coeffs = self.latent_to_coeffs(z)
        while 0.3 < coeffs[1] < 0.4 or 0.8 < coeffs[1] < 0.9:
            z = super().sample_task()
            coeffs = self.latent_to_coeffs(z)
        return z

if __name__ == '__main__':
    # r = OutOfDistFetchReachRelabelerWithGoalAndObs(sparse_reward=1.0)
    # coeffs = np.array([r.latent_to_coeffs(r.sample_task()) for _ in range(1000)])
    # y = coeffs[:,1]
    # import matplotlib.pyplot as plt
    # plt.hist(y, bins=100)
    # plt.show()
    # import ipdb; ipdb.set_trace()

    relabeler = FetchReachRelabelerWithGoalAndObs(sparse_reward=2)
    latents = [relabeler.sample_task() for _ in range(25)]
    import ipdb; ipdb.set_trace(context=10)