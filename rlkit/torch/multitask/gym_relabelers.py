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


class ContinuousRelabeler(RandomRelabeler):
    def __init__(self, dim=1, low=-1.0, high=1.0, **kwargs):
        super().__init__(**kwargs)
        self.latent_space = spaces.Box(low=low, high=high, shape=(dim,))

    def sample_task(self):
        return self.latent_space.sample()

    def get_features(self, path, latent=None):
        raise NotImplementedError


    def get_features_matrix(self, paths):
        return np.array([self.get_features(path) for path in paths])



class MountainCarRelabeler(ContinuousRelabeler):
    def __init__(self, dim=1, low=0.0, high=1.0, **kwargs):
        assert dim == 1
        super().__init__(dim, low, high, **kwargs)

    def reward_done(self, obs, action, latent, env_info=None):
        alpha = float(latent)
        return alpha * env_info['done_reward'] + (1 - alpha) * env_info['action_reward']

    def calculate_path_reward(self, path, latent):
        env_infos = path['env_infos']
        done_rewards = np.array([env_info['done_reward'] for env_info in env_infos])
        action_rewards = np.array([env_info['action_reward'] for env_info in env_infos])

        alpha = float(latent)
        return alpha * done_rewards + (1 - alpha) * action_rewards

class ReacherRelabeler(ContinuousRelabeler):
    def __init__(self, dim=2, low=0.0, high=1.0, **kwargs):
        super().__init__(dim, low, high, **kwargs)

    def latent_to_coeffs(self, latent):
        theta, phi = 2 * np.pi * latent[0], np.arccos(2 * latent[1] - 1)
        return np.array([np.cos(theta) * np.sin(phi), np.sin(phi) * np.sin(theta), np.cos(phi)])

    def coords_to_latent(self, coords):
        assert np.isclose(np.linalg.norm(coords), 1)
        theta, phi = np.arctan2(coords[1], coords[0]), np.arccos(coords[2])
        theta = np.where(theta < 0, 2 * np.pi + theta, theta)
        return np.array([theta / 2.0 / np.pi, 0.5 * (np.cos(phi) + 1)])

    def reward_done(self, obs, action, latent, env_info=None):
        coeffs = self.latent_to_coeffs(latent)
        return env_info['reward_dist'] * coeffs[0] + \
               env_info['reward_energy'] * coeffs[1] + \
               env_info['reward_safety'] * coeffs[2], False

    def plot_coeffs(self):
        from mpl_toolkits.mplot3d import Axes3D
        coords = np.array([self.latent_to_coeffs(self.sample_task()) for _ in range(10000)])
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(xs=coords[:, 0], ys=coords[:, 1], zs=coords[:, 2], s=1.0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def calculate_path_reward(self, path, latent):
        env_infos = path['env_infos']
        reward_dist = np.array([env_info['reward_dist'] for env_info in env_infos])
        reward_energy = np.array([env_info['reward_energy'] for env_info in env_infos])
        reward_safety = np.array([env_info['reward_safety'] for env_info in env_infos])
        coeffs = self.latent_to_coeffs(latent)
        return reward_dist * coeffs[0] + reward_energy * coeffs[1] + reward_safety * coeffs[2]

    def check_conversions(self):
        latents = np.array([self.sample_task() for _ in range(1000)])
        coords = [self.latent_to_coeffs(latent) for latent in latents]
        latents_back = np.array([self.coords_to_latent(coord) for coord in coords])
        sum = np.abs(latents - latents_back).sum()
        assert np.isclose(sum/100.0, 0)

class ReacherRelabelerWithGoal(ReacherRelabeler):
    def __init__(self, dim=4, low=0.0, high=1.0, **kwargs):
        self.num_parameters = 2
        self.maxs = np.zeros(2)
        self.mins = - np.ones(2)
        super().__init__(dim, low, high, **kwargs)

    def sample_task(self):
        # latent: [u, v, theta_goal, r_goal]
        return np.random.uniform(low=[0.0, 0.5, -np.pi, 0], high=[0.25, 1.0, np.pi, 0.3], size=4)

    def get_goal(self, latent):
        return latent[3] * np.array([np.cos(latent[2]), np.sin(latent[2])])

    def interpret_latent(self, latent):
        coeffs = self.latent_to_coeffs(latent)
        goal_pos = latent[3] * np.array([np.cos(latent[2]), np.sin(latent[2])])
        return "dist_weight:{:.2f}, energy_weight:{:.2f}, safety_weight:{:.2f}, goal pos:({:.2f}, {:.2f})".format(
            coeffs[0], coeffs[1], coeffs[2], goal_pos[0], goal_pos[1]
        )

    def coords_to_latent(self, coords, goal_params=np.array([0.0, 0.3])):
        assert np.isclose(np.linalg.norm(coords), 1)
        theta, phi = np.arctan2(coords[1], coords[0]), np.arccos(coords[2])
        theta = np.where(theta < 0, 2 * np.pi + theta, theta)
        return np.array([theta / 2.0 / np.pi, 0.5 * (np.cos(phi) + 1), goal_params[0], goal_params[1]])

    #todo: double check that this is right
    def reward_done(self, obs, action, latent, env_info=None):
        coeffs = self.latent_to_coeffs(latent)
        goal_pos = latent[3] * np.array([np.cos(latent[2]), np.sin(latent[2])])
        reward_dist = - np.linalg.norm(env_info['end_effector_loc'][:2] - goal_pos)
        return reward_dist * coeffs[0] + \
               env_info['reward_energy'] * coeffs[1] + \
               env_info['reward_safety'] * coeffs[2], False

    # todo: double check that this is right
    def calculate_path_reward(self, path, latent):
        env_infos = path['env_infos']
        goal_pos = latent[3] * np.array([np.cos(latent[2]), np.sin(latent[2])])
        end_effector_locs = np.array([env_info['end_effector_loc'][:2] for env_info in env_infos])
        reward_dist = - np.linalg.norm(end_effector_locs - goal_pos, axis=1)
        reward_energy = np.array([env_info['reward_energy'] for env_info in env_infos])
        reward_safety = np.array([env_info['reward_safety'] for env_info in env_infos])
        coeffs = self.latent_to_coeffs(latent)
        return reward_dist * coeffs[0] + reward_energy * coeffs[1] + reward_safety * coeffs[2]


class ReacherRelabelerWithFixedGoal(ReacherRelabeler):

    def sample_task(self):
        # latent: [alpha, theta_goal, r_goal]
        return np.array([np.random.uniform(low=0, high=np.pi/2), np.pi, 0.25])
        # return np.array([np.pi/2, np.pi, 0.25])  # set energy weight to 1

    def get_goal(self, latent):
        return latent[2] * np.array([np.cos(latent[1]), np.sin(latent[1])])

    def interpret_latent(self, latent):
        goal_pos = self.get_goal(latent)
        return "dist_weight:{:.2f}, energy_weight:{:.2f}, goal pos:({:.2f}, {:.2f})".format(
            np.cos(latent[0]), np.sin(latent[0]), goal_pos[0], goal_pos[1]
        )

    #todo: double check that this is right
    def reward_done(self, obs, action, latent, env_info=None):
        goal_pos = self.get_goal(latent)
        reward_dist = 0.2 - np.linalg.norm(env_info['end_effector_loc'][:2] - goal_pos)
        return reward_dist * np.cos(latent[0]) + env_info['reward_energy'] * np.sin(latent[0]), False

    # todo: double check that this is right
    def calculate_path_reward(self, path, latent):
        env_infos = path['env_infos']
        goal_pos = self.get_goal(latent)
        end_effector_locs = np.array([env_info['end_effector_loc'][:2] for env_info in env_infos])
        reward_dist = 0.2 - np.linalg.norm(end_effector_locs - goal_pos, axis=1)
        reward_energy = np.array([env_info['reward_energy'] for env_info in env_infos])
        result = reward_dist * np.cos(latent[0]) + reward_energy * np.sin(latent[0])
        # multiplier = np.array([np.cos(latent[0]), np.sin(latent[0])])
        # print(result, (self.calculate_path_features(path, latent) * multiplier.reshape([2, 1])).sum())
        return result
        # return reward_dist * np.cos(latent[0]) + reward_energy * np.sin(latent[0])


    def calculate_path_features(self, path, latent):
        env_infos = path['env_infos']
        goal_pos = self.get_goal(latent)
        end_effector_locs = np.array([env_info['end_effector_loc'][:2] for env_info in env_infos])
        reward_dist = 0.2 - np.linalg.norm(end_effector_locs - goal_pos, axis=1)
        reward_energy = np.array([env_info['reward_energy'] for env_info in env_infos])
        return np.array([reward_dist, reward_energy])

    def update_sliding_params(self, paths):
        print("original sliding params:", self.mins, self.maxs)
        latents = [path['latents'][0] for path in paths]
        all_features = np.array([self.calculate_path_features(path, latent) for path, latent in zip(paths, latents)])
        mins, maxes = np.amin(all_features), np.amax(all_features)
        self.maxs = self.maxs * (1 - self.tau) + maxes * self.tau
        self.mins = self.mins * (1 - self.tau) + mins * self.tau

    def get_reward_matrix(self, paths, latents):
        return np.array([[self.get_discounted_path_reward(path, latent) for latent in latents] for path in paths])

class ReacherRelabelerWithGoalSimple(ReacherRelabeler):
    def __init__(self, test=False, sparse_reward=False, fixed_ratio=None, **kwargs):
        super().__init__(**kwargs)
        self.test = test
        self.sparse_reward = sparse_reward
        print("sparse reward:", self.sparse_reward)
        self.fixed_ratio = fixed_ratio

    def sample_task(self):
        # latent: [alpha, theta_goal, r_goal]
        if self.test:
            return np.concatenate([np.array([np.pi/4]), np.random.uniform(low=[-np.pi, 0.15], high=[np.pi, 0.3], size=2)])
        elif self.fixed_ratio is not None:
            return np.concatenate([np.array([self.fixed_ratio]), np.random.uniform(low=[-np.pi, 0.15], high=[np.pi, 0.3], size=2)])
        return np.random.uniform(low=[0, -np.pi, 0.1], high=[np.pi/2, np.pi, 0.3], size=3)

    def get_goal(self, latent):
        return latent[2] * np.array([np.cos(latent[1]), np.sin(latent[1])])

    def interpret_latent(self, latent):
        goal_pos = self.get_goal(latent)
        return "dist_weight:{:.2f}, energy_weight:{:.2f}, goal pos:({:.2f}, {:.2f})".format(
            np.cos(latent[0]), np.sin(latent[0]), goal_pos[0], goal_pos[1]
        )

    #todo: double check that this is right
    def reward_done(self, obs, action, latent, env_info=None):
        goal_pos = self.get_goal(latent)
        dist = np.linalg.norm(env_info['end_effector_loc'][:2] - goal_pos)
        if self.sparse_reward:
            # reward_dist = 0.2 + 0.3 * (np.exp(-dist ** 2 / 4E-4) - 1)
            reward_dist = self.sparse_reward * (np.exp(-dist ** 2 / 0.08**2))
        else:
            reward_dist = 0.2 - dist
        # print('distance:{:.4f}, energy:{:.4f}'.format(reward_dist, env_info['reward_energy']))
        return reward_dist * np.cos(latent[0]) + env_info['reward_energy'] * np.sin(latent[0]), False

    # todo: double check that this is right
    def calculate_path_reward(self, path, latent):
        env_infos = path['env_infos']
        goal_pos = self.get_goal(latent)
        end_effector_locs = np.array([env_info['end_effector_loc'][:2] for env_info in env_infos])
        dists = np.linalg.norm(end_effector_locs - goal_pos, axis=1)
        if self.sparse_reward:
            # reward_dist = 0.2 + 0.3 * (np.exp(-dists ** 2 / 4E-4) - 1)
            reward_dist = self.sparse_reward * (np.exp(-dists ** 2 / 0.08**2))
        else:
            reward_dist = 0.2 - dists
        reward_energy = np.array([env_info['reward_energy'] for env_info in env_infos])
        result = reward_dist * np.cos(latent[0]) + reward_energy * np.sin(latent[0])
        # multiplier = np.array([np.cos(latent[0]), np.sin(latent[0])])
        # print(result, (self.calculate_path_features(path, latent) * multiplier.reshape([2, 1])).sum())
        return result
        # return reward_dist * np.cos(latent[0]) + reward_energy * np.sin(latent[0])


    def calculate_path_features(self, path, latent):
        raise RuntimeError
        env_infos = path['env_infos']
        goal_pos = self.get_goal(latent)
        end_effector_locs = np.array([env_info['end_effector_loc'][:2] for env_info in env_infos])
        reward_dist = 0.2 - np.linalg.norm(end_effector_locs - goal_pos, axis=1)
        reward_energy = np.array([env_info['reward_energy'] for env_info in env_infos])
        return np.array([reward_dist, reward_energy])

    def update_sliding_params(self, paths):
        print("original sliding params:", self.mins, self.maxs)
        latents = [path['latents'][0] for path in paths]
        all_features = np.array([self.calculate_path_features(path, latent) for path, latent in zip(paths, latents)])
        mins, maxes = np.amin(all_features), np.amax(all_features)
        self.maxs = self.maxs * (1 - self.tau) + maxes * self.tau
        self.mins = self.mins * (1 - self.tau) + mins * self.tau

    def get_reward_matrix(self, paths, latents):
        return np.array([[self.get_discounted_path_reward(path, latent) for latent in latents] for path in paths])
        # return np.array([[self.calculate_path_reward(path, latent).sum() for latent in latents] for path in paths])


    def plot_paths(self, paths, orig_latents, new_latents, title='Reacher'):
        # plot the first trajectory in blue, original goal as x, new goal as square
        # plot the rest of the trajectories in red

        print("plotting", title)
        # each element of trajectory_latent_lst is (path, [original_z, rest of zs])
        num_trajs = len(paths)
        print(num_trajs, 'trajectories')
        fig, axes = plt.subplots((num_trajs + 1) // 2, 2, figsize=(5, 10))

        for i in range(num_trajs):
            ax = axes[i // 2, i % 2]
            path = paths[i]
            locs = np.array([env_info['end_effector_loc'][:2] for env_info in path['env_infos']])
            color = list(plt.cm.rainbow(np.linspace(0, 1, len(locs))))
            # c = 'b' if i == 0 else 'r'
            ax.scatter(locs[:, 0], locs[:, 1], c=color, alpha=0.9, s=3)
            ax.scatter(x=locs[-1][0], y=locs[-1][1], marker='1', c=color[-1], s=30)
            orig_goal, new_goal = self.get_goal(orig_latents[i]), self.get_goal(new_latents[i])
            ax.scatter(x=orig_goal[0], y=orig_goal[1], marker='x', c='r', s=15)
            ax.scatter(x=new_goal[0], y=new_goal[1], marker='s', c='g', s=15)
            ax.set_aspect('equal')
            ax.set_xlim([-0.4, 0.4])
            ax.set_ylim([-0.4, 0.4])
        fig.tight_layout()
        exp_name = 'irl'
        plt.savefig(osp.join(logger.get_snapshot_dir(), '{}_{}'.format(exp_name, title)))
        plt.close('all')
        print("done plotting", title)

class ReacherRelabelerWithGoalAndObs(ReacherRelabeler):
    def __init__(self, test=False, sparse_reward=False, fixed_ratio=None, **kwargs):
        super().__init__(**kwargs)
        self.test = test
        self.sparse_reward = sparse_reward
        print("sparse reward:", self.sparse_reward)
        self.fixed_ratio = fixed_ratio
        assert not self.fixed_ratio

    def sample_task(self):
        # latent: [u, v, theta_goal, r_goal, theta_obs, r_obs]
        return np.random.uniform(low=[0.0, 0.5, -np.pi, 0.15, -np.pi, 0.15], high=[0.25, 1.0, np.pi, 0.3, np.pi, 0.3], size=6)

    def get_goal(self, latent):
        return latent[3] * np.array([np.cos(latent[2]), np.sin(latent[2])])

    def get_obstacle(self, latent):
        return latent[5] * np.array([np.cos(latent[4]), np.sin(latent[4])])

    def calculate_path_features(self, path, latent):
        env_infos = path['env_infos']
        goal_pos, obs_pos = self.get_goal(latent), self.get_obstacle(latent)
        end_effector_locs = np.array([env_info['end_effector_loc'][:2] for env_info in env_infos])
        dists = np.linalg.norm(end_effector_locs - goal_pos, axis=1)
        dists_obs = np.linalg.norm(end_effector_locs - obs_pos, axis=1)
        if self.sparse_reward:
            # reward_dist = 0.2 + 0.3 * (np.exp(-dists ** 2 / 4E-4) - 1)
            reward_dist = self.sparse_reward * (np.exp(-dists ** 2 / 0.08**2))
        else:
            reward_dist = 0.2 - dists
        reward_safety = np.log10(dists_obs + 1e-2) / 5.0
        reward_energy = np.array([env_info['reward_energy'] for env_info in env_infos])
        return np.concatenate([reward_dist[:, np.newaxis], reward_energy[:, np.newaxis], reward_safety[:, np.newaxis]], axis=1)


    def interpret_latent(self, latent):
        coeffs = self.latent_to_coeffs(latent)
        goal_pos = self.get_goal(latent)
        obs_pos = self.get_obstacle(latent)
        return "dist_weight:{:.2f}, energy_weight:{:.2f}, safety_weight:{:.2f}, goal pos:({:.2f}, {:.2f}), obs pos:({:.2f}, {:.2f})".format(
            coeffs[0], coeffs[1], coeffs[2], goal_pos[0], goal_pos[1], obs_pos[0], obs_pos[1]
        )

    def coords_to_latent(self, coords, goal_params=np.array([0.0, 0.3])):
        raise NotImplementedError
        assert np.isclose(np.linalg.norm(coords), 1)
        theta, phi = np.arctan2(coords[1], coords[0]), np.arccos(coords[2])
        theta = np.where(theta < 0, 2 * np.pi + theta, theta)
        return np.array([theta / 2.0 / np.pi, 0.5 * (np.cos(phi) + 1), goal_params[0], goal_params[1]])

    def coords_to_uv(self, coords):
        assert np.isclose(np.linalg.norm(coords), 1)
        theta, phi = np.arctan2(coords[1], coords[0]), np.arccos(coords[2])
        theta = np.where(theta < 0, 2 * np.pi + theta, theta)
        return np.array([theta / 2.0 / np.pi, 0.5 * (np.cos(phi) + 1)])

    def get_features(self, path, latent=None):
        return np.zeros(len(path))

    def plot_resampling_heatmaps(self, trajectory_latent_lst, title, traj_infos=None):
        print("plotting", title)
        # each element of trajectory_latent_lst is (path, [original_z, rest of zs])
        num_trajs = len(trajectory_latent_lst)
        num_lats = len(trajectory_latent_lst[0][1])

        # create titles for subplots:
        if traj_infos is not None:
            r = traj_infos['rewards']
            v1 = traj_infos['v1']
            v2 = traj_infos['v2']
            adv = traj_infos['adv']
            titles = [["r:{:.1f}_v1:{:.1f}_v2:{:.1f}_adv:{:.1f}".format(r[traj_i][lat_j],
                                                                    v1[traj_i][lat_j],
                                                                    v2[traj_i][lat_j],
                                                                    adv[traj_i][lat_j])
                       for lat_j in range(num_lats)] for traj_i in range(num_trajs)]


            fig, axs = plt.subplots(num_lats, num_trajs, sharex='col', sharey='row', figsize=(8,10))
        else:
            fig, axs = plt.subplots(2, num_trajs, sharex='col', sharey='row', figsize=(9, 6))
        for i in range(num_trajs):
            # locs = trajectory_latent_lst[i][0]['observations']
            path = trajectory_latent_lst[i][0]
            env_infos = path['env_infos']
            locs = np.array([env_info['end_effector_loc'][:2] for env_info in env_infos])
            for j in range(2): # num_lats
                latent = trajectory_latent_lst[i][1][j]
                dx, dy = 0.005, 0.005
                y, x = np.mgrid[slice(-1, 1 + dy, dy),
                                slice(-1, 1 + dx, dx)]
                mesh_xs = np.stack([x, y], axis=2).reshape(-1, 2)
                plotting_env_infos = []
                for xy in mesh_xs:
                    plotting_env_infos.append(dict(end_effector_loc=xy, reward_energy=0.0))
                path = dict(env_infos=plotting_env_infos)
                rewards = self.calculate_path_reward(path, latent)
                ax = axs[j, i]
                c = ax.pcolor(x, y, rewards.reshape([y.shape[0], y.shape[1]]), cmap='OrRd')
                ax.plot(locs[:,0], locs[:,1], c='g', linewidth=3)
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                if traj_infos is not None:
                    subtitle = titles[i][j]
                    ax.set_title(subtitle, size=8.0)
                # fig.colorbar(c, ax=ax)
        # fig.suptitle(title)
        fig.tight_layout()
        exp_name = "adv" if self.q1 is not None else "reward"
        plt.savefig(osp.join(logger.get_snapshot_dir(), '{}_{}'.format(exp_name, title)))
        plt.close('all')
        print("done plotting", title)

    #todo: double check that this is right
    def reward_done(self, obs, action, latent, env_info=None):
        goal_pos, obs_pos = self.get_goal(latent), self.get_obstacle(latent)
        dist = np.linalg.norm(env_info['end_effector_loc'][:2] - goal_pos)
        dist_obs = np.linalg.norm(env_info['end_effector_loc'][:2] - obs_pos)
        if self.sparse_reward:
            # reward_dist = 0.2 + 0.3 * (np.exp(-dist ** 2 / 4E-4) - 1)
            reward_dist = self.sparse_reward * (np.exp(-dist ** 2 / 0.08**2))
        else:
            reward_dist = 0.2 - dist
        reward_safety = np.log10(dist_obs + 1e-2) / 5.0

        coeffs = self.latent_to_coeffs(latent)
        return reward_dist * coeffs[0] + \
               env_info['reward_energy'] * coeffs[1] + \
               reward_safety * coeffs[2], False

    # todo: double check that this is right
    def calculate_path_reward(self, path, latent):
        env_infos = path['env_infos']
        goal_pos, obs_pos = self.get_goal(latent), self.get_obstacle(latent)
        end_effector_locs = np.array([env_info['end_effector_loc'][:2] for env_info in env_infos])
        dists = np.linalg.norm(end_effector_locs - goal_pos, axis=1)
        dists_obs = np.linalg.norm(end_effector_locs - obs_pos, axis=1)
        if self.sparse_reward:
            # reward_dist = 0.2 + 0.3 * (np.exp(-dists ** 2 / 4E-4) - 1)
            reward_dist = self.sparse_reward * (np.exp(-dists ** 2 / 0.08**2))
        else:
            reward_dist = 0.2 - dists
        reward_safety = np.log10(dists_obs + 1e-2) / 5.0
        reward_energy = np.array([env_info['reward_energy'] for env_info in env_infos])
        coeffs = self.latent_to_coeffs(latent)
        return reward_dist * coeffs[0] + reward_energy * coeffs[1] + reward_safety * coeffs[2]

    def get_reward_matrix(self, paths, latents):
        return np.array([[self.get_discounted_path_reward(path, latent) for latent in latents] for path in paths])


class ReacherRelabelerWithGoalSimpleDeterministic(ReacherRelabelerWithGoalSimple):
    def __init__(self, path=None, **kwargs):
        self.latents = self.get_reference_latents()
        self.i = 0
        if path is not None:
            self.normalizing_constants = np.load(path)
            assert len(self.latents) == len(self.normalizing_constants)
        super().__init__(**kwargs)

    def sample_task(self):
        self.i = (self.i + 1) % len(self.latents)
        return self.latents[self.i % len(self.latents)]

    def get_normalizing_constant(self, latent):
        index = np.where(np.isclose(self.latents, latent))[0]
        return self.normalizing_constants[index]

    def get_normalized_path_rewards(self, path):
        latent = path['latents'][0]
        return path['rewards'] / self.get_normalizing_constant(latent)

    def get_reference_latents(self):
        # latent: [alpha, theta_goal, r_goal]
        alphas = np.linspace(0, np.pi / 2, 6)[:-1] + np.pi/2/5/2
        thetas = np.linspace(-np.pi, np.pi, 6)[:-1] + np.pi*2/2/5
        rs = np.linspace(0.15, 0.3, 5)[:-1] + 0.15/4/2
        latents = list(product(alphas, thetas, rs))
        return [np.array(z) for z in latents]


class HERReacherRelabelerWithGoalSimple(ReacherRelabelerWithGoalSimple):

    def get_latents_and_rewards(self, path):
        latent = self.sample_task()
        env_infos = path['env_infos']
        last_pos = env_infos[-1]['end_effector_loc'][:2]
        # alpha, theta_goal, r_goal
        theta, r = np.arccos(last_pos[1]/np.arccos(last_pos[0])), np.linalg.norm(last_pos)
        latent[1] = theta
        latent[2] = r
        return [latent], [self.calculate_path_reward(path, latent)], [latent]

# discretized evenly
class ReacherRelabelerWithFixedGoalDiscretized(ReacherRelabelerWithFixedGoal):
    def __init__(self, n=10, **kwargs):
        self.n = n
        latents = np.linspace(0, np.pi / 2, n).reshape([n, -1])
        self.latents = [np.concatenate([latent, np.array([np.pi, 0.25])]) for latent in latents]

    def sample_task(self):
        return self.latents[np.random.choice(self.n)]


class HERReacherRelabelerWithGoalSimpleDiscretized(HERReacherRelabelerWithGoalSimple):
    def __init__(self, n=10, **kwargs):
        self.n = n
        latents = np.linspace(0, np.pi / 2, n).reshape([n, -1])
        self.latents = [np.concatenate([latent, np.array([np.pi, 0.25])]) for latent in latents]
        super().__init__(**kwargs)
        import ipdb; ipdb.set_trace()  # todo: make sure there isn't that weird problem with initialization


class HERReacherRelabelerWithGoalAndObs(ReacherRelabelerWithGoalAndObs):
    def get_latents_and_rewards(self, path):
        # sample n_to_take places along the trajectory and change that about the latent
        observations = path['observations']
        new_locs = observations[np.random.choice(len(observations), self.n_to_take)][:,:2]
        new_locs_polar = [np.linalg.norm(new_locs, axis=1), np.arctan2(new_locs[:,1], new_locs[:,0])]

        # latent: [u, v, theta_goal, r_goal, theta_obs, r_obs]
        orig_latent = path['latents'][0]
        new_latents = [np.concatenate([orig_latent[:2],
                                       new_locs_polar[1][i:i+1],
                                       new_locs_polar[0][i:i+1],
                                       orig_latent[-2:]]) for i in range(self.n_to_take)]
        return new_latents, [self.calculate_path_reward(path, latent) for latent in new_latents], []


class OutOfDistReacherRelabelerWithGoalAndObs(ReacherRelabelerWithGoalAndObs):
    def sample_task(self):
        z = super().sample_task()
        coeffs = self.latent_to_coeffs(z)
        while 0.3 < coeffs[1] < 0.4 or 0.8 < coeffs[1] < 0.9:
            z = super().sample_task()
            coeffs = self.latent_to_coeffs(z)
        return z

class GetOutOfDistReacherRelabelerWithGoalAndObs(ReacherRelabelerWithGoalAndObs):
    def sample_task(self):
        z = super().sample_task()
        coeffs = self.latent_to_coeffs(z)
        while not (0.3 < coeffs[1] < 0.4 or 0.8 < coeffs[1] < 0.9):
            z = super().sample_task()
            coeffs = self.latent_to_coeffs(z)
        return z


class OutOfDistHERReacherRelabelerWithGoalAndObs(HERReacherRelabelerWithGoalAndObs):
    def sample_task(self):
        z = super().sample_task()
        coeffs = self.latent_to_coeffs(z)
        while 0.3 < coeffs[1] < 0.4 or 0.8 < coeffs[1] < 0.9:
            z = super().sample_task()
            coeffs = self.latent_to_coeffs(z)
        return z


if __name__ == "__main__":
    # obs = np.random.uniform(size=(5,2))
    # obs = np.concatenate([obs, np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])])
    # print(obs)
    # relabeler = MountainCarRelabeler()
    # relabeler = ReacherRelabeler()
    # relabeler.check_conversions()
    # relabeler.plot_coeffs()
    relabeler = ReacherRelabelerWithGoal()
    relabeler.plot_coeffs()
    import ipdb; ipdb.set_trace()
