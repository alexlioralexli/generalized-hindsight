import numpy as np
from gym import spaces
from rlkit.torch.multitask.rewards import Relabeler, RandomRelabeler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm
from rlkit.core import logger
import os.path as osp

class PointMassBestRandomRelabeler(RandomRelabeler):
    def __init__(self, power=1, **kwargs):
        super().__init__(**kwargs)
        self.power = power
        self.latent = self.sample_task()
        self.rotation_matrix = self.calculate_rotation_matrix(self.latent)

    def calculate_rotation_matrix(self, latent):
        angle, _, _ = latent
        return np.array([np.cos(angle), -np.sin(angle), np.sin(angle), np.cos(angle)]).reshape([2,2])

    def reward_done(self, obs, action, latent, env_infos=None):
        # obs is shaped (d,)
        _, d, a = latent
        if np.array_equal(self.latent, latent):
            rotation_matrix = self.rotation_matrix
        else:
            rotation_matrix = self.calculate_rotation_matrix(latent)
        obs_new = obs[:2].dot(rotation_matrix.T)
        x, y = obs_new[0].flatten(), obs_new[1].flatten()
        alpha = x/d
        if alpha < 0:
            return -0.0, False
        else:
            return ((alpha+1e-12)**self.power * norm.pdf(y - a*np.sin(x*np.pi/d), scale=0.05))[0], False

    def train(self):
        pass

    def sample_task(self):
        angle = np.random.uniform(-np.pi, np.pi, 1)[0]
        d = np.random.uniform(0.75, 1, 1)[0]  # used to be 0.5
        a = np.random.uniform(-0.25, 0.25, 1)[0]
        return np.array([angle, d, a])

    def calculate_path_reward(self, path, latent):
        angle, d, a = latent
        rotation_matrix = self.calculate_rotation_matrix(latent)

        obs = path['observations']
        obs_new = obs[:,:2].dot(rotation_matrix.T)
        xs, ys = obs_new[:, 0].flatten(), obs_new[:, 1].flatten()
        alphas = xs/d
        sin_rewards = (np.abs(alphas) + 1e-12)**self.power * norm.pdf(ys - a*np.sin(xs*np.pi/d), scale=0.05)
        wrong_dir_rewards = np.zeros(len(alphas))
        rewards = np.where(alphas >= 0.0, sin_rewards.flatten(), wrong_dir_rewards.flatten())
        return rewards

    def get_reward_matrix(self, paths, latents):
        return np.array([[self.get_discounted_path_reward(path, latent) for latent in latents] for path in paths])


    def make_reward_heatmap(self, latent, title='/tmp/heatmap.png'):
        dx, dy = 0.01, 0.01
        y, x = np.mgrid[slice(-1, 1 + dy, dy),
                        slice(-1, 1 + dx, dx)]
        mesh_xs = np.stack([x, y], axis=2).reshape(-1, 2)
        rewards = self.calculate_path_reward(dict(observations=mesh_xs), latent)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        c = ax.pcolor(x, y, rewards.reshape([y.shape[0], y.shape[1]]))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Reward Heatmap")
        ax.set_aspect('equal')
        fig.colorbar(c, ax=ax)
        plt.savefig(title)

    def plot_trajectory_on_heatmap(self, latent, path, title):
        locs = path['observations']

        dx, dy = 0.01, 0.01
        y, x = np.mgrid[slice(-1, 1 + dy, dy),
                        slice(-1, 1 + dx, dx)]
        mesh_xs = np.stack([x, y], axis=2).reshape(-1, 2)
        # ll = torch.exp(self.made.get_log_prob(torch.tensor(mesh_xs).to(device).float())).cpu().detach().numpy()
        # raise NotImplementedError
        rewards = self.calculate_path_reward(dict(observations=mesh_xs), latent)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        c = ax.pcolor(x, y, rewards.reshape([y.shape[0], y.shape[1]]))
        ax.plot(locs[:,0], locs[:,1], c='r')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Reward Heatmap")
        ax.set_aspect('equal')
        fig.colorbar(c, ax=ax)
        plt.savefig('/tmp/heatmap_traj/{}.png'.format(title))
        plt.close('all')


    def plot_resampling_heatmaps(self, trajectory_latent_lst, title, traj_infos=None):
        pass


    def plot_irl_heatmaps(self, trajectory_latent_lst, title, traj_infos=None):
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
            locs = trajectory_latent_lst[i][0]['observations']
            for j in range(2): # num_lats
                latent = trajectory_latent_lst[i][1][j]
                dx, dy = 0.005, 0.005
                y, x = np.mgrid[slice(-1, 1 + dy, dy),
                                slice(-1, 1 + dx, dx)]
                mesh_xs = np.stack([x, y], axis=2).reshape(-1, 2)
                rewards = self.calculate_path_reward(dict(observations=mesh_xs), latent)
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

    def plot_multiple_heatmaps(self, latents, paths, title, ncols=5, mesh_size=0.05, grid_rewards=None):
        print("plotting", title, "eval trajs")
        assert len(paths) % ncols == 0
        nrows = len(paths) // ncols
        fig, axs = plt.subplots(nrows, ncols, sharex='col', sharey='row',figsize=(8,8))
        for i in range(len(latents)):
            col, row = i // ncols, i % ncols
            latent = latents[i]
            locs = paths[i]['observations']
            dx, dy = mesh_size, mesh_size
            y, x = np.mgrid[slice(-1, 1 + dy, dy),
                            slice(-1, 1 + dx, dx)]
            mesh_xs = np.stack([x, y], axis=2).reshape(-1, 2)
            if grid_rewards is None:
                rewards = self.calculate_path_reward(dict(observations=mesh_xs), latent)
            else:
                rewards = grid_rewards[i]
            ax = axs[row, col]
            c = ax.pcolor(x, y, rewards.reshape([y.shape[0], y.shape[1]]), cmap='OrRd')
            ax.plot(locs[:, 0], locs[:, 1], c='r')
            ax.set_aspect('equal')
            ax.set_title("r: {:.2f}".format(self.get_discounted_path_reward(paths[i], latents[i])), size=8.0)
        fig.suptitle(title)
        exp_name = "adv" if self.q1 is not None else "reward"
        plt.savefig(osp.join(logger.get_snapshot_dir(), 'eval_{}_{}'.format(exp_name, title)))
        plt.close('all')
        print("done plotting", title, "eval trajs")

    def get_features(self, paths):
        return np.zeros([len(paths), 1])

class PointmassRelabelerEarlyStopping(PointMassBestRandomRelabeler):

    def reward_done(self, obs, action, latent, env_infos=None):
        _, d, a = latent
        if np.array_equal(self.latent, latent):
            rotation_matrix = self.rotation_matrix
        else:
            rotation_matrix = self.calculate_rotation_matrix(latent)
        obs_new = obs.dot(rotation_matrix.T)
        x, y = obs_new[0].flatten(), obs_new[1].flatten()
        alpha = x/d
        if alpha < 0:
            return -0.0, False
        else:
            return ((alpha+1e-12)**self.power * norm.pdf(y - a*np.sin(x*np.pi/d), scale=0.05))[0], (y**2 + (x-d)**2) < 0.1


    def calculate_path_reward(self, path, latent):
        angle, d, a = latent
        rotation_matrix = self.calculate_rotation_matrix(latent)

        obs = path['observations']
        obs_new = obs.dot(rotation_matrix.T)
        xs, ys = obs_new[:, 0].flatten(), obs_new[:, 1].flatten()
        alphas = xs/d
        sin_rewards = (np.abs(alphas) + 1e-12)**self.power * norm.pdf(ys - a*np.sin(xs*np.pi/d), scale=0.05)
        wrong_dir_rewards = np.zeros(len(alphas))
        rewards = np.where(alphas >= 0.0, sin_rewards.flatten(), wrong_dir_rewards.flatten())
        return rewards


if __name__ == "__main__":
    # obs = np.random.uniform(size=(5,2))
    # obs = np.concatenate([obs, np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])])
    # print(obs)
    relabeler = PointMassBestRandomRelabeler()
    # relabeler.make_reward_heatmap(np.array([np.pi/4, 1.4, -0.2]))
    # relabeler.make_reward_heatmap(np.array([np.pi/4*3, 1.4, -0.2]))
    # relabeler.make_reward_heatmap(np.array([np.pi*5/4, 1.4, -0.2]))
    # relabeler.make_reward_heatmap(np.array([np.pi*7/4, 1.4, -0.2]))
    # relabeler.make_reward_heatmap(np.array([0.7, 0.7, -0.2]))
    # relabeler.make_reward_heatmap(np.array([1, -1, -0.2]))
    import ipdb; ipdb.set_trace()
