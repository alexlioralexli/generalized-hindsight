import numpy as np
import matplotlib
matplotlib.use("Agg")

from rlkit.torch.multitask.gym_relabelers import ContinuousRelabeler

class PendulumRelabeler(ContinuousRelabeler):  #todo: flip all the sin and cosines here
    def sample_task(self):
        return np.random.uniform(low=[0], high=[np.pi/2], size=(1,))

    def reward_done(self, obs, action, latent, env_info=None):
        alpha = float(latent[0])
        reward = np.cos(alpha) * (0.90 - env_info['theta'] - env_info['thdot']) + \
                 np.sin(alpha) * (0.80 - 500 * env_info['u'])
        return reward, False

    def calculate_path_reward(self, path, latent):
        env_infos = path['env_infos']
        theta_costs = np.array([env_info['theta'] for env_info in env_infos])
        thdot_costs = np.array([env_info['thdot'] for env_info in env_infos])
        u_costs = np.array([env_info['u'] for env_info in env_infos])
        alpha = float(latent[0])
        return np.cos(alpha) * (0.90 - theta_costs - thdot_costs) + np.sin(alpha) * (0.80 - 500*u_costs)

    def get_features(self, path, latent=None):  #todo: not sure about where to put the thdot costs, also was bug where i did the offset wrong
        env_infos = path['env_infos']
        theta_costs = np.array([env_info['theta'] for env_info in env_infos])
        thdot_costs = np.array([env_info['thdot'] for env_info in env_infos])
        u_costs = np.array([env_info['u'] for env_info in env_infos])
        return np.array([self.get_discounted_reward(0.9 - theta_costs) - self.get_discounted_reward(thdot_costs),
                         self.get_discounted_reward(0.8 - 500 * u_costs)])

    def get_weights(self, latents):  # todo: double check this
        latents = np.array(latents)
        return np.array([np.cos(latents[:,0]), np.sin(latents[:,0])])

    def get_reward_matrix(self, paths, latents):
        # |paths| rows, and |latents| columns
        features = self.get_features_matrix(paths)
        weights = self.get_weights(latents)
        result = features.dot(weights)
        return result
        # return np.array([[self.get_discounted_path_reward(path, latent) for latent in latents] for path in paths])


class NewPendulumRelabeler(ContinuousRelabeler):  #todo: flip all the sin and cosines here
    def sample_task(self):
        return np.random.uniform(low=[0], high=[1], size=(1,))

    def reward_done(self, obs, action, latent, env_info=None):
        alpha = float(latent[0])
        reward = - env_info['theta'] - env_info['thdot'] - (500 + alpha * 4500) * env_info['u']
        return reward, False

    def calculate_path_reward(self, path, latent):
        env_infos = path['env_infos']
        theta_costs = np.array([env_info['theta'] for env_info in env_infos])
        thdot_costs = np.array([env_info['thdot'] for env_info in env_infos])
        u_costs = np.array([env_info['u'] for env_info in env_infos])
        alpha = float(latent[0])
        return - theta_costs - thdot_costs - (500 + alpha * 4500) * u_costs

    def get_features(self, path, latent=None):  #todo: not sure about where to put the thdot costs, also was bug where i did the offset wrong
        env_infos = path['env_infos']
        theta_costs = np.array([env_info['theta'] for env_info in env_infos])
        thdot_costs = np.array([env_info['thdot'] for env_info in env_infos])
        u_costs = np.array([env_info['u'] for env_info in env_infos])
        return np.array([-self.get_discounted_reward(theta_costs) - self.get_discounted_reward(thdot_costs),
                         self.get_discounted_reward(u_costs)])

    def get_weights(self, latents):  # todo: double check this
        latents = np.array(latents)
        return np.array([np.ones_like(latents[:,0]), 500 + latents[:,0] * 4500])

    def get_reward_matrix(self, paths, latents):
        # |paths| rows, and |latents| columns
        features = self.get_features_matrix(paths)
        weights = self.get_weights(latents)
        result = features.dot(weights)
        return result
        # return np.array([[self.get_discounted_path_reward(path, latent) for latent in latents] for path in paths])