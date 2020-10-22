import numpy as np
import os.path as osp
from rlkit.torch.multitask.gym_relabelers import ContinuousRelabeler
from rlkit.torch.multitask.reference_latents import HALF_CHEETAH_HARD_LATENT

class HalfCheetahRelabeler(ContinuousRelabeler):
    def __init__(self, dim=1, low=0.0, high=np.pi/2.0, **kwargs):
        assert dim == 1
        super().__init__(dim, low, high, **kwargs)

    def reward_done(self, obs, action, latent, env_info=None):
        alpha = float(latent)
        return np.cos(alpha) * env_info['reward_run'] + np.sin(alpha) * env_info['reward_ctrl'], False

    def calculate_path_reward(self, path, latent):
        env_infos = path['env_infos']
        speed_rewards = np.array([env_info['reward_run'] for env_info in env_infos])
        action_rewards = np.array([env_info['reward_ctrl'] for env_info in env_infos])
        alpha = float(latent)
        return np.cos(alpha) * speed_rewards + np.sin(alpha) * action_rewards

    def get_features(self, path, latent=None):
        env_infos = path['env_infos']
        speed_rewards = np.array([env_info['reward_run'] for env_info in env_infos])
        action_rewards = np.array([env_info['reward_ctrl'] for env_info in env_infos])
        return np.array([self.get_discounted_reward(speed_rewards), self.get_discounted_reward(action_rewards)])

    def get_weights(self, latents): #todo: double check this
        latents = np.array(latents)
        return np.concatenate([np.cos(latents).reshape([1, -1]), np.sin(latents).reshape([1, -1])], axis=0)

    def get_reward_matrix(self, paths, latents):
        # |paths| rows, and |latents| columns
        features = self.get_features_matrix(paths)
        weights = self.get_weights(latents)
        result = features.dot(weights)
        return result


class HalfCheetahRelabelerMoreFeatures(ContinuousRelabeler):
    def __init__(self, dim=1, low=0.0, high=np.pi/2.0, **kwargs):
        super().__init__(dim, low, high, **kwargs)
        if self.is_eval:
            self.eval_latents = HALF_CHEETAH_HARD_LATENT.copy()
            self.curr_idx = 0

    def sample_task(self):
        if self.is_eval:
            self.curr_idx = (self.curr_idx + 1) % len(self.eval_latents)
            return self.eval_latents[self.curr_idx].copy()
        # sample gaussian, normalize to norm 1; first and last feature can be negative
        sample = np.random.multivariate_normal(np.zeros(4), np.diag(np.ones(4)))
        while np.linalg.norm(sample) < 0.0001:
            sample = np.random.multivariate_normal(np.zeros(4), np.diag(np.ones(4)))
        return np.array([sample[0], np.abs(sample[1]), np.abs(sample[2]), sample[3]]) / np.linalg.norm(sample)

    def reward_done(self, obs, action, latent, env_info=None):
        features = np.array([env_info['reward_run'], env_info['reward_ctrl'], env_info['height'], env_info['reward_angular']])
        return latent.dot(features), False

    def get_features(self, path, latent=None):
        env_infos = path['env_infos']
        speed_rewards = np.array([env_info['reward_run'] for env_info in env_infos])
        action_rewards = np.array([env_info['reward_ctrl'] for env_info in env_infos])
        height_rewards = np.array([env_info['height'] for env_info in env_infos])
        angular_rewards = np.array([env_info['reward_angular'] for env_info in env_infos])
        return np.array([self.get_discounted_reward(speed_rewards),
                         self.get_discounted_reward(action_rewards),
                         self.get_discounted_reward(height_rewards),
                         self.get_discounted_reward(angular_rewards)])

    def calculate_path_reward(self, path, latent):
        env_infos = path['env_infos']
        speed_rewards = np.array([env_info['reward_run'] for env_info in env_infos])
        action_rewards = np.array([env_info['reward_ctrl'] for env_info in env_infos])
        height_rewards = np.array([env_info['height'] for env_info in env_infos])
        angular_rewards = np.array([env_info['reward_angular'] for env_info in env_infos])
        return latent[0] * speed_rewards + latent[1] * action_rewards + latent[2] * height_rewards + latent[3] * angular_rewards

    def get_reward_matrix(self, paths, latents):
        # |paths| rows, and |latents| columns
        features = self.get_features_matrix(paths)
        weights = np.array(latents).T
        result = features.dot(weights)
        return result


class HalfCheetahDeterministicRelabeler(HalfCheetahRelabeler):
    def __init__(self, latents, path, **kwargs):
        self.latents = latents
        self.i = 0
        self.normalizing_constants = np.load(path)
        super().__init__(**kwargs)
        assert len(latents) == len(self.normalizing_constants)

    def sample_task(self):
        self.i = (self.i + 1) % len(self.latents)
        return self.latents[self.i % len(self.latents)]

    def reward_done(self, obs, action, latent, env_info=None):
        # get normalizing constant
        r, d = super().reward_done(obs, action, latent, env_info=env_info)
        return r/self.get_normalizing_constant(latent), d

    def calculate_path_reward(self, path, latent):
        return super().calculate_path_reward(path, latent) / self.get_normalizing_constant(latent)

    def get_normalizing_constant(self, latent):
        index = np.where(np.isclose(self.latents, latent))[0]
        return self.normalizing_constants[index]


class HalfCheetahVelocityRelabeler(ContinuousRelabeler):
    def __init__(self, dim=1, low=0.0, high=2.0, **kwargs):
        assert dim == 1
        super().__init__(dim, low, high, **kwargs)

    def reward_done(self, obs, action, latent, env_info=None):
        alpha = float(latent)
        return - (env_info['reward_run'] - alpha) ** 2 + env_info['reward_ctrl'], False

    def calculate_path_reward(self, path, latent):
        env_infos = path['env_infos']
        velocities = np.array([env_info['reward_run'] for env_info in env_infos])
        action_rewards = np.array([env_info['reward_ctrl'] for env_info in env_infos])
        alpha = float(latent)
        return - (velocities - alpha) ** 2 + action_rewards

    def get_features(self, path, latent=None):
        return np.zeros(1)

    def get_reward_matrix(self, paths, latents):
        return np.array([[self.get_discounted_path_reward(path, latent) for latent in latents] for path in paths])

class HalfCheetahVelocityRelabelerSparse(HalfCheetahVelocityRelabeler):
    def __init__(self, dim=1, low=-2.0, high=2.0, **kwargs):
        assert dim == 1
        super().__init__(dim, low, high, **kwargs)

    def reward_done(self, obs, action, latent, env_info=None):
        alpha = float(latent)
        return (np.abs(env_info['reward_run'] - alpha) < 0.125).astype(np.float32) + env_info['reward_ctrl'], False

    def calculate_path_reward(self, path, latent):
        env_infos = path['env_infos']
        velocities = np.array([env_info['reward_run'] for env_info in env_infos])
        action_rewards = np.array([env_info['reward_ctrl'] for env_info in env_infos])
        alpha = float(latent)
        rewards = (np.abs(velocities - alpha) < 0.125).astype(np.float32) + action_rewards
        return rewards


if __name__ == '__main__':
    # for half cheetah with more features
    relabeler = HalfCheetahRelabelerMoreFeatures()
    latents = [relabeler.sample_task() for _ in range(25)]

    import ipdb; ipdb.set_trace(context=10)