import numpy as np
import matplotlib

matplotlib.use("Agg")

from rlkit.torch.multitask.gym_relabelers import ContinuousRelabeler


class AntDirectionRelabeler(ContinuousRelabeler):  # todo: flip all the sin and cosines here

    def sample_task(self):
        return np.random.uniform(low=[0.0, -np.pi], high=[np.pi / 2, np.pi / 2],
                                 size=2)  # todo: is the second thing accurate? why is it pi/2

    def reward_done(self, obs, action, latent, env_info=None):
        theta = float(latent[0])
        alpha = float(latent[1])
        # return np.cos(alpha) * env_info['reward_run'] + np.sin(alpha) * (1 + env_info['reward_ctrl']), False
        reward_run = env_info['torso_velocity'][0] * np.cos(theta) + env_info['torso_velocity'][1] * np.sin(theta)
        return np.sin(alpha) * reward_run + np.cos(alpha) * (env_info['reward_ctrl']), False

    def calculate_path_reward(self, path, latent):
        env_infos = path['env_infos']
        # done_rewards = np.array([env_info['reward_run'] for env_info in env_infos])
        action_rewards = np.array([env_info['reward_ctrl'] for env_info in env_infos])
        theta = float(latent[0])
        alpha = float(latent[1])

        torso_velocities = np.array([env_info['torso_velocity'][:2] for env_info in env_infos])
        done_rewards = torso_velocities[:, 0] * np.cos(theta) + torso_velocities[:, 1] * np.sin(theta)

        # return np.cos(alpha) * done_rewards + np.sin(alpha) * action_rewards
        return np.sin(alpha) * done_rewards + np.cos(alpha) * action_rewards

    def get_features(self, path, latent=None):
        env_infos = path['env_infos']
        action_rewards = np.array([env_info['reward_ctrl'] for env_info in env_infos])
        theta = float(latent[0])
        torso_velocities = np.array([env_info['torso_velocity'][:2] for env_info in env_infos])
        done_rewards = torso_velocities[:, 0] * np.cos(theta) + torso_velocities[:, 1] * np.sin(theta)
        return np.array([self.get_discounted_reward(done_rewards), self.get_discounted_reward(action_rewards)])

    def get_weights(self, latents):
        latents = np.array(latents)
        # return np.concatenate([np.cos(latents).reshape([1, -1]), np.sin(latents).reshape([1, -1])], axis=0)
        return np.concatenate([np.sin(latents).reshape([1, -1]), np.cos(latents).reshape([1, -1])], axis=0)

    def get_reward_matrix(self, paths, latents):
        # |paths| rows, and |latents| columns
        # features = self.get_features_matrix(paths)
        # weights = self.get_weights(latents)
        # result = features.dot(weights)
        # return result
        return np.array([[self.get_discounted_path_reward(path, latent) for latent in latents] for path in paths])


# no energy
class AntDirectionRelabelerNoEnergy(ContinuousRelabeler):
    def sample_task(self):
        return np.random.uniform(low=[-np.pi], high=[np.pi], size=(1,))

    def reward_done(self, obs, action, latent, env_info=None):
        theta = float(latent[0])
        # return np.cos(alpha) * env_info['reward_run'] + np.sin(alpha) * (1 + env_info['reward_ctrl']), False
        reward_run = env_info['torso_velocity'][0] * np.cos(theta) + env_info['torso_velocity'][1] * np.sin(theta)
        return reward_run, False

    def calculate_path_reward(self, path, latent):
        env_infos = path['env_infos']
        # done_rewards = np.array([env_info['reward_run'] for env_info in env_infos])
        # action_rewards = np.array([env_info['reward_ctrl'] for env_info in env_infos])
        theta = float(latent[0])
        # alpha = float(latent[1])

        torso_velocities = np.array([env_info['torso_velocity'][:2] for env_info in env_infos])
        done_rewards = torso_velocities[:, 0] * np.cos(theta) + torso_velocities[:, 1] * np.sin(theta)

        # return np.cos(alpha) * done_rewards + np.sin(alpha) * action_rewards

        return done_rewards

    def get_features(self, path, latent=None):
        env_infos = path['env_infos']
        torso_vel = np.array([env_info['torso_velocity'][:2] for env_info in env_infos])
        return np.array([self.get_discounted_reward(torso_vel[:, 0]), self.get_discounted_reward(torso_vel[:, 1])])

    def get_weights(self, latents):
        return np.array([np.cos(latents[:, 0]), np.sin(latents[:, 0])])

    def get_reward_matrix(self, paths, latents):
        # |paths| rows, and |latents| columns
        features = self.get_features_matrix(paths)
        weights = self.get_weights(latents)
        result = features.dot(weights)
        return result
        # return np.array([[self.get_discounted_path_reward(path, latent) for latent in latents] for path in paths])


# latent controls direction, but contact and energy terms still there
class AntDirectionRelabelerNew(ContinuousRelabeler):

    def __init__(self, type='360', **kwargs):
        super().__init__(**kwargs)
        assert type in {'90', '180', '360'}
        self.type = type
        if self.is_eval:
            assert type == '360'
            self.eval_latents = np.linspace(-np.pi, np.pi, 25, endpoint=False) + np.pi / 25.0
            self.eval_latents = self.eval_latents.reshape(-1, 1)
            self.curr_idx = 0

    def sample_task(self):
        if self.is_eval:
            self.curr_idx = (self.curr_idx + 1) % len(self.eval_latents)
            return self.eval_latents[self.curr_idx].copy()
        if self.type == '90':
            return np.random.uniform(low=[-np.pi / 4.0], high=[np.pi / 4.0], size=(1,))
        elif self.type == '180':
            return np.random.uniform(low=[-np.pi / 2.0], high=[np.pi / 2.0], size=(1,))
        elif self.type == '360':
            return np.random.uniform(low=[-np.pi], high=[np.pi], size=(1,))
        else:
            raise RuntimeError

    def reward_done(self, obs, action, latent, env_info=None):
        theta = float(latent[0])
        # return np.cos(alpha) * env_info['reward_run'] + np.sin(alpha) * (1 + env_info['reward_ctrl']), False
        reward_run = env_info['torso_velocity'][0] * np.cos(theta) + env_info['torso_velocity'][1] * np.sin(theta) \
                     + env_info['reward_ctrl'] + env_info['reward_contact'] + 1
        return reward_run, False

    def calculate_path_reward(self, path, latent):
        env_infos = path['env_infos']
        ctrl_rewards = np.array([env_info['reward_ctrl'] for env_info in env_infos])
        contact_rewards = np.array([env_info['reward_contact'] for env_info in env_infos])
        theta = float(latent[0])
        torso_velocities = np.array([env_info['torso_velocity'][:2] for env_info in env_infos])
        rewards = torso_velocities[:, 0] * np.cos(theta) + torso_velocities[:, 1] * np.sin(theta) \
                  + ctrl_rewards + contact_rewards + 1
        return rewards

    def get_features(self, path, latent=None):
        env_infos = path['env_infos']
        torso_vel = np.array([env_info['torso_velocity'][:2] for env_info in env_infos])
        ctrl_rewards = np.array([env_info['reward_ctrl'] for env_info in env_infos])
        contact_rewards = np.array([env_info['reward_contact'] for env_info in env_infos])
        return np.array([self.get_discounted_reward(torso_vel[:, 0]),
                         self.get_discounted_reward(torso_vel[:, 1]),
                         self.get_discounted_reward(ctrl_rewards + contact_rewards + 1)])

    def get_weights(self, latents):
        latents = np.array(latents)
        return np.array([np.cos(latents[:, 0]), np.sin(latents[:, 0]), np.ones(len(latents))])

    def get_reward_matrix(self, paths, latents):
        # |paths| rows, and |latents| columns
        features = self.get_features_matrix(paths)
        weights = self.get_weights(latents)
        result = features.dot(weights)
        return result
        # return np.array([[self.get_discounted_path_reward(path, latent) for latent in latents] for path in paths])

    def to_save_video(self, epoch):
        """
        :return: boolean whether to save rgb_video for the epoch
        """
        if epoch < 10:
            return True
        else:
            return epoch % 10 == 0

class AntDirectionRelabelerNewSquared(AntDirectionRelabelerNew):
    def reward_done(self, obs, action, latent, env_info=None):
        theta = float(latent[0])

        speed = np.linalg.norm(env_info['torso_velocity'][:2])
        cosine = (env_info['torso_velocity'][:2] / speed).dot(np.array([np.cos(theta), np.sin(theta)]))
        reward_run = speed * (max(0, cosine) ** 2) + env_info['reward_ctrl'] + env_info['reward_contact'] + 1
        return reward_run, False

    def calculate_path_reward(self, path, latent):
        env_infos = path['env_infos']
        ctrl_rewards = np.array([env_info['reward_ctrl'] for env_info in env_infos])
        contact_rewards = np.array([env_info['reward_contact'] for env_info in env_infos])
        theta = float(latent[0])
        torso_velocities = np.array([env_info['torso_velocity'][:2] for env_info in env_infos])
        speeds = np.linalg.norm(torso_velocities, axis=1, keepdims=True)
        cosines = (torso_velocities / speeds).dot((np.array([np.cos(theta), np.sin(theta)])).reshape([-1,1])).flatten()
        cosines[cosines < 0] = 0
        rewards = speeds.flatten() * (cosines ** 2) + ctrl_rewards + contact_rewards + 1
        return rewards

    def get_features(self, path, latent=None):
        return np.zeros([len(path), 1])

    def get_reward_matrix(self, paths, latents):
        return np.array([[self.get_discounted_path_reward(path, latent) for latent in latents] for path in paths])

class AntDirectionRelabelerNewSparse(AntDirectionRelabelerNew):
    def reward_done(self, obs, action, latent, env_info=None):
        theta = float(latent[0])
        speed = np.linalg.norm(env_info['torso_velocity'][:2])
        cosine = (env_info['torso_velocity'][:2] / speed).dot(np.array([np.cos(theta), np.sin(theta)]))
        reward_run = speed * (cosine > 0.9659).astype(np.float32) + env_info['reward_ctrl'] + env_info['reward_contact'] + 1
        return reward_run, False

    def calculate_path_reward(self, path, latent):
        env_infos = path['env_infos']
        ctrl_rewards = np.array([env_info['reward_ctrl'] for env_info in env_infos])
        contact_rewards = np.array([env_info['reward_contact'] for env_info in env_infos])
        theta = float(latent[0])
        torso_velocities = np.array([env_info['torso_velocity'][:2] for env_info in env_infos])
        speeds = np.linalg.norm(torso_velocities, axis=1, keepdims=True)
        cosines = (torso_velocities / speeds).dot((np.array([np.cos(theta), np.sin(theta)])).reshape([-1, 1])).flatten()
        rewards = speeds.flatten() * (cosines > 0.9659).astype(np.float32) + ctrl_rewards + contact_rewards + 1
        return rewards

    def get_features(self, path, latent=None):
        return np.zeros([len(path), 1])

    def get_reward_matrix(self, paths, latents):
        return np.array([[self.get_discounted_path_reward(path, latent) for latent in latents] for path in paths])


class DiscretizedAntDirectionRelabelerNoEnergy(AntDirectionRelabelerNoEnergy):
    def __init__(self, index, num_bins=30, **kwargs):
        low, high = -np.pi / 2, np.pi / 2
        self.latent = np.array([low + (high - low) * index / num_bins])
        super().__init__(**kwargs)

    def sample_task(self):
        return self.latent.copy()

class SingleLatentAntDirectionRelabelerNew(AntDirectionRelabelerNew):
    def sample_task(self):
        return np.array([0.0])

# for debugging the discrepancy between sac + gym ant and sac_gher + our ant
class SingleLatentAntDirectionRelabelerNew(AntDirectionRelabelerNew):
    def sample_task(self):
        return np.array([0.0])

class AntDirectionRelabelerRestricted(AntDirectionRelabelerNew):
    def sample_task(self):
        return np.random.uniform(low=[-np.pi/4.0], high=[np.pi/4.0], size=(1,))

if __name__ == '__main__':
    x = np.linspace(-np.pi, np.pi, 25, endpoint=False) + np.pi / 25.0
    import ipdb; ipdb.set_trace(context=10)
