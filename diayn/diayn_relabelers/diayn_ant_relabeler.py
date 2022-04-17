import numpy as np
import matplotlib

matplotlib.use("Agg")

from rlkit.torch.multitask.gym_relabelers import ContinuousRelabeler, DIAYNRandomRelabeler

#Relabeler : AntDirectionNewRelabelerSparse:

class DIAYNAntDirectionRelabelerNew(DIAYNRandomRelabeler):

    def __init__(self, type='360', **kwargs):
        super().__init__(**kwargs)
        assert type in {'90', '180', '360'}
        self.type = type
        if self.is_eval:
            assert type == '360'
            self.eval_latents = np.linspace(-np.pi, np.pi, 25, endpoint=False) + np.pi / 25.0
            self.eval_latents = self.eval_latents.reshape(-1, 1)
            self.curr_idx = 0


    """

        DO WE sample task with DIAYN or the GHER?

        How should I get the sample tasks from DIAYN here?


    """
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
class DIAYNAntDirectionRelabelerNewSparse(DIAYNAntDirectionRelabelerNew):

    """
        Where are these methods used. 
        Then you can understand DIAYN reward fits.

    """
    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent 
        print(f"Agent received is : {self.agent}")
        # self.sample_task = None
        # self.diversity_reward = None


    def reward_done(self, obs, action, latent, skill, next_obs, env_info=None):


        #USE AGENT HERE NOT THE, DON'T PASS DIVERSITY      
        # theta = float(latent[0])
        # print(f"env: info {env_info['torso_velocity']}")
        # print(f"Env info, when sliced: {env_info['torso_velocity'][:2]} ")
        # speed = np.linalg.norm(env_info['torso_velocity'][:2])
        # cosine = (env_info['torso_velocity'][:2] / speed).dot(np.array([np.cos(theta), np.sin(theta)]))
        # reward_run = speed * (cosine > 0.9659).astype(np.float32) + env_info['reward_ctrl'] + env_info['reward_contact'] + 1
        reward_run = self.agent.compute_lone_diversity_reward(skill, next_obs)

        return reward_run, False
    def calculate_best_path_reward(self, path, skill):
     


        rewards = self.agent.compute_diversity_reward(skill, path["next_observations"])

        return rewards

    def calculate_path_reward(self, path, latent):
        # env_infos = path['env_infos']
        # ctrl_rewards = np.array([env_info['reward_ctrl'] for env_info in env_infos])
        # contact_rewards = np.array([env_info['reward_contact'] for env_info in env_infos])
        # theta = float(latent[0])
        # torso_velocities = np.array([env_info['torso_velocity'][:2] for env_info in env_infos])
        # speeds = np.linalg.norm(torso_velocities, axis=1, keepdims=True)
        # cosines = (torso_velocities / speeds).dot((np.array([np.cos(theta), np.sin(theta)])).reshape([-1, 1])).flatten()
        # rewards = speeds.flatten() * self.agent.compute_diversity_reward(skill, next_obs) + ctrl_rewards + contact_rewards + 1
        # print(f"Keys are : {path.keys()}")
        # print(f"The next_observations are: {path["next_observations"]}")
        # print(f"The shape of next_observations are: {path["next_observations"].shape}")
        # print(f"The type of path is : {type(path)}")
        # print(f"The skills are: {path["skills"]}")
        # print(f"The shape of skills are: {path["skills"].shape}")


        rewards = self.agent.compute_diversity_reward(path["skills"], path["next_observations"])

        return rewards

    def get_features(self, path, latent=None):
        return np.zeros([len(path), 1])

    def get_reward_matrix(self, paths, latents):
        return np.array([[self.get_discounted_path_reward(path, latent) for latent in latents] for path in paths])


