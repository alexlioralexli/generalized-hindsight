import numpy as np
import matplotlib

matplotlib.use("Agg")

from rlkit.torch.multitask.ant_direction_relabeler import AntDirectionRelabelerNew
#Relabeler : AntDirectionNewRelabelerSparse:

class DIAYNAntDirectionRelabelerNewSparse(AntDirectionRelabelerNew):

    """
        Where are these methods used. 
        Then you can understand DIAYN reward fits.

    """
    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent 

    def sample_task(self):
        if self.is_eval:
            self.curr_idx = (self.curr_idx + 1) % len(self.eval_latents)
            return self.eval_latents[self.curr_idx].copy()
        else:
            return self.agent.skill_dist.sample()
      

    def reward_done(self, obs, action, latent, skill, next_obs, env_info=None):
        reward_run = self.agent.compute_lone_diversity_reward(skill, next_obs)
        return reward_run, False
    def calculate_best_path_reward(self, path, skill):
        rewards = self.agent.compute_diversity_reward(skill, path["next_observations"])
        return rewards


    def calculate_lone_path_reward(self, path, skill):
        #print(f"IN CALCULATE LONE PATH REWARD")
        #print(f"GIVEN SKILLS : {skill}")
        path_skill = path["skills"]
        #print(f"The path skill is: {path_skill}")
        reward = self.agent.compute_lone_diversity_reward(skill, path["next_observations"])
        return reward

    def calculate_path_reward(self, path, latent, currentSkill=False):
        skills = path["skills"]
        # print(f"THE LATENT IN CALC REWARD IS: {latent}, the one attached to skills is: {skills}")
        if not currentSkill:
            rewards = self.agent.compute_diversity_reward(path["skills"], path["next_observations"])
        else:
            # print(f"CURRENT LATENT IN CALC PATH REWARD: {latent}")
            rewards = self.agent.compute_diversity_reward(latent, path["next_observations"], True)
        return rewards

    def get_features(self, path, latent=None):
        return np.zeros([len(path), 1])

    def get_reward_matrix(self, paths, latents):
        return np.array([[self.get_discounted_path_reward(path, latent) for latent in latents] for path in paths])


