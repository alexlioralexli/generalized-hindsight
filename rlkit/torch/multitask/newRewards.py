
class DIAYNRelabeler(object):
    def __init__(self,
                 discount=0.99,
                 relabel=True,
                 use_adv=False,
                 cache=False,
                 subtract_final_value=False,
                 q1=None,
                 q2=None,
                 agent=None,
                 vf=None,
                 action_fn=None,
                 test=False,
                 sliding_normalization=False,
                 is_eval=False,
                 ):
        self.discount = discount
        self.relabel = relabel
        self.use_adv = use_adv
        self.q1 = q1
        self.q2 = q2
        self.vf = vf
        self.action_fn = action_fn
        self.test = test
        self.subtract_final_value = subtract_final_value
        self.sliding_normalization = sliding_normalization

        self.agent = agent

        if self.use_adv:
            assert (q1 is not None and q2 is not None and action_fn is not None) or vf is not None
            self.use_vf_baseline = (vf is not None)
        self.cache = cache
        if self.cache:
            self.cached_paths = []
        self.is_eval = is_eval


    def reward_done(self, obs, action, latent, env_infos=None):
        raise NotImplementedError

    def train(self):
        pass

    def sample_task(self):
        raise NotImplementedError

    def calculate_reward(self, path, latent):
        raise NotImplementedError

    def calculate_path_reward(self, path, latent):
        raise NotImplementedError

    def get_discounted_reward(self, rewards):

        # NEED TO CHANGE FOR THE DIVERSITY REWARDS. 
        #assert len(rewards.shape) == 1

        # print("ORIGINAL VALUES :")
        # print(f"rewards are: {rewards}, rewards shape is: {rewards.shape}, shape of len: {len(rewards)}")
        if not (isinstance(rewards, np.ndarray)):
            rewards = rewards.cpu().detach().numpy()
            rewards = np.asarray(rewards)
            self.discount = 0.99
         
        # print(f"length of rewards is: {len(rewards)}")
        # print(f"type of rewards is: {type(rewards)}")
        # print(f"Type of length: {type(rewards)}, length is : {reward_length}")
        reward_list = np.arange(len(rewards))
        # print(f"Type Study: Type of self.discount : {type(self.discount)}, Type of reward_list: {type(reward_list)}, Type of data type: {reward_list.dtype}")
        multipliers = np.power(self.discount, np.arange(len(rewards)))
        result = np.sum(rewards * multipliers)



        # VALUE STUDY -> COMPARE IT IN THE ORIGINAL GHER + DIAYN 
        return np.sum(rewards * multipliers)

    def get_discounted_path_reward(self, path, latent):
        #print(f"I AM IN THE CURRENT GET DISCOUNTED REWARD: ")
        latent_path = path["skills"]
        #print(f"The path is: {latent_path}, the latent is: {latent}")
        
        path_rewards = self.calculate_path_reward(path, latent, True)
        #print(f"The path rewards is : {path_rewards}")
        return self.get_discounted_reward(path_rewards)

    # use this if advantage_fn is provided
    def get_baseline_estimate(self, obs, latent):
        obs, latent = ptu.from_numpy(obs).unsqueeze(0), ptu.from_numpy(latent).unsqueeze(0)

        if self.use_vf_baseline:
            estimate = ptu.get_numpy(self.vf(obs, latent))
        else:
            actions = self.action_fn(obs, latent, deterministic=True)[0]
            estimate = ptu.get_numpy(torch.min(self.q1(obs, actions, latent), self.q2(obs, actions, latent)))
        return estimate

    def get_baseline_estimates(self, obs, skills):
        v1, v2 = self.get_both_values(obs, skills)
        return np.minimum(v1, v2)

    def get_both_values(self, obs, skills):
        """
            SKILLS NEEDS TO BE A 2D TENSOR
            OBS NEEDS TO BE A REPEAT, FOR THE SKILLS
            ACTIONS need to be a result of the multiple skills for the same OBS.


        """
        device = torch.device("cuda")

        # print(f"Skills received is : {skills}, its shape is: {skills.shape}")

        #CONVERTING SKILLS TO PROPER SHAPE AND TENSOR. 
        skill_array = []
        for index in range(skills.shape[0]):
            skill_array.append(skills[index].tolist())

        skill_tensor = torch.DoubleTensor(skill_array).to(device)
        #print(f"Skill tensor is : {skill_tensor}, the shape is : {skill_tensor.shape}")
        obs_tensor = torch.from_numpy(obs).type(torch.DoubleTensor).unsqueeze(0).repeat(len(skills), 1).to(device)
        #print(f"obs_tensor obs_tensor is : {obs_tensor}, the shape is : {obs_tensor.shape}")

        action_list = []
        for index in range(len(skills)):
            action_np = self.agent.act(obs, skills[index], sample=True)
            action_pyList = action_np.tolist()
            action_list.append(action_pyList)
        action_tensor = torch.DoubleTensor(action_list).to(device)
        #print(f"action_tensor is : {action_tensor}, the shape is : {action_tensor.shape}")





        # print(f"Type of skills is : {type(skills)}, skills is: {skills.shape}, {skills}")
        # new_skills = []
        # for skill in skills:
        #    new_skill = skill.cpu().detach().numpy()
        #    new_skills.append(new_skill)

        # print(f"new skills: {new_skills}")

        # print(f"Obs first value is {obs}, then the skills value is : {skills[0]}")


        
        # obs = ptu.from_numpy(obs).unsqueeze(0).repeat(len(skills), 1)

        # print(f"OBS from np repeat is: {obs}")
        #print(f"Agent in get_both_values : {self.agent}")
        # action_list = []
        # for index in range(len(skills)):
        #     action_list.append(utils.to_np(self.agent.act(obs, skills[index], sample=True)))

        # action_list = np.asarray(action_list)

        # print(f"type of action list: {action_list}")
        
        # action_torch = torch.from_numpy(action_list).type(torch.DoubleTensor).to(device)
        # print(f"Actions from action_fn are as follows: {actions}")
        # obs_torch = torch.from_numpy(obs).type(torch.DoubleTensor).unsqueeze(0).repeat(len(skills), 1).to(device)
        # print(f"Actions from action_fn are as follows: {actions}")
        # print(f"Type of the actions is: {type(actions)}")
        #obs_torch = torch.from_numpy(obs).type(torch.DoubleTensor).to(device)
        # actions_torch = torch.from_numpy(actions_list).type(torch.DoubleTensor).to(device)
        
        
        #skills_torch = torch.tensor(skills).type(torch.DoubleTensor).to(device)
        # print(f"dtype for obs_torch :{obs_torch.dtype}")
        # print(f"dtype for actions_torch :{type(actions_list_torch)}")
        # print(f"dtype for skills_torch :{skills.dtype}")
        # print(f"Skills torch : {skills[0]}")
        # print(f"Actions torch: {actions_list_torch}")
        # print(f"Obs torch : {obs_torch}")

        obs_action_skill = torch.cat([obs_tensor, action_tensor, skill_tensor], dim=-1).float().to(device)
        # obs_action_skill = obs_action_skill.t()
        
        #print(f"OBSACTIONSKILL INFO is: {obs_action_skill}, its shape is : {obs_action_skill.shape}, its type is: {type(obs_action_skill)}, its dtype is : {obs_action_skill.dtype}")


        # return ptu.get_numpy(self.agent.critic.Q1(obs_action_skill)), ptu.get_numpy(self.agent.critic.Q2(obs_action_skill))
        return ptu.get_numpy(self.agent.critic.Q1(obs_action_skill)), ptu.get_numpy(self.agent.critic.Q2(obs_action_skill))
    def get_latents_and_rewards(self, path):
        raise NotImplementedError

    def approx_irl_relabeling(self, paths):
        raise NotImplementedError

class DIAYNRandomRelabeler(DIAYNRelabeler):

    def __init__(self, n_sampled_latents=5, n_to_take=1, do_cem=False, cem_itrs=3, n_cem_elites=5, **kwargs):
        super().__init__(**kwargs)
        self.n_sampled_latents = n_sampled_latents
        self.n_to_take = n_to_take
        self.do_cem = do_cem
        self.cem_itrs = cem_itrs
        self.n_cem_elites = n_cem_elites
        

    def get_latents_and_rewards(self, path):
        if not self.relabel:
            return [], [], []
        if self.n_sampled_latents == 1:
            latents = [self.agent.skill_dist.sample()]
        else:
            latents = [self.agent.skill_dist.sample() for _ in range(self.n_sampled_latents - 1)]
            latents.append(path['skills'][0])
        rewards = [self.calculate_path_reward(path, latent) for latent in latents]
        
        
        """
            SELF ADVANTAGES.

        """
        
        
        if self.use_adv:  # calculate advantages
            baselines = self.get_baseline_estimates(path['observations'][0], np.array(latents)).flatten()
            if self.subtract_final_value:
                final_baselines = self.get_baseline_estimates(path['next_observations'][-1], np.array(latents)).flatten() * self.discount**len(path['observations'])
            else:
                final_baselines = np.zeros(len(latents))
            trios = [(self.get_discounted_reward(reward) - baseline + final_baseline, reward, latent)
                     for baseline, final_baseline, reward, latent in zip(baselines, final_baselines, rewards, latents)]
        else:
            trios = [(self.get_discounted_reward(reward), reward, latent) for reward, latent in zip(rewards, latents)]
        # ValueError if just sorted(), since default is to tiebreak on the second element of the tuple
        trios = list(reversed(sorted(trios, key=lambda x: x[0])))
        return [trios[i][2] for i in range(self.n_to_take)], [trios[i][1] for i in range(self.n_to_take)], [trio[2] for trio in trios]



    """
            NEED TO COMPLETE THIS FUNCTION

    """
    def normalize_path_returns(self, paths, use_grid=False):
        assert self.relabel
        if self.n_sampled_latents == 1:
            latents = [self.agent.skill_dist.sample()]
        else:
            latents = [self.agent.skill_dist.sample() for _ in range(self.n_sampled_latents - len(paths))]
        for path in paths:
            latents.append(path['latents'][0])
        reward_means_for_latents = self.get_reward_matrix(paths, latents).T
        if self.test:
            print("dividing by 1")
            means = np.ones(len(latents))
        else:
            means = np.mean(reward_means_for_latents, axis=1)
        normalized_rewards = reward_means_for_latents / (np.abs(means).reshape([-1, 1]) + 1e-6)
        indices = np.argmax(normalized_rewards, axis=0)
        best_latents = [latents[idx] for idx in indices]
        rewards = [self.calculate_path_reward(path, latent, True) for path, latent in zip(paths, best_latents)]
        return [[z] for z in best_latents], [[r] for r in rewards]


    def plot_normalize_hist(self, grid_means, traj_means):
        # want to plot epoch by epoch, and over all the trajectories
        pass

    def get_grid_reward_mean(self, latent, interval=0.05):
        dx, dy = interval, interval
        y, x = np.mgrid[slice(-1, 1 + dy, dy),
                        slice(-1, 1 + dx, dx)]
        mesh_xs = np.stack([x, y], axis=2).reshape(-1, 2)
        return np.mean(self.calculate_path_reward(dict(observations=mesh_xs), latent))

    def plot_distribution(self, rewards):
        # histogram of average rewards
        pass

    def dist_values(self, fitness):
        pass 
    def calculate_fitness(self, skills, path):
        pass 
    def skill_distribution(self, mu, variance):
        """
              if self.skill_type == 'discrete':
            # If the skill type is discrete, the shape of the skill gives us 
            # the number of different skills
            self.skill_dist = torch.distributions.OneHotCategorical(
                probs=torch.ones(self.skill_dim).to(self.device))
        else:
            # The skills are a contunious hypercube where every axis is 
            # between zero and one
            self.skill_dist = torch.distributions.Uniform(low=torch.zeros(self.skill_dim).to(self.device), 
                                                          high=torch.ones(self.skill_dim).to(self.device))
        self.discriminator_update_frequency = discriminator_update_frequency

        """

        pass 


    def cem_relabeler(self, paths, rho=0.1, k=5,n=100):
        """
            *NOTES:

            Rho -> population percentage
            n -> sample size
            k -> iterations

            PSEUDO ALGORITHM :
            INPUT = paths, rho, n, k
            mean, variance
            run for k generations:
                skills <- generate_skills from mu, rho
                fitness <- 
                    use next_obs from ORIGINAL path
                    USE ORIGINAL ACTION SKILL ALWAYS 
                    find REWARD with SKILL, OBS, ACTION
                reduce (from 100 by Rho)
                Adjust mu, variance from reduce (rho)

            
            #ADD SINGLE SAMPLE ORIGINAL SKILL AND BEST SKILL FROM original
            return single path with best skill + ORIGINAL SKILL, (ADD SINGLE PATH) max(from rho)

        """
        
        mu = 0
        variance = 100 # * I Identity Matrix 

        # Random Sampling uses mu and variance
        """
            100

        """
        for _ in range(k):
            original_skill = path[0]["skill"]
            new_skills = [self.skill_distribution(mu, variance) for _ in n-1]
            fitness = calculate_fitness(new_skills, path)
            mu, variance = dist_values(fitness)


            

            pass 
        pass 
    def approx_irl_relabeling(self, paths):
        device = torch.device("cuda")
        assert self.relabel
        """

            ALGORITHM:

            1. Given Paths, with skills, obs, next_obs, and other information
            2. Sample more skills from the same distribution
            3. Get Reward Matrix -> with more skills, REWARD MATRIX is IN: DIAYNAntRelabeler
                    features = self.get_features_matrix(paths) -> GET FEATURES MATRIX -> INSIDE THE RELABELER SPARSE GET REWARD MATRIX -> GET DISCOUNTED REWARDS -> GET DISCOUNT REWARD
                    weights = self.get_weights(latents)
                    result = features.dot(weights)


        """

        if self.n_sampled_latents == 1:
            skills = self.agent.skill_dist.sample()
           #skills = [utils.to_np(self.agent.skill_dist.sample())]


        else:
            skills = [self.agent.skill_dist.sample() for _ in range(self.n_sampled_latents - len(paths))]


        # print(f"Skills received is :{skills}")
            #skills = [utils.to_np(self.agent.skill_dist.sample()) for _ in range(self.n_sampled_latents - len(paths))]
        # print(f"SKILLS from sample: {self.agent.skill_dist.sample()}")
        # util_skill = utils.to_np(self.agent.skill_dist.sample())
        # print(f"SKILLS USING UTILTON: {util_skill}")
        # print(f"SKILLS FROM TENSOR AS WELL {torch.as_tensor(util_skill, device=device).float()}")
        # print(f"SKILLS AFTER TEDETACH IS FROM SAMPLE IS: {skills}")

        for path in paths:
            skills.append(path['skills'][0])


        # latents = [np.concatenate([latent, np.array([np.pi, 0.25])]) for latent in x]
        # form matrix with |paths| rows x |latents| cols
        if self.cache:
            #new_paths = remove_extra_trajectory_info(paths)
            self.cached_paths = (paths + self.cached_paths)[:500]
            reward_matrix = self.get_reward_matrix(self.cached_paths, skills)
        else:
            reward_matrix = self.get_reward_matrix(paths, skills)
        '''
        array([[ 6,  0,  3],
                [14,  7, 12]])
        >>> y = x.argsort()
        array([[1, 2, 0],
           [1, 2, 0]])
        '''
        best_skills = []

        """
            NEED TO CHANGE CALCULATE PATH TRAJECTORY, TO SINGULAR, SO THAT IT USES THE CORRECT SKILL
            FOR REWARD CALCULATION, NOT THE ONE ASSOCIATED WITH PATH BUT WITH BEST SKILL

        """

        # strategy with percentiles
        temp = reward_matrix.T.argsort()
        ranks = np.empty_like(temp)
        for i in range(len(temp)):
            ranks[i, temp[i]] = np.arange(len(temp[i]))
        ranks = ranks.T
        if self.n_to_take == 1:
            for i, path in enumerate(paths):
                # best_latent_index = np.argmax(ranks[i])
                winners = np.argwhere(ranks[i] == np.amax(ranks[i]))
                if len(winners) == 1:
                    best_skill_index = winners[0]
                    best_skills.append(skills[int(best_skill_index)])
                else:
                    winnner_traj_rewards = reward_matrix[i, winners]


                    """
                        USE SELF ADVANTAGES IN RELABELING.

                    """
                    if self.use_adv:  # break ties by traj advantage
                        """
                            IN NORMAL GHER CODE, the getbaseline estimates uses 
                            latent vector and obs tensors, both are 2d 
                            to calculate action function and value functions

                            Our skill can only take in 1D vectors for skill and 
                            obs

                        """
                        baselines = self.get_baseline_estimates(path['observations'][0], np.array([skills[int(winner)] for winner in winners])).flatten()
                        advantages = winnner_traj_rewards.flatten() - baselines
                        best_skill_index = winners[np.argmax(advantages)]
                    else:  # break ties by traj reward
                        best_skill_index = winners[np.argmax(winnner_traj_rewards)]  # break ties by traj reward

                    best_skills.append(skills[int(best_skill_index)])
            return [[z] for z in best_skills], \
                   [[self.calculate_path_reward(path, skill, True)] for path, skill in zip(paths, best_skills)]
        else:
            sorted_indices = ranks.argsort(axis=1)  # goes from low to high
            for i, path in enumerate(paths):
                num_needed = self.n_to_take
                # strategy:
                n_taken = self.n_to_take
                nth_largest = ranks[i, sorted_indices[i, -n_taken]]
                while n_taken > 0 and ranks[i, sorted_indices[i, -n_taken]] == nth_largest:
                    n_taken -= 1
                # cases
                if n_taken == 0:  # case 1 or 2, group of maxes has size >= n_to_take
                    winners = np.argwhere(ranks[i] == np.amax(ranks[i]))
                    winnner_traj_rewards = reward_matrix[i, winners]
                    if self.use_adv:  # break ties by traj advantage
                        baselines = self.get_baseline_estimates(path['observations'][0], np.array([skills[int(winner)] for winner in winners])).flatten()
                        advantages = winnner_traj_rewards.flatten() - baselines
                        scores = advantages
                    else:  # break ties by traj reward
                        scores = winnner_traj_rewards
                    top_ntotake_indices = scores.argsort()[-self.n_to_take:]  # break ties by either traj reward or advantage
                    top_indices = [winners[int(idx)] for idx in top_ntotake_indices]
                    best_skills.append([skills[int(idx)] for idx in top_indices])
                else:  # case 3 or 4, group of maxes has size < n_to_take
                    # need to get the size of the group strictly better
                    best_skills.append([skills[sorted_indices[i, -j]] for j in range(1, n_taken + 1)])
                    n_needed = self.n_to_take - len(best_skills)
                    winners = np.argwhere(ranks[i] == nth_largest)
                    winnner_traj_rewards = reward_matrix[i, winners]
                    if self.use_adv:  # break ties by traj advantage
                        baselines = self.get_baseline_estimates(path['observations'][0], np.array([skills[int(winner)] for winner in winners])).flatten()
                        advantages = winnner_traj_rewards.flatten() - baselines
                        scores = advantages
                    else:  # break ties by traj reward
                        scores = winnner_traj_rewards
                    top_ntotake_indices = scores.argsort()[-n_needed:]  # break ties by either traj reward or advantage
                    top_indices = [winners[int(idx)] for idx in top_ntotake_indices]
                    best_skills[-1].extend([latents[int(idx)] for idx in top_indices])
            return best_skills, [[self.calculate_path_reward(path, z, True) for z in lat_list] for path, lat_list in zip(paths, best_skills)]


    def get_reward_matrix(self, paths, latents):
        raise NotImplementedError

    def plot_reward_matrix_histogram(self, reward_matrix, latents, title='histogram'):
        pass