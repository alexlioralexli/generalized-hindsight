import numpy as np
import os.path as osp
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from gym.spaces import Box, Discrete, Tuple, MultiBinary
from rlkit.torch.multitask.pointmass_rewards import PointMassBestRandomRelabeler
from rlkit.torch.multitask.gym_relabelers import ReacherRelabelerWithGoalAndObs
from rlkit.util.utils import save_video
from rlkit.core import logger
import torch


class MultiTaskReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            relabeler,
            alg="SAC",
            cfg=None,
            skill_dim=False,
            latent_dim=None,
            normalize_rewards=False,
            grid_normalize=False,
            on_policy=False,
            plot=False,
            dads=False,
            approx_irl=False,
            hide_latent=False,
            permute_relabeling=False,
            add_random_relabeling=False,
            cem=False,
            agent=None,

    ):
        self.alg = alg
        if self.alg == "DIAYN":
            max_replay_buffer_size = cfg.max_replay_buffer_size
            self._skills = np.zeros((max_replay_buffer_size, skill_dim))
            self._not_dones_no_max = np.zeros((max_replay_buffer_size, 1))
            # self._pureSkills = np.zeros((max_replay_buffer_size, skill_dim))
            self.pureSkills = np.zeros((cfg.max_replay_buffer_size, skill_dim))

        
        # print(f"Algorithm received is: {self.alg}")
        """
        :param max_replay_buffer_size:
        """
        self.env = env
        self._action_space = env.action_space
        self._observation_space = env.observation_space
        if isinstance(self.env.action_space, Discrete):
            action_dim = env.action_space.n
        else:
            action_dim = env.action_space.low.size

        if isinstance(self.env.observation_space, Discrete) or isinstance(self.env.observation_space, MultiBinary):
            obs_dim = env.observation_space.n
        else:
            obs_dim = env.observation_space.low.size

        self.latent_dim = latent_dim
        self._latents = np.zeros((max_replay_buffer_size, latent_dim))
        self.relabeler = relabeler

        self._observation_dim = obs_dim
        self._action_dim = action_dim
        
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, obs_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, obs_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        # Define self._env_infos[key][i] to be the return value of env_info[key]
        # at time i
        self.normalize_rewards = normalize_rewards
        self.grid_normalize = grid_normalize
        self.on_policy = on_policy
        self.dads = dads
        self.hide_latent = hide_latent
        self.permute_relabeling = permute_relabeling
        self.add_random_relabeling = add_random_relabeling
        self.permutation_list = []
        if dads:
            self._qpos = np.zeros((max_replay_buffer_size, 2))
            self._next_qpos = np.zeros((max_replay_buffer_size, 2))
        self.plot = plot
        if plot:
            self.trajectory_latents = []
        self.approx_irl = approx_irl

        self.cem = cem
        if self.cem:
            # Set the rho, k, n
            self.rho, self.k, self.n = None, None, None
        
        self._top = 0
        self._size = 0
        self.all_relabeling_info = dict()
        self.original_latents = []
        self.relabeled_latents = []
        self.features = []
        self.original_rewards = []
        self.relabeled_rewards = []
        float_formatter = "{:.2f}".format
        np.set_printoptions(formatter={'float_kind': float_formatter})

        self.epoch = 0
        self.paths_this_epoch = 0


    def add_sample(self, observation, action, reward, terminal, next_observation, **kwargs):
        raise NotImplementedError("Only use add_path")

    def add_single_sample(self, latent, observation, action, reward, next_observation, terminal=None, pureSkill=None, **kwargs):
        #RECEIVES SINGLE ELEMENTS OF EACH. NOT 2D ARRAYS.
        # print(f"Obs in single sample is: {observation}")
        # print(f"Next_obs in single sample is: {next_observation}")
        if isinstance(self._action_space, Discrete):
            action = np.eye(self._action_space.n)[action]
        if self.alg == "DIAYN":
            if (torch.is_tensor(latent)):
                latent = latent.cpu().detach().numpy()
            reward = reward.cpu().detach().numpy()
            # print(f"Latent received is : {latent}, the type is : {type(latent)}")
            self._next_obs[self._top] = next_observation
            self._skills[self._top] = latent
            self.pureSkills[self._top] = pureSkill
            self._observations[self._top] = observation
            self._actions[self._top] = action
        elif self.alg == "SAC":
            self._latents[self._top] = latent
            self._terminals[self._top] = terminal
            self._rewards[self._top] = reward
            self._observations[self._top] = observation
            self._actions[self._top] = action
            self._next_obs[self._top] = next_observation

        self._advance()

    


    def add_single_dads_sample(self, latent, observation, action, terminal, next_observation, qpos, next_qpos):
        if isinstance(self._action_space, Discrete):
            action = np.eye(self._action_space.n)[action]
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._terminals[self._top] = terminal
        self._latents[self._top] = latent
        self._next_obs[self._top] = next_observation
        self._qpos[self._top] = qpos
        self._next_qpos[self._top] = next_qpos
        self._advance()

    def sample_data(self, indices):
        if self.alg ==  "SAC":
            return dict(
                observations=self._observations[indices],
                actions=self._actions[indices],
                latents=self._latents[indices],
                rewards=self._rewards[indices],
                terminals=self._terminals[indices],
                next_observations=self._next_obs[indices],
            )
        elif self.alg == "DIAYN":
            return dict(
                skills = self._skills[indices],
                not_dones_no_max = self._not_dones_no_max[indices],
                observations=self._observations[indices],
                actions=self._actions[indices],
                latents=self._latents[indices],
                rewards=self._rewards[indices],
                terminals=self._terminals[indices],
                next_observations=self._next_obs[indices],
            )



    # THIS ADD PATH IS JUST TO ADD RANDOM RELABELING.
    def add_path(self, path):
        # add with original z and resampled z
        original_z = path["latents"][0]
        """
                GET LATENTS AND REWARDS.

        """
        resampled_zs, reward_list, ranked_latents = self.relabeler.get_latents_and_rewards(path)
        if self.permute_relabeling:
            self.permutation_list.append((path, resampled_zs[0]))
        if not any([np.array_equal(original_z, z) for z in resampled_zs]):
            resampled_zs.append(original_z)
            reward_list.append(self.relabeler.calculate_path_reward(path, original_z))
        
        
        """
            RANDOM RELABELING LOGIC

        """
        
        if self.add_random_relabeling:

            
            random_z = self.relabeler.sample_task()
            # print(f"Random_Z sampeled is: {random_z}")
            resampled_zs.append(random_z)
            
            if self.alg == "DIAYN":
                reward_list.append(self.relabeler.calculate_path_reward(path, random_z, True))
            elif self.alg == "SAC":
                reward_list.append(self.relabeler.calculate_path_reward(path, random_z))

        # save video if applicable, definitely save
        original_discounted_reward = self.relabeler.get_discounted_reward(reward_list[-1])
        if 'rgb_array' in path.keys():
            save_path = osp.join(logger.get_snapshot_dir(), "epoch{}_{}.mp4".format(self.epoch, self.paths_this_epoch))
            text_original = "{}, {}, {:.2f}".format(str(original_z),
                                       str(self.relabeler.get_features(path, latent=original_z)),
                                       original_discounted_reward)
            text_relabeled = "{}, {:.2f}".format(str(resampled_zs[0]), self.relabeler.get_discounted_reward(reward_list[0]))
            text = text_original + "; " + text_relabeled
            save_video(path['rgb_array'], save_path, text)
        self.original_latents.append(original_z)
        self.relabeled_latents.append(resampled_zs[:-1])
        self.features.append(self.relabeler.get_features(path))
        self.original_rewards.append(original_discounted_reward)
        self.relabeled_rewards.append([self.relabeler.get_discounted_reward(reward) for reward in reward_list[:-1]])

        if self.hide_latent:
            resampled_zs = [np.zeros_like(z) for z in resampled_zs]
        for z, rewards in zip(resampled_zs, reward_list):
            if self.alg == "SAC":
                for i, (
                        obs,
                        action,
                        reward,
                        next_obs,
                        terminal,
                        agent_info,
                ) in enumerate(zip(
                    path["observations"],
                    path["actions"],
                    rewards,
                    path["next_observations"],
                    path["terminals"],
                    path["agent_infos"],
                )):
                    self.add_single_sample(
                        z,
                        obs,
                        action,
                        reward,
                        terminal,
                        next_obs,
                    )
            elif self.alg == "DIAYN":
                    for i, (
                        obs,
                        action,
                        reward,
                        next_obs,
                        terminal,
                        # skills,
                        done,
                        # agent_info,
                    ) in enumerate(zip(
                        path["observations"],
                        path["actions"],
                        rewards,
                        path["next_observations"],
                        path["terminals"],
                        # path["skills"],
                        path["done_no_max"]

                        # path["agent_infos"],
                        
                    )):
                        self.add_single_sample(
                            z,
                            obs,
                            action,
                            reward,
                            next_obs,
                            pureSkill = original_z
                            # skills, 
                        )

        self.terminate_episode()


    def add_dads_path(self, path):
        z = path["latents"][0]
        for i, (
            obs,
            action,
            next_obs,
            terminal,
            agent_info,
            qpos,
            next_qpos,
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path['qpos'],
            path['next_qpos']
        )):
            self.add_single_dads_sample(
                z,
                obs,
                action,
                terminal,
                next_obs,
                qpos,
                next_qpos,
            )

        self.terminate_episode()
    
    def add_path_fixed_latent(self, path, rewards, latent, originalSkill=None):
        if self.alg == "SAC":
            for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
                agent_info,
            ) in enumerate(zip(
                path["observations"],
                path["actions"],
                rewards,
                path["next_observations"],
                path["terminals"],
                path["agent_infos"],
            )):
                # self, latent, observation, action, reward, next_observation, terminal=None,
                # print(f"In add latent: obs: {obs}, next_obs: {next_obs}")
                self.add_single_sample(
                    latent = latent,
                    observation = obs,
                    action = action,
                    reward = reward,
                    next_observation = next_obs,
                    terminal = terminal

                )

        elif self.alg == "DIAYN":
            for i, (
                obs,
                action,
                reward,
                next_obs,
                # terminal,
                # agent_info,
            ) in enumerate(zip(
                path["observations"],
                path["actions"],
                rewards,
                path["next_observations"],
            )):
                self.add_single_sample(
                    latent,
                    obs,
                    action,
                    reward,
                    next_obs,
                    pureSkill = originalSkill
                )


        self.terminate_episode()




    def add_paths(self, paths, epoch):
        print(f"Length of paths IN GHER are: {len(paths)}")
        # print(f"The paths received for epoch: {epoch}, are: {paths}")
        if self.dads:
            for path in paths:
                self.add_dads_path(path)
        elif self.normalize_rewards or self.approx_irl or self.cem:
            # print("I am in approx_irl")
            assert not self.hide_latent
            if self.normalize_rewards:
                # sample a bunch of latents
                # calculate the rewards of each trajectory under each latent
                # take each of them and normalize
                # label based on the best normalized
                if self.alg == "SAC":

                    latents, rewards = self.relabeler.normalize_path_returns(paths, use_grid=self.grid_normalize)  #latents should be a list of lists, same for rewards
                    print(f"Len of latents are: {latents}")
                elif self.alg == "DIAYN":
                    skills, rewards = self.relabeler.normalize_path_returns(paths, use_grid=self.grid_normalize)  #latents should be a list of lists, same for rewards

            elif self.approx_irl or self.cem:
                """
                    ALGORITHM: 

                    1. APPROX IRL RELABEL FUNC 
                    2. GET REWARD MATRIX 
                    3. IF USE ADV: get_baseline_estimates
                        GET BOTH VALUES
                    4. CALCULATE PATH REWARD 
                    5. COMPUTE DIVERSITY REWARD (LONE), with the correct skill
                    


                """
                if self.cem:
                    if self.alg == "SAC":
                        latents, rewards = self.relabeler.cem_relabeler(paths)
                        print(f"Latents receievd after CEM relabeling are: {latents}, its length is : {len(latents)}")
                    elif self.alg == "DIAYN":
                        print(f"I am in CEM, DIAYN")
                        skills, rewards = self.relabeler.cem_relabeler(paths)
                        print(f"The skill received after CEM relabeleing is: {skills}")
                elif self.approx_irl:
                    if self.alg == "SAC":
                        latents, rewards = self.relabeler.approx_irl_relabeling(paths)
                        print(f"Latents receievd after IRL relabeling are: {latents}, its length is : {len(latents)}")
                    elif self.alg == "DIAYN":
                        skills, rewards = self.relabeler.approx_irl_relabeling(paths)
                        print(f"The skill received after IRL relabeleing is: {skills}")
            else:
                raise RuntimeError


            

            if self.alg == "SAC":
                print(f"Len of paths in GHER is: {len(paths)}")
                self.relabeled_latents.extend(latents)
                orig_latents = [[path['latents'][0]] for path in paths]

                orig_rewards = [[self.relabeler.calculate_path_reward(path, z[0])] for path, z in zip(paths, orig_latents)]
                # print(f"Original Latents were: {orig_latents}")
                assert len(latents) == len(rewards)
                assert len(orig_latents) == len(orig_rewards)
                assert len(latents) == len(orig_latents)
                for i in range(len(latents)):
                    latents[i].extend(orig_latents[i])
                    rewards[i].extend(orig_rewards[i])
            elif self.alg == "DIAYN":

                """
                    THESE SKILLS, COME FROM THE APPROX IRL FUNC in rewards.py

                """
                self.relabeled_latents.extend(skills)
                # print(f"The relabeled latents arE: {self.relabeled_latents}")
                
                orig_skills = [[path['skills'][0]] for path in paths]
                # print(f"The original skills are:{orig_skills}")
                orig_rewards = [[self.relabeler.calculate_path_reward(path, z[0], True)] for path, z in zip(paths, orig_skills)]
                assert len(skills) == len(rewards)
                assert len(orig_skills) == len(orig_rewards)
                assert len(skills) == len(orig_skills)
                for i in range(len(skills)):

                    """
                        THE SKILLS AND REWARDS

                        GET APPENDED WITH THE ORIGINAL SKILLS AND THE ORIGINAL REWARDS

                    """
                    skills[i].extend(orig_skills[i])
                    rewards[i].extend(orig_rewards[i])

            # print(f"Skills after extend: {skills}")
            if self.plot and isinstance(self.relabeler, PointMassBestRandomRelabeler):
                self.relabeler.plot_paths(paths, orig_latents, latents, title='Epoch {}'.format(str(self.epoch)))
                self.trajectory_latents = None


            

            # save videos if necessary
            for i, path in enumerate(paths):
                if self.alg == "SAC":
                    original_z = path['latents'][0]
                    # print(f"Original Z was: {original_z}")
                elif self.alg == "DIAYN":
                    original_z = path['skills'][0]
                    # print(f"Original Z is: {original_z}")
                original_discounted_reward = self.relabeler.get_discounted_reward(rewards[i][-1])
                
                
                
                """
                    SAVE VIDEO LOGIC

                """
                
                if 'rgb_array' in path.keys():
                    save_path = osp.join(logger.get_snapshot_dir(),
                                         "epoch{}_{}.mp4".format(self.epoch, self.paths_this_epoch))
                    text_original = "{}, {}, {:.2f}".format(str(original_z),
                                                        str(self.relabeler.get_features(path, latent=original_z)),
                                                        original_discounted_reward)
                    text_relabeled = "{}, {:.2f}".format(str(latents[i][0]),
                                                        self.relabeler.get_discounted_reward(rewards[i][0]))
                    text = text_original + "; " + text_relabeled
                    save_video(path['rgb_array'], save_path, text)
                
                self.original_latents.append(original_z)
                

                if self.alg == "SAC":
                    self.relabeled_latents.append(latents[i][:-1])
                
                
                elif self.alg == "DIAYN":
                    self.relabeled_latents.append(skills[i][:-1])

                
                self.features.append(self.relabeler.get_features(path))
                self.original_rewards.append(original_discounted_reward)
                self.relabeled_rewards.append([self.relabeler.get_discounted_reward(reward) for reward in rewards[i][:-1]])
                self.paths_this_epoch += 1
            
            if self.alg == "SAC":
                for path, reward_list, latent_list in zip(paths, rewards, latents):
                    assert len(reward_list) == len(latent_list)
                    for r, z in zip(reward_list, latent_list):
                        #print(f"Reward is : {r}, its shape is: {r.shape}, its type is: {type(r)}")

                        self.add_path_fixed_latent(path, r, z)
                    if self.add_random_relabeling:
                        random_z = self.relabeler.sample_task()
                        random_r = self.relabeler.calculate_path_reward(path, random_z)
                        self.add_path_fixed_latent(path, random_r, random_z)
                    if self.permute_relabeling:
                        self.permutation_list.append((path, z))
            
            
            elif self.alg == "DIAYN":
                print(f"Skill list is: {skills}")
                random_z = None
                for path, reward_list, skill_list in zip(paths, rewards, skills):
                    assert len(reward_list) == len(skill_list)
                    for r, z in zip(reward_list, skill_list):
                        # print(f"Reward is : {r}, its shape is: {r.shape}, its type is: {type(r)}")
                        # print(f"Path is: {path}, counter is: {counter}")

                        """
                            What does the add path fixed latent do?

                            IT ADDS THE NEW SKILLS AND THE REWARDS TO THE REPLAY BUFFER

                        """
                        # print(f"Z added in the loop is: {z}")
                        # print(f"The original skill is: {original_z}")
                        self.add_path_fixed_latent(path, r, z, originalSkill=original_z)



                    if self.add_random_relabeling:

                        """
                            HOW DO WE SAMPLE THIS TASK? in random relabeling from the distribution?

                            FROM THE ORIGINAL DIAYN SAMPLE TASK OR THE ONE IN THE CLASS FOR GHER.
                        """

                        random_z = self.relabeler.sample_task()
                        print(f"Random Z relabeled is: {random_z}")
                        random_r = self.relabeler.calculate_path_reward(path, random_z, True)
                        self.add_path_fixed_latent(path, random_r, random_z)
                    if self.permute_relabeling:
                        self.permutation_list.append((path,z))
       

        else:
            if self.relabeler.sliding_normalization:
                self.relabeler.update_sliding_params(paths)
            for path in paths:
                self.add_path(path)
        if self.permute_relabeling:
            self.handle_permuting()


    def random_batch(self, batch_size):
        # print(f"The self size is in the replay buffer: {self._size}, the batch size is: {batch_size}")
        indices = np.random.randint(0, self._size, batch_size)
        if self.dads:
            return dict(
                observations=self._observations[indices],
                actions=self._actions[indices],
                terminals=self._terminals[indices],
                next_observations=self._next_obs[indices],
                latents=self._latents[indices],
                qpos=self._qpos[indices],
                next_qpos=self._next_qpos[indices],
            )
        else:
            if self.alg == "SAC":
                return dict(
                    # observations=np.concatenate([self._observations[indices], self._latents[indices]], axis=1),
                    observations=self._observations[indices],
                    actions=self._actions[indices],
                    rewards=self._rewards[indices],
                    terminals=self._terminals[indices],
                    next_observations=self._next_obs[indices],
                    # next_observations=np.concatenate([self._next_obs[indices], self._latents[indices]], axis=1),
                    latents=self._latents[indices],
                )
            elif self.alg == "DIAYN":
                return dict(
                # observations=np.concatenate([self._observations[indices], self._latents[indices]], axis=1),
                    observations=self._observations[indices],
                    actions=self._actions[indices],
                    rewards=self._rewards[indices],
                    terminals=self._terminals[indices],
                    next_observations=self._next_obs[indices],
                    # next_observations=np.concatenate([self._next_obs[indices], self._latents[indices]], axis=1),
                    latents=self._latents[indices],
                    skill = self._skills[indices],
                    # not_done = self._not_dones[indices],
                    not_dones_no_max = self._not_dones_no_max[indices],
                    pureSkill=self.pureSkills[indices]
                )   

    def handle_permuting(self):
        latents = [pair[1] for pair in self.permutation_list]
        paths = [pair[0] for pair in self.permutation_list]
        if len(latents) <= 1:
            print("No permuting happening, only {} paths".format(len(latents)))
            return
        print("permuting {} paths".format(len(latents)))
        new_latents = latents[1:] + [latents[0]]
        assert len(new_latents) == len(paths)
        for path, z in zip(paths, new_latents):
            r = self.relabeler.calculate_path_reward(path, z)
            self.add_path_fixed_latent(path, r, z)
        self.permutation_list = []

    def end_epoch(self, epoch):
        if self.on_policy:
            self._top = 0
            self._size = 0
        if self.plot:
            # plot everything
            if self.relabeler.use_adv and False:
                # need to calculate the rewards, q1, q2, and advantage
                disc_rewards = np.array([[self.relabeler.get_discounted_path_reward(path, latent) for latent in latents]
                                for path, latents in self.trajectory_latents])
                values = np.array([self.relabeler.get_both_values(path['observations'][0], np.array(latents))
                                   for path, latents in self.trajectory_latents]).squeeze()
                v1, v2 = values[:,0,:], values[:,1,:]
                if self.relabeler.subtract_final_value:
                    print('subtracting final value')
                    final_values = np.array([self.relabeler.get_both_values(path['next_observations'][-1], np.array(latents))
                                   for path, latents in self.trajectory_latents]).squeeze()
                    v1 -= final_values[:,0,:] * self.relabeler.discount ** \
                                        np.array([len(path['observations']) for path, _ in self.trajectory_latents])[...,np.newaxis]
                    v2 -= final_values[:,1,:] * self.relabeler.discount ** \
                                        np.array([len(path['observations']) for path, _ in self.trajectory_latents])[...,np.newaxis]
                adv = disc_rewards - values.min(axis=1)
                traj_infos = dict(
                    rewards=disc_rewards,
                    v1=v1,
                    v2=v2,
                    adv=adv,
                )
            else:
                traj_infos = None
            if isinstance(self.relabeler, PointMassBestRandomRelabeler):
                self.relabeler.plot_resampling_heatmaps(self.trajectory_latents, "Epoch" + str(self.epoch), traj_infos=traj_infos)
            if isinstance(self.relabeler, ReacherRelabelerWithGoalAndObs):
                self.relabeler.plot_resampling_heatmaps(self.trajectory_latents, "Epoch" + str(self.epoch), traj_infos=traj_infos)
            self.trajectory_latents = []
        self.permutation_list = []

        # save the relabeling information here
        relabeling_info = {"original_latents": np.array(self.original_latents),
                            "relabeled_latents": np.array(self.relabeled_latents),
                            "features": np.array(self.features),
                            "original_rewards": np.array(self.original_rewards),
                            "relabeled_rewards": np.array(self.relabeled_rewards)}
        self.all_relabeling_info['epoch{}'.format(self.epoch)] = relabeling_info
        logger.save_extra_data(self.all_relabeling_info, 'relabeling_info.pkl')
        self.original_latents = []
        self.relabeled_latents = []
        self.features = []
        self.original_rewards = []
        self.relabeled_rewards = []
        self.epoch += 1
        self.paths_this_epoch = 0

    def get_snapshot(self):
        return dict(relabeler=self.relabeler)