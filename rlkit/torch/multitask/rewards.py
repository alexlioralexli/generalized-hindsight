import numpy as np
import rlkit.torch.pytorch_util as ptu
import torch
from gym import spaces
import matplotlib
matplotlib.use('Agg')

class Relabeler(object):
    def __init__(self,
                 discount=0.99,
                 relabel=True,
                 use_adv=False,
                 cache=False,
                 subtract_final_value=False,
                 q1=None,
                 q2=None,
                 vf=None,
                 action_fn=None,
                 test=False,
                 sliding_normalization=False,
                 is_eval=False
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
        assert len(rewards.shape) == 1
        multipliers = np.power(self.discount, np.arange(len(rewards)))
        return np.sum(rewards * multipliers)

    def get_discounted_path_reward(self, path, latent):
        path_rewards = self.calculate_path_reward(path, latent)
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

    def get_baseline_estimates(self, obs, latents):
        v1, v2 = self.get_both_values(obs, latents)
        return np.minimum(v1, v2)

    def get_both_values(self, obs, latents):
        obs, latent = ptu.from_numpy(obs).unsqueeze(0).repeat(len(latents), 1), ptu.from_numpy(latents)
        actions = self.action_fn(obs, latent, deterministic=True)[0]
        return ptu.get_numpy(self.q1(obs, actions, latent)), ptu.get_numpy(self.q2(obs, actions, latent))

    def get_latents_and_rewards(self, path):
        raise NotImplementedError

    def approx_irl_relabeling(self, paths):
        raise NotImplementedError

class RandomRelabeler(Relabeler):

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
            latents = [self.sample_task()]
        else:
            latents = [self.sample_task() for _ in range(self.n_sampled_latents - 1)]
            latents.append(path['latents'][0])
        rewards = [self.calculate_path_reward(path, latent) for latent in latents]
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

    def normalize_path_returns(self, paths, use_grid=False):
        assert self.relabel
        if self.n_sampled_latents == 1:
            latents = [self.sample_task()]
        else:
            latents = [self.sample_task() for _ in range(self.n_sampled_latents - len(paths))]
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
        rewards = [self.calculate_path_reward(path, latent) for path, latent in zip(paths, best_latents)]
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

    def approx_irl_relabeling(self, paths):
        assert self.relabel
        if self.n_sampled_latents == 1:
            latents = [self.sample_task()]
        else:
            latents = [self.sample_task() for _ in range(self.n_sampled_latents - len(paths))]
        for path in paths:
            latents.append(path['latents'][0])
        # latents = [np.concatenate([latent, np.array([np.pi, 0.25])]) for latent in x]
        # form matrix with |paths| rows x |latents| cols
        if self.cache:
            new_paths = remove_extra_trajectory_info(paths)
            self.cached_paths = (new_paths + self.cached_paths)[:500]
            reward_matrix = self.get_reward_matrix(self.cached_paths, latents)
        else:
            reward_matrix = self.get_reward_matrix(paths, latents)
        '''
        array([[ 6,  0,  3],
                [14,  7, 12]])
        >>> y = x.argsort()
        array([[1, 2, 0],
           [1, 2, 0]])
        '''
        best_latents = []

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
                    best_latent_index = winners[0]
                    best_latents.append(latents[int(best_latent_index)])
                else:
                    winnner_traj_rewards = reward_matrix[i, winners]
                    if self.use_adv:  # break ties by traj advantage
                        baselines = self.get_baseline_estimates(path['observations'][0], np.array([latents[int(winner)] for winner in winners])).flatten()
                        advantages = winnner_traj_rewards.flatten() - baselines
                        best_latent_index = winners[np.argmax(advantages)]
                    else:  # break ties by traj reward
                        best_latent_index = winners[np.argmax(winnner_traj_rewards)]  # break ties by traj reward

                    best_latents.append(latents[int(best_latent_index)])
            return [[z] for z in best_latents], \
                   [[self.calculate_path_reward(path, latent)] for path, latent in zip(paths, best_latents)]
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
                        baselines = self.get_baseline_estimates(path['observations'][0], np.array([latents[int(winner)] for winner in winners])).flatten()
                        advantages = winnner_traj_rewards.flatten() - baselines
                        scores = advantages
                    else:  # break ties by traj reward
                        scores = winnner_traj_rewards
                    top_ntotake_indices = scores.argsort()[-self.n_to_take:]  # break ties by either traj reward or advantage
                    top_indices = [winners[int(idx)] for idx in top_ntotake_indices]
                    best_latents.append([latents[int(idx)] for idx in top_indices])
                else:  # case 3 or 4, group of maxes has size < n_to_take
                    # need to get the size of the group strictly better
                    best_latents.append([latents[sorted_indices[i, -j]] for j in range(1, n_taken + 1)])
                    n_needed = self.n_to_take - len(best_latents)
                    winners = np.argwhere(ranks[i] == nth_largest)
                    winnner_traj_rewards = reward_matrix[i, winners]
                    if self.use_adv:  # break ties by traj advantage
                        baselines = self.get_baseline_estimates(path['observations'][0], np.array([latents[int(winner)] for winner in winners])).flatten()
                        advantages = winnner_traj_rewards.flatten() - baselines
                        scores = advantages
                    else:  # break ties by traj reward
                        scores = winnner_traj_rewards
                    top_ntotake_indices = scores.argsort()[-n_needed:]  # break ties by either traj reward or advantage
                    top_indices = [winners[int(idx)] for idx in top_ntotake_indices]
                    best_latents[-1].extend([latents[int(idx)] for idx in top_indices])
            return best_latents, [[self.calculate_path_reward(path, z) for z in lat_list] for path, lat_list in zip(paths, best_latents)]


    def get_reward_matrix(self, paths, latents):
        raise NotImplementedError

    def plot_reward_matrix_histogram(self, reward_matrix, latents, title='histogram'):
        pass


class FunctionRelabeler(Relabeler):
    def __init__(self,
                 latent_dist,
                 reward_fn):
        super().__init__()
        self.reward_fn = reward_fn
        self.latent_dist = latent_dist

    def calculate_reward(self, trajectory, latent):
        return self.reward_fn(trajectory, latent)

    def train(self):
        raise NotImplementedError


class BitstringRelabeler(Relabeler):
    def __init__(self, length):
        super().__init__()
        self.length = length
        self.latent_space = spaces.MultiBinary(length)

    def reward_done(self, obs, action, latent, env_info=None):
        if np.array_equal(latent, obs):
            action = int(action)
            if action == self.length:
                if (obs == 1).all():
                    return 1.0
                else:
                    return 0.0
            elif (obs[:action] == 1).all() and obs[action] == 0:
                return 1.0
        return 0.0

    def calculate_path_reward(self, path, latent):
        obs, actions = path['observations'], path['actions']
        indices = np.arange(len(obs))

        # want places where the obs matches the latent, and the action flips the first 0
        successful_indices = []
        for o, a, i in zip(obs, actions, indices):
            if np.array_equal(o, latent):
                a = int(a)
                if a == self.length:
                    if (o == 1).all():
                        successful_indices.append(i)
                elif (o[:a] == 1).all() and o[a] == 0:
                    successful_indices.append(i)
        rewards = np.zeros(len(path['observations']))
        rewards[successful_indices] = 1.0
        return rewards

    def sample_task(self):
        return self.latent_space.sample()

    def get_latents_and_rewards(self, path):
        obs, actions = path['observations'], path['actions']
        latents = []
        for o, a in zip(obs, actions):
            a = int(a)
            if a == self.length:
                if (o == 1).all() and not any([np.array_equal(o, lat) for lat in latents]):
                    latents.append(o)
            elif (o[:a] == 1).all() and o[a] == 0 and not any([np.array_equal(o, lat) for lat in latents]):
                latents.append(o)
        rewards = [self.calculate_path_reward(path, latent) for latent in latents]
        return latents, rewards


class DoNothingBitstringRelabeler(BitstringRelabeler):

    def get_latents_and_rewards(self, path):
        return [], []

class MakeDeterministicRelabeler(RandomRelabeler):

    def __init__(self, relabeler, latent):
        # self.__dict__['_wrapped_relabeler'] = relabeler
        self._wrapped_relabeler = relabeler
        self.latent = latent
        self.sliding_normalization = False

    def sample_task(self):
        return self.latent

    def reward_done(self, obs, action, latent, env_info=None):
        return self._wrapped_relabeler.reward_done(obs, action, latent, env_info=env_info)

    def train(self):
        return self._wrapped_relabeler.train()

    def calculate_reward(self, path, latent):
        return self._wrapped_relabeler.calculate_reward(path, latent)

    def calculate_path_reward(self, path, latent):
        return self._wrapped_relabeler.calculate_path_reward(path, latent)

    def get_discounted_reward(self, rewards):
        return self._wrapped_relabeler.get_discounted_reward(rewards)

    def get_baseline_estimate(self, obs, latent):
        return self._wrapped_relabeler.get_baseline_estimate(obs, latent)

    def get_baseline_estimates(self, obs, latents):
        return self._wrapped_relabeler.get_baseline_estimates(obs, latents)

    def get_both_values(self, obs, latents):
        return self._wrapped_relabeler.get_both_values(obs, latents)

    def get_latents_and_rewards(self, path):
        return self._wrapped_relabeler.get_latents_and_rewards(path)

    @property
    def discount(self):
        return self._wrapped_relabeler.discount

    @property
    def relabel(self):
        return self._wrapped_relabeler.relabel

    @property
    def n_sampled_latents(self):
        return self._wrapped_relabeler.n_sampled_latents

    def get_reward_matrix(self, paths, latents):
        return self._wrapped_relabeler.get_reward_matrix(paths, latents)

# make caching paths more memory efficient
def remove_extra_trajectory_info(paths):
    return [dict(env_infos=path['env_infos'],
                 actions=path['actions'],
                 next_observations=path['next_observations'],
                 observations=path['observations'],
                 latents=path['latents'],
                 terminals=path['terminals'],
                 rewards=path['rewards']) for path in paths]

if __name__ == "__main__":
    import ipdb;
    ipdb.set_trace()
