import os
import numpy as np
import random
from gym import utils, spaces
from gym.envs.robotics import fetch_env

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')


class FetchReachEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', truncate_obs=False):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=100,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.25, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        self.truncate_obs = truncate_obs
        if self.truncate_obs:
            high = np.inf * np.ones((3,))
        else:
            high = np.inf * np.ones((10,))
        low = -high
        self.observation_space = spaces.Box(low, high)


    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            'end_effector_loc': obs['achieved_goal'],
            'reward_energy': -np.linalg.norm(action[:3]),
            'reward_safety': 0
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        if self.truncate_obs:
            obs_to_return = obs['observation'][:3]
        else:
            obs_to_return = obs['observation']

        return obs_to_return, reward, done, info

    def reset(self, goal_pos=None):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        if goal_pos is not None:
            self.goal = goal_pos.copy()
        else:
            self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        if self.truncate_obs:
            return obs['observation'][:3]
        else:
            return obs['observation']


if __name__ == '__main__':
    env = FetchReachEnv()
    # random.seed(0)
    # np.random.seed(0)
    #
    # # stage 1: checking the the variance of the start state
    # x = np.array([env.reset() for _ in range(1000)])
    # mean = x.mean(axis=0)
    # # y = x - mean
    # # print(y.max())
    # import ipdb; ipdb.set_trace()
    # env._get_obs()
    #
    # # stage 2: checking the variance of the step
    # differences = []
    # locs = []
    # o, r, d, env_info = env.step(env.action_space.sample())
    # init_loc = env_info['end_effector_loc']
    # for _ in range(1000):
    #     a = env.action_space.sample()
    #     obs, r, d, env_info = env.step(a)
    #     next_loc = env_info['end_effector_loc']
    #     locs.append(next_loc.copy())
    #     print(init_loc, a[:3]/20.0, next_loc)
    #     differences.append(np.abs(next_loc - init_loc - a[:3]/20.0))
    #     init_loc = next_loc
    # differences = np.array(differences)
    # norms = np.linalg.norm(differences, axis=1)
    # import ipdb; ipdb.set_trace()

    while True:
        env.reset()
        for i in range(100):
            env.step(env.action_space.sample())
            env.render()
    # import ipdb; ipdb.set_trace()