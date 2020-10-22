import numpy as np
import gym
from gym import spaces
from gym import Env


# from . import register_env


# @register_env('PointMassDense-v0')
class PointEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane

     - tasks sampled from unit square
     - reward is L2 distance
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, horizon=15):

        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.t = 0
        self.horizon = horizon
        self._max_episode_steps = horizon
        self.spec = gym.envs.registration.EnvSpec('pointmass2-v0', max_episode_steps=horizon)

    def reset_model(self):
        self._state = np.zeros(shape=(3,))
        self.t = 0
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):
        # print('before', self._get_obs())
        # print('action', action)
        self.t += 1
        self._state[:2] = self._state[:2] + action/10.0
        self._state = np.clip(self._state, -1.0, 1.0)
        self._state[2] = self.t/self.horizon
        done = self.t >= self.horizon
        ob = self._get_obs()
        # print('after', self._get_obs())
        return ob, 0.0, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)

