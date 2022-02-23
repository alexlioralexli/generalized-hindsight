import numpy as np
from gym import spaces
from gym import Env

# from . import register_env


# @register_env('point-robot')
class PointEnvOld(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane

     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(self):

        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.t = 0
        self.H = 15

    def reset_model(self):
        self._state = np.zeros(shape=(2,))
        return self._get_obs()

    def reset(self):
        self.t = 0
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):
        self._state = self._state + action/10.0
        self._state = np.clip(self._state, -1.0, 1.0)
        # done = False
        ob = self._get_obs()
        self.t += 1
        done = (self.t >= self.H)
        return ob, 0.0, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)