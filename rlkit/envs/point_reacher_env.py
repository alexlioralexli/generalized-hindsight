import numpy as np
from gym import spaces
from gym import Env
import matplotlib
from rlkit.torch.multitask.gym_relabelers import ReacherRelabelerWithGoalAndObs, ReacherRelabelerWithGoalSimple

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# from . import register_env


# @register_env('point-robot')
class PointReacherEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane
    """

    def __init__(self, horizon=20):
        # x, y, xvel, yvel, t
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(5,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.t = 0
        self.horizon = horizon
        self.goal_pos = np.zeros(2)

    def reset_model(self, goal_pos=None):
        self._state = np.zeros(shape=(5,))
        self.t = 0
        self.goal_pos = goal_pos
        return self._get_obs()

    def reset(self, goal_pos=None):
        return self.reset_model(goal_pos=goal_pos)

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):
        self.t += 1
        self._state[2:4] = np.clip(self._state[2:4] + action / 20.0, -0.25, 0.25)  # update velocity and clip
        self._state[:2] = np.clip(self._state[:2] + self._state[2:4], -1.0, 1.0)  # update position and clip within wall
        self._state[4] = self.t / self.horizon  # normalized timestep
        done = self.t >= self.horizon
        ob = self._get_obs()
        return ob, 0.0, done, dict(reward_dist=(np.exp(-np.linalg.norm(self._state[:2] - self.goal_pos) ** 2 / 0.08 ** 2)),
                                   reward_energy=-np.linalg.norm(action),
                                   reward_safety=0,
                                   end_effector_loc=self._state[:2].copy())

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self, mode='human'):
        print('current state: (x={.2f},y={.2f}), (xvel={.2f},yvel={.2f}), t={:.2f}'.format(self._state[0],
                                                                                           self._state[1],
                                                                                           self._state[2],
                                                                                           self._state[3],
                                                                                           self._state[4],
                                                                                           self._state[5]))

    def render_path(self, path):
        pass

    def plot_trajectory_on_heatmap(self, latent, path, title, relabeler=None):
        pass
