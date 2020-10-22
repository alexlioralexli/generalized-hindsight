import numpy as np
from gym import spaces
from gym import Env
import matplotlib
from rlkit.torch.multitask.gym_relabelers import ReacherRelabelerWithGoalAndObs
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class PointReacherEnv3D(Env):
    """
    point robot on a 3-D plane with position control
    tasks (aka goals) are positions on the plane
    """

    def __init__(self, horizon=50):
        # x, y, z, t
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))
        self.t = 0
        self.horizon = horizon
        self.goal_pos = np.zeros(3)

    def _sample_goal(self):
        return np.random.uniform(-0.15, 0.15, size=(3,))

    def reset_model(self, goal_pos=None):
        self._state = np.zeros(shape=(4,))
        self.t = 0
        self.goal_pos = goal_pos
        return self._get_obs()

    def reset(self, goal_pos=None):
        return self.reset_model(goal_pos=goal_pos)

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):
        self.t += 1
        self._state[:3] = np.clip(self._state[:3] + action / 20.0, -0.15, 0.15)
        self._state[3] = self.t / self.horizon  # normalized timestep
        done = self.t >= self.horizon
        ob = self._get_obs()
        return ob, 0.0, done, dict(reward_dist=(np.exp(-np.linalg.norm(self._state[:3] - self.goal_pos) ** 2 / 0.08 ** 2)),
                                   reward_energy=-np.linalg.norm(action),
                                   reward_safety=0,
                                   end_effector_loc=self._state[:3].copy())

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
        locs = np.array([env_info['end_effector_loc'][:2] for env_info in path['env_infos']])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        color = list(plt.cm.rainbow(np.linspace(0, 1, len(locs))))
        ax.scatter(locs[:, 0], locs[:, 1], c=color)

        orig_goal = relabeler.get_goal(path['latents'][0])
        ax.scatter(x=orig_goal[0], y=orig_goal[1], marker='o', c='g', s=30, alpha=0.6)
        if isinstance(relabeler, ReacherRelabelerWithGoalAndObs):
            orig_obstacle = relabeler.get_obstacle(path['latents'][0])
        ax.scatter(x=orig_obstacle[0], y=orig_obstacle[1], marker='x', c='g', s=30, alpha=0.6)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim([-0.4, 0.4])
        ax.set_ylim([-0.4, 0.4])
        ax.set_title(title)
        ax.set_aspect('equal')
        plt.savefig('/tmp/pointreacher_plots/{}.png'.format(title))
        plt.close('all')
