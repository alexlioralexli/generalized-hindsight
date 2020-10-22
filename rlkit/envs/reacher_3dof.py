import numpy as np
import os.path as osp
from gym import utils
from gym.envs.mujoco import mujoco_env
import time

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, safety_fn='nosafety', energy_fn='kinetic', energy_factor=4.0, multitask=True, random_goal=False, random_obstacle=False):
        self.safety_fn = safety_fn
        self.energy_fn = energy_fn
        self.energy_factor = energy_factor
        self.multitask = multitask
        self.random_goal = random_goal
        self.random_obstacle = random_obstacle
        print(self.energy_fn, self.safety_fn, self.energy_factor)
        assert safety_fn in {'log', 'linear', 'inverse', 'nosafety', 'newlog'}
        assert energy_fn in {'work', 'norm', 'oldwork', 'kinetic', 'velocity'}
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, osp.expanduser('~/workspace/rlkit/rlkit/envs/assets/reacher_3dof.xml'), 2)

    def step(self, a):
        init_joint_pos = self.sim.data.qpos.flat[:3]
        init_kinetic_energy = self.sim.data.energy[1]

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        # distance to goal
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)

        # safety
        dist_to_obstacle = np.linalg.norm(self.get_body_com('fingertip') - self.get_body_com('obstacle'))
        if self.safety_fn == 'log':
            # print('log')
            reward_safety = np.log10(dist_to_obstacle + 1e-3)/5.0
        elif self.safety_fn == 'newlog':
            reward_safety = np.log10(dist_to_obstacle + 1e-2) / 5.0
        elif self.safety_fn == 'linear':
            reward_safety = dist_to_obstacle / 2.0
        elif self.safety_fn == 'inverse':
            reward_safety = - 0.01 / (dist_to_obstacle + 0.01)
        elif self.safety_fn == 'nosafety':
            reward_safety = 0
        else:
            raise NotImplementedError

        # energy
        final_joint_pos = self.sim.data.qpos.flat[:3]
        ke_diff = (self.sim.data.energy[1] - init_kinetic_energy).clip(min=0.0)
        clipped_work = (a * (final_joint_pos - init_joint_pos)).clip(min=0.0).sum()
        if self.energy_fn == 'work':
            reward_energy = - clipped_work
        elif self.energy_fn == 'velocity':
            reward_energy = - 0.01 * np.square(self.sim.data.qvel.flat[:3]).sum()
        elif self.energy_fn == 'kinetic':
            reward_energy = - ke_diff / 150.0 * 8
        elif self.energy_fn == 'oldwork':
            reward_energy = - np.dot(a, final_joint_pos - init_joint_pos).clip(min=0.0)
        elif self.energy_fn == 'norm':
            reward_energy = - np.square(a).sum()  # not precisely the right energy metric
        else:
            raise NotImplementedError
        reward_energy *= self.energy_factor
        reward_dist += 0.2
        reward_energy += 0.2
        reward_safety += 0.2
        reward = reward_dist + reward_energy + reward_safety
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_energy=reward_energy,
                                      reward_safety=reward_safety, dist_to_obstacle=dist_to_obstacle,
                                      ke_diff=ke_diff, clipped_work=clipped_work, end_effector_loc=self.get_body_com("fingertip").copy())

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset(self, goal_pos=None):
        self.sim.reset()
        ob = self.reset_model(goal_pos=goal_pos)
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def reset_model(self, goal_pos=None):
        # sample arm, obstacle, and goal location independently
        arm_qpos = self.np_random.uniform(low=-0.8, high=0.8, size=3) + self.init_qpos[:3]
        # arm_qpos = self.init_qpos[:3]

        # obstacle qpos
        if self.random_obstacle:
            obstacle_qpos = self.sample_reachable_point()
        else:
            obstacle_qpos = np.array([0.225, 0.13])

        # goal qpos
        if goal_pos is not None:
            goal_qpos = goal_pos
        elif self.random_goal:
            goal_qpos = self.sample_reachable_point()
        else:
            goal_qpos = np.array([-0.25, 0])

        qpos = np.concatenate([arm_qpos, obstacle_qpos, goal_qpos])
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-4:] = 0  # don't let the goal or obstacle have velocity
        self.set_state(qpos, qvel)
        return self._get_obs()

    def sample_reachable_point(self):
        theta = self.np_random.uniform(low=-np.pi, high=np.pi, size=1),
        r = self.np_random.uniform(low=0.05, high=0.30, size=1)  # total arm length is 0.3
        return r * np.array([np.cos(theta), np.sin(theta)]).flatten()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:3]  # the three joint angles?
        if self.multitask:
            return np.concatenate([
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[3:5],  # the obstacle position
                self.sim.data.qvel.flat[:3],  # the three joint angular velocities or something?
                self.get_body_com('fingertip')[:2],  # fingertip position
                self.get_body_com("fingertip") - self.get_body_com("obstacle")
            ])

        else:
            return np.concatenate([
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[3:],  # the goal position
                self.sim.data.qvel.flat[:3], # the three joint angular velocities or something?
                self.get_body_com("fingertip") - self.get_body_com("target")
            ])

    def render(self, mode='human'):
        if mode == 'rgb_array':
            # self._get_viewer().render()
            data = self.sim.render(width=1000, height=1000, camera_name='top_cam')
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()

if __name__ == '__main__':
    env = ReacherEnv(random_goal=True, random_obstacle=True)
    env.reset()
    env.render()
    factor = 1
    for i in range(1000):
        factor *= -1
        env.reset()
        for t in range(200):
            a = env.action_space.high * factor
            env.render()
            env.step(a)
