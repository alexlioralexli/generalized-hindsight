import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, use_xy=True, contact_forces=True):
        self.use_xy = use_xy
        self.contact_forces = contact_forces
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        torso_xyz_before = np.array(self.get_body_com("torso"))
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = (torso_xyz_after - torso_xyz_before) / self.dt
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity)

    # def _get_obs(self):
    #     return np.concatenate([
    #         self.sim.data.qpos.flat[2:],
    #         self.sim.data.qvel.flat,
    #         np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
    #     ])

    def _get_obs(self):
        if self.use_xy:
            elements = [self.sim.data.qpos.flat, self.sim.data.qvel.flat]
        else:
            elements = [self.sim.data.qpos.flat[2:], self.sim.data.qvel.flat]
        if self.contact_forces:
            elements.append(np.clip(self.sim.data.cfrc_ext, -1, 1).flat)
        return np.concatenate(elements)

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

if __name__ == '__main__':
    env = AntEnv()
    env.reset()
    # import IPython; IPython.embed()
    while True:
        a = env.action_space.sample()
        a = np.zeros_like(a)
        env.step(a)
        env.render()