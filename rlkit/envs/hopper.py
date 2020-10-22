import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore, angle_before = self.sim.data.qpos[0], self.sim.data.qpos[2]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]

        reward_run = (posafter - posbefore) / self.dt
        # reward_ctrl = 1e-3 * np.square(a).sum()
        reward_ctrl = - np.linalg.norm(a)  # np.square(a).sum()
        # print(ang, angle_before)
        s = self.state_vector()
        done = not np.isfinite(s).all()
        ob = self._get_obs()
        return ob, 0.0, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl, height=self.sim.data.qpos[1] - 1.2,
                                  reward_angular=self.sim.data.qvel[2].copy())

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


if __name__ == '__main__':
    env = HopperEnv()
    import ipdb; ipdb.set_trace(context=10)
    for _ in range(1000):
        env.reset()
        print('reset')
        for j in range(200):
            o, r, d, env_info = env.step(env.action_space.sample())
            if d:
                break
            env.render()
    import ipdb; ipdb.set_trace()