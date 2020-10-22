import numpy as np
import os.path as osp
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env

class PendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, energy_multiplier=1.0, velocity_term=True):
        self.energy_multiplier = energy_multiplier
        self.velocity_term = velocity_term
        self.action_space = spaces.Box(low=np.array([-10.0]), high=np.array([10.0]))
        utils.EzPickle.__init__(self)
        xml_path = osp.expanduser('~/workspace/rlkit/rlkit/envs/assets/pendulum.xml')
        mujoco_env.MujocoEnv.__init__(self, xml_path, 2)

    def step(self, a):
        # scale a
        a = normalize_action(a, self.action_space.low, self.action_space.high)

        th, thdot = self.sim.data.qpos, self.sim.data.qvel
        costs = angle_normalize(th)**2 + .1*thdot**2 * float(self.velocity_term) + self.energy_multiplier * .001*(a**2)
        cost_dict = dict(theta=float(angle_normalize(th)**2), thdot=float(0.1*thdot**2), u=float(0.001*(a**2)))
        # print(a)
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all()
        done = not notdone
        return ob, -costs, done, cost_dict

    def reset_model(self):
        qpos = self.init_qpos - np.pi
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([np.cos(self.sim.data.qpos), np.sin(self.sim.data.qpos), self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

# The value returned by tolerance() at `margin` distance from `bounds` interval.
_DEFAULT_VALUE_AT_MARGIN = 0.1
def tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid='gaussian',
              value_at_margin=_DEFAULT_VALUE_AT_MARGIN):
  """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.
  Args:
    x: A scalar or numpy array.
    bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
      the target interval. These can be infinite if the interval is unbounded
      at one or both ends, or they can be equal to one another if the target
      value is exact.
    margin: Float. Parameter that controls how steeply the output decreases as
      `x` moves out-of-bounds.
      * If `margin == 0` then the output will be 0 for all values of `x`
        outside of `bounds`.
      * If `margin > 0` then the output will decrease sigmoidally with
        increasing distance from the nearest bound.
    sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
       'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
    value_at_margin: A float between 0 and 1 specifying the output value when
      the distance from `x` to the nearest bound is equal to `margin`. Ignored
      if `margin == 0`.
  Returns:
    A float or numpy array with values between 0.0 and 1.0.
  Raises:
    ValueError: If `bounds[0] > bounds[1]`.
    ValueError: If `margin` is negative.
  """
  lower, upper = bounds
  if lower > upper:
    raise ValueError('Lower bound must be <= upper bound.')
  if margin < 0:
    raise ValueError('`margin` must be non-negative.')

  in_bounds = np.logical_and(lower <= x, x <= upper)
  if margin == 0:
    value = np.where(in_bounds, 1.0, 0.0)
  else:
    d = np.where(x < lower, lower - x, x - upper) / margin
    value = np.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

  return float(value) if np.isscalar(x) else value

def normalize_action(action, lb, ub):
    scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
    return np.clip(scaled_action, lb, ub)

if __name__ == "__main__":
    env = PendulumEnv(energy_multiplier=2.0)
    env.reset()
    import ipdb; ipdb.set_trace(context=10)
    while True:
        print('resetting')
        env.reset()
        for _ in range(200):
            env.step(env.action_space.sample())
            env.render()