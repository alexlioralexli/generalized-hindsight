"""
Simple square environment where the agent can run around however it wants.
"""

import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import matplotlib.pyplot as plt

class SquareEnv(gym.Env):
    metadata = {
        "render.modes": ["rgb_array"],
    }

    def __init__(self, size, episode_length, enable_render=True):
        self.enable_render = enable_render
        self.size = size
        self._max_episode_steps = episode_length
        self.episode_length = episode_length

        self.action_space = spaces.Box(low=np.array([-1., -1.]),
                                        high=np.array([1., 1.]), dtype=np.float32)
        
        self.observation_space = spaces.Box(low=-self.size, high=self.size, 
                                            shape=(2,), 
                                            dtype=np.float32)

        # Initial conditions
        self.state = None
        self.movement_memory = np.empty(shape=(episode_length+1, 2))
        self._current_step = 0
        self.fig = None

    def step(self, action):
        self.state += action
        self.state = np.clip(self.state, 
                             self.observation_space.low, 
                             self.observation_space.high)
        self._current_step += 1
        self.movement_memory[self._current_step] = self.state
        reward = 0
        done = self._current_step >= self.episode_length
        info = {}

        return self.state, reward, done, info

    def reset(self):
        # Start at the zero state.
        self.state = np.zeros(2)
        self._current_step = 0
        # And reset the movement phase
        self.movement_memory[self._current_step] = self.state

        # Rendering details
        if self.fig is not None:
            plt.close()
        fig = plt.figure()
        self._inch_size = 2.56
        fig.set_size_inches((self._inch_size, self._inch_size))
        plt.axis('off')
        plt.xlim([-self.size,self.size])
        plt.ylim([-self.size,self.size])
        self.fig = fig
        self.plt = plt
        self.line_plot = self.plt.plot(self.movement_memory[:1])[0]
        self.fig.canvas.draw()

        return self.state

    def render(self, height, width, camera_id, mode="rgb_array"):
        if mode=="rgb_array":
            self.fig.set_dpi(height / self._inch_size)
            self.line_plot.set_data(self.movement_memory[:self._current_step, 0], 
                                    self.movement_memory[:self._current_step, 1])
            self.fig.canvas.draw()
            # this rasterizes the figure
            return np.array(self.fig.canvas.renderer._renderer)
        else:

            return None