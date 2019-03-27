"""
    File name: circle.py
    Author: Christopher Villalta
"""
import logging

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle

class CircleEnv(object):
    """Circle reinforcement learning environment.

    An agent is tasked with finding and remaining within a circular area.
    """

    def __init__(self, continuous=True, sparse=False):
        self.state = np.array([0, 0])
        self.actions = {
            0: np.array([0, 0.25]),
            1: np.array([0, -0.25]),
            2: np.array([-0.25, 0]),
            3: np.array([0.25, 0])
        }
        self.continuous = continuous
        self.sparse = sparse
        self.observation_size = 2
        self.action_size = 4
        self.circle_center = np.array([0, 0])
        self.circle_radius = 1
        self.episode_length = 200
        self.steps = 0
        self.agent_render_pos = None
        self.render_initialized = False

    def step(self, action):
        self.steps += 1
        next_state, reward, done = self._take_action(action)
        return next_state, reward, done

    def _get_reward(self):
        x, y = self.state
        h, k = self.circle_center
        d = ((x-h)**2 + (y-k)**2)**(1/2)
        if d <= self.circle_radius:
            return 1
        else:
            if self.sparse:
                return 0
            else:
                return 1/(1+d)**2

    def _take_action(self, action):
        reward = self._get_reward()
        next_state = self.state + self.actions[action]
        self.state = next_state
        done = True if self.steps >= self.episode_length else False
        return next_state, reward, done

    def reset(self):
        if self.continuous:
            self.state = np.random.uniform(low=-4, high=4, size=2)
        else:
            self.state = np.array((np.random.choice(np.arange(-4, 4.1, 0.25)), np.random.choice(np.arange(-4, 4.1, 0.25))))
        self.steps = 0
        return self.state

    # TODO: consider having multiple different views of the agent
    # TODO: consider displaying summary statistics alongside agent (i.e: value of state, etc.)
    def _init_render(self):
        """Initializes plot render with circle region and proper axis."""
        goal = plt.Circle((self.circle_center[0], self.circle_center[1]), self.circle_radius, color='blue', fill=False, linewidth=0.5)
        fig, ax = plt.subplots()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.add_artist(goal)
        ax.set_aspect('equal')
        plt.axis([-10 , 10, -10, 10])

    def render(self):
        """Renders the agent at the current timestep on the plot."""
        if not self.render_initialized:
            self._init_render()
            self.render_initialized = True
        x, y = self.state
        if self.agent_render_pos:
            self.agent_render_pos = self.agent_render_pos.pop(0)
            self.agent_render_pos.remove()
        self.agent_render_pos = plt.plot([x], [y], marker='.', color='red')
        plt.pause(0.0001)

    def close(self):
        """Closes the render."""
        if self.render_initialized:
            plt.close()
            self.render_initialized = False
