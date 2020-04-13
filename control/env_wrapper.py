# =============================================================================
# Wrapper for the OpenAI gym environment
#
# Author: Anthony G. Chen
# =============================================================================

import numpy as np

import gym
from gym import spaces
from gym import ObservationWrapper


class MiniGridImgWrapper(gym.core.Wrapper):
    def __init__(self, env):
        super(MiniGridImgWrapper, self).__init__(env)

        # Update the observation space
        self.observation_space = self._init_observation_space()

        # TODO set seed

    def _init_observation_space(self):
        """
        Helper method to initialize image observation space with
        shape (channel, image width, image height)
        :return: gym.spaces.Box with the right observation space
        """
        # Get the old observation space
        obs_space = self.env.observation_space
        obs_shape = obs_space.shape

        # Ensure that this is a image (width, height, channel)
        # NOTE: assumed last dimension is channel
        assert len(obs_shape) == 3

        # Shift the shape dimensions around
        new_obs_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[2], obs_shape[0], obs_shape[1]),
            dtype='uint8'
        )

        return new_obs_space

    def reset(self, **kwargs):
        """
        Wraps around the reset function to change observation
        :param kwargs: any input to the reset function
        :return: Modified observation
        """
        obs = self.env.reset(**kwargs)
        obs = np.moveaxis(obs, 2, 0)
        return obs

    def step(self, action):
        """
        Wrapper for the step function
        :param action: action to be taken
        :return: modified obs, reward, done, info
        """
        obs, reward, done, info = self.env.step(action)
        obs = np.moveaxis(obs, 2, 0)
        return obs, reward, done, info
