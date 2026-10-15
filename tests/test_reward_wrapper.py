"""
Unit tests for CustomRewardWrapper and environment creation.
"""

import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest

from reward_wrapper import CustomRewardWrapper, make_lunar_lander_env


class DummyEnv(gym.Env):
    """Mock environment simulating LunarLander-v3 for fast unit testing."""
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

    def reset(self, *, seed=None, options=None):
        obs = np.array([0.5, 0.8, -0.1, -0.2, 0.05, 0.01, 0.0, 0.0], dtype=np.float32)
        return obs, {}

    def step(self, action):
        obs = np.array([0.4, 0.7, -0.05, -0.1, 0.02, 0.0, 0.0, 0.0], dtype=np.float32)
        base_reward = 1.5
        terminated = False
        truncated = False
        info = {}
        return obs, base_reward, terminated, truncated, info


def test_invalid_reward_mode():
    dummy = DummyEnv()
    with pytest.raises(ValueError, match="Invalid reward_mode"):
        CustomRewardWrapper(dummy, reward_mode="non_existent")


def test_hack_mode_adds_survival_bonus():
    dummy = DummyEnv()
    wrapped = CustomRewardWrapper(dummy, reward_mode="hack")
    obs, info = wrapped.reset()
    _, reward, _, _, info = wrapped.step(0)

    # Base reward is 1.5, hack mode adds +1.5
    assert pytest.approx(reward, 0.001) == 3.0
    assert info["original_reward"] == 1.5
    assert info["shaped_reward"] == 3.0
    assert info["reward_mode"] == "hack"


def test_fixed_mode_step_penalty_and_leg_bonus():
    dummy = DummyEnv()
    step_penalty = 0.05
    wrapped = CustomRewardWrapper(dummy, reward_mode="fixed", step_penalty=step_penalty)
    obs, info = wrapped.reset()

    obs2, reward, terminated, truncated, info = wrapped.step(0)
    # Base reward is 1.5, step penalty is 0.05, legs are [0, 0] so leg_bonus is 0
    expected_reward = 1.5 - 0.05
    assert pytest.approx(reward, 0.001) == expected_reward
    assert info["reward_mode"] == "fixed"


def test_shaping_calculation_logic():
    dummy = DummyEnv()
    wrapped = CustomRewardWrapper(dummy, reward_mode="fixed")
    obs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    score = wrapped._calculate_shaping(obs)
    # distance=0, speed=0, tilt=0, legs=1+1 => score should be 20.0
    assert pytest.approx(score, 0.001) == 20.0


def test_make_lunar_lander_env_default():
    env = make_lunar_lander_env(reward_mode="default")
    assert env is not None
    obs, info = env.reset(seed=42)
    assert len(obs) == 8
    env.close()


def test_make_lunar_lander_env_hack():
    env = make_lunar_lander_env(reward_mode="hack")
    assert isinstance(env, CustomRewardWrapper)
    assert env.reward_mode == "hack"
    obs, info = env.reset(seed=42)
    obs, reward, term, trunc, info = env.step(0)
    assert "shaped_reward" in info
    env.close()
