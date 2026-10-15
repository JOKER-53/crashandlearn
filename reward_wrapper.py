"""
Custom Reward Wrapper for LunarLander-v3.

Demonstrates reward shaping dynamics and intentional reward hacking (specification gaming):
- 'default': Standard Gymnasium LunarLander reward function.
- 'hack': Adds a flat survival reward (+1.0/frame) regardless of progress.
          The agent discovers that hovering for 1000 steps yields up to +1000 reward,
          far exceeding the ~200 landing bonus, causing it to never land.
- 'fixed': Replaces the survival incentive with delta-based progress shaping and
          a step penalty (-0.05/frame), making hovering costly and guiding the lander safely down.
"""

from typing import Any, Dict, Optional, Tuple
import math
import gymnasium as gym
import numpy as np


class CustomRewardWrapper(gym.Wrapper):
    """
    Wraps LunarLander-v3 to modify reward signals for research experiments.
    """

    def __init__(
        self,
        env: gym.Env,
        reward_mode: str = "default",
        step_penalty: float = 0.05,
    ) -> None:
        super().__init__(env)
        self.reward_mode = reward_mode.lower()
        self.step_penalty = step_penalty
        self.prev_shaping: Optional[float] = None

        valid_modes = {"default", "hack", "fixed"}
        if self.reward_mode not in valid_modes:
            raise ValueError(
                f"Invalid reward_mode '{reward_mode}'. Choose from: {sorted(valid_modes)}"
            )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.prev_shaping = self._calculate_shaping(obs)
        return obs, info

    def _calculate_shaping(self, obs: np.ndarray) -> float:
        """
        Potential-based shaping score based on distance, velocity, and angle.
        Observation layout:
          obs[0]: x position
          obs[1]: y position
          obs[2]: x velocity
          obs[3]: y velocity
          obs[4]: angle
          obs[5]: angular velocity
          obs[6]: left leg contact (0 or 1)
          obs[7]: right leg contact (0 or 1)
        """
        x, y, vx, vy, angle, _, leg_left, leg_right = obs[:8]
        dist = math.sqrt(x * x + y * y)
        speed = math.sqrt(vx * vx + vy * vy)
        tilt = abs(angle)

        # Higher score the closer the lander is to pad center, slow, and upright
        score = -100.0 * dist - 100.0 * speed - 100.0 * tilt + 10.0 * leg_left + 10.0 * leg_right
        return score

    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = reward

        if self.reward_mode == "hack":
            # Survival bonus exploit: +1.0 per frame regardless of progress
            # Over 1000 frames, hovering yields ~1000 pts without risking landing
            shaped_reward += 1.0

        elif self.reward_mode == "fixed":
            # Fixed shaping:
            # 1. Delta reward from progress towards target pad
            # 2. Time penalty per step to eliminate hover incentives
            current_shaping = self._calculate_shaping(obs)
            delta_shaping = 0.0
            if self.prev_shaping is not None:
                delta_shaping = current_shaping - self.prev_shaping
            self.prev_shaping = current_shaping

            # Base environment already awards +100 for landing and -100 for crash on termination
            if terminated:
                # Retain clean terminal bonus/penalty
                shaped_reward = reward
            else:
                shaped_reward = delta_shaping - self.step_penalty

        info["original_reward"] = reward
        info["shaped_reward"] = shaped_reward
        info["reward_mode"] = self.reward_mode

        return obs, shaped_reward, terminated, truncated, info


def make_lunar_lander_env(
    reward_mode: str = "default",
    render_mode: Optional[str] = None,
    step_penalty: float = 0.05,
) -> gym.Env:
    """
    Factory helper to create a configured LunarLander-v3 environment.
    """
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    if reward_mode != "default":
        env = CustomRewardWrapper(
            env, reward_mode=reward_mode, step_penalty=step_penalty
        )
    return env
