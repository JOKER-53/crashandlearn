#!/usr/bin/env python3
"""
stress_test.py - Stochastic Robustness & Environmental Perturbation Stress Testing

Benchmarks trained LunarLander PPO policies under severe stochastic stressors:
1. Crosswind & Aerodynamic Turbulence: Native Box2D wind and rotational turbulence
2. Sensor Jitter (State Noise): Gaussian noise added to telemetry observations
3. Actuator Degradation: Random thruster misfires and thrust attenuation
4. Extreme Storm: Compound multi-axis stressor combining all perturbations
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


class SensorNoiseWrapper(gym.ObservationWrapper):
    """Adds zero-mean Gaussian noise to telemetry observations simulating noisy IMU/radar."""

    def __init__(self, env: gym.Env, noise_std: float = 0.05):
        super().__init__(env)
        self.noise_std = noise_std

    def observation(self, observation: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0.0, self.noise_std, size=observation.shape).astype(np.float32)
        # Keep leg contact booleans binary (indices 6 and 7)
        noise[6:] = 0.0
        return np.clip(observation + noise, -5.0, 5.0)


class ActuatorDegradationWrapper(gym.ActionWrapper):
    """Simulates mechanical actuator faults (power attenuation and intermittent misfire)."""

    def __init__(self, env: gym.Env, power_multiplier: float = 0.85, misfire_prob: float = 0.05):
        super().__init__(env)
        self.power_multiplier = power_multiplier
        self.misfire_prob = misfire_prob
        self.is_continuous = isinstance(env.action_space, gym.spaces.Box)

    def action(self, action: Any) -> Any:
        # Intermittent misfire: engine command is dropped to no-op
        if np.random.random() < self.misfire_prob:
            if self.is_continuous:
                return np.zeros_like(action)
            return 0  # No-op discrete

        if self.is_continuous:
            return np.clip(action * self.power_multiplier, -1.0, 1.0)
        return action


def create_stressed_env(
    profile: str,
    continuous: bool = False,
    render_mode: Optional[str] = None,
) -> gym.Env:
    """Builds a LunarLander environment configured for the selected stress profile."""
    if profile == "nominal":
        env = gym.make("LunarLander-v3", continuous=continuous, render_mode=render_mode)
    elif profile == "wind":
        env = gym.make(
            "LunarLander-v3",
            continuous=continuous,
            render_mode=render_mode,
            enable_wind=True,
            wind_power=15.0,
            turbulence_power=1.5,
        )
    elif profile == "sensor_noise":
        env = gym.make("LunarLander-v3", continuous=continuous, render_mode=render_mode)
        env = SensorNoiseWrapper(env, noise_std=0.06)
    elif profile == "engine_degradation":
        env = gym.make("LunarLander-v3", continuous=continuous, render_mode=render_mode)
        env = ActuatorDegradationWrapper(env, power_multiplier=0.80, misfire_prob=0.08)
    elif profile == "extreme":
        env = gym.make(
            "LunarLander-v3",
            continuous=continuous,
            render_mode=render_mode,
            enable_wind=True,
            wind_power=20.0,
            turbulence_power=2.0,
        )
        env = SensorNoiseWrapper(env, noise_std=0.08)
        env = ActuatorDegradationWrapper(env, power_multiplier=0.75, misfire_prob=0.10)
    else:
        raise ValueError(f"Unknown stress profile: {profile}")

    return env


def benchmark_policy(
    model_path: str,
    profile: str,
    episodes: int = 20,
    seed: int = 42,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """Runs N evaluation episodes on the given model and stress profile."""
    if os.path.exists(model_path) or os.path.exists(model_path + ".zip"):
        if not model_path.endswith(".zip") and os.path.exists(model_path + ".zip"):
            model_path = model_path + ".zip"
        model = PPO.load(model_path)
    else:
        print(f"[!] Warning: Model '{model_path}' not found on disk. Testing with baseline PPO.")
        dummy_env = gym.make("LunarLander-v3")
        model = PPO("MlpPolicy", dummy_env, n_steps=64)
        dummy_env.close()
    is_continuous = isinstance(model.action_space, gym.spaces.Box)
    env = create_stressed_env(profile=profile, continuous=is_continuous)

    rewards = []
    successes = 0
    crashes = 0
    timeouts = 0

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            steps += 1
            done = terminated or truncated

        rewards.append(ep_reward)
        if ep_reward >= 200.0:
            successes += 1
        elif steps >= 1000:
            timeouts += 1
        else:
            crashes += 1

    env.close()

    return {
        "model": os.path.basename(model_path),
        "profile": profile,
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(successes / episodes * 100.0),
        "crash_rate": float(crashes / episodes * 100.0),
        "timeout_rate": float(timeouts / episodes * 100.0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run stochastic environmental stress tests on trained PPO policies."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["models/ppo_deep_discrete.zip", "models/ppo_default.zip"],
        help="List of model files to benchmark.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["nominal", "wind", "sensor_noise", "engine_degradation", "extreme"],
        choices=["nominal", "wind", "sensor_noise", "engine_degradation", "extreme"],
        help="Stress profiles to test against.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Episodes per scenario (default: 20).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write benchmark results as JSON.",
    )
    args = parser.parse_args()

    results = []
    print("\n" + "=" * 80)
    print(" LUNAR LANDER STOCHASTIC ROBUSTNESS STRESS TEST BENCHMARK")
    print("=" * 80)
    print(f"{'Model':<25} | {'Profile':<18} | {'Mean ± Std':<18} | {'Success':<8} | {'Crash':<8}")
    print("-" * 80)

    for model_path in args.models:
        if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
            print(f"[!] Warning: Skipping non-existent model: {model_path}")
            continue

        for profile in args.profiles:
            res = benchmark_policy(model_path, profile, episodes=args.episodes)
            results.append(res)
            score_str = f"{res['mean_reward']:6.1f} ± {res['std_reward']:<5.1f}"
            print(
                f"{res['model']:<25} | {res['profile']:<18} | {score_str:<18} | {res['success_rate']:5.1f}% | {res['crash_rate']:5.1f}%"
            )

    print("=" * 80 + "\n")

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[+] Benchmark results exported to: {args.output_json}")


if __name__ == "__main__":
    main()
