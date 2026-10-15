#!/usr/bin/env python3
"""
evaluate.py - Benchmark and record trained LunarLander agents.

Evaluates an agent over N episodes, reporting:
- Mean and standard deviation of episode rewards
- Success rate (score >= 200 solved threshold)
- Crash rate vs. Hover/Timeout rate (1000 steps reached)
- High-definition MP4 video recordings using RecordVideo
"""

import argparse
import os
import sys
from typing import List, Tuple
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from reward_wrapper import make_lunar_lander_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and record a trained PPO agent on LunarLander-v3."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model .zip (e.g. models/ppo_default.zip or models/ppo_default).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes (default: 20).",
    )
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="default",
        choices=["default", "hack", "fixed"],
        help="Reward mode wrapper to apply during evaluation (default: raw env score).",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record episode playthroughs as MP4 videos.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="videos",
        help="Output directory for recorded video clips (default: videos/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Base random seed for reproducible evaluation episodes.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy actions during evaluation.",
    )
    return parser.parse_args()


def evaluate_agent(
    model_path: str,
    episodes: int = 20,
    reward_mode: str = "default",
    record: bool = False,
    video_dir: str = "videos",
    seed: int = 1337,
    deterministic: bool = True,
) -> None:
    if not model_path.endswith(".zip") and os.path.exists(model_path + ".zip"):
        model_path = model_path + ".zip"

    if not os.path.isfile(model_path):
        print(f"[!] Error: Model file not found at '{model_path}'")
        sys.exit(1)

    print(f"[*] Loading PPO model from: {model_path}")
    model = PPO.load(model_path)

    render_mode = "rgb_array" if record else None
    env = make_lunar_lander_env(reward_mode=reward_mode, render_mode=render_mode)

    if record:
        os.makedirs(video_dir, exist_ok=True)
        # Record every episode requested
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda ep: True,
            name_prefix=f"eval_{os.path.basename(model_path).replace('.zip', '')}",
        )
        print(f"[*] Recording enabled. Saving video clips to '{video_dir}/'")

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    landed_count = 0
    crashed_count = 0
    hover_timeout_count = 0

    print(f"\n[*] Commencing evaluation over {episodes} episodes (Seed: {seed})...\n")
    print(f"{'Ep':>3} | {'Steps':>5} | {'Score':>8} | {'Outcome':<15}")
    print("-" * 38)

    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        # Categorize outcome
        if steps >= 1000:
            outcome = "HOVER / TIMEOUT"
            hover_timeout_count += 1
        elif total_reward >= 200.0:
            outcome = "SAFE LANDING"
            landed_count += 1
        elif total_reward < 0.0:
            outcome = "CRASHED"
            crashed_count += 1
        else:
            outcome = "TOUCHDOWN"
            landed_count += 1

        print(f"{ep:3d} | {steps:5d} | {total_reward:8.2f} | {outcome:<15}")

    env.close()

    mean_score = float(np.mean(episode_rewards))
    std_score = float(np.std(episode_rewards))
    min_score = float(np.min(episode_rewards))
    max_score = float(np.max(episode_rewards))
    avg_steps = float(np.mean(episode_lengths))

    print("\n" + "=" * 45)
    print("           EVALUATION SUMMARY                ")
    print("=" * 45)
    print(f" Episodes Evaluated : {episodes}")
    print(f" Average Score      : {mean_score:6.2f} +/- {std_score:.2f}")
    print(f" Min / Max Score    : {min_score:.2f} / {max_score:.2f}")
    print(f" Average Steps      : {avg_steps:.1f} frames")
    print(f" Safe Landings      : {landed_count}/{episodes} ({100*landed_count/episodes:.1f}%)")
    print(f" Crashes            : {crashed_count}/{episodes} ({100*crashed_count/episodes:.1f}%)")
    print(f" Hover / Timeouts   : {hover_timeout_count}/{episodes} ({100*hover_timeout_count/episodes:.1f}%)")
    print(f" Solved (Score>=200): {'YES (PASSED)' if mean_score >= 200 else 'NO (BELOW 200)'}")
    print("=" * 45)


def main() -> None:
    args = parse_args()
    evaluate_agent(
        model_path=args.model,
        episodes=args.episodes,
        reward_mode=args.reward_mode,
        record=args.record,
        video_dir=args.video_dir,
        seed=args.seed,
        deterministic=args.deterministic,
    )


if __name__ == "__main__":
    main()
