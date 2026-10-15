#!/usr/bin/env python3
"""
random_agent.py - Watch an untrained agent interact with LunarLander-v3.

Demonstrates the baseline behavior: without training, random engine firings
lead almost inevitably to high-velocity crashes and negative scores (-100 to -400).
"""

import argparse
import sys
import gymnasium as gym


def run_random_agent(episodes: int = 5, render: bool = True, seed: int = 42) -> None:
    render_mode = "human" if render else None
    print(f"[*] Initializing LunarLander-v3 (render_mode={render_mode}, episodes={episodes})...")

    try:
        env = gym.make("LunarLander-v3", render_mode=render_mode)
    except Exception as exc:
        if render and "human" in str(exc).lower():
            print("[!] Display not available for human rendering. Falling back to headless mode.")
            env = gym.make("LunarLander-v3", render_mode=None)
        else:
            raise exc

    scores = []
    print("\n--- Running Random Agent ---")
    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=seed + ep if seed is not None else None)
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = env.action_space.sample()  # Random action [0: nothing, 1: left, 2: main, 3: right]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        scores.append(total_reward)
        outcome = "CRASHED" if terminated and total_reward < 0 else ("LANDED" if terminated else "TRUNCATED")
        print(f"Episode {ep:02d} | Steps: {steps:03d} | Total Score: {total_reward:7.2f} | Outcome: {outcome}")

    env.close()
    avg_score = sum(scores) / len(scores)
    print("----------------------------")
    print(f"Average Score across {episodes} episodes: {avg_score:.2f} (Untrained baseline)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an untrained random agent in LunarLander-v3")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run (default: 5)")
    parser.add_argument("--no-render", action="store_true", help="Disable graphical rendering")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    run_random_agent(episodes=args.episodes, render=not args.no_render, seed=args.seed)


if __name__ == "__main__":
    main()
