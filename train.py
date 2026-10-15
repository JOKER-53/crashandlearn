#!/usr/bin/env python3
"""
train.py - Train a PPO agent on LunarLander-v3 with customizable reward modes.

Supported Modes:
  --reward-mode default : Train with standard environment reward (solves lander ~200+ pts)
  --reward-mode hack    : Train with +1/frame survival reward (induces hover gaming)
  --reward-mode fixed   : Train with delta shaping + time penalty (restores safe landing)
"""

import argparse
import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from reward_wrapper import make_lunar_lander_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on LunarLander-v3 with reward shaping experiments."
    )
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="default",
        choices=["default", "hack", "fixed"],
        help="Reward shaping configuration to train on (default, hack, fixed).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total environment interaction timesteps (default: 500000).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Destination path for trained model checkpoint (e.g. models/ppo_lander).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for Adam optimizer (default: 0.0003).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default="./runs",
        help="Directory to write TensorBoard telemetry logs (default: ./runs).",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    reward_mode = args.reward_mode.lower()

    if args.save_path is None:
        save_path = os.path.join("models", f"ppo_{reward_mode}")
    else:
        save_path = args.save_path

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    os.makedirs(args.tensorboard_log, exist_ok=True)

    print("==================================================")
    print("           LUNAR LANDER PPO TRAINING              ")
    print("==================================================")
    print(f" Reward Mode    : {reward_mode}")
    print(f" Total Timesteps: {args.timesteps:,}")
    print(f" Random Seed    : {args.seed}")
    print(f" Learning Rate  : {args.learning_rate}")
    print(f" Checkpoint Path: {save_path}.zip")
    print(f" Tensorboard Log: {args.tensorboard_log}")
    print("==================================================")

    def env_fn():
        env = make_lunar_lander_env(reward_mode=reward_mode)
        return Monitor(env)

    vec_env = DummyVecEnv([env_fn])

    # Check if tensorboard is available
    tb_log = args.tensorboard_log
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        print("[!] Note: tensorboard not installed. Disabling TensorBoard logging.")
        tb_log = None

    # Standard optimized PPO hyperparameters for LunarLander
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=args.learning_rate,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1,
        seed=args.seed,
        tensorboard_log=tb_log,
    )

    print("\n[*] Starting training loop. Monitor ep_rew_mean in console logs...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            tb_log_name=f"ppo_{reward_mode}",
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user. Saving current checkpoint...")

    model.save(save_path)
    print(f"\n[+] Model successfully saved to {save_path}.zip")
    vec_env.close()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
