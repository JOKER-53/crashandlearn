#!/usr/bin/env python3
"""
scripts/push_to_hub.py - Publish Trained Policy & Interactive Space to Hugging Face Hub

Uploads the trained PPO policy checkpoints, ONNX WebAssembly model,
publication curves, and Model Card to Hugging Face Hub (Model or Space).
"""

import argparse
import os
import sys
from typing import Optional

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("\n[!] Error: 'huggingface_hub' is not installed in your current Python environment.")
    print("[*] Please activate the virtual environment or run with venv:")
    print("      source venv/bin/activate")
    print("      python scripts/push_to_hub.py ...")
    print("   Or run directly with:")
    print("      venv/bin/python scripts/push_to_hub.py ...\n")
    sys.exit(1)


def generate_model_card(repo_id: str, model_filename: str) -> str:
    """Generates a standardized Hugging Face Model Card with full metadata tags."""
    return fr"""---
tags:
- deep-reinforcement-learning
- reinforcement-learning
- stable-baselines3
- ppo
- gymnasium
- lunar-lander-v3
- onnx
- robotics
model-index:
- name: PPO-Deep-ResMLP-LunarLander-v3
  results:
  - task:
      type: reinforcement-learning
      name: Autonomous Precision Rocket Landing
    dataset:
      name: LunarLander-v3
      type: LunarLander-v3
    metrics:
    - type: mean_reward
      value: 274.75 +/- 20.31
      name: Mean Evaluation Return (10 seeds)
---

# 🚀 PPO Deep Residual Guidance Policy for Gymnasium LunarLander-v3

An empirical deep reinforcement learning policy achieving precision touchdowns on **Gymnasium's LunarLander-v3** environment ($\mu = 274.75 \pm 20.31$ points, **100% landing success**).

Developed as part of the [crashandlearn research lab](https://github.com/JOKER-53/crashandlearn) by **Kadhiravan E. G.**

* **Interactive Live Web Arena & Human vs AI Duel:** [https://joker53.me/crashandlearn/](https://joker53.me/crashandlearn/)
* **GitHub Repository:** [https://github.com/JOKER-53/crashandlearn](https://github.com/JOKER-53/crashandlearn)

---

## 🧠 Neural Architecture: Deep ResMLP

Unlike standard 2-layer MLPs, this policy leverages a **Deep Decoupled Residual MLP**:
* **Actor Head:** `[256] -> [256 (Residual Block)] -> [128] -> 4 discrete thruster logits`
* **Critic Head:** `[256] -> [256 (Residual Block)] -> [128] -> 1 scalar state-value`
* **Activations:** GELU non-linearities with LayerNorm stabilization.
* **Parameters:** ~186,000 trainable weights.

---

## 📊 Empirical Benchmarks

| Policy Model | Architecture | Mean Return (μ ± σ) | Touchdown Rate | Crash Rate |
| :--- | :--- | :---: | :---: | :---: |
| **Deep ResMLP Guidance** | Decoupled [256, 256, 128] | **274.75 ± 20.31** | **100.0%** | **0.0%** |
| **Standard PPO Baseline** | Standard MLP [64, 64] | **277.10 ± 19.90** | **100.0%** | **0.0%** |
| **Reward Hacking (+1.5)** | Standard MLP [64, 64] | 1,480.0 (shaped) | 0.0% (Hover) | 0.0% |
| **Untrained Random** | Uniform Random Actions | -183.17 ± 95.40 | 0.0% | 100.0% |

*The official Gymnasium LunarLander-v3 solved threshold is $\ge 200.0$ points.*

---

## 💻 Quickstart: Load & Run Policy in Python

### 1. Download & Evaluate with Stable-Baselines3

```bash
pip install stable-baselines3 gymnasium[box2d] huggingface_hub
```

```python
from huggingface_hub import hf_hub_download
from stable_baselines3 import PPO
import gymnasium as gym

# Download checkpoint from Hugging Face Hub
model_path = hf_hub_download(repo_id="{repo_id}", filename="{model_filename}")
model = PPO.load(model_path)

# Run test episode
env = gym.make("LunarLander-v3", render_mode="human")
obs, _ = env.reset(seed=42)
done = False
total_reward = 0.0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Episode Completed! Total Score: {{total_reward:.2f}}")
env.close()
```

### 2. Edge / Web Deployment via ONNX

This repository also includes a self-contained **ONNX graph (`ppo_policy.onnx`)** ready for zero-dependency inference in C++, Rust, Go, or client browsers via WebAssembly.

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("ppo_policy.onnx")
# Dummy 8-dimensional observation vector
observation = np.zeros((1, 8), dtype=np.float32)
outputs = session.run(["action_logits"], {{"observation": observation}})
action = np.argmax(outputs[0])
print(f"Selected Thruster Action: {{action}}")
```

---

## 📄 License & Attribution

Released under the **MIT License**. Created by [Kadhiravan E. G.](https://github.com/KADHIRAVANEG).
"""


def validate_auth_and_repo(api: HfApi, token: str, repo_id: str) -> None:
    """Verifies token identity and alerts if repo-id namespace does not match."""
    try:
        user_info = api.whoami(token=token)
        username = user_info.get("name")
        print(f"[*] Authenticated as Hugging Face user: '{username}'")
        expected_prefix = repo_id.split("/")[0] if "/" in repo_id else repo_id
        if username.lower() != expected_prefix.lower():
            print(f"\n[!] Notice: Your token belongs to '{username}', but repo-id namespace is '{expected_prefix}'.")
            print(f"[*] To upload under your personal account, use: --repo-id {username}/{repo_id.split('/')[-1]}\n")
    except Exception as e:
        print(f"[*] Token verification note: {e}")


def push_model_to_hub(
    repo_id: str,
    token: Optional[str] = None,
    model_path: str = "models/ppo_deep_discrete.zip",
    dry_run: bool = False,
) -> None:
    """Publishes policy files, ONNX runtime, and Model Card to Hugging Face Model Hub."""
    token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    model_filename = os.path.basename(model_path)
    card_content = generate_model_card(repo_id, model_filename)

    if dry_run:
        print("[*] DRY RUN: Model card preview generated:")
        print("=" * 60)
        print(card_content[:500] + "\n... [truncated] ...")
        print("=" * 60)
        print(f"[+] Files scheduled for upload to 'https://huggingface.co/{repo_id}':")
        print(f"  - {model_path} -> {model_filename}")
        if os.path.exists("assets/ppo_policy.onnx"):
            print("  - assets/ppo_policy.onnx -> ppo_policy.onnx")
        if os.path.exists("assets/learning_curves.png"):
            print("  - assets/learning_curves.png -> learning_curves.png")
        if os.path.exists("assets/demo_deep.gif"):
            print("  - assets/demo_deep.gif -> demo_deep.gif")
        print("  - README.md (Model Card)")
        return

    if not token:
        print("[!] Error: No Hugging Face token provided. Pass --token or set export HF_TOKEN='hf_...'")
        sys.exit(1)

    validate_auth_and_repo(api, token, repo_id)

    print(f"[*] Creating / Verifying Hugging Face repository: {repo_id}")
    try:
        create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)
    except Exception as e:
        if "403" in str(e) or "Forbidden" in str(e):
            print("\n" + "=" * 72)
            print(" [!] HUGGING FACE PERMISSION ERROR (403 Forbidden)")
            print("=" * 72)
            print(" 1. WRONG USERNAME / NAMESPACE:")
            print("    Check your exact Hugging Face username at: https://huggingface.co/settings/profile")
            print(f"    Ensure --repo-id begins with your HF username: --repo-id <YOUR-HF-USERNAME>/{repo_id.split('/')[-1]}")
            print("\n 2. TOKEN NEEDS 'WRITE' PERMISSION:")
            print("    Tokens created as 'Read' (default) cannot create repositories.")
            print("    Go to: https://huggingface.co/settings/tokens")
            print("    Create a new token with Type set to 'Write' (or check repo creation permissions).")
            print("=" * 72 + "\n")
        sys.exit(1)

    # 1. Upload Model Card
    print("[*] Uploading Model Card (README.md)...")
    api.upload_file(
        path_or_fileobj=card_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    # 2. Upload Checkpoint .zip
    if os.path.exists(model_path):
        print(f"[*] Uploading model checkpoint: {model_path}...")
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=model_filename,
            repo_id=repo_id,
            repo_type="model",
        )

    # 3. Upload ONNX Model
    if os.path.exists("assets/ppo_policy.onnx"):
        print("[*] Uploading ONNX WebAssembly policy: assets/ppo_policy.onnx...")
        api.upload_file(
            path_or_fileobj="assets/ppo_policy.onnx",
            path_in_repo="ppo_policy.onnx",
            repo_id=repo_id,
            repo_type="model",
        )

    # 4. Upload Publication Assets (Learning Curves & Replay GIF)
    for asset, target in [
        ("assets/learning_curves.png", "learning_curves.png"),
        ("assets/demo_deep.gif", "demo_deep.gif"),
    ]:
        if os.path.exists(asset):
            print(f"[*] Uploading publication asset: {asset}...")
            api.upload_file(
                path_or_fileobj=asset,
                path_in_repo=target,
                repo_id=repo_id,
                repo_type="model",
            )

    print(f"\n[✓] Successfully deployed model to Hugging Face: https://huggingface.co/{repo_id}\n")


def push_space_to_hub(
    repo_id: str,
    token: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Publishes the interactive flight duel and telemetry lab as a Hugging Face Space."""
    token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    space_readme = """---
title: Lunar Lander AI vs Human Flight Arena
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: static
pinned: false
---

# Autonomous Rocket Landing Research Lab & Flight Duel
Interactive in-browser WebAssembly flight simulator pitting human pilots against trained PPO neural networks.
"""

    if dry_run:
        print(f"[*] DRY RUN: Space deployment preview for 'https://huggingface.co/spaces/{repo_id}':")
        print("  - README.md (Static Space SDK)")
        print("  - index.html")
        print("  - assets/ (ONNX weights, figures, replay GIFs)")
        return

    if not token:
        print("[!] Error: No Hugging Face token provided. Pass --token or set export HF_TOKEN='hf_...'")
        sys.exit(1)

    validate_auth_and_repo(api, token, repo_id)

    print(f"[*] Creating / Verifying Hugging Face Space: {repo_id}")
    try:
        create_repo(repo_id=repo_id, token=token, repo_type="space", space_sdk="static", exist_ok=True)
    except Exception as e:
        if "403" in str(e) or "Forbidden" in str(e):
            print("\n" + "=" * 72)
            print(" [!] HUGGING FACE PERMISSION ERROR (403 Forbidden)")
            print("=" * 72)
            print(" 1. WRONG USERNAME / NAMESPACE:")
            print("    Check your exact Hugging Face username at: https://huggingface.co/settings/profile")
            print(f"    Ensure --repo-id begins with your HF username: --repo-id <YOUR-HF-USERNAME>/{repo_id.split('/')[-1]}")
            print("\n 2. TOKEN NEEDS 'WRITE' PERMISSION:")
            print("    Tokens created as 'Read' (default) cannot create Space repositories.")
            print("    Go to: https://huggingface.co/settings/tokens")
            print("    Create a new token with Type set to 'Write' (or check repo creation permissions).")
            print("=" * 72 + "\n")
        sys.exit(1)

    print("[*] Uploading Space README.md...")
    api.upload_file(
        path_or_fileobj=space_readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
    )

    print("[*] Uploading index.html to Space...")
    api.upload_file(
        path_or_fileobj="index.html",
        path_in_repo="index.html",
        repo_id=repo_id,
        repo_type="space",
    )

    print("[*] Uploading assets directory...")
    api.upload_folder(
        folder_path="assets",
        path_in_repo="assets",
        repo_id=repo_id,
        repo_type="space",
    )

    print(f"\n[✓] Successfully published Space: https://huggingface.co/spaces/{repo_id}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Publish LunarLander PPO models and interactive spaces to Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repo ID in the format 'username/repo-name'.",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["model", "space"],
        default="model",
        help="Target Hugging Face repository type: 'model' or 'space' (default: model).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/ppo_deep_discrete.zip",
        help="Path to trained model checkpoint (default: models/ppo_deep_discrete.zip).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face access token (or set HF_TOKEN environment variable).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate model card and validate files locally without connecting to network.",
    )
    args = parser.parse_args()

    if args.type == "model":
        push_model_to_hub(
            repo_id=args.repo_id,
            token=args.token,
            model_path=args.model_path,
            dry_run=args.dry_run,
        )
    else:
        push_space_to_hub(
            repo_id=args.repo_id,
            token=args.token,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
