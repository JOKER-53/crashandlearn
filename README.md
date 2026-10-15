# 🚀 lunar-lander-rl

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![RL](https://img.shields.io/badge/Algorithm-PPO-orange?style=flat-square)
![Gymnasium](https://img.shields.io/badge/Gymnasium-LunarLander--v3-purple?style=flat-square)
![Colab](https://img.shields.io/badge/Open%20in-Colab-f9ab00?style=flat-square&logo=googlecolab)

> Training an AI agent to land a rocket using Proximal Policy Optimization (PPO) — with hands-on reward shaping experiments that deliberately break and fix the agent.

---

## 📌 Overview

This project trains a reinforcement learning agent to autonomously land a rocket in OpenAI's Lunar Lander environment (via [Gymnasium](https://gymnasium.farama.org/)). The core focus is **reward shaping** — exploring how the choice of reward function dramatically affects learned behavior, including intentional reward hacking and how to fix it.

The same principles powering this simulation underlie real-world systems. Tesla's Optimus robot reportedly learned to dance entirely through reinforcement learning in simulation.

---

## 🎥 Demo

| Default Reward | Reward Hacking | Fixed Reward |
|:-:|:-:|:-:|
| ![Default Reward](assets/demo_default.gif) | ![Reward Hacking](assets/demo_hack.gif) | ![Fixed Reward](assets/demo_fixed.gif) |
| Agent learns to land | Agent learns to hover | Agent lands again |
| ✅ Solved (~200 pts avg) | ❌ Hovers indefinitely | ✅ Solved again |

> Clips recorded using `gymnasium.wrappers.RecordVideo` and converted to lightweight web GIFs.

---

## ✨ Features

- **PPO training** via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — no GPU needed
- **Custom reward wrapper** — plug-in layer to experiment with shaping rules
- **Reward hacking demo** — intentionally triggers hover behavior with `+1/frame` rule
- **Fix included** — delta-based shaping and per-frame cost to eliminate hacking
- **Evaluation script** — scores over N episodes with video recording
- **Beginner-friendly** — clear code structure, commented throughout

---

## ⚙️ Installation

```bash
# 1. Clone the repo
git clone https://github.com/KADHIRAVANEG/lunar-lander-rl.git
cd lunar-lander-rl

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install swig first (required for Box2D physics)
# Ubuntu/Debian:
sudo apt install swig
# macOS:
brew install swig
# Windows: download from http://www.swig.org/download.html

# 4. Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`**
```
gymnasium[box2d]>=0.29.0
stable-baselines3>=2.0.0
torch>=2.0.0
moviepy>=1.0.3
numpy>=1.24.0
```

---

## 🚀 Usage

### 1. Watch an untrained agent crash

```bash
python random_agent.py
```

### 2. Train the agent (default reward)

```bash
python train.py --timesteps 1000000 --save-path models/ppo_lander
```

Training takes ~20 minutes on CPU. Watch `ep_rew_mean` climb above 200 — that's the solve threshold.

### 3. Evaluate and record

```bash
python evaluate.py --model models/ppo_lander --episodes 20 --record
```

### 4. Trigger reward hacking

```bash
python train.py --reward-mode hack --timesteps 1000000 --save-path models/ppo_hover
python evaluate.py --model models/ppo_hover --record
# Agent will hover — never land
```

### 5. Fix it

```bash
python train.py --reward-mode fixed --timesteps 1000000 --save-path models/ppo_fixed
python evaluate.py --model models/ppo_fixed --record
```

---

## 🧠 Reward Shaping

The environment gives 8 observations per frame:

| # | Observation | Example |
|---|-------------|---------|
| 1 | X position (left/right of pad) | 0.20 |
| 2 | Y position (height) | 0.90 |
| 3 | X velocity | -0.10 |
| 4 | Y velocity | -0.50 |
| 5 | Tilt angle (radians) | 0.05 |
| 6 | Angular velocity | 0.00 |
| 7 | Left leg contact | 0 |
| 8 | Right leg contact | 0 |

### Built-in Reward (default)

```
position_score = -100 * distance - 100 * speed - 100 * tilt + 10 * legs_down
frame_reward   = Δposition_score - fuel_cost
terminal       = +100 (safe landing) or -100 (crash)
```

### Reward Wrapper (`reward_wrapper.py`)

```python
class CustomRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward += self.custom_reward(obs, action, reward)
        return obs, reward, terminated, truncated, info
```

**Mode: `hack`** — adds `+1` per frame regardless of progress. Agent learns to hover (1000 frames × 1pt = 1000pts > landing bonus).

**Mode: `fixed`** — replaces absolute reward with delta reward + small time penalty (`-0.01/frame`). Hovering costs; improvement pays.

---

## 📁 Project Structure

```
lunar-lander-rl/
├── random_agent.py       # Watch untrained random agent crash
├── train.py              # PPO training with reward mode flag
├── evaluate.py           # Score + record evaluation clips
├── reward_wrapper.py     # Custom reward shaping layer
├── models/               # Saved model checkpoints
├── videos/               # Recorded episode clips
├── requirements.txt
└── README.md
```

---

## 📊 Results

| Reward Mode | Avg Score (10 eps) | Landing Rate | Behavior |
|-------------|:------------------:|:------------:|----------|
| **Default** | **~277.10** | **100%** | Lands dead-center consistently |
| **Hack (+1.5/frame)** | **~1,480 (shaped)** | **0%** | Hovers indefinitely for 1,000 frames |
| **Fixed (step penalty + legs)** | **~259.71** | **100%** | Lands safely within ~320 frames |

> Score ≥ 200 is the official "solved" threshold for LunarLander-v3. Both Default and Fixed modes achieve 100% safe landings.

---

## 🐛 Common Issues

| Problem | Fix |
|---------|-----|
| `swig` install error during `box2d` install | Install `swig` system-wide first (see Installation) |
| `LunarLander-v2` not found | Use `v3` — Gymnasium dropped v2 |
| Agent hovers and never lands | You triggered reward hacking — run `--reward-mode fixed` |
| Training result varies wildly | Normal — run 2–3 seeds before judging a reward function |

---

## 📚 References

- [Gymnasium: LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [PPO Paper — Schulman et al. 2017](https://arxiv.org/abs/1707.06347)
- [Hugging Face Deep RL Course, Unit 1](https://huggingface.co/learn/deep-rl-course/unit1/introduction)

---

## 📄 License

MIT — free to use, fork, and build on.

---

> Built by [KADHIR](https://github.com/KADHIRAVANEG) · Contributions welcome
