"""
play.py - Interactive Human vs AI Lunar Lander Playable Experience.

Features:
- Dual-pilot split-screen race: Human (keyboard) vs AI (trained PPO agent) on identical terrain & seed.
- Solo mode: Human pilot practice with real-time aerospace HUD telemetry.
- Real-time 50 FPS physics loop powered by Gymnasium Box2D & Pygame.
- Live telemetry: Altitude, Descent Velocity, Pitch, Fuel usage, and Cumulative Scores.
- Automatic victory adjudication (Safe Touchdown vs Crash vs Score).
"""

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import PPO

from reward_wrapper import make_lunar_lander_env

# Color Palette (Aerospace Mission Control Dark)
COLOR_BG = (10, 15, 26)
COLOR_PANEL = (17, 24, 39)
COLOR_BORDER = (31, 41, 55)
COLOR_TEXT_PRIMARY = (243, 244, 246)
COLOR_TEXT_SECONDARY = (156, 163, 175)
COLOR_TEXT_MUTED = (107, 114, 128)
COLOR_CYAN = (0, 229, 255)
COLOR_GREEN = (16, 185, 129)
COLOR_AMBER = (245, 158, 11)
COLOR_RED = (239, 68, 68)
COLOR_BLUE = (59, 130, 246)
COLOR_PURPLE = (168, 85, 247)


def get_human_action(keys) -> int:
    """
    Translates keyboard state into discrete LunarLander actions:
    0: Coast / No-op
    1: Fire left orientation engine (rotates lander right)
    2: Fire main engine (upward thrust)
    3: Fire right orientation engine (rotates lander left)
    """
    up = keys[pygame.K_UP] or keys[pygame.K_w]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]

    if up:
        return 2
    if left:
        return 1
    if right:
        return 3
    return 0


class LanderDuelGame:
    """
    Manages the split-screen Human vs AI Lunar Lander game loop.
    """

    def __init__(
        self,
        model_path: str,
        mode: str = "duel",
        seed: Optional[int] = None,
        target_fps: int = 50,
    ):
        self.mode = mode
        self.target_fps = target_fps
        self.current_seed = seed if seed is not None else int(time.time()) % 100000

        # Load AI model
        print(f"[*] Loading AI model from: {model_path}")
        if not os.path.exists(model_path) and not os.path.exists(f"{model_path}.zip"):
            print(f"[!] Warning: Model file '{model_path}' not found.")
            if os.path.exists("models/ppo_deep_discrete.zip"):
                model_path = "models/ppo_deep_discrete"
                print(f"[*] Falling back to default: {model_path}")
            elif os.path.exists("models/ppo_default.zip"):
                model_path = "models/ppo_default"
                print(f"[*] Falling back to default: {model_path}")

        self.model = PPO.load(model_path)
        self.model_name = os.path.basename(model_path).replace(".zip", "")

        # Initialize environments
        self.human_env = gym.make("LunarLander-v3", render_mode="rgb_array")
        if self.mode == "duel":
            self.ai_env = gym.make("LunarLander-v3", render_mode="rgb_array")
        else:
            self.ai_env = None

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.font_title = pygame.font.SysFont("DejaVu Sans, Helvetica, Arial", 16, bold=True)
        self.font_data = pygame.font.SysFont("Courier, Monospace", 14, bold=True)
        self.font_hud = pygame.font.SysFont("Courier, Monospace", 13)
        self.font_banner = pygame.font.SysFont("DejaVu Sans, Helvetica, Arial", 20, bold=True)

        if self.mode == "duel":
            self.win_width = 1200
            self.win_height = 540
            self.header_height = 80
            self.footer_height = 60
        else:
            self.win_width = 600
            self.win_height = 540
            self.header_height = 80
            self.footer_height = 60

        self.screen = pygame.display.set_mode((self.win_width, self.win_height))
        pygame.display.set_caption(f"Lunar Lander - Human vs AI ({self.model_name})")
        self.clock = pygame.time.Clock()

        self.reset_match()

    def reset_match(self) -> None:
        """Resets both environments with the exact same initial seed."""
        self.human_obs, _ = self.human_env.reset(seed=self.current_seed)
        self.human_score = 0.0
        self.human_steps = 0
        self.human_done = False
        self.human_outcome = "FLYING"

        if self.mode == "duel":
            self.ai_obs, _ = self.ai_env.reset(seed=self.current_seed)
            self.ai_score = 0.0
            self.ai_steps = 0
            self.ai_done = False
            self.ai_outcome = "FLYING"

        self.paused = False

    def step_human(self, action: int) -> None:
        """Advances the human environment by one step."""
        if not self.human_done:
            self.human_steps += 1
            obs, reward, term, trunc, _ = self.human_env.step(action)
            self.human_score += float(reward)
            self.human_obs = obs
            if term or trunc:
                self.human_done = True
                if reward >= 100:
                    self.human_outcome = "SAFE TOUCHDOWN"
                elif reward <= -100:
                    self.human_outcome = "CRASHED"
                else:
                    self.human_outcome = "TIMEOUT" if trunc else "LANDED"

    def step_ai(self) -> None:
        """Advances the AI environment by one step using the policy network."""
        if self.mode == "duel" and not self.ai_done:
            self.ai_steps += 1
            action, _ = self.model.predict(self.ai_obs, deterministic=True)
            obs, reward, term, trunc, _ = self.ai_env.step(int(action))
            self.ai_score += float(reward)
            self.ai_obs = obs
            if term or trunc:
                self.ai_done = True
                if reward >= 100:
                    self.ai_outcome = "SAFE TOUCHDOWN"
                elif reward <= -100:
                    self.ai_outcome = "CRASHED"
                else:
                    self.ai_outcome = "TIMEOUT" if trunc else "LANDED"

    def render_hud_overlay(
        self,
        surface: pygame.Surface,
        obs: np.ndarray,
        score: float,
        steps: int,
        action: int,
        pilot_label: str,
        pilot_color: Tuple[int, int, int],
        outcome: str,
    ) -> None:
        """Overlays live telemetry indicators onto the given simulation surface."""
        # Telemetry: x, y, vx, vy, angle, v_angular, leg_l, leg_r
        x, y, vx, vy, angle, v_ang, leg_l, leg_r = obs

        alt_m = max(0.0, y * 100.0)
        v_speed = vy * 10.0
        h_drift = vx * 10.0
        pitch_deg = np.degrees(angle)

        # Header tag on the simulation box
        tag = self.font_title.render(f" {pilot_label} ", True, (10, 15, 26), pilot_color)
        surface.blit(tag, (12, 10))

        # Status Pill
        status_color = COLOR_GREEN if "SAFE" in outcome else (COLOR_RED if "CRASH" in outcome else COLOR_CYAN)
        status_surf = self.font_data.render(f"STATUS: {outcome}", True, status_color)
        surface.blit(status_surf, (12, 34))

        # Action Indicator
        action_names = {0: "COAST (0)", 1: "LEFT THRUST (1)", 2: "MAIN THRUST (2)", 3: "RIGHT THRUST (3)"}
        act_color = COLOR_AMBER if action == 2 else (COLOR_CYAN if action in (1, 3) else COLOR_TEXT_MUTED)
        act_surf = self.font_hud.render(f"THRUSTER: {action_names.get(action, 'NONE')}", True, act_color)
        surface.blit(act_surf, (12, 54))

        # Telemetry box on the right
        telemetry_lines = [
            f"ALTITUDE : {alt_m:5.1f} m",
            f"V.SPEED  : {v_speed:+5.2f} m/s",
            f"DRIFT    : {h_drift:+5.2f} m/s",
            f"PITCH    : {pitch_deg:+5.1f} deg",
            f"GEAR L/R : {'[X]' if leg_l > 0.5 else '[ ]'} {'[X]' if leg_r > 0.5 else '[ ]'}",
            f"FRAMES   : {steps:04d}",
            f"SCORE    : {score:+7.1f} pts",
        ]

        box_w = 175
        box_h = len(telemetry_lines) * 16 + 10
        box_x = surface.get_width() - box_w - 10
        box_y = 10

        # Semi-transparent dark background
        hud_bg = pygame.Surface((box_w, box_h))
        hud_bg.set_alpha(190)
        hud_bg.fill((10, 15, 26))
        surface.blit(hud_bg, (box_x, box_y))
        pygame.draw.rect(surface, COLOR_BORDER, (box_x, box_y, box_w, box_h), 1)

        for i, line in enumerate(telemetry_lines):
            line_color = COLOR_GREEN if "SCORE" in line and score >= 200 else COLOR_TEXT_PRIMARY
            txt = self.font_hud.render(line, True, line_color)
            surface.blit(txt, (box_x + 8, box_y + 6 + i * 16))

    def draw(self, human_action: int) -> None:
        """Renders the split-screen viewports, header, and comparison stats."""
        self.screen.fill(COLOR_BG)

        # 1. Header Bar
        pygame.draw.rect(self.screen, COLOR_PANEL, (0, 0, self.win_width, self.header_height))
        pygame.draw.line(self.screen, COLOR_BORDER, (0, self.header_height), (self.win_width, self.header_height), 1)

        title_text = "HUMAN vs AI: LUNAR LANDER RACE" if self.mode == "duel" else "LUNAR LANDER: HUMAN FLIGHT SIMULATOR"
        title_surf = self.font_banner.render(title_text, True, COLOR_CYAN)
        self.screen.blit(title_surf, (20, 12))

        controls_text = "CONTROLS: [↑ / W] Main Engine | [← / A] Left Thruster | [→ / D] Right Thruster | [R] New Seed | [Q] Quit"
        controls_surf = self.font_hud.render(controls_text, True, COLOR_TEXT_SECONDARY)
        self.screen.blit(controls_surf, (20, 44))

        seed_text = f"SEED: {self.current_seed} | AI AGENT: {self.model_name}"
        seed_surf = self.font_hud.render(seed_text, True, COLOR_TEXT_MUTED)
        self.screen.blit(seed_surf, (self.win_width - seed_surf.get_width() - 20, 44))

        # 2. Viewports
        viewport_y = self.header_height

        # Render Human Frame
        human_frame = self.human_env.render()
        human_surf = pygame.image.frombuffer(human_frame.tobytes(), (600, 400), "RGB")
        self.render_hud_overlay(
            human_surf,
            self.human_obs,
            self.human_score,
            self.human_steps,
            human_action,
            "HUMAN PILOT (YOU)",
            COLOR_BLUE,
            self.human_outcome,
        )
        self.screen.blit(human_surf, (0, viewport_y))

        # Render AI Frame (if in duel mode)
        if self.mode == "duel":
            ai_frame = self.ai_env.render()
            ai_surf = pygame.image.frombuffer(ai_frame.tobytes(), (600, 400), "RGB")
            ai_action, _ = self.model.predict(self.ai_obs, deterministic=True)
            self.render_hud_overlay(
                ai_surf,
                self.ai_obs,
                self.ai_score,
                self.ai_steps,
                int(ai_action),
                f"AI PILOT ({self.model_name})",
                COLOR_PURPLE,
                self.ai_outcome,
            )
            self.screen.blit(ai_surf, (600, viewport_y))
            # Divider line between Human and AI viewports
            pygame.draw.line(self.screen, COLOR_CYAN, (600, viewport_y), (600, viewport_y + 400), 2)

        # 3. Footer Comparison Bar
        footer_y = viewport_y + 400
        pygame.draw.rect(self.screen, COLOR_PANEL, (0, footer_y, self.win_width, self.footer_height))
        pygame.draw.line(self.screen, COLOR_BORDER, (0, footer_y), (self.win_width, footer_y), 1)

        if self.mode == "duel":
            score_diff = self.human_score - self.ai_score
            diff_color = COLOR_GREEN if score_diff > 0 else (COLOR_RED if score_diff < 0 else COLOR_TEXT_MUTED)
            diff_str = f"{score_diff:+6.1f} pts"

            # Determine race winner
            if self.human_done and self.ai_done:
                if self.human_outcome == "SAFE TOUCHDOWN" and self.ai_outcome != "SAFE TOUCHDOWN":
                    result_msg = "🏆 VICTORY! Human landed safely while AI failed!"
                    msg_color = COLOR_GREEN
                elif self.ai_outcome == "SAFE TOUCHDOWN" and self.human_outcome != "SAFE TOUCHDOWN":
                    result_msg = "🤖 AI VICTORY! Machine achieved precision touchdown!"
                    msg_color = COLOR_PURPLE
                elif self.human_outcome == "SAFE TOUCHDOWN" and self.ai_outcome == "SAFE TOUCHDOWN":
                    if score_diff > 0:
                        result_msg = f"🏆 HUMAN WINS ON SCORE! (+{score_diff:.1f} pts higher than AI)"
                        msg_color = COLOR_GREEN
                    else:
                        result_msg = f"🤖 AI WINS ON EFFICIENCY! (+{-score_diff:.1f} pts higher than Human)"
                        msg_color = COLOR_PURPLE
                else:
                    result_msg = "💥 BOTH CRASHED! Press [R] to try a new seed."
                    msg_color = COLOR_AMBER
            elif self.human_done and not self.ai_done:
                result_msg = f"Human flight finished ({self.human_outcome}). AI still in descent..."
                msg_color = COLOR_CYAN
            elif not self.human_done and self.ai_done:
                result_msg = f"AI flight finished ({self.ai_outcome}). Pilot the lander to the pad!"
                msg_color = COLOR_AMBER
            else:
                result_msg = "⚡ RACE IN PROGRESS — Touch down gently on the pad between flags!"
                msg_color = COLOR_CYAN

            res_surf = self.font_title.render(result_msg, True, msg_color)
            self.screen.blit(res_surf, (20, footer_y + 18))

            scores_text = f"HUMAN: {self.human_score:+6.1f}  |  AI: {self.ai_score:+6.1f}  |  DELTA: {diff_str}"
            scores_surf = self.font_data.render(scores_text, True, diff_color)
            self.screen.blit(scores_surf, (self.win_width - scores_surf.get_width() - 20, footer_y + 18))
        else:
            solo_msg = f"STATUS: {self.human_outcome} | SCORE: {self.human_score:+6.1f} pts | Press [R] for New Seed"
            solo_color = COLOR_GREEN if "SAFE" in self.human_outcome else COLOR_CYAN
            res_surf = self.font_title.render(solo_msg, True, solo_color)
            self.screen.blit(res_surf, (20, footer_y + 18))

        pygame.display.flip()

    def run(self) -> None:
        """Main interactive game loop running at target FPS."""
        running = True
        print("\n" + "=" * 60)
        print("          HUMAN vs AI LUNAR LANDER DUEL STARTED")
        print("=" * 60)
        print(" Controls:")
        print("   [UP / W]    : Fire Main Rocket Engine (Thrust)")
        print("   [LEFT / A]  : Fire Left Engine (Tilt Right)")
        print("   [RIGHT / D] : Fire Right Engine (Tilt Left)")
        print("   [R]         : Restart Match with New Random Seed")
        print("   [P]         : Pause / Unpause")
        print("   [Q / ESC]   : Quit Match")
        print("=" * 60)

        while running:
            human_action = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_r:
                        self.current_seed = (self.current_seed + 1337) % 100000
                        print(f"[*] Resetting with seed: {self.current_seed}")
                        self.reset_match()
                    elif event.key == pygame.K_p:
                        self.paused = not self.paused

            if not self.paused:
                keys = pygame.key.get_pressed()
                human_action = get_human_action(keys)

                self.step_human(human_action)
                if self.mode == "duel":
                    self.step_ai()

            self.draw(human_action)
            self.clock.tick(self.target_fps)

        self.close()

    def close(self) -> None:
        """Cleans up Gymnasium and Pygame resources."""
        self.human_env.close()
        if self.ai_env is not None:
            self.ai_env.close()
        pygame.quit()
        print("\n[*] Flight duel session closed. Fly again soon!\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive Human vs AI Lunar Lander Duel (Playable Box2D Simulation)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/ppo_deep_discrete",
        help="Path to trained PPO model (default: models/ppo_deep_discrete).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="duel",
        choices=["duel", "solo"],
        help="Game mode: 'duel' (split-screen Human vs AI race) or 'solo' (Human pilot only).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional initial random seed (default: random).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Target frames per second (default: 50).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    game = LanderDuelGame(
        model_path=args.model,
        mode=args.mode,
        seed=args.seed,
        target_fps=args.fps,
    )
    game.run()
