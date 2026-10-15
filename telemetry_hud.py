"""
telemetry_hud.py - Flight Telemetry HUD Overlay for Lunar Lander Video Recordings.

Adds a mission-control aerospace Heads-Up Display (HUD) directly onto video frames:
- Flight mode indicator (Default, Reward Hacking, Fixed)
- Altitude, vertical speed, horizontal drift, and pitch angle
- Real-time cumulative score and step ticker
- Landing gear touchdown status indicators
- Terminal outcome banner (Safe Landing, Hover Timeout, Crash)
"""

import math
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class TelemetryHUD:
    """
    Renders aerospace telemetry overlays onto RGB video frames.
    """

    def __init__(self, mode: str = "default"):
        self.mode = mode.lower()
        # Fallback to default bitmap font if custom TTF is not available
        try:
            self.font_large = ImageFont.load_default(size=14)
            self.font_small = ImageFont.load_default(size=11)
            self.font_bold = ImageFont.load_default(size=13)
        except TypeError:
            self.font_large = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
            self.font_bold = ImageFont.load_default()

    def draw_hud(
        self,
        frame: np.ndarray,
        obs: np.ndarray,
        step: int,
        cumulative_reward: float,
        terminal_status: Optional[str] = None,
    ) -> np.ndarray:
        """
        Draws telemetry onto an RGB numpy array (H, W, 3) and returns modified frame.
        """
        h, w, c = frame.shape
        img = Image.fromarray(frame).convert("RGBA")
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Parse observation state:
        # [x, y, vx, vy, angle, angular_vel, left_leg, right_leg]
        x = float(obs[0])
        y = float(obs[1])
        vx = float(obs[2])
        vy = float(obs[3])
        angle = float(obs[4])
        left_leg = bool(obs[6])
        right_leg = bool(obs[7])

        altitude_m = max(0.0, y * 50.0)
        v_speed = vy * 20.0
        h_speed = vx * 20.0
        pitch_deg = angle * 180.0 / math.pi

        # ----------------------------------------------------------------------
        # 1. TOP HEADER HUD (Translucent Glass Bar)
        # ----------------------------------------------------------------------
        top_bar_h = 36
        draw.rectangle([(0, 0), (w, top_bar_h)], fill=(10, 15, 25, 210))
        draw.line([(0, top_bar_h), (w, top_bar_h)], fill=(40, 70, 110, 255), width=1)

        # Mode Tag & Color
        if self.mode == "hack":
            mode_badge = "REWARD HACK [+1.5/FRAME]"
            badge_color = (255, 110, 60, 255)  # Neon Orange / Warning
        elif self.mode == "fixed":
            mode_badge = "FIXED SHAPING [STEP PENALTY]"
            badge_color = (60, 230, 160, 255)  # Neon Emerald
        else:
            mode_badge = "DEFAULT SOLVER [PPO]"
            badge_color = (0, 210, 255, 255)   # Neon Cyan

        draw.text((12, 10), mode_badge, fill=badge_color, font=self.font_bold)

        # Step Ticker and Score (Top Right)
        score_text = f"PTS: {cumulative_reward:+7.1f}"
        step_text = f"T+ {step:03d} / 1000"
        score_color = (80, 240, 120, 255) if cumulative_reward >= 0 else (255, 90, 90, 255)

        draw.text((w - 230, 10), score_text, fill=score_color, font=self.font_bold)
        draw.text((w - 100, 10), step_text, fill=(180, 205, 230, 240), font=self.font_small)

        # ----------------------------------------------------------------------
        # 2. BOTTOM TELEMETRY BAR
        # ----------------------------------------------------------------------
        bot_bar_h = 34
        bot_y = h - bot_bar_h
        draw.rectangle([(0, bot_y), (w, h)], fill=(10, 15, 25, 210))
        draw.line([(0, bot_y), (w, bot_y)], fill=(40, 70, 110, 255), width=1)

        # Telemetry Metrics
        v_speed_color = (80, 240, 120, 255) if abs(v_speed) < 3.0 else (255, 80, 80, 255)
        pitch_color = (80, 240, 120, 255) if abs(pitch_deg) < 15.0 else (255, 180, 50, 255)

        telemetry_str = f"ALT: {altitude_m:4.1f}m   VY: {v_speed:+4.1f} m/s   VX: {h_speed:+4.1f} m/s   PITCH: {pitch_deg:+4.1f}°"
        draw.text((12, bot_y + 9), telemetry_str, fill=(200, 220, 240, 255), font=self.font_small)

        # Gear Touchdown Indicators
        gear_l_color = (80, 255, 120, 255) if left_leg else (100, 120, 140, 180)
        gear_r_color = (80, 255, 120, 255) if right_leg else (100, 120, 140, 180)
        draw.text((w - 140, bot_y + 9), "GEAR:", fill=(160, 180, 200, 255), font=self.font_small)
        draw.text((w - 95, bot_y + 9), "[L]", fill=gear_l_color, font=self.font_bold)
        draw.text((w - 65, bot_y + 9), "[R]", fill=gear_r_color, font=self.font_bold)

        # ----------------------------------------------------------------------
        # 3. TERMINAL OUTCOME OVERLAY BANNER
        # ----------------------------------------------------------------------
        if terminal_status:
            banner_w = 340
            banner_h = 44
            bx1 = (w - banner_w) // 2
            by1 = (h - banner_h) // 2 - 20
            bx2 = bx1 + banner_w
            by2 = by1 + banner_h

            if "SAFE" in terminal_status or "LANDED" in terminal_status or "TOUCHDOWN" in terminal_status:
                bg = (10, 60, 30, 230)
                border = (80, 255, 120, 255)
                text = "✓ TOUCHDOWN CONFIRMED"
                tc = (100, 255, 150, 255)
            elif "HOVER" in terminal_status or "TIMEOUT" in terminal_status:
                bg = (70, 45, 10, 230)
                border = (255, 180, 50, 255)
                text = "⚠ REWARD HACK: HOVER TIMEOUT"
                tc = (255, 210, 90, 255)
            else:
                bg = (70, 15, 15, 230)
                border = (255, 60, 60, 255)
                text = "✗ CRASH DETECTED"
                tc = (255, 120, 120, 255)

            draw.rectangle([(bx1, by1), (bx2, by2)], fill=bg, outline=border, width=2)
            draw.text((bx1 + 25, by1 + 14), text, fill=tc, font=self.font_bold)

        # Merge overlay onto base frame
        combined = Image.alpha_composite(img, overlay).convert("RGB")
        return np.array(combined)
