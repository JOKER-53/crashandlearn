"""
Unit tests for play.py - Human vs AI Lunar Lander Playable Experience.
"""

import os
import pygame
import pytest
from unittest.mock import MagicMock

from play import get_human_action, LanderDuelGame


def test_get_human_action_mapping():
    # Mock keys dict
    keys = {
        pygame.K_UP: False,
        pygame.K_w: False,
        pygame.K_LEFT: False,
        pygame.K_a: False,
        pygame.K_RIGHT: False,
        pygame.K_d: False,
    }

    # Default should be coast (0)
    assert get_human_action(keys) == 0

    # UP or W -> 2 (Main engine)
    keys[pygame.K_UP] = True
    assert get_human_action(keys) == 2
    keys[pygame.K_UP] = False
    keys[pygame.K_w] = True
    assert get_human_action(keys) == 2
    keys[pygame.K_w] = False

    # LEFT or A -> 1 (Left orientation engine)
    keys[pygame.K_LEFT] = True
    assert get_human_action(keys) == 1
    keys[pygame.K_LEFT] = False
    keys[pygame.K_a] = True
    assert get_human_action(keys) == 1
    keys[pygame.K_a] = False

    # RIGHT or D -> 3 (Right orientation engine)
    keys[pygame.K_RIGHT] = True
    assert get_human_action(keys) == 3
    keys[pygame.K_RIGHT] = False
    keys[pygame.K_d] = True
    assert get_human_action(keys) == 3


def test_lander_duel_game_step():
    # Test headless duel game step logic
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    game = LanderDuelGame(model_path="models/ppo_default", mode="duel", seed=42)

    try:
        assert game.human_steps == 0
        assert game.ai_steps == 0
        assert not game.human_done
        assert not game.ai_done

        # Step human with main engine thrust (action 2)
        game.step_human(2)
        assert game.human_steps == 1
        assert len(game.human_obs) == 8

        # Step AI with policy network
        game.step_ai()
        assert game.ai_steps == 1
        assert len(game.ai_obs) == 8
    finally:
        game.close()


def test_lander_duel_game_wind():
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    game = LanderDuelGame(model_path="models/ppo_default", mode="duel", seed=123, enable_wind=True, wind_power=20.0)
    try:
        assert game.enable_wind is True
        assert game.wind_power == 20.0
        game.step_human(1)
        game.step_ai()
        assert game.human_steps == 1
        assert game.ai_steps == 1
    finally:
        game.close()

