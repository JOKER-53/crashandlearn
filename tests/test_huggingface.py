"""
tests/test_huggingface.py - Unit tests for Hugging Face publishing script and model card generator.
"""

import pytest
from scripts.push_to_hub import generate_model_card, push_model_to_hub, push_space_to_hub


def test_generate_model_card_tags():
    card = generate_model_card(repo_id="testuser/test-model", model_filename="ppo_deep_discrete.zip")
    assert "tags:" in card
    assert "deep-reinforcement-learning" in card
    assert "stable-baselines3" in card
    assert "LunarLander-v3" in card
    assert "model-index:" in card
    assert "274.75" in card
    assert "testuser/test-model" in card
    assert "ppo_deep_discrete.zip" in card


def test_push_model_dry_run(capsys):
    push_model_to_hub(repo_id="mock/repo", dry_run=True)
    captured = capsys.readouterr()
    assert "DRY RUN: Model card preview generated" in captured.out
    assert "ppo_policy.onnx" in captured.out


def test_push_space_dry_run(capsys):
    push_space_to_hub(repo_id="mock/space", dry_run=True)
    captured = capsys.readouterr()
    assert "DRY RUN: Space deployment preview" in captured.out
    assert "index.html" in captured.out
