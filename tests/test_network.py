"""
Unit tests for the High-Definition Deep Neural Guidance Network.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import pytest

from network import ResMLPBlock, DeepGuidanceExtractor, get_deep_policy_kwargs


def test_resmlp_block_forward():
    dim = 64
    block = ResMLPBlock(dim)
    x = torch.randn(8, dim)
    out = block(x)
    assert out.shape == (8, dim)
    # Verify gradient flows back to input
    loss = out.sum()
    loss.backward()
    assert block.fc1.weight.grad is not None


def test_deep_guidance_extractor_forward():
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
    extractor = DeepGuidanceExtractor(obs_space, features_dim=256)

    dummy_obs = torch.randn(16, 8)
    features = extractor(dummy_obs)
    assert features.shape == (16, 256)


def test_deep_policy_kwargs():
    kwargs = get_deep_policy_kwargs()
    assert "pi" in kwargs["net_arch"]
    assert "vf" in kwargs["net_arch"]
    assert kwargs["net_arch"]["pi"] == [256, 256, 128]
    assert kwargs["net_arch"]["vf"] == [256, 256, 128]
    assert kwargs["ortho_init"] is True
