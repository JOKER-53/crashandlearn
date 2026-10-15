"""
tests/test_engineering.py - Unit tests for stress testing wrappers, environment profiles, and ONNX artifacts.
"""

import os
import pytest
import numpy as np
import gymnasium as gym

from stress_test import SensorNoiseWrapper, ActuatorDegradationWrapper, create_stressed_env


def test_sensor_noise_wrapper():
    base_env = gym.make("LunarLander-v3")
    wrapped_env = SensorNoiseWrapper(base_env, noise_std=0.1)

    obs, _ = wrapped_env.reset(seed=42)
    assert obs.shape == (8,)
    assert obs.dtype == np.float32

    # Verify that observation modification works and leg flags remain binary
    dummy_obs = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    perturbed = wrapped_env.observation(dummy_obs.copy())
    assert perturbed.shape == (8,)
    assert perturbed[6] == 1.0
    assert perturbed[7] == 0.0
    wrapped_env.close()


def test_actuator_degradation_wrapper():
    base_env = gym.make("LunarLander-v3", continuous=True)
    wrapped_env = ActuatorDegradationWrapper(base_env, power_multiplier=0.8, misfire_prob=0.0)

    action = np.array([1.0, 0.5], dtype=np.float32)
    processed = wrapped_env.action(action)
    np.testing.assert_allclose(processed, [0.8, 0.4], rtol=1e-5)
    wrapped_env.close()


def test_create_stressed_env_profiles():
    profiles = ["nominal", "wind", "sensor_noise", "engine_degradation", "extreme"]
    for prof in profiles:
        env = create_stressed_env(profile=prof)
        obs, _ = env.reset(seed=123)
        assert obs.shape == (8,)
        env.close()


def test_onnx_model_validity():
    import onnx
    onnx_path = "assets/ppo_policy.onnx"
    assert os.path.exists(onnx_path), f"Expected ONNX model at {onnx_path}"

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    # Check graph inputs and outputs
    assert len(model.graph.input) == 1
    assert model.graph.input[0].name == "observation"
    assert len(model.graph.output) == 1
    assert model.graph.output[0].name == "action_logits"
