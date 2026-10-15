"""
network.py - High-Definition Deep Neural Guidance Network for Lunar Lander RL.

Implements an advanced Residual MLP (ResMLP) architecture:
- Deep multi-layer representation with LayerNorm and GELU activations
- Residual skip connections (x + f(x)) for stable gradient flow
- High-capacity decoupled feature representations for actor (policy) and critic (value)
- Scaled orthogonal initialization for robust reinforcement learning stability
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResMLPBlock(nn.Module):
    """
    Residual Multi-Layer Perceptron Block with contraction scaling.
    Allows deep representation without disrupting low-level flight telemetry gradients.
    """

    def __init__(self, dim: int, scale: float = 0.1):
        super().__init__()
        self.scale = scale
        self.fc1 = nn.Linear(dim, dim)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.act2 = nn.GELU()

        # Initialize residual branch output near zero for identity-first propagation
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight, gain=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.fc2(self.act1(self.fc1(x)))
        return self.act2(x + self.scale * residual)


class DeepGuidanceExtractor(BaseFeaturesExtractor):
    """
    High-Definition Deep Aerospace Guidance Feature Extractor.
    Encodes 8-dimensional flight telemetry through dense projections and ResMLP blocks.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        in_dim = int(np.prod(observation_space.shape))

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
        )

        self.res1 = ResMLPBlock(256, scale=0.1)
        self.res2 = ResMLPBlock(256, scale=0.1)

        self.latent_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, features_dim),
            nn.GELU(),
        )

        self._init_projections()

    def _init_projections(self) -> None:
        """Applies orthogonal initialization to input and output projections."""
        for layer in [self.input_proj[0], self.latent_head[1]]:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(observations)
        x = self.res1(x)
        x = self.res2(x)
        return self.latent_head(x)


def get_deep_policy_kwargs() -> dict:
    """
    Returns policy_kwargs configuring PPO to use high-capacity decoupled
    actor-critic architectures (256x256x128) with orthogonal initialization
    and Tanh activations for optimal gradient propagation in Box2D aerospace dynamics.
    """
    return dict(
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
        activation_fn=nn.Tanh,
        ortho_init=True,
    )
