"""
scripts/export_onnx.py - Export trained PyTorch PPO Actor Policy to ONNX for Web & Edge Deployment.
"""

import argparse
import os
import torch
import torch.nn as nn
import onnx
from stable_baselines3 import PPO


class OnnxActorPolicy(nn.Module):
    """
    Wraps the Actor Policy from Stable-Baselines3 to output action logits
    from an 8-dimensional observation vector.
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.policy.eval()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.policy.extract_features(obs)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        logits = self.policy.action_net(latent_pi)
        return logits


def export_model_to_onnx(model_path: str, output_path: str, opset_version: int = 14) -> None:
    print(f"[*] Loading PPO model from: {model_path}")
    model = PPO.load(model_path)
    policy = model.policy

    onnx_module = OnnxActorPolicy(policy)
    dummy_input = torch.randn(1, 8, dtype=torch.float32)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temp_path = f"{output_path}.tmp"

    print(f"[*] Exporting Actor policy to ONNX (opset={opset_version})...")
    torch.onnx.export(
        onnx_module,
        dummy_input,
        temp_path,
        opset_version=opset_version,
        input_names=["observation"],
        output_names=["action_logits"],
        dynamic_axes={"observation": {0: "batch_size"}, "action_logits": {0: "batch_size"}},
    )

    # Convert to single self-contained file with embedded initializers
    onnx_model = onnx.load(temp_path, load_external_data=True)
    onnx.save_model(onnx_model, output_path, save_as_external_data=False)

    if os.path.exists(temp_path):
        os.remove(temp_path)
    if os.path.exists(f"{temp_path}.data"):
        os.remove(f"{temp_path}.data")

    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"[+] Successfully exported self-contained ONNX model to '{output_path}' ({file_size_kb:.1f} KB)")


def parse_args():
    parser = argparse.ArgumentParser(description="Export trained PPO policy to ONNX format.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/ppo_deep_discrete",
        help="Path to trained PPO model (default: models/ppo_deep_discrete).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="assets/ppo_policy.onnx",
        help="Target ONNX export destination (default: assets/ppo_policy.onnx).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX operator set version (default: 14).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_model_to_onnx(args.model, args.output, args.opset)
