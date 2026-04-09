from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .constants import DEFAULT_MODEL_BLOCKS, DEFAULT_MODEL_CHANNELS, INPUT_CHANNELS


class CNNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.bn(self.conv(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        y = torch.cat([avg, mx], dim=1)
        y = self.conv(y)
        y = self.bn(y)
        y = torch.sigmoid(y)
        return x * y


class ResnetLayer(nn.Module):
    def __init__(self, channels: int, mid_channels: int) -> None:
        super().__init__()
        self.conv1 = CNNLayer(channels, mid_channels)
        self.conv2 = CNNLayer(mid_channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x)) + x


class OutputHeadResNet(nn.Module):
    def __init__(self, trunk_channels: int, head_mid_channels: int) -> None:
        super().__init__()
        self.cnn = CNNLayer(trunk_channels, head_mid_channels)
        self.valueHeadLinear = nn.Linear(head_mid_channels, 4)
        self.policyHeadLinear = nn.Conv2d(head_mid_channels, 1, kernel_size=1, bias=False)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.cnn(h)
        value = self.valueHeadLinear(x.mean((2, 3)))[:, :3]
        policy = self.policyHeadLinear(x).flatten(1)
        return value, policy


class CompetitionResNet(nn.Module):
    def __init__(self, blocks: int, channels: int) -> None:
        super().__init__()
        self.model_type = "res"
        self.model_param = (blocks, channels)
        self.input_c = INPUT_CHANNELS
        self.inputhead = CNNLayer(self.input_c, channels)
        self.trunk = nn.ModuleList(ResnetLayer(channels, channels) for _ in range(blocks))
        self.outputhead = OutputHeadResNet(channels, channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.inputhead(x)
        for block in self.trunk:
            h = block(h)
        return self.outputhead(h)


MODEL_REGISTRY = {
    "res": CompetitionResNet,
}


def build_model(
    blocks: int = DEFAULT_MODEL_BLOCKS,
    channels: int = DEFAULT_MODEL_CHANNELS,
    model_type: str = "res",
) -> nn.Module:
    if model_type not in MODEL_REGISTRY:
        raise KeyError(f"unknown model type: {model_type}")
    return MODEL_REGISTRY[model_type](blocks, channels)


def _extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    raise KeyError("checkpoint does not contain state_dict or model_state_dict")


def load_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model_type = checkpoint.get("model_type", "res")
    model_param = checkpoint.get(
        "model_param",
        (DEFAULT_MODEL_BLOCKS, DEFAULT_MODEL_CHANNELS),
    )
    if isinstance(model_param, list):
        model_param = tuple(model_param)
    model = build_model(blocks=model_param[0], channels=model_param[1], model_type=model_type)
    model.load_state_dict(_extract_state_dict(checkpoint))
    return model, checkpoint


def export_competition_checkpoint(
    model: nn.Module,
    output_path: str | Path,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "model_type": getattr(model, "model_type", "res"),
        "model_param": list(getattr(model, "model_param", (DEFAULT_MODEL_BLOCKS, DEFAULT_MODEL_CHANNELS))),
        "state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, output_path)
