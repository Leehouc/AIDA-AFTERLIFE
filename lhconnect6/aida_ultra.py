from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import DEFAULT_MODEL_BLOCKS, DEFAULT_MODEL_CHANNELS, INPUT_CHANNELS


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int | None = None, groups: int = 1, bias: bool = False) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = x.mean(dim=(2, 3))
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = torch.sigmoid(y).view(b, c, 1, 1)
        return x * y


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


class ChannelShuffle(nn.Module):
    def __init__(self, groups: int = 2) -> None:
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        if c % self.groups != 0:
            return x
        x = x.view(b, self.groups, c // self.groups, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, c, h, w)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()
        return x / keep_prob * random_tensor


class Bottleneck(nn.Module):
    def __init__(self, channels: int, expansion: int = 4, drop_prob: float = 0.0) -> None:
        super().__init__()
        mid = max(4, channels // expansion)
        self.conv1 = ConvBNAct(channels, mid, 1)
        self.conv2 = ConvBNAct(mid, mid, 3)
        self.conv3 = nn.Conv2d(mid, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.drop = DropPath(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = self.drop(y)
        return F.relu(x + y)


class DilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.drop = DropPath(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.bn(y)
        y = F.relu(y)
        y = self.drop(y)
        return x + y


class MultiScaleBlock(nn.Module):
    def __init__(self, channels: int, drop_prob: float = 0.0) -> None:
        super().__init__()
        c = channels
        self.branch1 = ConvBNAct(c, c, 1)
        self.branch2 = ConvBNAct(c, c, 3)
        self.branch3 = ConvBNAct(c, c, 5)
        self.mix = nn.Conv2d(c * 3, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.drop = DropPath(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        y = torch.cat([b1, b2, b3], dim=1)
        y = self.mix(y)
        y = self.bn(y)
        y = F.relu(y)
        y = self.drop(y)
        return x + y


class CoordAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(4, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv_h = nn.Conv2d(mid, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = F.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.transpose(2, 3)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        out = x * a_h * a_w
        return out


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, 3)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.drop = DropPath(drop_prob)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.se(y)
        y = self.drop(y)
        return F.relu(x + y)


class GatedUnit(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        a, b = torch.chunk(y, 2, dim=1)
        return torch.tanh(a) * torch.sigmoid(b)


class MixerBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.ln1 = LayerNorm2d(channels)
        self.ln2 = LayerNorm2d(channels)
        self.mlp1 = nn.Conv2d(channels, channels * 2, 1)
        self.mlp2 = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln1(x)
        y = self.mlp1(y)
        y = F.gelu(y)
        y = self.mlp2(y)
        x = x + y
        y = self.ln2(x)
        y = self.mlp1(y)
        y = F.gelu(y)
        y = self.mlp2(y)
        return x + y


class DepthwiseSeparable(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.pw(y)
        y = self.bn(y)
        return F.relu(x + y)


class PyramidPooling(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.conv3 = nn.Conv2d(channels, channels, 1, bias=False)
        self.conv4 = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels * 5)
        self.fuse = nn.Conv2d(channels * 5, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        p1 = F.adaptive_avg_pool2d(x, 1)
        p2 = F.adaptive_avg_pool2d(x, 2)
        p3 = F.adaptive_avg_pool2d(x, 3)
        p4 = F.adaptive_avg_pool2d(x, 4)
        p1 = F.interpolate(self.conv1(p1), size=(h, w), mode="bilinear", align_corners=False)
        p2 = F.interpolate(self.conv2(p2), size=(h, w), mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.conv3(p3), size=(h, w), mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.conv4(p4), size=(h, w), mode="bilinear", align_corners=False)
        y = torch.cat([x, p1, p2, p3, p4], dim=1)
        y = self.bn(y)
        y = F.relu(y)
        y = self.fuse(y)
        return y


class ComplexTrunk(nn.Module):
    def __init__(self, channels: int, blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(blocks):
            if i % 6 == 0:
                self.blocks.append(ResidualUnit(channels, drop_prob=0.05))
            elif i % 6 == 1:
                self.blocks.append(Bottleneck(channels, expansion=4, drop_prob=0.05))
            elif i % 6 == 2:
                self.blocks.append(MultiScaleBlock(channels, drop_prob=0.05))
            elif i % 6 == 3:
                self.blocks.append(DilatedBlock(channels, dilation=2, drop_prob=0.05))
            elif i % 6 == 4:
                self.blocks.append(DepthwiseSeparable(channels))
            else:
                self.blocks.append(MixerBlock(channels))
        self.coord = CoordAttention(channels)
        self.spatial = SpatialAttention()
        self.shuffle = ChannelShuffle(groups=2)
        self.gate = GatedUnit(channels)
        self.ppm = PyramidPooling(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i % 2 == 0:
                feats.append(x)
        if feats:
            x = sum(feats) / float(len(feats))
        x = self.coord(x)
        x = self.spatial(x)
        x = self.shuffle(x)
        x = self.gate(x)
        x = self.ppm(x)
        return x


class PolicyHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, 3)
        self.conv2 = nn.Conv2d(channels, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x.flatten(1)


class ValueHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, 3)
        self.fc1 = nn.Linear(channels, channels)
        self.fc2 = nn.Linear(channels, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = x.mean((2, 3))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x[:, :3]


class CompetitionResNet(nn.Module):
    def __init__(self, blocks: int, channels: int) -> None:
        super().__init__()
        self.model_type = "res"
        self.model_param = (blocks, channels)
        self.input_c = INPUT_CHANNELS
        self.stem = ConvBNAct(self.input_c, channels, 3)
        self.trunk = ComplexTrunk(channels, blocks)
        self.policy_head = PolicyHead(channels)
        self.value_head = ValueHead(channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.trunk(x)
        value = self.value_head(x)
        policy = self.policy_head(x)
        return value, policy


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
