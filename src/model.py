"""Small ResNet for single-channel 32x32 input.

Implements the classic ResNet-18 topology (four stages of BasicBlock with
[2,2,2,2] blocks) with a configurable width multiplier so the model can be
trained on CPU within minutes.  The stem is adjusted for 1-channel (or 3-
channel) 32x32 inputs: a 3x3 conv at stride 1 replaces the stock 7x7 stride-2
"ImageNet stem", followed by BatchNorm + ReLU and the standard stages.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


def _round_width(c: int, width: float) -> int:
    """Scale a channel count by the width multiplier, keeping it a multiple of 8.

    Keeps the final pooled feature count aligned so the classifier head works
    for width values like 0.25 / 0.5 / 1.0.
    """
    return max(8, int(round(c * width / 8) * 8))


def make_resnet18(
    num_classes: int = config.NUM_CLASSES,
    in_channels: int = config.IN_CHANNELS,
    width: float = config.WIDTH,
) -> nn.Module:
    return _ResNet(
        block=_BasicBlock,
        layers=config.BLOCKS,
        in_channels=in_channels,
        width=width,
        num_classes=num_classes,
    )


def load_fp32_model(checkpoint: str | None = None, **kwargs) -> nn.Module:
    """Build the (unquantized) model and optionally load FP32 weights."""
    model = make_resnet18(**kwargs)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"checkpoint mismatch: missing={list(missing)} unexpected={list(unexpected)}"
            )
    return model


class _BasicBlock(nn.Module):
    """Basic residual block: conv->bn->relu->conv->bn + identity shortcut."""

    expansion = 1

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class _ResNet(nn.Module):
    def __init__(
        self,
        block: type[_BasicBlock],
        layers: tuple[int, ...],
        in_channels: int,
        width: float,
        num_classes: int,
    ):
        super().__init__()
        self.width = width
        base = _round_width(64, width)
        self.in_planes = base
        self.conv1 = nn.Conv2d(in_channels, base, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(base)
        self.layer1 = self._make_layer(block, base * 1, layers[0], stride=1)
        self.layer2 = self._make_layer(block, base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes: int, n_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x