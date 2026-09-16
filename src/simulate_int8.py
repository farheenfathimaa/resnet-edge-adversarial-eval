"""Differentiable INT8 "simulation" of the quantized model.

The real static-INT8 model (models/<MODEL_INT8>) uses true QuantizedConv
kernels that have no autograd, so gradient-based adversarial attacks cannot be
run against it directly.  This module provides a SimulatedInt8ResNet18 twin of
the same architecture in which every Conv2d/Linear weight is rounded to INT8
(per-output-channel, symmetric) and every intermediate activation is clamped
to an INT8 grid (per-tensor, asymmetric) using straight-through estimators.

It matches the real INT8 model's weights and accuracy closely (validated in
the README results) while remaining fully differentiable -- the standard
"simulated quantization / QAT-style" approximation used in the literature for
robustness analysis of quantized networks.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Straight-through-estimator round: forward=round, backward=identity."""
    return x + (x.round() - x).detach()


def quantize_weight_per_channel(w: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Symmetric per-output-channel INT8 weight quantization (dequantized)."""
    qmax = 2 ** (bits - 1) - 1
    dims = list(range(1, w.dim()))
    scale = w.abs().amax(dim=dims, keepdim=True).clamp_min(1e-8) / qmax
    return _ste_round(w / scale).clamp(-qmax, qmax) * scale


def quantize_activation(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Asymmetric per-tensor INT8 activation quantization (dequantized)."""
    qmax = float(2 ** bits) - 1.0
    x_min = x.min()
    x_max = x.max()
    scale = ((x_max - x_min) / qmax).clamp_min(1e-8)
    zero = -x_min / scale
    return (_ste_round(x / scale + zero).clamp(0, qmax) - zero) * scale


class _QuantConv2d(nn.Module):
    """Conv2d whose weight is fake-quantized per-channel (PARAM NAMES MATCH Conv2d)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1,
                 padding: int = 0, bias: bool = False):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = 1
        self.groups = 1
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, kernel, kernel))
        self.bias: nn.Parameter | None = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_ch))
        # init like nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            bound = 1 / (in_ch * kernel * kernel) ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wq = quantize_weight_per_channel(self.weight)
        out = F.conv2d(x, wq, self.bias, self.stride, self.padding,
                       self.dilation, self.groups)
        return quantize_activation(out)


class _QuantLinear(nn.Module):
    """Linear whose weight is fake-quantized per-channel (PARAM NAMES MATCH Linear)."""

    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.empty(out_f))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        bound = 1 / in_f ** 0.5
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wq = quantize_weight_per_channel(self.weight)
        return F.linear(x, wq, self.bias)


class _Block(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int):
        super().__init__()
        self.conv1 = _QuantConv2d(in_planes, out_planes, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = _QuantConv2d(out_planes, out_planes, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                _QuantConv2d(in_planes, out_planes, 1, stride),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class SimulatedInt8ResNet18(nn.Module):
    """INT8-simulated twin of the trained FP32 ResNet18x<width> model.

    Weights are copied from the FP32 checkpoint; every weight is fake-quantized
    to INT8 at forward time and every activation is clamped to an INT8 grid.
    """

    def __init__(self, _fp32_state: dict | None = None,
                 num_classes: int = config.NUM_CLASSES,
                 in_channels: int = config.IN_CHANNELS,
                 width: float = config.WIDTH):
        super().__init__()
        width = width
        base = max(8, int(round(64 * width / 8) * 8))
        self.in_planes = base
        self.conv1 = _QuantConv2d(in_channels, base, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(base)
        self.layer1 = self._make_layer(base * 1, 2, 1)
        self.layer2 = self._make_layer(base * 2, 2, 2)
        self.layer3 = self._make_layer(base * 4, 2, 2)
        self.layer4 = self._make_layer(base * 8, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = _QuantLinear(base * 8, num_classes)
        if _fp32_state is not None:
            missing, unexpected = self.load_state_dict(_fp32_state, strict=False)
            if missing or unexpected:
                raise ValueError(
                    f"state mismatch missing={list(missing)} unexpected={list(unexpected)}"
                )

    def _make_layer(self, planes: int, n: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (n - 1)
        layers = []
        for s in strides:
            layers.append(_Block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_simulated_int8(checkpoint: str | None = None,
                         eval_mode: bool = True) -> SimulatedInt8ResNet18:
    from .model import make_resnet18

    fp = make_resnet18()
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        fp.load_state_dict(state, strict=True)
    sim = SimulatedInt8ResNet18(_fp32_state=fp.state_dict())
    if eval_mode:
        sim.eval()
    return sim