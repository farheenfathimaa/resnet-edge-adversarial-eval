"""Shared fixtures: tiny model + quantized twin for fast, offline tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src import config


@pytest.fixture(scope="session")
def tiny_input():
    return torch.randn(4, config.IN_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory, tiny_input, name="models"):
    from src.model import make_resnet18

    torch.manual_seed(0)
    net = make_resnet18()
    # a few random-optimizer steps so weights differ from init (shape-only tests)
    opt = torch.optim.SGD(net.parameters(), lr=1e-3)
    net.train()
    y = torch.randint(0, config.NUM_CLASSES, (tiny_input.shape[0],))
    for _ in range(2):
        opt.zero_grad()
        nn.functional.cross_entropy(net(tiny_input), y).backward()
        opt.step()
    path = tmp_path_factory.mktemp(name) / "test_fp32.pth"
    torch.save(net.state_dict(), path)
    return str(path)


@pytest.fixture(scope="session")
def quantized_tiny(tiny_checkpoint, tiny_input, tmp_path_factory):
    """FX static-INT8 quantized version of the tiny model (state_dict file)."""
    warnings = __import__("warnings")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from src.model import load_fp32_model
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
        from torch.ao.quantization import QConfigMapping
        from torch.ao.quantization import get_default_qconfig

        net = load_fp32_model(tiny_checkpoint).eval()
        qcfg = get_default_qconfig("onednn")
        prepared = prepare_fx(net, QConfigMapping().set_global(qcfg),
                              example_inputs=[tiny_input])
        with torch.no_grad():
            prepared(tiny_input)
        quant = convert_fx(prepared).eval()
    path = tmp_path_factory.mktemp("quantized") / "test_int8.pt"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.save(quant.state_dict(), path)
    return str(path)


@pytest.fixture(scope="session")
def val_batch():
    """A single normalized batch that looks like the model's input domain."""
    data = torch.randn(8, config.IN_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)
    targets = torch.randint(0, config.NUM_CLASSES, (8,))
    return data, targets


@pytest.fixture(scope="session")
def app_client(quantized_tiny):
    """FastAPI test client wired to the quantized tiny model."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:  # pragma: no cover
        pytest.skip("fastapi TestClient not available")
    from src.serve import create_app
    from src.quantize import _load_quant_nn

    return TestClient(create_app(provided=_load_quant_nn(quantized_tiny)))