"""Model construction, persistence and quantized-shape consistency tests."""
from __future__ import annotations

import torch

from src import config
from src.model import load_fp32_model, make_resnet18


def test_model_loads_and_forward(tiny_checkpoint, tiny_input):
    net = load_fp32_model(tiny_checkpoint)
    net.eval()
    out = net(tiny_input)
    assert tuple(out.shape) == (tiny_input.shape[0], config.NUM_CLASSES)
    assert out.dtype == torch.float32


def test_checkpoint_roundtrip(tiny_checkpoint, tiny_input):
    net = load_fp32_model(tiny_checkpoint).eval()
    with torch.no_grad():
        a = net(tiny_input)
    # reload a second time from the same file
    again = load_fp32_model(tiny_checkpoint).eval()
    with torch.no_grad():
        b = again(tiny_input)
    assert torch.allclose(a, b, atol=1e-6)


def test_quantized_output_shape_matches_fp32(tiny_checkpoint, quantized_tiny, tiny_input):
    from src.model import load_fp32_model
    from src.quantize import _load_quant_nn

    fp = load_fp32_model(tiny_checkpoint).eval()
    qm = _load_quant_nn(quantized_tiny).eval()
    with torch.no_grad():
        fp_out = fp(tiny_input)
        q_out = qm(tiny_input)
    assert q_out.shape == fp_out.shape
    assert tuple(q_out.shape) == (tiny_input.shape[0], config.NUM_CLASSES)
    from src.simulate_int8 import build_simulated_int8

    fp = load_fp32_model(tiny_checkpoint).eval()
    sim = build_simulated_int8(tiny_checkpoint).eval()
    with torch.no_grad():
        fp_out = fp(tiny_input)
        sim_out = sim(tiny_input)
    assert sim_out.shape == fp_out.shape
    assert torch.isfinite(sim_out).all()