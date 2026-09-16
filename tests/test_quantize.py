"""ONNX export + onnxruntime quantization pipeline tests (offline)."""
from __future__ import annotations

import os
from pathlib import Path

import onnx
import pytest
import torch

from src.model import load_fp32_model


def test_fp32_onnx_export_validates(tmp_path, tiny_checkpoint, tiny_input):
    from src.quantize import export_onnx

    out = Path(tmp_path) / "model.onnx"
    export_onnx(tiny_checkpoint, out)
    assert out.exists() and out.stat().st_size > 0
    onnx_model = onnx.load(str(out))
    onnx.checker.check_model(onnx_model)
    assert onnx_model.graph.input[0].name == "input"


def test_ort_dynamic_quantize_runs(tmp_path, tiny_checkpoint, tiny_input):
    onnxruntime = pytest.importorskip("onnxruntime")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    from src.quantize import export_onnx

    onnx_path = Path(tmp_path) / "model.onnx"
    export_onnx(tiny_checkpoint, onnx_path)
    out = Path(tmp_path) / "model_int8.onnx"
    quantize_dynamic(str(onnx_path), str(out), weight_type=QuantType.QInt8)
    assert out.exists()
    assert out.stat().st_size < onnx_path.stat().st_size

    so = onnxruntime.SessionOptions()
    sess = onnxruntime.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
    out_arr = sess.run(None, {"input": tiny_input.numpy()})[0]
    assert out_arr.shape == (tiny_input.shape[0], 10)


def test_ort_session_outputs_match_torch(tmp_path, tiny_checkpoint, tiny_input):
    onnxruntime = pytest.importorskip("onnxruntime")

    from src.quantize import export_onnx

    onnx_path = Path(tmp_path) / "model.onnx"
    export_onnx(tiny_checkpoint, onnx_path)
    sess = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    with torch.no_grad():
        torch_out = load_fp32_model(tiny_checkpoint).eval()(tiny_input)
    ort_out = torch.from_numpy(sess.run(None, {"input": tiny_input.numpy()})[0])
    # logits closely match between torch eager and the ONNX graph
    assert torch.allclose(ort_out, torch_out, atol=1e-4)