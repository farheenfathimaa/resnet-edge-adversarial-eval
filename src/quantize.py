"""INT8 quantization of the trained FP32 model.

Three OPT variants are produced, all fully local:

1. PyTorch static post-training quantization (FX graph mode, oneDNN backend):
   models/<MODEL_INT8>  -- a real INT8 network (QuantizedConvReLU2d kernels).
2. ONNX FP32 export (legacy TorchScript exporter, dynamic batch axis):
   models/<ONNX_FP32>
3. ONNX Runtime INT8 (dynamic + static), produced with onnxruntime's
   quantization tools: models/<ONNX_INT8_DYN> and models/<ONNX_INT8_STATIC>.

Accuracy of every variant on the validation subset is written to
results/quant_accuracy.json so the quantization accuracy drop is reported.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from . import config, data, model

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module="torch.*"
    )
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    from torch.ao.quantization import QConfigMapping

MAX_INT8 = 127


def _fx_prepare(net: torch.nn.Module, example: torch.Tensor):
    qconfig = torch.ao.quantization.get_default_qconfig("onednn")
    mapping = QConfigMapping().set_global(qconfig)
    prepared = prepare_fx(net, mapping, example_inputs=[example])
    return prepared


def _convert_structure(example: torch.Tensor | None = None):
    """Build a fresh, un-calibrated converted INT8 network.

    Used to load the saved INT8 *state_dict*: the graph structure of an FX
    GraphModule cannot be pickled reliably in this torch build (see
    torch.fx.graph_module pickling of string globals), but its int8 weights /
    scale / zero-point are fully captured in state_dict.  We rebuild the
    identical graph (same architecture -> same prepare/convert trace) and load
    the state.  No calibration is needed for reconstruction because the saved
    activation ranges overwrite whatever the fresh observers contained.
    """
    example = example or torch.randn(2, config.IN_CHANNELS, config.IMG_SIZE,
                                     config.IMG_SIZE)
    fresh = model.make_resnet18().eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared = _fx_prepare(fresh, example)
        converted = convert_fx(prepared)
        # convert runs the model once with fake-quant observers in eval mode;
        # done below to satisfy the trace and clean up extraneous state.
    converted.eval()
    return converted


def quantize_torch_int8(ckpt: Path | str | None = None, out: Path | str | None = None) -> Path:
    """Static post-training INT8 quantization via FX graph mode."""
    ckpt = Path(ckpt or config.MODELS_DIR / config.MODEL_FP32)
    if not ckpt.exists():
        raise FileNotFoundError(f"FP32 checkpoint not found at {ckpt}")

    net = model.load_fp32_model(str(ckpt))
    net.eval()
    example = torch.randn(2, config.IN_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared = _fx_prepare(net, example)

        # calibration: run a small representative batch through the graph
        train_ds, _ = data.load_dataset()
        prepared.eval()
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(train_ds), generator=g)[: config.CALIBRATION_SAMPLES]
        cal_batch = torch.stack([train_ds[i][0] for i in idx], dim=0)
        with torch.no_grad():
            prepared(data.preprocess_batch(cal_batch))

        int8_model = convert_fx(prepared)
    int8_model.eval()

    out = Path(out or config.MODELS_DIR / config.MODEL_INT8)
    torch.save(int8_model.state_dict(), out)
    size_mb = out.stat().st_size / 1e6
    print(f"[quantize] PyTorch static INT8 model saved -> {out} ({size_mb:.2f} MB)")
    return out


def export_onnx(ckpt: Path | str | None = None, out: Path | str | None = None) -> Path:
    """Export the FP32 model to ONNX (legacy exporter, dynamic batch)."""
    ckpt = Path(ckpt or config.MODELS_DIR / config.MODEL_FP32)
    if not ckpt.exists():
        raise FileNotFoundError(f"FP32 checkpoint not found at {ckpt}")

    net = model.load_fp32_model(str(ckpt))
    net.eval()
    x = torch.randn(1, config.IN_CHANNELS, config.IMG_SIZE, config.IMG_SIZE, dtype=torch.float32)
    out = Path(out or config.MODELS_DIR / config.ONNX_FP32)
    torch.onnx.export(
        net, x, str(out),
        input_names=["input"], output_names=["output"],
        opset_version=13,
        dynamo=False,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"[quantize] ONNX FP32 exported -> {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return out


def ort_quantize_models(onnx_path: Path | str | None = None) -> dict[str, Path]:
    """Create INT8 ONNX models (dynamic + static) with onnxruntime quantizers."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )

    onnx_path = Path(onnx_path or config.MODELS_DIR / config.ONNX_FP32)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

    out_dyn = Path(config.MODELS_DIR / config.ONNX_INT8_DYN)
    quantize_dynamic(str(onnx_path), str(out_dyn), weight_type=QuantType.QInt8)
    print(f"[quantize] ONNX Runtime dynamic INT8 -> {out_dyn} "
          f"({out_dyn.stat().st_size / 1e6:.2f} MB)")

    class _Reader(CalibrationDataReader):
        def __init__(self, batch: np.ndarray):
            self._rows = [
                {"input": batch[i : i + 1]} for i in range(0, len(batch), 8)
            ]
            self._i = 0

        def get_next(self):
            if self._i < len(self._rows):
                row = self._rows[self._i]
                self._i += 1
                return row
            return None

        def rewind(self):
            self._i = 0

    train_ds, _ = data.load_dataset()
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(train_ds), generator=g)[: config.CALIBRATION_SAMPLES]
    cal = data.preprocess_batch(torch.stack([train_ds[j][0] for j in idx], dim=0))
    reader = _Reader(cal.numpy())

    out_static = Path(config.MODELS_DIR / config.ONNX_INT8_STATIC)
    quantize_static(
        str(onnx_path), str(out_static), reader,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    print(f"[quantize] ONNX Runtime static INT8 -> {out_static} "
          f"({out_static.stat().st_size / 1e6:.2f} MB)")
    return {"dynamic": out_dyn, "static": out_static}


def _load_quant_nn(model_path: Path | str | None = None) -> torch.nn.Module:
    """Load the saved torch static-INT8 model (reconstructed from state_dict)."""
    model_path = str(Path(model_path or config.MODELS_DIR / config.MODEL_INT8))
    if not Path(model_path).exists():
        raise FileNotFoundError(f"INT8 checkpoint not found at {model_path}")
    converted = _convert_structure()
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    missing, unexpected = converted.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"INT8 state mismatch: missing={list(missing)[:5]} "
            f"unexpected={list(unexpected)[:5]}"
        )
    converted.eval()
    return converted


def evaluate_variants(
    val_batches: int | None = None,
) -> dict:
    """Accuracy of all variants on the validation subset.

    Returns {"fp32_torch": acc, "int8_torch_ptq": acc,
             "fp32_onnx_ort": acc, "int8_onnx_ort_dynamic": acc,
             "int8_onnx_ort_static": acc} plus elapsed time.
    """
    import onnxruntime as ort

    t0 = time.perf_counter()
    train_ds, val_ds = data.load_dataset()
    _, dl_val = data.make_loaders(
        train_ds, val_ds, num_workers=0,
        val_subset=config.VAL_SUBSET, train_subset=0,
    )

    fp = model.load_fp32_model(str(config.MODELS_DIR / config.MODEL_FP32)).eval()
    results: dict = {}

    def acc_fn(fn) -> float:
        correct = total = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                out = fn(data.preprocess_batch(xb))
                if isinstance(out, tuple):
                    out = out[0]
                correct += (out.argmax(1) == yb).sum().item()
                total += yb.size(0)
        return correct / max(total, 1)

    results["fp32_torch"] = acc_fn(fp)
    print(f"  fp32 torch         acc={results['fp32_torch']:.4f}")

    int8_path = config.MODELS_DIR / config.MODEL_INT8
    if int8_path.exists():
        int8_model = _load_quant_nn(int8_path)
        results["int8_torch_ptq"] = acc_fn(int8_model)
        print(f"  int8 torch (PTQ)   acc={results['int8_torch_ptq']:.4f}")

    def make_ort_sess(path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = config.EDGE_THREADS
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    onnx_fp32 = config.MODELS_DIR / config.ONNX_FP32
    if onnx_fp32.exists():
        sess = make_ort_sess(onnx_fp32)

        def fn(x):
            return torch.from_numpy(sess.run(None, {"input": x.numpy()})[0])

        results["fp32_onnx_ort"] = acc_fn(fn)
        print(f"  fp32 onnx (ORT)    acc={results['fp32_onnx_ort']:.4f}")

    for key, name in [
        ("int8_onnx_ort_dynamic", config.ONNX_INT8_DYN),
        ("int8_onnx_ort_static", config.ONNX_INT8_STATIC),
    ]:
        p = config.MODELS_DIR / name
        if p.exists():
            sess = make_ort_sess(p)

            def fn(x, sess=sess):
                return torch.from_numpy(sess.run(None, {"input": x.numpy()})[0])

            results[key] = acc_fn(fn)
            print(f"  {key:<18} acc={results[key]:.4f}")

    results["elapsed_s"] = round(time.perf_counter() - t0, 1)
    results["val_subset"] = config.VAL_SUBSET
    results["calibration_samples"] = config.CALIBRATION_SAMPLES

    out = config.RESULTS_DIR / "quant_accuracy.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[quantize] accuracy report -> {out}")
    if "int8_torch_ptq" in results:
        drop = results["fp32_torch"] - results["int8_torch_ptq"]
        print(f"[quantize] torch INT8 accuracy drop vs FP32: {drop:.4f}")
    return results


def run_all(ckpt: Path | str | None = None) -> dict:
    """Full quantization pipeline: torch INT8 + ONNX export + ORT INT8 + eval."""
    ckpt = Path(ckpt or config.MODELS_DIR / config.MODEL_FP32)
    quantize_torch_int8(ckpt)
    export_onnx(ckpt)
    ort_quantize_models()
    return evaluate_variants()


if __name__ == "__main__":
    run_all()