"""Inference latency + model-size benchmark on CPU (edge-style constraints).

Simulates an ARM/NPU-less edge device by pinning inference to a single CPU
thread (config.EDGE_THREADS).  Variants benchmarked:

  * torch fp32   (eager)
  * torch int8   (FX static PTQ)
  * onnx fp32    (onnxruntime)
  * onnx int8    (onnxruntime dynamic quantization)
  * onnx int8 st.(onnxruntime static quantization)

The same normalized-input preprocessing used at training time is applied.
Results are written to results/benchmark_result.json and a Markdown table.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

from . import config, data, model
from .quantize import _load_quant_nn


def _make_input() -> torch.Tensor:
    img = torch.randn(1, config.IN_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)
    return data.preprocess_batch(img)


def _time_fn(fn, iters: int = config.BENCH_ITERS, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return round(statistics.median(times), 3)


def _benchmark_torch(name: str, net: torch.nn.Module, x: torch.Tensor) -> float:
    torch.set_num_threads(config.EDGE_THREADS)
    net.eval()
    with torch.no_grad():
        _time_fn(lambda: net(x), warmup=3)
        return _time_fn(lambda: net(x))


def _benchmark_ort(sess) -> float:
    x = _make_input().numpy()
    return _time_fn(lambda: sess.run(None, {"input": x}))


def _ort_session(path: Path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = config.EDGE_THREADS  # cap == edge simulation
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _size_mb(path: Path) -> float:
    return round(path.stat().st_size / 1e6, 3)


def _variant(name: str, kind: str, latency: float | None, size_mb: float | None,
             note: str = "") -> dict:
    return {"name": name, "kind": kind, "latency_ms": latency,
            "size_mb": size_mb, "note": note}


def run_all() -> dict:
    x = _make_input()
    results = {"edge_threads": config.EDGE_THREADS, "iters": config.BENCH_ITERS,
               "input_shape": list(x.shape), "variants": []}
    add = results["variants"].append

    fp32_path = config.MODELS_DIR / config.MODEL_FP32
    int8_path = config.MODELS_DIR / config.MODEL_INT8
    onnx_fp32 = config.MODELS_DIR / config.ONNX_FP32
    onnx_dyn = config.MODELS_DIR / config.ONNX_INT8_DYN
    onnx_static = config.MODELS_DIR / config.ONNX_INT8_STATIC

    if fp32_path.exists():
        net = model.load_fp32_model(str(fp32_path))
        lat = _benchmark_torch("fp32", net, x)
        add(_variant("torch fp32", "torch", lat, _size_mb(fp32_path)))
        print(f"  torch fp32 : {lat} ms  ({_size_mb(fp32_path)} MB)")

    if int8_path.exists():
        net = _load_quant_nn(int8_path)
        lat = _benchmark_torch("int8", net, x)
        add(_variant("torch int8 (static ptq)", "torch", lat, _size_mb(int8_path)))
        print(f"  torch int8 : {lat} ms  ({_size_mb(int8_path)} MB)")

    for label, path, kind in [
        ("onnx fp32", onnx_fp32, "onnx"),
        ("onnx int8 (dynamic)", onnx_dyn, "onnx"),
        ("onnx int8 (static)", onnx_static, "onnx"),
    ]:
        if path.exists():
            sess = _ort_session(path)
            lat = _benchmark_ort(sess)
            add(_variant(label, kind, lat, _size_mb(path)))
            print(f"  {label:<20}: {lat} ms  ({_size_mb(path)} MB)")

    out = config.RESULTS_DIR / "benchmark_result.json"
    out.write_text(json.dumps(results, indent=2))
    _write_md(results)
    print(f"[benchmark] report -> {out} and results/benchmark_result.md")
    return results


def _write_md(results: dict) -> None:
    lines = [
        "# Latency & model-size benchmark",
        "",
        f"- Edge simulation: {results['edge_threads']} CPU thread(s), "
        f"input {results['input_shape']}, {results['iters']} iters (median)",
        "",
        "| variant | kind | latency (ms) | size (MB) |",
        "|---|---|---|---|",
    ]
    for v in results["variants"]:
        lines.append(
            f"| {v['name']} | {v['kind']} | {v['latency_ms']} | {v['size_mb']} |"
        )
    (config.RESULTS_DIR / "benchmark_result.md").write_text("\n".join(lines))


if __name__ == "__main__":
    run_all()