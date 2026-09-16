# resnet-edge-adversarial-eval

Edge-optimized deployment and adversarial robustness evaluation of a small ResNet
classifier. Trains a half-width ResNet-18 on **FashionMNIST** (CPU-only), quantizes it
to INT8 with PyTorch FX static PTQ **and** onnxruntime (dynamic + static), serves
predictions via a minimal **FastAPI** endpoint, and benchmarks accuracy under
**FGSM/PGD** adversarial perturbation for both the FP32 and INT8 models — simulating a
Raspberry Pi / Jetson-class device on a laptop CPU.

## Why this setup

- **FashionMNIST instead of CIFAR-10**: the official CIFAR-10 server is frequently
  unavailable and extremely slow (~60 KB/s), making it impractical for a
  budget-constrained local pipeline. FashionMNIST ships from a fast AWS mirror and is
  cached on first use.
- **Half-width ResNet-18 (width 0.5, 2+2+2+2 blocks)**: a full ResNet-18 needs ~6-10x
  more FLOPs and trains at ~2.5 s/step on a 12-thread laptop CPU. The half-width model
  trains in under 10 minutes and still reaches ~84% top-1 on FashionMNIST — a realistic
  stand-in for fine-tuned-family-view classifiers (e.g., a ResNet-50 with an eager
  classification head) that live on edge hardware.

## What this repo does

1. **Train** a small FP32 ResNet on FashionMNIST (subset: 12k train / 2k val, 2 epochs).
2. **Quantize** to INT8 three ways:
   - PyTorch **FX static PTQ** (`torch.ao.quantization.prepare_fx` / `convert_fx`, onednn
     qconfig) — the only static path that works for Conv2d in torch 2.13 CPU builds
     (`quantize_dynamic` skips Conv2d; eager static fails on `aten::add.out` for the
     QuantizedCPU backend).
   - onnxruntime **dynamic** INT8.
   - onnxruntime **static** INT8 (MinMax calibration, 128 samples).
3. **Benchmark** latency (median over 30 runs, 1 thread = simulated edge) and on-disk
   size for every variant.
4. **Serve** the quantized model via FastAPI `POST /predict` (image upload → class id +
   name + latency) and `GET /health`.
5. **Attack** FP32, a STE fake-quantized INT8 twin, and cross-model transfer with FGSM
   and PGD-8 at multiple epsilons, and persist tables + a chart.

All commands are CLI-driven:

```bash
python main.py train        # train FP32 checkpoint (models/resnet18w05_fp32.pth)
python main.py quantize     # FX INT8 + ONNX FP32 + ORT dynamic/static INT8, accuracy report
python main.py benchmark    # latency + size table (1 thread / batch 1)
python main.py adversarial  # FGSM/PGD robustness, saves table + chart to results/
python main.py serve        # FastAPI dev server (or: uvicorn main:app ...)
python main.py all          # train -> quantize -> benchmark -> adversarial
python -m pytest -q         # offline unit/integration tests
```

First run downloads FashionMNIST (~60 MB, cached in `data/`).

## Results

Hardware: Intel i5-1235U, 12 logical cores, **no GPU**; single CPU thread for edge-mode
inference. Input is a normalized `[1, 1, 32, 32]` tensor.

### Quantization accuracy (2,000 validation images)

| variant | top-1 | vs FP32 |
|---|---|---|
| FP32 torch | 0.8115 | — |
| **INT8 torch (FX static PTQ)** | **0.8175** | +0.6 pp (noise) |
| FP32 ONNX (onnxruntime) | 0.8115 | — |
| INT8 ONNX dynamic | 0.8125 | ±0.0 |
| INT8 ONNX static | 0.8120 | ±0.0 |

Quantization is **lossless for accuracy** at this width (drop is within random-subset
noise; the FX INT8 model even edges FP32 by 0.6 pp on this subset).

### Latency & size (batch 1, 1 thread, median of 30)

| variant | kind | latency (ms) | size (MB) | vs FP32 time | vs FP32 size |
|---|---|---|---|---|---|
| torch fp32 | torch | 33.51 | 11.25 | 1.0x | 1.0x |
| torch int8 (static ptq) | torch | 17.55 | 2.89 | **1.9x faster** | 3.9x smaller |
| onnx fp32 | onnx | 19.65 | 11.19 | 1.0x | 1.0x |
| onnx int8 (dynamic) | onnx | 170.69 | 2.84 | 0.1x slower | 3.9x smaller |
| onnx int8 (static) | onnx | 9.48 | 2.84 | **2.1x faster** | 3.9x smaller |

Key findings:

- **Static INT8 wins**: ~4x smaller on disk and ~2x faster end-to-end for this tiny
  conv net. Kernels are `s8::s8::u8`-fused on x86-64.
- **Dynamic INT8 is a trap** for small conv nets: the per-call input quantization
  overhead dominates the compute time of a ~4 MFLOP forward pass, so it is ~10x slower
  than FP32. For image-classification-style (per-tensor activation) edge models prefer
  static calibration.
- These numbers are for a CPU laptop; INT8 advantage grows on actual edge HW (Jetson /
  RPi with SIMD/NEON), and size savings (4x) matter most for flash-constrained
  deployments.

### Adversarial robustness (200-image subset; white-box FGSM + PGD-8; epsilon in normalized units)

| attack | eps | FP32 clean | FP32 fgsm | FP32 pgd | INT8(sim) fgsm | INT8(sim) pgd | INT8 transfer fgsm | transfer pgd |
|---|---|---|---|---|---|---|---|---|
| fp32 | 0.005 | 0.84 | 0.815 | 0.815 | — | — | 0.810 | 0.815 |
| fp32 | 0.010 | 0.84 | 0.785 | 0.785 | 0.790 | 0.785 | 0.785 | 0.785 |
| fp32 | 0.020 | 0.84 | 0.710 | 0.710 | 0.710 | 0.705 | 0.710 | 0.710 |
| int8 (sim) | 0.005 | 0.83 | 0.815 | 0.810 | — | — | — | — |
| int8 (sim) | 0.010 | 0.83 | 0.790 | 0.785 | — | — | — | — |
| int8 (sim) | 0.020 | 0.83 | 0.710 | 0.705 | — | — | — | — |

Key findings:

- **Quantization does not change robustness at all** here: FP32, the fake-quant INT8
  twin, and cross-model transfer all track the same accuracy-vs-epsilon curve
  (≈0.81 / 0.79 / 0.71). INT8 errors are far below the perturbation budget, so they do
  not add meaningful attack surface.
- The model is already brittle: a tiny eps=0.005 (≈0.002 in raw pixels) buys ~3 pp of
  accuracy loss, and eps=0.05 collapses it to random (~11.5%) — early stopping /
  stronger training would be the honest robustness lever, not quantization.

## Repository layout

```
main.py                 CLI entrypoint (train/quantize/benchmark/adversarial/serve/all/status)
src/
  config.py             paths + hyperparameters
  data.py               FashionMNIST loader, (0..1)->normalized preprocessing
  model.py              half-width ResNet-18 (1-ch, 32x32 stem / 10 classes)
  train.py              SGD (w/ momentum, cosine) training loop
  quantize.py           FX static PTQ + state-dict persistence, ONNX export, ORT qnt, accuracy report
  benchmark.py          latency/size benchmarking (1 thread, median)
  adversarial.py        FGSM/PGD runs, transfer, table+chart writers
  simulate_int8.py      differentiable STE fake-quant twin (adversarial access to INT8)
  serve.py              FastAPI app (/predict, /health)
tests/                  offline fixtures + model/quantize/hosted-API tests
models/                 trained + quantized artifacts (gitignored)
results/                JSON/MD/CSV tables + robustness chart (gitignored)
Dockerfile, .dockerignore, requirements.txt
```

## Notes & limitations

- **Accuracy numbers fluctuate ±1-2 pp** between runs because the eval subsets are
  randomly re-sampled (seeded per run); treat exact values as indicative.
- **FGSM ≈ PGD-8** in the table because the training-margin is tiny and both saturate
  the same losses; a stronger classifier would separate the two.
- The white-box INT8 attack uses a **simulated (fake-quant, STE) twin** because the real
  static-INT8 FX graph is non-differentiable; standard practice for gradient-based
  analysis of quantized models. Clean-accuracy of the twin matches the real INT8 model
  to within noise.
- Docker image is CPU-only and x86_64-optimized (int8 kernels use AVX2). For aarch64
  edge targets rebuild on an ARM host with `pytorch/pytorch:latest-cpu` (no prebuilt
  int8 // onednn kernels exist for ARM PyTorch wheels; ONNX (onnxruntime) is the
  recommended ARM path).
- FaceNet-style *vertex embeddings* and *fine-tuned-family-view* matching were out of
  scope: this repo evaluates the *classifier+quantization+adversarial* pipeline end to
  end on a public dataset. See `results/` for the machine-generated tables/chart and
  `ROBUSTNESS_FINDINGS.md` for the narrative.

## Reproduction

```bash
pip install -r requirements.txt
python main.py all          # full pipeline (~15-20 min CPU)
python -m pytest -q         # 10 offline tests
docker build -t resnet-edge-eval .   # serve workload in a container
docker run --rm -p 8000:8000 resnet-edge-eval
curl -X POST -F "file=@img.png" http://localhost:8000/predict
```