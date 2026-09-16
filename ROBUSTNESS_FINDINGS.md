# Robustness findings

_How edge quantization (INT8) interacts with adversarial perturbations — from this
repo's empirical runs._

## TL;DR

1. **Static INT8 quantization is essentially free** for this model: top-1 accuracy is
   unchanged within noise (0.8115 → 0.8175 / 0.8120 across torch-FX / ORT variants).
2. **Quantization does not make the model more (or less) vulnerable.** FP32, the
   fake-quant INT8 twin, and cross-model transfer all lose accuracy at the same rate
   versus epsilon (≈0.81 / 0.79 / 0.71 at eps 0.005 / 0.010 / 0.020). INT8 rounding error
   is orders of magnitude below the perturbation budget.
3. **The real robustness problem is the model, not the encoding.** At eps = 0.05 (≈0.018
   raw pixels) accuracy collapses to ~random (11.5%). Even eps = 0.005 strips ~3 pp.
   Adversarial training / stronger training is the only honest lever.

## Quantization accuracy

Measured on a 2,000-image validation subset (single-threaded eval):

| variant | top-1 |
|---|---|
| FP32 torch | 0.8115 |
| INT8 torch (FX static PTQ) | 0.8175 |
| FP32 ONNX (ORT) | 0.8115 |
| INT8 ONNX dynamic | 0.8125 |
| INT8 ONNX static | 0.8120 |

Speed (batch 1, 1 thread): static INT8 is **1.9x (torch) / 2.1x (ORT)** faster than FP32
and **3.9x smaller**. ORT **dynamic** INT8 is ~10x *slower*: per-call input quantization
dominates a ~4 MFLOP forward pass — prefer static calibration for image-classification
edge workloads.

## Method

- Whites-box FGSM + PGD-8 (alpha = eps / 4) on a 200-image subset, 3 budgets.
- Perturbations applied in normalized space, clipped to the valid [0,1] pixel range via
  `torchattacks.set_normalization_used` (without it, torchattacks clamps the
  already-normalized tensor to [0,1], collapsing every epsilon to the same image).
- Three evaluations: FP32 (white-box), **INT8 (sim)** (white-box via a straight-through
  estimator fake-quantized twin — the real static-INT8 FX graph is non-differentiable),
  and **transfer** (FP32-crafted perturbations evaluated on INT8, the realistic case when
  an attacker has no gradient access to the quantized artifact).
- FGSM ≈ PGD-8 at every epsilon: this tiny model's decision boundaries are so thin that a
  single sign step saturates the same losses PGD converges to.

## Recommendations

- Deploy the **static** INT8 artifact (torch-FX or ORT-static); expect no accuracy drop
  on this architecture and a ~2x latency / 4x size win on CPU.
- For security: treat the deployed INT8 model as **equally attackable** as FP32 — don't
  claim quantization as a mitigation. Invest in adversarial training / input
  preprocessing, or raise the epsilon-budget question per task.
- If the target hardware is aarch64 (Jetson / RPi), benchmark the ORT-static path there;
  the x86-64 int8 kernels measured here do not map 1:1 to ARM.