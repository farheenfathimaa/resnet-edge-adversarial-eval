"""Adversarial robustness evaluation (FGSM + PGD) on FP32 vs INT8 models.

White-box attacks are run with `torchattacks` (fully local, no API keys)
against both:
  * the trained FP32 model, and
  * the quantized-model twin SimulatedInt8ResNet18 (differentiable INT8).

The real static-INT8 model is not directly attackable (true quantized kernels
have no autograd), so the INT8 model is represented by its differentiable
INT8-simulated twin -- the standard approximation used in the robustness-of-
quantized-networks literature.  Crafted perturbations are also *transferred*
between the two models to show cross-model behavior.

Results are written to results/adversarial_result.{json,md,csv} and a chart to
results/adversarial_robustness.png.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config, data, model
from .simulate_int8 import build_simulated_int8

try:
    import torchattacks
except ImportError:  # pragma: no cover
    torchattacks = None


def _load_clean_data() -> DataLoader:
    train_ds, val_ds = data.load_dataset()
    sub = data.make_val_subset(val_ds, config.ADV_SUBSAMPLE, seed=7)
    return DataLoader(sub, batch_size=config.BATCH_SIZE)


def _accuracy(model, loader) -> float:
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            out = model(data.preprocess_batch(xb))
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return correct / max(total, 1)


def run_attacks(
    fp32_model,
    int8_model,
    loader: DataLoader,
    epsilons=config.ADV_EPSILONS,
    pgd_steps: int = config.PGD_STEPS,
    pgd_alpha_ratio: float = config.PGD_ALPHA_RATIO,
) -> dict:
    """FGSM/PGD white-box accuracy per epsilon, plus cross-model transfer.

    "Transfer" means: perturbations crafted against FP32, then evaluated on
    the INT8 model -- this is what an attacker would experience if they did
    not have gradient access to the quantized deployment artifact.
    """
    if torchattacks is None:
        raise RuntimeError("torchattacks is not installed")

    rows: list[dict] = []
    for eps in epsilons:
        fp32_fgsm = torchattacks.FGSM(fp32_model, eps=eps)
        fp32_pgd = torchattacks.PGD(
            fp32_model, eps=eps, alpha=eps * pgd_alpha_ratio, steps=pgd_steps
        )
        # Our model operates on mean/std-normalized inputs. Let torchattacks
        # clip to the [0,1] pixel range *in pixel space* (it otherwise clamps
        # the already-normalized tensor to [0,1], which collapses every eps).
        for atk_ in (fp32_fgsm, fp32_pgd):
            atk_.set_normalization_used(mean=[config.MEAN], std=[config.STD])
        n_f32_fgsm = n_f32_pgd = n_i8_fgsm = n_i8_pgd = n_f32_clean = n_i8_clean = 0
        n_total = 0
        for xb, yb in loader:
            x = data.preprocess_batch(xb)
            n_total += yb.size(0)
            n_f32_clean += (fp32_model(x).argmax(1) == yb).sum().item()
            n_i8_clean += (int8_model(x).argmax(1) == yb).sum().item()
            adv_f = fp32_fgsm(x, yb)
            adv_p = fp32_pgd(x, yb)
            n_f32_fgsm += (fp32_model(adv_f).argmax(1) == yb).sum().item()
            n_f32_pgd += (fp32_model(adv_p).argmax(1) == yb).sum().item()
            n_i8_fgsm += (int8_model(adv_f).argmax(1) == yb).sum().item()
            n_i8_pgd += (int8_model(adv_p).argmax(1) == yb).sum().item()

        rows.append({"attack": "fp32", "eps": eps,
                     "clean_acc": n_f32_clean / n_total,
                     "fgsm_acc": n_f32_fgsm / n_total,
                     "pgd_acc": n_f32_pgd / n_total})
        rows.append({"attack": "fp32->int8 (transfer)", "eps": eps,
                     "fgsm_acc": n_i8_fgsm / n_total,
                     "pgd_acc": n_i8_pgd / n_total})
        print(f"  [eps={eps:.3f}] fp32 clean={n_f32_clean / n_total:.4f} "
              f"fgsm={n_f32_fgsm / n_total:.4f} pgd={n_f32_pgd / n_total:.4f} | "
              f"int8 (transfer) fgsm={n_i8_fgsm / n_total:.4f} "
              f"pgd={n_i8_pgd / n_total:.4f}")

    for eps in epsilons:
        int8_fgsm = torchattacks.FGSM(int8_model, eps=eps)
        int8_pgd = torchattacks.PGD(
            int8_model, eps=eps, alpha=eps * pgd_alpha_ratio, steps=pgd_steps
        )
        for atk_ in (int8_fgsm, int8_pgd):
            atk_.set_normalization_used(mean=[config.MEAN], std=[config.STD])
        n_fgsm = n_pgd = n_clean = n_total = 0
        for xb, yb in loader:
            x = data.preprocess_batch(xb)
            n_total += yb.size(0)
            n_clean += (int8_model(x).argmax(1) == yb).sum().item()
            adv_f = int8_fgsm(x, yb)
            adv_p = int8_pgd(x, yb)
            n_fgsm += (int8_model(adv_f).argmax(1) == yb).sum().item()
            n_pgd += (int8_model(adv_p).argmax(1) == yb).sum().item()
        rows.append({"attack": "int8 (sim)", "eps": eps,
                     "clean_acc": n_clean / n_total,
                     "fgsm_acc": n_fgsm / n_total,
                     "pgd_acc": n_pgd / n_total})
        print(f"  [eps={eps:.3f}] int8 clean={n_clean / n_total:.4f} "
              f"fgsm={n_fgsm / n_total:.4f} pgd={n_pgd / n_total:.4f}")

    return {
        "epsilons": list(epsilons),
        "pgd_steps": pgd_steps,
        "subsample": config.ADV_SUBSAMPLE,
        "rows": rows,
    }


def write_results(results: dict) -> dict:
    """Persist JSON + CSV + Markdown table and the matplotlib chart."""
    out_json = config.RESULTS_DIR / "adversarial_result.json"
    out_csv = config.RESULTS_DIR / "adversarial_result.csv"
    out_md = config.RESULTS_DIR / "adversarial_result.md"
    out_png = config.RESULTS_DIR / "adversarial_robustness.png"

    out_json.write_text(json.dumps(results, indent=2))

    header = ["attack", "eps", "clean_acc", "fgsm_acc", "pgd_acc"]
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in results["rows"]:
            w.writerow([r.get(k, "") for k in header])

    lines = ["| attack | eps | clean | fgsm | pgd |", "|---|---|---|---|---|"]
    for r in results["rows"]:
        lines.append(
            f"| {r['attack']} | {r.get('eps', '')} | "
            f"{r.get('clean_acc', '-')} | {r.get('fgsm_acc', '-')} | "
            f"{r.get('pgd_acc', '-')} |"
        )
    out_md.write_text("\n".join(lines))

    _render_chart(results, out_png)
    return {"json": str(out_json), "csv": str(out_csv), "md": str(out_md),
            "png": str(out_png)}


def _render_chart(results: dict, out_png: Path) -> None:
    epsilons = results["epsilons"]
    fp32 = {r["eps"]: r for r in results["rows"] if r["attack"] == "fp32"}
    int8 = {r["eps"]: r for r in results["rows"] if r["attack"] == "int8 (sim)"}
    x = np.arange(len(epsilons))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, metric, title in [
        (axes[0], "fgsm_acc", "FGSM"), (axes[1], "pgd_acc", f"PGD ({results['pgd_steps']} steps)"),
    ]:
        fp = [fp32[e][metric] for e in epsilons]
        i8 = [int8[e][metric] for e in epsilons]
        ax.bar(x - width / 2, fp, width, label="FP32", color="#4c72b0")
        ax.bar(x + width / 2, i8, width, label="INT8 (sim)", color="#c44e52")
        clean_fp = fp32[epsilons[0]]["clean_acc"]
        clean_i8 = int8[epsilons[0]]["clean_acc"]
        ax.axhline(clean_fp, color="#4c72b0", ls=":", lw=1)
        ax.axhline(clean_i8, color="#c44e52", ls=":", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{e:.3f}" for e in epsilons])
        ax.set_xlabel("perturbation budget eps")
        ax.set_ylabel("top-1 accuracy")
        ax.set_ylim(0.0, 1.02)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Adversarial accuracy vs epsilon (quantization robustness)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def run_all(checkpoint: str | None = None) -> dict:
    """Full adversarial run and persist results."""
    checkpoint = checkpoint or str(config.MODELS_DIR / config.MODEL_FP32)
    fp32_model = model.load_fp32_model(checkpoint).eval()
    int8_model = build_simulated_int8(checkpoint).eval()
    print("[adversarial] clean accuracy check")
    loader = _load_clean_data()
    print(f"  fp32 clean   : {_accuracy(fp32_model, loader):.4f}")
    print(f"  int8 clean   : {_accuracy(int8_model, loader):.4f}")
    print("[adversarial] running FGSM + PGD (white-box, both models)")
    results = run_attacks(fp32_model, int8_model, loader)
    paths = write_results(results)
    print("[adversarial] results written:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return results


if __name__ == "__main__":
    run_all()