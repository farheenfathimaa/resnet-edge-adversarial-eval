"""Global project configuration: paths, dataset stats, model & experiment constants."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"
TESTS_DATA_DIR = ROOT / "tests" / "data"

ROOT.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# --- dataset ---------------------------------------------------------------
# Substitute medium-size public dataset. Historically scored on CIFAR-10 /
# Oxford-Pets; the official CIFAR-10 host (cs.toronto.edu) has been unstable,
# so the default is FashionMNIST (mirrored on a fast AWS bucket). Pass
# --dataset cifar10 to train/quantize on CIFAR-10 instead.
DATASET = "fashionmnist"
IMG_SIZE = 32
PAD = 2
MEAN = 0.2860
STD = 0.3530
FASHION_MNIST_CLASSES = [
    "t-shirt/top", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "ankle boot",
]
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def class_names(dataset: str = DATASET) -> list[str]:
    return CIFAR10_CLASSES if dataset == "cifar10" else FASHION_MNIST_CLASSES


# --- model -----------------------------------------------------------------
# A "small ResNet": ResNet-18 depth topology ([2,2,2,2] BasicBlocks) with a
# width multiplier (0.5 => half the channels of a stock ResNet-18) and a
# single-channel stem adapted for 32x32 grayscale input. Set WIDTH=1.0 for a
# stock ResNet-18 (slower to train on CPU).
IN_CHANNELS = 1
NUM_CLASSES = 10
BLOCKS = (2, 2, 2, 2)
WIDTH = 0.5

MODEL_FP32 = "resnet18w05_fp32.pth"
MODEL_INT8 = "resnet18w05_int8_ptq.pt"
ONNX_FP32 = "resnet18w05_fp32.onnx"
ONNX_INT8_DYN = "resnet18w05_int8_dynamic.onnx"
ONNX_INT8_STATIC = "resnet18w05_int8_static.onnx"

# --- training --------------------------------------------------------------
# Defaults are sized for a laptop-CPU run ("~15 minutes end-to-end"). Bump
# TRAIN_SUBSET / EPOCHS / VAL_SUBSET for more accurate numbers.
TRAIN_SUBSET = 12000          # subsample of the 60k train set (CPU budget)
VAL_SUBSET = 2000             # subsample of the 10k val set
EPOCHS = 2
BATCH_SIZE = 128
LR = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
SEED = 42
# Training uses up to this many threads; the EDGE_THREADS cap only applies to
# inference/benchmarking/serving (it simulates constrained ARM-class devices).
TRAIN_THREADS = 8

# --- edge simulation ---------------------------------------------------------
EDGE_THREADS = 1
CALIBRATION_SAMPLES = 128
BENCH_ITERS = 30
BENCH_BATCH = 1

# --- adversarial -------------------------------------------------------------
ADV_EPSILONS = (0.005, 0.01, 0.02)
ADV_SUBSAMPLE = 200
PGD_STEPS = 8
PGD_ALPHA_RATIO = 0.25  # alpha = ratio * eps