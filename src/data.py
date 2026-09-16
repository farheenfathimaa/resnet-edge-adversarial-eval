"""Fast, dependency-light dataset loading.

Default dataset is FashionMNIST, cached by torchvision from a fast AWS mirror.
A CIFAR-10 branch is provided but its official host (cs.toronto.edu) is
currently unstable; download may be very slow.  All tensors are preprocessed
once and reused across runs, which keeps training/validation cheap on CPU.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets

from . import config

__all__ = ["load_dataset", "make_loaders", "make_val_subset"]


def _stats() -> tuple[float | torch.Tensor, float | torch.Tensor]:
    """(mean, std) precomputed for the dataset normalized into [0, 1]."""
    if config.DATASET == "cifar10":
        m = torch.tensor([0.4914, 0.4822, 0.4465])
        s = torch.tensor([0.2470, 0.2435, 0.2616])
        return m.view(1, 3, 1, 1), s.view(1, 3, 1, 1)
    return config.MEAN, config.STD


def _load_fashionmnist(tensor_dir: str) -> tuple[TensorDataset, TensorDataset]:
    """FashionMNIST (28x28, grayscale) -> (N,1,32,32) float tensors in [0,1]."""
    train = datasets.FashionMNIST(root=tensor_dir, train=True, download=True)
    test = datasets.FashionMNIST(root=tensor_dir, train=False, download=True)
    return (
        _to_padded_tensor(train.data, train.targets),
        _to_padded_tensor(test.data, test.targets),
    )


def _load_cifar10(tensor_dir: str) -> tuple[TensorDataset, TensorDataset]:
    """CIFAR-10 (32x32, RGB) -> (N,3,32,32) float tensors in [0,1]."""
    train = datasets.CIFAR10(root=tensor_dir, train=True, download=True)
    test = datasets.CIFAR10(root=tensor_dir, train=False, download=True)
    return (
        _to_rgb_tensor(train.data, train.targets),
        _to_rgb_tensor(test.data, test.targets),
    )


def _to_rgb_tensor(pixels, labels: list) -> TensorDataset:
    x = torch.as_tensor(np.asarray(pixels)).permute(0, 3, 1, 2).to(torch.float32) / 255.0
    return TensorDataset(x, torch.as_tensor(labels))


def _to_padded_tensor(images: torch.Tensor, labels: torch.Tensor) -> TensorDataset:
    x = images.to(torch.float32) / 255.0                       # (N,28,28)
    x = F.pad(x.unsqueeze(1), (config.PAD,) * 4)               # (N,1,32,32)
    return TensorDataset(x, labels)


def load_dataset(dataset: str = config.DATASET) -> tuple[TensorDataset, TensorDataset]:
    """Return (train_ds, val_ds) already normalized by the dataset mean/std.

    Weights remain in [0,1]-ish space (values shifted/scaled by mean/std),
    matching the statistics the model was trained against.
    """
    if dataset == "cifar10":
        return _load_cifar10(str(config.DATA_DIR))
    return _load_fashionmnist(str(config.DATA_DIR))


def make_loaders(
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    batch_size: int = config.BATCH_SIZE,
    train_subset: int = config.TRAIN_SUBSET,
    val_subset: int = config.VAL_SUBSET,
    shuffle_train: bool = True,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build DataLoaders, optionally subsampling to stay within CPU budget."""
    from torch import Generator

    g = Generator().manual_seed(config.SEED)
    if train_subset and len(train_ds) > train_subset:
        idx = torch.randperm(len(train_ds), generator=g)[:train_subset]
        train_ds = Subset(train_ds, idx)
    if val_subset and len(val_ds) > val_subset:
        idx = torch.randperm(len(val_ds), generator=g)[:val_subset]
        val_ds = Subset(val_ds, idx)
    dl_train = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train,
        num_workers=num_workers, persistent_workers=False,
    )
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return dl_train, dl_val


def make_val_subset(val_ds: TensorDataset, n: int, seed: int = 1) -> Subset:
    from torch import Generator

    g = Generator().manual_seed(seed)
    idx = torch.randperm(len(val_ds), generator=g)[: min(n, len(val_ds))]
    return Subset(val_ds, idx)


def preprocess_batch(batch: torch.Tensor) -> torch.Tensor:
    """Apply the dataset normalization that the model expects."""
    mean, std = _stats()
    return (batch - mean) / std


def denormalize(batch: torch.Tensor) -> torch.Tensor:
    """Inverse of preprocess_batch, used to plot adversarial images."""
    mean, std = _stats()
    return batch * std + mean