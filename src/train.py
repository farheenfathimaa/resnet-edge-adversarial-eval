"""Training + validation for the small ResNet on the local dataset."""
from __future__ import annotations

import time

import torch
import torch.nn as nn

from . import config, data, model


def evaluate(model: nn.Module, loader, device: str = "cpu") -> float:
    """Top-1 accuracy on a DataLoader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = data.preprocess_batch(xb)
            out = model(xb.to(device))
            correct += (out.argmax(dim=1) == yb.to(device)).sum().item()
            total += yb.size(0)
    return correct / max(total, 1)


def train(
    epochs: int = config.EPOCHS,
    train_subset: int = config.TRAIN_SUBSET,
    val_subset: int = config.VAL_SUBSET,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LR,
    out_path: str | None = None,
    seed: int = config.SEED,
    quiet: bool = False,
) -> dict:
    """Train the small ResNet on CPU and persist FP32 weights."""
    torch.manual_seed(seed)
    torch.set_num_threads(config.TRAIN_THREADS)

    train_ds, val_ds = data.load_dataset()
    dl_train, dl_val = data.make_loaders(
        train_ds, val_ds, batch_size=batch_size,
        train_subset=train_subset, val_subset=val_subset,
        num_workers=0,
    )

    net = model.make_resnet18()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(), lr=lr, momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * len(dl_train))
    )
    net.train()
    print(
        f"[train] {config.DATASET} | model=ResNet18x{config.WIDTH} "
        f"| {len(dl_train.dataset)} train imgs | {epochs} epochs "
        f"| bs={batch_size} | lr={lr} | threads={config.TRAIN_THREADS}"
    )
    t0 = time.perf_counter()
    best_acc = 0.0
    for epoch in range(epochs):
        running = correct = total = 0
        for xb, yb in dl_train:
            xb = data.preprocess_batch(xb)
            optimizer.zero_grad()
            out = net(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()
            correct += (out.argmax(dim=1) == yb).sum().item()
            total += yb.size(0)
        val_acc = evaluate(net, dl_val)
        best_acc = max(best_acc, val_acc)
        print(
            f"  epoch {epoch + 1}/{epochs}: loss={running / max(total, 1):.4f} "
            f"train_acc={correct / max(total, 1):.4f} val_acc={val_acc:.4f} "
            f"({time.perf_counter() - t0:.0f}s elapsed)"
        )

    out_path = out_path or str(config.MODELS_DIR / config.MODEL_FP32)
    torch.save(net.state_dict(), out_path)
    result = {
        "val_acc": best_acc,
        "checkpoint": out_path,
        "train_time_s": round(time.perf_counter() - t0, 1),
        "epochs": epochs,
        "train_images_seen": epochs * total,
    }
    print(f"[train] saved FP32 checkpoint -> {out_path}")
    print(
        f"[train] done in {result['train_time_s']:.0f}s | "
        f"best val accuracy = {best_acc:.4f}"
    )
    if not quiet:
        print(result)
    return result


if __name__ == "__main__":
    train()