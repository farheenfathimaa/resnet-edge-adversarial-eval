"""resnet-edge-adversarial-eval CLI.

Usage examples
--------------
    python main.py train                        # train small ResNet on FashionMNIST
    python main.py quantize                     # INT8 (torch FX) + ONNX/ORT INT8 + accuracy report
    python main.py benchmark                    # latency & size, FP32 vs quantized, 1 CPU thread
    python main.py adversarial                  # FGSM + PGD on FP32 vs INT8, table + chart
    python main.py all                          # train -> quantize -> benchmark -> adversarial
    python main.py serve --port 8000            # local edge-style FastAPI endpoint
    pytest -q                                  # run the test suite
"""
from __future__ import annotations

import argparse
import time

from src import config


def _cmd_train(args) -> None:
    from src import train
    from src import config as cfg

    train.train(
        epochs=args.epochs or cfg.EPOCHS,
        train_subset=args.train_subset or cfg.TRAIN_SUBSET,
        val_subset=args.val_subset or cfg.VAL_SUBSET,
        batch_size=args.batch_size or cfg.BATCH_SIZE,
        lr=args.lr or cfg.LR,
    )


def _cmd_quantize(args) -> None:
    from src import quantize

    if args.compact:
        quantize.quantize_torch_int8()
        quantize.evaluate_variants(include_onnx=False)
    else:
        quantize.run_all()


def _cmd_benchmark(args) -> None:
    from src import benchmark

    benchmark.run_all()


def _cmd_adversarial(args) -> None:
    from src import adversarial

    adversarial.run_all()


def _cmd_all(args) -> None:
    print("=" * 60)
    print("STEP 1/4 : train")
    print("=" * 60)
    _cmd_train(args)
    print("\n" + "=" * 60)
    print("STEP 2/4 : quantize")
    print("=" * 60)
    _cmd_quantize(args)
    print("\n" + "=" * 60)
    print("STEP 3/4 : benchmark")
    print("=" * 60)
    _cmd_benchmark(args)
    print("\n" + "=" * 60)
    print("STEP 4/4 : adversarial")
    print("=" * 60)
    _cmd_adversarial(args)
    print("\nPipeline complete.")


def _cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run(
        "src.serve:make_live_app",
        host=args.host,
        port=args.port,
        factory=True,
        workers=1,
    )


def _cmd_status(args) -> None:
    from src import config

    print(f"data dir    : {config.DATA_DIR}")
    print(f"models dir  : {config.MODELS_DIR}")
    print(f"results dir : {config.RESULTS_DIR}")
    for name in dir(config):
        if not (name.startswith("MODEL_") or name.startswith("ONNX_")):
            continue
        p = config.MODELS_DIR / getattr(config, name)
        if p.exists():
            print(f"  {name:<26}: OK ({p.stat().st_size / 1e6:.2f} MB)")
        else:
            print(f"  {name:<26}: missing")


def _add_train_args(p) -> None:
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--train-subset", type=int, default=config.TRAIN_SUBSET)
    p.add_argument("--val-subset", type=int, default=config.VAL_SUBSET)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="resnet-edge-adversarial-eval",
        description="Local-only quantization, edge serving and adversarial "
                    "robustness evaluation of a small ResNet.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train the small ResNet (FP32)")
    _add_train_args(p)
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("quantize", help="INT8 quantization + accuracy report")
    p.add_argument("--compact", action="store_true",
                   help="skip ONNX export / ORT quantization; only torch INT8")
    p.set_defaults(func=_cmd_quantize)

    p = sub.add_parser("benchmark", help="latency + size benchmark")
    p.set_defaults(func=_cmd_benchmark)

    p = sub.add_parser("adversarial", help="FGSM/PGD robustness evaluation")
    p.set_defaults(func=_cmd_adversarial)

    p = sub.add_parser("serve", help="start the local FastAPI service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("all", help="run: train -> quantize -> benchmark -> adversarial")
    _add_train_args(p)
    p.add_argument("--compact", action="store_true",
                   help="skip ONNX export / ORT quantization")
    p.set_defaults(func=_cmd_all)

    p = sub.add_parser("status", help="show which artifacts exist")
    p.set_defaults(func=_cmd_status)

    args = parser.parse_args()
    t0 = time.perf_counter()
    args.func(args)
    print(f"\n[cli] finished in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()