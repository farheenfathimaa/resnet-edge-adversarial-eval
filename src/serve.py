"""Minimal local edge-style inference service (CPU-only) via FastAPI.

Exposes:
  * POST /predict  -- upload an image, get {class_id, class_name, latency_ms}
  * GET  /health   -- model status

Model resolution order:
  1) an explicitly provided model object (tests),
  2) the real static-INT8 model if present (preferred for the "edge" story),
  3) the FP32 checkpoint,
  4) a freshly built (untrained) model so the app still boots without weights.

There is no GPU and no external service dependency -- everything stays on the
local CPU with a bounded thread pool (edge threads), mirroring a Jetson /
Raspberry Pi deployment target.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from . import config, data, model
from .quantize import _load_quant_nn

STATUS_OK = {"health": "ok"}


def _preprocess_image(raw: bytes) -> "torch.Tensor":
    """Decode an uploaded image into the model's normalized input tensor."""
    import torch

    img = Image.open(io.BytesIO(raw)).convert("L")          # grayscale
    img = img.resize((config.IMG_SIZE - 2 * config.PAD,) * 2, Image.BILINEAR)
    img = torch.from_numpy(np.asarray(img, dtype=np.float32)) / 255.0
    img = torch.nn.functional.pad(img.unsqueeze(0).unsqueeze(0),
                                  (config.PAD,) * 4)
    return data.preprocess_batch(img)


def _resolve_model(provided=None):
    """Pick the best available model per the priority list above."""
    if provided is not None:
        return provided, "explicit"
    int8 = config.MODELS_DIR / config.MODEL_INT8
    if int8.exists():
        return _load_quant_nn(int8), "torch int8 (static ptq)"
    fp32 = config.MODELS_DIR / config.MODEL_FP32
    if fp32.exists():
        net = model.load_fp32_model(str(fp32))
        net.eval()
        return net, "torch fp32"
    net = model.make_resnet18()
    net.eval()
    return net, "untrained fallback"


def create_app(provided=None) -> FastAPI:
    import torch

    net, source = _resolve_model(provided)
    torch.set_num_threads(config.EDGE_THREADS)
    app = FastAPI(title="resnet-edge-adversarial-eval",
                  description="CPU-only edge inference service (INT8 quantized ResNet)")

    @app.get("/health")
    def health() -> dict:
        return STATUS_OK | {"model_source": source}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> JSONResponse:
        raw = await file.read()
        if not raw:
            return JSONResponse({"error": "empty upload"}, status_code=400)
        x = _preprocess_image(raw)
        with torch.no_grad():
            t0 = time.perf_counter()
            logits = net(x)
            latency_ms = (time.perf_counter() - t0) * 1000.0
        class_id = int(logits.argmax(dim=1).item())
        names = config.class_names(config.DATASET)
        label = names[class_id] if class_id < len(names) else str(class_id)
        return JSONResponse({
            "class_id": class_id,
            "class_name": label,
            "latency_ms": round(latency_ms, 2),
            "model_source": source,
        })

    return app


def make_live_app() -> FastAPI:
    """Entry point used by uvicorn (e.g. `uvicorn src.serve:make_live_app --factory`)."""
    return create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(make_live_app(), host="0.0.0.0", port=8000)