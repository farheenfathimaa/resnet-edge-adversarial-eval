"""API tests for the local edge-style /predict endpoint."""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from src import config


def _make_png_bytes(gray: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(gray).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def fake_image():
    rng = np.random.default_rng(0)
    return _make_png_bytes(rng.integers(0, 256, size=(28, 28), dtype=np.uint8))


def test_predict_endpoint_returns_valid_output(app_client, fake_image):
    resp = app_client.post("/predict", files={"file": ("img.png", fake_image, "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert "class_id" in body
    assert "class_name" in body
    assert "latency_ms" in body
    assert isinstance(body["class_id"], int)
    assert 0 <= body["class_id"] < config.NUM_CLASSES
    assert body["latency_ms"] >= 0


def test_predict_rejects_empty_upload(app_client):
    resp = app_client.post("/predict", files={"file": ("img.png", b"", "image/png")})
    assert resp.status_code == 400


def test_health_endpoint(app_client):
    resp = app_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["health"] == "ok"


def test_predict_round_trip_consistency(app_client, fake_image):
    """Same image, same model => same prediction."""
    a = app_client.post("/predict", files={"file": ("a.png", fake_image, "image/png")})
    b = app_client.post("/predict", files={"file": ("b.png", fake_image, "image/png")})
    assert a.json()["class_id"] == b.json()["class_id"]