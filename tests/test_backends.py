"""Unit tests for the pluggable VLM backends (no network calls)."""

from __future__ import annotations

import base64
import io

from PIL import Image

from formextract.backends import (
    HFInferenceBackend,
    OllamaBackend,
    _encode_image_data_uri,
    get_backend,
)
from formextract.config import Settings


def test_get_backend_returns_ollama_by_default():
    backend = get_backend(Settings(vlm_backend="ollama"))
    assert isinstance(backend, OllamaBackend)
    assert backend.model == "qwen2.5vl:7b"


def test_get_backend_returns_hf_when_configured():
    # InferenceClient construction is offline (provider resolved at call time).
    backend = get_backend(Settings(vlm_backend="hf", hf_token="dummy"))
    assert isinstance(backend, HFInferenceBackend)
    assert backend.model == "Qwen/Qwen2.5-VL-72B-Instruct"


def test_encode_image_data_uri_downscales(tmp_path):
    path = tmp_path / "big.png"
    Image.new("RGB", (4000, 3000), "white").save(path)  # 12 MP

    uri = _encode_image_data_uri(str(path), max_pixels=1_000_000)
    assert uri.startswith("data:image/jpeg;base64,")

    raw = base64.b64decode(uri.split(",", 1)[1])
    decoded = Image.open(io.BytesIO(raw))
    assert decoded.size[0] * decoded.size[1] <= 1_000_000  # honored the cap
