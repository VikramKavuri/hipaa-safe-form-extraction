"""Pluggable vision-language-model backends.

Two backends expose the same ``chat`` interface so the rest of the pipeline is
backend-agnostic:

* :class:`OllamaBackend` — local Ollama. Fully private; the default. Behaviour is
  identical to the original code (schema passed as a hard decoding grammar via
  ``format=``).
* :class:`HFInferenceBackend` — Hugging Face Inference Providers (hosted), used by
  the public demo. Sends the page as a base64 data URI and asks for JSON matching
  the schema. Images are **downscaled** first because vision-token cost scales with
  resolution.

Neither ``ollama`` nor ``huggingface_hub`` is imported at module top level — each
backend imports its own dependency lazily, so an environment that only uses one
backend does not need the other installed.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any, Protocol

from .config import Settings
from .logging_utils import get_logger

log = get_logger(__name__)


class VLMBackend(Protocol):
    """A backend that answers a prompt about one or more images."""

    def chat(
        self,
        *,
        prompt: str,
        image_paths: list[str],
        json_schema: dict | None,
        temperature: float,
        max_tokens: int,
    ) -> str: ...


class OllamaBackend:
    """Local Ollama backend (private, default). Preserves original behaviour."""

    def __init__(self, model: str) -> None:
        self.model = model

    def chat(
        self,
        *,
        prompt: str,
        image_paths: list[str],
        json_schema: dict | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        import ollama

        resp = ollama.chat(
            model=self.model,
            messages=[{"role": "system", "content": prompt, "images": image_paths}],
            format=json_schema,
            options={
                "temperature": temperature,
                "top_p": 1.0,
                "top_k": 1,
                "do_sample": False,
                "max_new_tokens": max_tokens,
            },
        )
        return resp.get("message", {}).get("content", "").strip()


def _encode_image_data_uri(path: str, max_pixels: int) -> str:
    """Downscale to ``max_pixels`` and return a base64 ``data:`` JPEG URI."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


class HFInferenceBackend:
    """Hosted backend via Hugging Face Inference Providers (the demo backend)."""

    def __init__(self, model: str, provider: str, token: str | None, max_image_pixels: int) -> None:
        from huggingface_hub import InferenceClient

        self.model = model
        self.max_image_pixels = max_image_pixels
        # token=None lets huggingface_hub fall back to HF_TOKEN env / stored login.
        # provider is a runtime-configurable string; HF validates it at call time.
        self.client = InferenceClient(provider=provider, api_key=token or None)  # type: ignore[arg-type]

    def chat(
        self,
        *,
        prompt: str,
        image_paths: list[str],
        json_schema: dict | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        # No decoding grammar over the wire, so fold the schema into the prompt.
        text = prompt
        if json_schema is not None:
            text += "\n\nReturn ONLY a JSON object matching this JSON Schema:\n"
            text += json.dumps(json_schema)
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for p in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _encode_image_data_uri(p, self.max_image_pixels)},
                }
            )

        resp = self.client.chat_completion(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


def get_backend(settings: Settings) -> VLMBackend:
    """Construct the configured backend."""
    if settings.vlm_backend == "hf":
        log.info(
            "Using HF Inference backend: %s (provider=%s)",
            settings.hf_vlm_model,
            settings.hf_provider,
        )
        return HFInferenceBackend(
            model=settings.hf_vlm_model,
            provider=settings.hf_provider,
            token=settings.hf_token,
            max_image_pixels=settings.hf_max_image_pixels,
        )
    log.info("Using Ollama backend: %s", settings.vlm_model)
    return OllamaBackend(model=settings.vlm_model)
