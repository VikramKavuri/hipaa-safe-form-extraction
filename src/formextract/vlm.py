"""Vision-language-model extraction via Ollama with constrained decoding.

The form spans two pages; we split the schema by field order so each page only
has to fill the fields that actually appear on it. The per-page Pydantic model's
JSON Schema is passed as ``format=...``, forcing the model to emit valid,
typed JSON (constrained decoding) instead of free-form text.
"""

from __future__ import annotations

import os

import ollama
from pydantic import BaseModel, create_model

from .logging_utils import get_logger
from .parsing import parse_model_json
from .schema import FireDrillFields

log = get_logger(__name__)

# Page 1 carries the first 24 fields; page 2 the remainder.
_PAGE1_FIELD_COUNT = 24


def get_partial_model_for_page(page_num: int) -> type[BaseModel]:
    """Build a Pydantic model containing only the fields for ``page_num``."""
    items = list(FireDrillFields.model_fields.items())
    page_fields = dict(items[:_PAGE1_FIELD_COUNT]) if page_num == 1 else dict(items[_PAGE1_FIELD_COUNT:])
    return create_model(
        f"PartialFields_Page_{page_num}",
        **{name: (field.annotation, ...) for name, field in page_fields.items()},
    )


def run_qwen_extraction(
    page_num: int,
    model_input_path: str,
    work_dir: str,
    file_tag: str,
    *,
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
) -> dict:
    """Send one page image to the VLM and parse its structured JSON response."""
    partial_model = get_partial_model_for_page(page_num)
    log.info("Sending page %d image to VLM (%s)...", page_num, model_name)
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "system", "content": str(prompt), "images": [model_input_path]}],
        format=partial_model.model_json_schema(),
        options={
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": 1,
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
        },
    )

    model_text = response.get("message", {}).get("content", "").strip()
    log.debug("Model output (truncated): %s", model_text[:500])

    out_path = os.path.join(work_dir, f"{file_tag}_page_{page_num}_output.json.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(model_text)

    extracted = parse_model_json(model_text)
    log.info("Extracted %d fields from page %d", len(extracted), page_num)
    return extracted
