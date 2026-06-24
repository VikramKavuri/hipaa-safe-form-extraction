<div align="center">

# 🔒 Privacy-First Document AI

**Structured field extraction from handwritten compliance forms — using a vision-language model + classical computer vision, running 100% locally.**

[![CI](https://github.com/VikramKavuri/hipaa-safe-form-extraction/actions/workflows/ci.yml/badge.svg)](https://github.com/VikramKavuri/hipaa-safe-form-extraction/actions/workflows/ci.yml)
[![Live demo on HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20demo-Hugging%20Face%20Spaces-yellow)](https://huggingface.co/spaces/VikramKavur/hipaa-safe-form-extraction)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**▶️ Try the live demo:** https://huggingface.co/spaces/VikramKavur/hipaa-safe-form-extraction

</div>

---

## What it does

Messy, handwritten fire-drill / evacuation compliance forms (PDFs and scans) go in →
clean, schema-validated **CSV rows** come out. Every field — names, times, narratives,
and notoriously-hard **checkboxes** — is extracted by a pipeline that combines a
**vision-language model (Qwen2.5-VL)** with a **classical-CV front end**, with
**zero data leaving the machine**.

> **Why this is engineered the way it is:** these forms contain PHI-adjacent data from
> a regulated (developmental-disability services) setting. A cloud OCR API is a
> compliance non-starter. So the entire stack — rendering, OCR, and the VLM — runs
> locally via [Ollama](https://ollama.com). Privacy is an *architectural constraint*,
> not an afterthought.

## Why it's interesting (the engineering)

| Technique | What & why |
|---|---|
| **Constrained decoding** | The Pydantic schema is compiled to JSON Schema and passed to the model as a hard grammar (`format=...`), so the VLM is *forced* to emit valid, typed JSON — no brittle regex-scraping of free prose. |
| **Hybrid CV + VLM** | VLMs are great at reading prose but unreliable on checkboxes (they transcribe every printed option). A dedicated **Tesseract-localized checkbox classifier** crops a tight ROI per question and asks a narrow yes/no — then *overrides* the VLM for those fields. |
| **Page-aware schema sharding** | The 2-page form is split by field order into per-page partial models, so each page only decodes the fields that actually appear on it (smaller, more reliable generations). |
| **Robust scan front-end** | Tesseract OSD orientation correction → grayscale/normalize → CLAHE (recovers faint handwriting) → Hough-line deskew. Each stage is independently toggleable for **ablation studies**. |
| **Batch-robust by design** | Any single page/field failure degrades to `"NA"` and is logged — one bad scan never kills a batch run. |

## Architecture

```mermaid
flowchart LR
    A[PDF / image] --> B[Render @ 600 DPI]
    B --> C[OSD orientation<br/>correction]
    C --> D{Checkbox pass}
    C --> E[CV preprocess<br/>CLAHE + deskew]
    D -->|Tesseract localize ROI| F[VLM checkbox<br/>classifier]
    E --> G[Qwen2.5-VL<br/>schema-constrained]
    F --> H[Merge overrides]
    G --> H
    H --> I[Pydantic-validated<br/>CSV row]
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full data flow and design
rationale, and [`docs/WRITEUP.md`](docs/WRITEUP.md) for the technical deep-dive.

## Quickstart

**Prerequisites:** [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) and
[Ollama](https://ollama.com) installed locally, with the VLM pulled:

```bash
ollama pull qwen2.5vl:7b
```

**Install & run:**

```bash
pip install -e ".[dev,eval,demo]"

# point the tool at your forms (or use the bundled synthetic samples)
export FORMEXTRACT_TESSERACT_CMD="/path/to/tesseract"   # omit if on PATH
formextract run --input data/sample --output outputs/run.csv

formextract info        # show resolved configuration
```

All configuration is environment-driven (prefix `FORMEXTRACT_`); see [`.env.example`](.env.example).

## Demo

**Hosted (no install):** [Live demo on Hugging Face Spaces](https://huggingface.co/spaces/VikramKavur/hipaa-safe-form-extraction)
— upload a form and watch it get extracted field-by-field. The hosted demo routes
inference to **Qwen2.5-VL-72B** via HF Inference Providers (so it needs no GPU); the
production pipeline runs the model **fully locally** for HIPAA-safety.

**Run locally** (fully private, via Ollama):

```bash
pip install -e ".[demo]"
python -m eval.generate_synthetic_forms --n 12 --seed 7   # if not already generated
streamlit run app/streamlit_app.py
```

The VLM backend is pluggable via `FORMEXTRACT_VLM_BACKEND`:
- `ollama` (default) — local Qwen2.5-VL, nothing leaves the machine.
- `hf` — hosted Qwen2.5-VL-72B via HF Inference Providers (what the demo uses).

## Results

The original prototype reported ~82% field-level accuracy on a private internal set.
That claim is now **re-derivable by anyone** via a reproducible evaluation harness
([`eval/`](eval)) that runs on *public, synthetic* forms — so no PHI is needed to
reproduce the numbers. It reports **per-field precision / recall / exact-match**,
fuzzy similarity for free-text, and an **ablation study** isolating each component.

```bash
python -m eval.generate_synthetic_forms --n 12 --seed 7   # public-safe data
python -m eval.run_eval --predictor real                  # real numbers (needs Ollama)
python -m eval.ablations                                  # component contributions
```

<p align="center"><img src="eval/reports/per_field.png" width="640" alt="Per-field performance"></p>

> ⚠️ **Honesty note:** the committed report (`eval/reports/metrics.md`) and the chart
> above are produced by a **mock predictor** so the harness runs in CI on any machine
> without a GPU — *they are not model-accuracy numbers.* Run `--predictor real` on a
> machine with Ollama to generate the true field-level results, which then replace the
> demo report. See [`eval/README.md`](eval/README.md) for the mock-vs-real distinction.

## Project structure

```
src/formextract/
  config.py         # env-driven settings (no hardcoded paths)
  schema.py         # Pydantic model = constrained-decoding grammar
  preprocessing.py  # OSD orient, CLAHE, Hough deskew (ablatable)
  ocr.py            # Tesseract spatial localization
  checkbox.py       # ROI crop + narrow VLM checkbox classifier
  vlm.py            # Qwen2.5-VL calls, page-aware schema sharding
  pipeline.py       # end-to-end orchestration
  cli.py            # `formextract` CLI (Typer)
app/                # Streamlit demo (sample mode runs without a model)
tests/              # pytest unit tests (pure logic, no model needed)
eval/               # ground-truth + metrics + ablations
docs/               # architecture, roadmap, technical write-up
legacy/             # original single-file prototype (provenance)
```

## Privacy & data

- **No PHI is ever committed.** `.gitignore` blocks real forms; only synthetic,
  hand-built sample forms live under `data/sample/`.
- The full pipeline is offline-capable — useful evidence that the design actually
  honors its HIPAA-safe premise.

## Roadmap

This repo is being deliberately built to a production-portfolio standard. See
[`docs/ROADMAP.md`](docs/ROADMAP.md) for the phased plan (eval rigor, CI, demo).

## License

MIT — see [LICENSE](LICENSE).
