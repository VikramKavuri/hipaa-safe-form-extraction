# Building a HIPAA-safe document extractor with a local vision-language model

> A technical deep-dive into the design decisions behind this project. Written to
> double as interview talking points.

## The problem

Residential service providers run fire/evacuation drills and record them on paper
forms — handwritten, scanned at varying quality, sometimes rotated, with checkboxes
and free-text narratives. Someone then manually keys ~30 fields per form into a
spreadsheet. It's slow, error-prone, and the data is PHI-adjacent (it identifies
residents of a regulated care setting).

**Goal:** automate the form → structured-data step. **Hard constraint:** no document
data may leave the premises, which rules out every cloud OCR/LLM API.

## Why a vision-language model (and why local)

The classic document-AI stack (Tesseract OCR → regex/heuristics, or layout models
like LayoutLMv3/Donut) struggles with *handwriting* + *heterogeneous layouts* +
*free-text narratives*. A modern VLM reads handwriting and prose well and can be
steered with natural-language instructions.

The privacy constraint then forces local inference. [Ollama](https://ollama.com)
serving **Qwen2.5-VL 7B** gives a capable multimodal model on commodity hardware
with zero network egress. Privacy here is an *architectural decision*, not a feature
bolted on later.

## Three techniques that make it work

### 1. Constrained decoding (don't prompt-and-pray)
Free-form LLM output is a parsing nightmare. Instead, the Pydantic schema
(`schema.py`) is compiled to JSON Schema and passed to the model as a grammar
(`format=...`). The model is *forced* to emit valid, typed JSON. A defensive
`parse_model_json` (strips fences/prose) remains as a backstop but rarely fires.

### 2. A separate, narrow model call for checkboxes
This was the key empirical lesson. Ask a general VLM "what's checked here?" and it
tends to transcribe *every printed option*. So checkboxes get their own path:

1. **Tesseract** locates the labelled row and the colon (`ocr.py`).
2. We crop a tight ROI to the right of the colon.
3. The VLM answers one narrow question about *that crop*: "which of these specific
   options is marked? return `NA` if unsure."
4. Single-/multi-select rules (`_coerce_selection`) reject ambiguous multi-marks
   instead of guessing, and the result *overrides* the page-level VLM for those fields.

### 3. Page-aware schema sharding
A 30-field, 2-page form is split by field order (24 fields on page 1, 6 on page 2).
Each page decodes a smaller schema → more focused, less cross-page hallucination.

Supporting all of this is a classical-CV front-end (OSD orientation correction →
grayscale/normalize → CLAHE → Hough deskew), with each stage independently toggleable
so its contribution can be *measured*, not assumed.

## How I evaluate it (the part most portfolios skip)

A number like "82% accuracy" is meaningless without a reproducible method. So:

- **Public-safe data.** Real forms are PHI and can never be committed. A generator
  (`eval/generate_synthetic_forms.py`) produces synthetic forms + ground-truth labels
  that anyone can regenerate. This *also* demonstrates data-governance awareness.
- **Field-aware metrics** (`eval/metrics.py`). Scalar fields → exact-match + fuzzy
  similarity (normalized edit distance); set/list/checkbox fields → micro
  precision/recall/F1 + exact-set match. Different fields fail differently, so they're
  scored differently.
- **Ablations** (`eval/ablations.py`). Toggle one component at a time (deskew, CLAHE,
  checkbox pass, OSD, 600↔300 DPI, VLM-only) and report the delta in exact-match. This
  is the evidence for *why* the pipeline is shaped the way it is.
- **CI-runnable harness.** A mock predictor lets the whole metrics/report/plot pipeline
  run in CI without a GPU — and the numbers are clearly labeled mock-vs-real so nobody
  is misled.

## What I'd do next

- Run the real model on the synthetic set and publish the per-field table + ablations.
- Fix the duplicate checkbox anchor (`Evacuation_Type` vs `Type_of_Evacuation` share an
  order-insensitive keyword set) — caught precisely *because* the eval surfaces it.
- Try active-learning-style triage: route low-confidence fields to human review.
- Benchmark Qwen2.5-VL against a smaller quantized model to trade accuracy for latency.

## What this project demonstrates

Problem framing under a real constraint, judgment about *when* to reach for classical
CV vs a VLM, production-grade techniques (constrained decoding), and — most importantly
— the discipline to **measure** rather than assert.
