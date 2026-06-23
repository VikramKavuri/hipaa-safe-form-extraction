# Roadmap

This project is being deliberately built from a single-file prototype into a
production-grade portfolio piece. Phases are ordered by recruiter signal-per-effort.

## ✅ Phase 1 — Repository foundation & restructure
- [x] Git, `.gitignore` (PHI-safe), MIT license
- [x] `pyproject.toml` + `requirements.txt`, installable package, `formextract` CLI
- [x] Split the 720-line monolith into typed `src/formextract/` modules
- [x] Env-driven config (no hardcoded machine paths)
- [x] Documented VLM extraction prompt template
- [x] Recruiter-grade README + architecture docs

## ✅ Phase 2 — Code quality (in progress)
- [x] `logging` instead of `print`
- [x] Ruff lint (clean) + mypy config
- [x] Pytest unit tests for pure logic (parsing, checkbox rules, deskew) — *Ollama not required*
- [ ] Raise type-hint coverage; mypy in CI

## ✅ Phase 3 — Evaluation harness, metrics, ablations  ← the differentiator
- [x] Synthetic, public-safe form generator + ground-truth labels (`eval/generate_synthetic_forms.py`)
- [x] Per-field **precision / recall / exact-match**; normalized edit distance for free-text (`eval/metrics.py`)
- [x] Markdown report + per-field plot (`eval/run_eval.py`, `eval/reporting.py`)
- [x] **Ablation runner**: deskew / CLAHE / checkbox-pass / OSD / DPI / VLM-only (`eval/ablations.py`)
- [x] Mock predictor so the whole harness runs in CI without a GPU; unit-tested metrics
- [ ] Run `--predictor real` on an Ollama box to publish the true headline number
- [ ] Error-analysis write-up with real failure examples (after the real run)

## Phase 4 — Reproducibility & CI
- [ ] `Makefile` targets (`setup`, `test`, `lint`, `eval`, `demo`)
- [ ] `pre-commit` hooks; GitHub Actions (lint + type-check + tests) with badge
- [ ] Optional `Dockerfile` (Ollama noted as external dependency)

## Phase 5 — Demo & communication
- [ ] Streamlit demo: upload a form → JSON + annotated checkbox crops
- [ ] Architecture GIF / screen recording in README
- [ ] "Design Decisions" deep-dive write-up (doubles as interview talking points)

## Phase 6 — Polish
- [ ] `CONTRIBUTING.md`, issue templates, clean tagged release `v0.1.0`
