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

## 🎯 Phase 3 — Evaluation harness, metrics, ablations  ← the differentiator
- [ ] Synthetic, public-safe sample forms + hand-labeled ground truth
- [ ] Per-field **precision / recall / exact-match**; normalized edit distance for free-text
- [ ] **Ablation study**: deskew / CLAHE / checkbox-pass / DPI on↔off, VLM-only vs hybrid
- [ ] Error-analysis report with failure examples and plots
- [ ] Re-derive and substantiate the headline accuracy number

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
