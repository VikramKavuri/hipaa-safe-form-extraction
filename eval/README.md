# Evaluation harness

This is the part of the project that turns *"~82% accuracy"* (a number that used
to live in a filename) into a **reproducible, per-field measurement** on
public-safe data.

## What it measures

| Field family | Schema type | Metric |
|---|---|---|
| Scalar text (addresses, times, Yes/No, narratives) | `str` | exact-match + fuzzy similarity (normalized edit distance) |
| Sets / lists (resident names, per-person times, **checkboxes**) | `list[str]` | micro precision / recall / F1 + exact-set match |

Checkbox fields are also broken out separately so the contribution of the
hybrid Tesseract-localized checkbox classifier is visible on its own.

## How to run

```bash
pip install -e ".[eval]"

# 1) Generate public-safe synthetic forms + ground truth (no real PHI ever).
python -m eval.generate_synthetic_forms --n 12 --seed 7

# 2a) Run the harness end-to-end WITHOUT a model (mock predictor).
#     Proves the metrics/report/plot pipeline works; used in CI.
python -m eval.run_eval --predictor mock

# 2b) Run the REAL pipeline (needs Ollama + Tesseract) for true numbers.
python -m eval.run_eval --predictor real

# 3) Ablation study — quantify each component (needs the real pipeline).
python -m eval.ablations
```

Outputs land in `eval/reports/`:
- `metrics.md` — the full per-field table
- `per_field.png` — per-field bar chart
- `ablations.md` — component contribution table

## Mock vs. real numbers

The committed `metrics.md` is produced by the **mock predictor** — it perturbs
the ground truth with a fixed error profile so the harness is fully runnable in
CI on any machine. **These are not model accuracy numbers.** The headline
accuracy is produced by `--predictor real` on a machine with Ollama; that report
replaces the demo one and is what the README cites.

## Ablation study

`eval/ablations.py` re-runs the real pipeline toggling one component at a time
(deskew, CLAHE, checkbox pass, OSD, 600↔300 DPI, VLM-only) and tabulates the
delta in overall exact-match — the evidence for *why* the pipeline is built the
way it is. See [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) for the
hypotheses each ablation tests.
