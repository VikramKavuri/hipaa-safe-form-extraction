# Legacy prototype

`Qwen_main_82_Accuracy.py` is the **original single-file prototype** (~720 lines)
that this project grew out of. It is kept here unchanged for provenance and to
show the "before" state.

The maintained, importable, tested implementation now lives in
[`src/formextract/`](../src/formextract). The logic is identical — it was
refactored into modules, made environment-driven (no hardcoded paths), typed,
linted, logged, and unit-tested.

Do not run this file for new work; use the CLI instead:

```bash
formextract run --input data/sample --output outputs/run.csv
```
