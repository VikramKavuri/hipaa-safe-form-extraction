"""Render an ``EvalReport`` to markdown and plots."""

from __future__ import annotations

import sys
from pathlib import Path

from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from formextract.config import CHECKBOX_FIELDS  # noqa: E402

from .metrics import EvalReport  # noqa: E402


def render_markdown(report: EvalReport, title: str = "Evaluation report") -> str:
    lines: list[str] = [f"# {title}", ""]
    lines.append(f"- **Documents evaluated:** {report.n_docs}")
    lines.append(f"- **Overall exact-match (all cells):** {report.overall_exact_match:.1%}")
    lines.append(f"- **Macro-F1 over set fields:** {report.macro_f1_sets:.3f}")
    lines.append("")

    # --- Checkbox fields (the hybrid-classifier story) --------------------
    lines.append("## Checkbox fields")
    cb_rows = []
    for fld in sorted(CHECKBOX_FIELDS):
        if fld in report.sets:
            s = report.sets[fld]
            cb_rows.append([fld, "set", f"{s.exact_rate:.1%}", f"{s.f1:.3f}"])
        elif fld in report.scalars:
            s = report.scalars[fld]
            cb_rows.append([fld, "scalar", f"{s.exact_rate:.1%}", f"{s.mean_similarity:.3f}"])
    lines.append(
        tabulate(cb_rows, headers=["field", "type", "exact", "F1 / sim"], tablefmt="github")
    )
    lines.append("")

    # --- Scalar fields ----------------------------------------------------
    lines.append("## Scalar (text) fields")
    rows = sorted(
        (
            [f, f"{s.exact_rate:.1%}", f"{s.mean_similarity:.3f}", s.n]
            for f, s in report.scalars.items()
        ),
        key=lambda r: r[1],
    )
    lines.append(
        tabulate(rows, headers=["field", "exact-match", "mean sim", "n"], tablefmt="github")
    )
    lines.append("")

    # --- Set / list fields ------------------------------------------------
    lines.append("## Set / list fields")
    rows = sorted(
        (
            [f, f"{s.precision:.3f}", f"{s.recall:.3f}", f"{s.f1:.3f}", f"{s.exact_rate:.1%}"]
            for f, s in report.sets.items()
        ),
        key=lambda r: r[3],
    )
    lines.append(
        tabulate(
            rows, headers=["field", "precision", "recall", "F1", "exact-set"], tablefmt="github"
        )
    )
    lines.append("")
    return "\n".join(lines)


def save_field_plot(report: EvalReport, out_path: str | Path) -> None:
    """Bar chart of per-field exact-match (scalars) / F1 (sets)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, values, colors = [], [], []
    for f, s in sorted(report.scalars.items(), key=lambda kv: kv[1].exact_rate):
        labels.append(f)
        values.append(s.exact_rate)
        colors.append("#4C72B0")
    for f, s in sorted(report.sets.items(), key=lambda kv: kv[1].f1):
        labels.append(f"{f} (set)")
        values.append(s.f1)
        colors.append("#DD8452")

    fig_h = max(4, 0.32 * len(labels))
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.barh(labels, values, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("exact-match (scalar) / F1 (set)")
    ax.set_title("Per-field performance")
    ax.axvline(
        report.overall_exact_match,
        color="black",
        ls="--",
        lw=1,
        label=f"overall {report.overall_exact_match:.0%}",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
