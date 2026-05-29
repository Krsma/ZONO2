"""Formatted comparison tables for benchmark results."""

from __future__ import annotations

import pandas as pd


def format_comparison_table(
    aggregate: pd.DataFrame,
    metrics: list[str] | None = None,
) -> str:
    """Format aggregate results as a Markdown table with mean ± CI."""
    if metrics is None:
        metrics = ["mean_trigger_width", "max_trigger_width", "mean_generator_count", "total_time_ms"]

    header = ["Method"]
    for m in metrics:
        header.append(m.replace("_", " ").title())
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for _, row in aggregate.iterrows():
        cells = [str(row["method"])]
        for m in metrics:
            mean_col = f"{m}_mean"
            lo_col = f"{m}_ci95_lo"
            hi_col = f"{m}_ci95_hi"
            if mean_col in row:
                mean = row[mean_col]
                if lo_col in row and hi_col in row:
                    cells.append(f"{mean:.3f} [{row[lo_col]:.3f}, {row[hi_col]:.3f}]")
                else:
                    cells.append(f"{mean:.3f}")
            else:
                cells.append("--")
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def format_latex_table(
    aggregate: pd.DataFrame,
    metrics: list[str] | None = None,
    caption: str = "Method comparison",
    label: str = "tab:comparison",
) -> str:
    """Format aggregate results as a LaTeX tabular."""
    if metrics is None:
        metrics = ["mean_trigger_width", "mean_generator_count", "total_time_ms"]

    n_cols = 1 + len(metrics)
    col_spec = "l" + "r" * len(metrics)
    header_cells = ["Method"]
    for m in metrics:
        parts = m.replace("_", " ").split()
        header_cells.append(" ".join(w.capitalize() for w in parts))

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(header_cells) + r" \\",
        r"\midrule",
    ]

    for _, row in aggregate.iterrows():
        cells = [str(row["method"])]
        for m in metrics:
            mean_col = f"{m}_mean"
            lo_col = f"{m}_ci95_lo"
            hi_col = f"{m}_ci95_hi"
            if mean_col in row:
                mean = row[mean_col]
                if lo_col in row and hi_col in row:
                    pm = (row[hi_col] - row[lo_col]) / 2
                    cells.append(f"${mean:.3f} \\pm {pm:.3f}$")
                else:
                    cells.append(f"${mean:.3f}$")
            else:
                cells.append("--")
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def format_soundness_report(summary: pd.DataFrame) -> str:
    """Summarize soundness invariants."""
    total_violations = int(summary["budget_violations"].sum())
    total_unsound = int(summary["unsound_certificates"].sum())
    total_runs = len(summary)

    lines = [
        "## Soundness Report",
        f"- Total runs: {total_runs}",
        f"- Budget violations: {total_violations}",
        f"- Unsound certificates: {total_unsound}",
    ]
    if total_violations == 0 and total_unsound == 0:
        lines.append("- **All soundness invariants hold.**")
    else:
        lines.append("- **WARNING: Soundness violations detected!**")
        if total_violations > 0:
            bad = summary[summary["budget_violations"] > 0][["method", "seed", "budget_violations"]]
            lines.append(f"  Budget violations:\n{bad.to_string(index=False)}")
        if total_unsound > 0:
            bad = summary[summary["unsound_certificates"] > 0][["method", "seed", "unsound_certificates"]]
            lines.append(f"  Unsound certificates:\n{bad.to_string(index=False)}")
    return "\n".join(lines)
