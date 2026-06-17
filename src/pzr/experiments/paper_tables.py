"""Normalize experiment outputs and emit paper-oriented LaTeX tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

STATIC_METHODS = {"girard", "combastel", "pca", "methA", "scott", "box"}


def collect_experiment_tables(inputs: Sequence[Path]) -> pd.DataFrame:
    """Collect benchmark and robotics sweep outputs into one normalized table."""
    frames: list[pd.DataFrame] = []
    seen: set[Path] = set()
    for root in inputs:
        root = root.resolve()
        for path in _candidate_result_files(root):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if path.name == "budget_sweep_summary.csv":
                frames.append(_load_robotics_budget_sweep(path))
            elif path.name == "aggregate.csv":
                loaded = _load_benchmark_aggregate(path)
                if loaded is not None:
                    frames.append(loaded)
    if not frames:
        return pd.DataFrame(columns=_NORMALIZED_COLUMNS)
    table = pd.concat(frames, ignore_index=True)
    for col in _NORMALIZED_COLUMNS:
        if col not in table:
            table[col] = np.nan
    table = table[_NORMALIZED_COLUMNS]
    return table.sort_values(
        ["environment", "trace_source", "horizon", "budget", "method"],
        kind="stable",
    ).reset_index(drop=True)


def write_paper_tables(inputs: Sequence[Path], output: Path) -> dict[str, Path]:
    """Write normalized CSV plus LaTeX table fragments."""
    output.mkdir(parents=True, exist_ok=True)
    combined = collect_experiment_tables(inputs)
    combined_path = output / "combined_summary.csv"
    combined.to_csv(combined_path, index=False)

    artifacts = {"combined_summary": combined_path}
    artifacts["main_k_sweep"] = _write_table(
        _main_k_sweep(combined),
        output / "main_k_sweep.tex",
        caption="Generator-budget sweep at the primary horizon.",
        label="tab:pzr-main-k-sweep",
    )
    artifacts["horizon_sweep"] = _write_table(
        _horizon_sweep(combined),
        output / "horizon_sweep.tex",
        caption="Prediction-horizon sweep at representative budgets.",
        label="tab:pzr-horizon-sweep",
    )
    artifacts["full_methods_h4"] = _write_table(
        _full_methods_h4(combined),
        output / "full_methods_h4.tex",
        caption="Full method comparison at horizon 4.",
        label="tab:pzr-full-methods-h4",
    )
    artifacts["distillation"] = _write_table(
        _distillation_table(combined),
        output / "distillation.tex",
        caption="Learned regret/ranking policy diagnostics.",
        label="tab:pzr-distillation",
    )
    artifacts["overview"] = _write_overview(output / "overview.tex", artifacts)
    return artifacts


def _candidate_result_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    files: list[Path] = []
    files.extend(root.glob("**/budget_sweep_summary.csv"))
    files.extend(root.glob("**/aggregate.csv"))
    return sorted(files)


def _load_robotics_budget_sweep(path: Path) -> pd.DataFrame:
    metadata = _read_json(path.parent / "budget_sweep_metadata.json")
    raw = pd.read_csv(path)
    if raw.empty:
        return pd.DataFrame(columns=_NORMALIZED_COLUMNS)
    grouped = raw.groupby(["candidate", "budget", "method"], dropna=False)
    rows = []
    for (candidate, budget, method), group in grouped:
        rows.append({
            "environment": str(candidate),
            "trace_source": metadata.get("trace_source", "robotics_replay"),
            "monitor_model": metadata.get("monitor_model", ""),
            "length": metadata.get("length", np.nan),
            "seed_count": group["seed"].nunique() if "seed" in group else len(group),
            "budget": int(budget),
            "horizon": metadata.get("horizon", np.nan),
            "method": str(method),
            "method_family": _method_family(str(method)),
            "mean_trigger_width": _mean(group, "mean_trigger_width"),
            "max_trigger_width": _mean(group, "max_trigger_width"),
            "false_positive_rate": _mean(group, "false_positive_rate"),
            "mean_approx_error": _mean(group, "mean_approx_error"),
            "total_time_ms": _mean(group, "total_time_ms"),
            "budget_violations": _sum(group, "budget_violations"),
            "unsound_certificates": _sum(group, "unsound_certificates"),
            "source_path": str(path),
        })
    return pd.DataFrame(rows)


def _load_benchmark_aggregate(path: Path) -> pd.DataFrame | None:
    if "learning" in path.parts:
        return None
    scenario = path.parent.name
    if scenario.startswith("budget_"):
        return None
    if scenario in {"figures", "budget_sweep"}:
        return None
    config = _read_yaml(_nearest_config(path))
    budget = _budget_from_path(path, config)
    horizon = config.get("horizon", np.nan)
    length = config.get("length", np.nan)
    seeds = config.get("seeds", np.nan)
    raw = pd.read_csv(path)
    if raw.empty or "method" not in raw:
        return pd.DataFrame(columns=_NORMALIZED_COLUMNS)
    rows = []
    for _, row in raw.iterrows():
        method = str(row["method"])
        rows.append({
            "environment": scenario,
            "trace_source": "benchmark",
            "monitor_model": "benchmark",
            "length": length,
            "seed_count": seeds,
            "budget": budget,
            "horizon": horizon,
            "method": method,
            "method_family": _method_family(method),
            "mean_trigger_width": _row_metric(row, "mean_trigger_width"),
            "max_trigger_width": _row_metric(row, "max_trigger_width"),
            "false_positive_rate": _row_metric(row, "false_positive_rate"),
            "mean_approx_error": _row_metric(row, "mean_approx_error"),
            "total_time_ms": _row_metric(row, "total_time_ms"),
            "budget_violations": _row_metric(row, "budget_violations", default=0.0),
            "unsound_certificates": _row_metric(row, "unsound_certificates", default=0.0),
            "source_path": str(path),
        })
    return pd.DataFrame(rows)


def _main_k_sweep(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame()
    primary_h = _primary_horizon(combined)
    subset = _longest_per_environment(combined[combined["horizon"] == primary_h].copy())
    rows = []
    for (env, trace_source, length, budget), group in subset.groupby(
        ["environment", "trace_source", "length", "budget"], dropna=False,
    ):
        static = _best(group[group["method_family"] == "static"])
        mpc = _best(group[group["method_family"] == "mpc"])
        if static is None or mpc is None:
            continue
        rows.append(_gain_row(env, trace_source, length, budget, primary_h, static, mpc))
    return pd.DataFrame(rows)


def _horizon_sweep(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame()
    rows = []
    for (env, trace_source, length, budget, horizon), group in combined.groupby(
        ["environment", "trace_source", "length", "budget", "horizon"], dropna=False,
    ):
        static = _best(group[group["method_family"] == "static"])
        mpc = _best(group[group["method_family"] == "mpc"])
        if static is None or mpc is None:
            continue
        rows.append(_gain_row(env, trace_source, length, budget, horizon, static, mpc))
    return pd.DataFrame(rows)


def _full_methods_h4(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame()
    h = _primary_horizon(combined)
    subset = _longest_per_environment(combined[combined["horizon"] == h].copy())
    if subset.empty:
        return pd.DataFrame()
    return subset[[
        "environment", "trace_source", "length", "budget", "method", "method_family",
        "mean_trigger_width", "false_positive_rate",
        "mean_approx_error", "total_time_ms",
    ]].sort_values(["environment", "budget", "method_family", "method"])


def _distillation_table(combined: pd.DataFrame) -> pd.DataFrame:
    learned = combined[combined["method_family"] == "learned"].copy()
    if learned.empty:
        return pd.DataFrame()
    rows = []
    for (env, trace_source, length, budget, horizon), group in combined.groupby(
        ["environment", "trace_source", "length", "budget", "horizon"], dropna=False,
    ):
        static = _best(group[group["method_family"] == "static"])
        learned_rows = group[group["method_family"] == "learned"]
        for _, row in learned_rows.iterrows():
            if static is None:
                continue
            rows.append(_gain_row(env, trace_source, length, budget, horizon, static, row))
    return pd.DataFrame(rows)


def _gain_row(
    env: str,
    trace_source: str,
    length: Any,
    budget: Any,
    horizon: Any,
    static: pd.Series,
    method: pd.Series,
) -> dict[str, Any]:
    static_width = float(static["mean_trigger_width"])
    method_width = float(method["mean_trigger_width"])
    gain = static_width - method_width
    percent = 100.0 * gain / static_width if abs(static_width) > 1e-12 else np.nan
    return {
        "environment": env,
        "trace_source": trace_source,
        "length": length,
        "budget": budget,
        "horizon": horizon,
        "best_static": static["method"],
        "method": method["method"],
        "static_width": static_width,
        "method_width": method_width,
        "width_gain": gain,
        "width_gain_percent": percent,
        "static_fpr": float(static["false_positive_rate"]),
        "method_fpr": float(method["false_positive_rate"]),
        "fpr_gain": float(static["false_positive_rate"] - method["false_positive_rate"]),
        "method_time_ms": float(method["total_time_ms"]),
    }


def _write_table(df: pd.DataFrame, path: Path, *, caption: str, label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        body = "% No rows available for this table.\n"
    else:
        body = df.to_latex(
            index=False,
            escape=True,
            float_format=lambda value: f"{value:.3f}",
            longtable=True,
            caption=caption,
            label=label,
        )
    path.write_text(body, encoding="utf-8")
    return path


def _write_overview(path: Path, artifacts: dict[str, Path]) -> Path:
    lines = [
        "% Auto-generated experiment table overview.",
        "% Include the fragments below from the paper source as needed.",
        "",
    ]
    for name, artifact in artifacts.items():
        if name == "overview":
            continue
        lines.append(f"% {name}: {artifact.name}")
        if artifact.suffix == ".tex":
            lines.append(f"\\input{{{artifact.name}}}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _primary_horizon(df: pd.DataFrame) -> Any:
    horizons = [h for h in df["horizon"].dropna().unique()]
    if 4 in horizons:
        return 4
    if 4.0 in horizons:
        return 4.0
    return sorted(horizons)[0] if horizons else np.nan


def _longest_per_environment(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "length" not in df:
        return df
    pieces = []
    for (_, env_rows) in df.groupby(["environment", "trace_source"], dropna=False):
        lengths = env_rows["length"].dropna()
        if lengths.empty:
            pieces.append(env_rows)
        else:
            pieces.append(env_rows[env_rows["length"] == lengths.max()])
    return pd.concat(pieces, ignore_index=True) if pieces else df


def _best(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    return df.sort_values(
        ["mean_trigger_width", "false_positive_rate", "method"],
        kind="stable",
    ).iloc[0]


def _method_family(method: str) -> str:
    if method in STATIC_METHODS:
        return "static"
    if method.startswith("mpc"):
        return "mpc"
    if method.startswith("learned"):
        return "learned"
    return "other"


def _nearest_config(path: Path) -> Path | None:
    for parent in [path.parent, *path.parents]:
        candidate = parent / "config.yaml"
        if candidate.exists():
            return candidate
    return None


def _budget_from_path(path: Path, config: dict[str, Any]) -> Any:
    for part in path.parts:
        if part.startswith("budget_"):
            try:
                return int(part.split("_", 1)[1])
            except ValueError:
                pass
    return config.get("budget", np.nan)


def _read_yaml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _row_metric(row: pd.Series, metric: str, *, default: float = np.nan) -> float:
    mean_col = f"{metric}_mean"
    if mean_col in row and pd.notna(row[mean_col]):
        return float(row[mean_col])
    if metric in row and pd.notna(row[metric]):
        return float(row[metric])
    return default


def _mean(df: pd.DataFrame, col: str) -> float:
    return float(df[col].mean()) if col in df and not df.empty else np.nan


def _sum(df: pd.DataFrame, col: str) -> float:
    return float(df[col].sum()) if col in df and not df.empty else 0.0


_NORMALIZED_COLUMNS = [
    "environment",
    "trace_source",
    "monitor_model",
    "length",
    "seed_count",
    "budget",
    "horizon",
    "method",
    "method_family",
    "mean_trigger_width",
    "max_trigger_width",
    "false_positive_rate",
    "mean_approx_error",
    "total_time_ms",
    "budget_violations",
    "unsound_certificates",
    "source_path",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build paper tables from PZR results.")
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    artifacts = write_paper_tables(args.input, args.output)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
