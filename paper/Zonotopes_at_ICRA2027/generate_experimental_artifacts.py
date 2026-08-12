#!/usr/bin/env python3
"""Generate paper-local experimental figures and LaTeX tables.

This script reads the frozen paper-evaluation and guarded-Vote3 artifacts.  It
never writes under ``results/``; generated paper assets live next to root.tex.

Run with the project runtime::

    external/miniconda3/envs/pzr-robot-arm/bin/python \
        paper/Zonotopes_at_ICRA2027/generate_experimental_artifacts.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[1]
OUTPUT = PAPER / "generated"
STYLE = PAPER / "academic.mplstyle"

CANONICAL = ROOT / "results/paper-evaluation-v3"
GUARDED = ROOT / "results/prp-vote3-guarded-paper-sweep-v1"
V4 = ROOT / "results/paper-evaluation-v4"
BUDGETS = (40, 80, 120, 150, 200, 250, 500)
REDUCERS = ("girard", "scott", "pca", "combastel")

# The headline table reports one representative budget; Figure 2 carries the
# full sweep.  150 is the mid budget and the one the H/W ablation fixes.
HEADLINE_BUDGET = 150
# Every trace advances at dt = 0.1 s, so a monitor must sustain 10 events/s to
# keep up with the system it observes.
EVENT_RATE_HZ = 10.0
# ieeeconf.cls: \textwidth 7.0in, \columnsep 0.2in.
COLUMN_WIDTH = 3.4
TEXT_WIDTH = 7.0

CANONICAL_METHODS = (
    "girard",
    "scott",
    "pca",
    "combastel",
    "mpc_terminal_beam",
    "mpc_terminal_full_width",
    "mpc_terminal_beam_predictive_linear",
    "pairwise_ranking_policy",
)
CONFIRMATION_METHODS = (
    "g15_clean148",
    "dagger05_vote3",
    "dagger05_vote3_guarded",
    "mpc_terminal_beam_predictive_linear",
)
# The two voting policies exist only on the confirmation cohort (seeds 328-347),
# so the canonical headline table and Fig. 1 cannot yet show the policy the paper
# promotes.  Closing that gap needs one canonical run over seeds 100-119.  They
# are listed here rather than added to CANONICAL_METHODS so every consumer keeps
# a reserved slot for them: the table prints a placeholder row, the figure prints
# a standing note, and both fill themselves in when the cells appear.
PENDING_CANONICAL_METHODS = (
    "dagger05_vote3",
    "dagger05_vote3_guarded",
)

METHOD_LABELS = {
    "girard": "Girard",
    "scott": "Scott",
    "pca": "PCA",
    "combastel": "Combastel",
    "mpc_terminal_beam": "MPC-B",
    "mpc_terminal_full_width": "MPC-F",
    "mpc_terminal_beam_predictive_linear": "MPC-L",
    "pairwise_ranking_policy": "G15/Clean148",
    "g15_clean148": "G15/Clean148",
    "dagger05_vote3": "Vote3",
    "dagger05_vote3_guarded": "Vote3-Guarded",
}
# Wide tables must fit \textwidth = 505.9pt, so they use abbreviated headers.
SHORT_LABELS = dict(METHOD_LABELS, **{
    "pairwise_ranking_policy": "G15",
    "g15_clean148": "G15",
    "dagger05_vote3_guarded": "V3-Guard",
    "combastel": "Combast.",
})

# One colour per method, shared by every figure.  The previous revision reused
# Girard's blue for Vote3 and Scott's orange for Vote3-Guarded, so the same
# colour named different methods in adjacent figures.
COLORS = {
    "girard": "#4C72B0",
    "scott": "#937860",
    "pca": "#55A868",
    "combastel": "#8172B3",
    "mpc_terminal_beam": "#E69F00",
    "mpc_terminal_full_width": "#999999",
    "mpc_terminal_beam_predictive_linear": "#000000",
    "pairwise_ranking_policy": "#D55E00",
    "g15_clean148": "#D55E00",
    "dagger05_vote3": "#0072B2",
    "dagger05_vote3_guarded": "#009E73",
}
# Redundant with colour so the figures survive grayscale reduction.
MARKERS = {
    "girard": "o",
    "scott": "s",
    "pca": "^",
    "combastel": "D",
    "mpc_terminal_beam": "v",
    "mpc_terminal_full_width": "X",
    "mpc_terminal_beam_predictive_linear": "P",
    "pairwise_ranking_policy": "*",
    "g15_clean148": "*",
    # Distinct from Girard's circle and Combastel's diamond: once the canonical
    # cells land, all ten methods share the plane in Fig. 1.
    "dagger05_vote3": "p",
    "dagger05_vote3_guarded": "h",
}
LINESTYLES = {
    "girard": "-",
    "scott": "-",
    "pca": "-",
    "combastel": "-",
    "mpc_terminal_beam": "--",
    "mpc_terminal_full_width": "--",
    "mpc_terminal_beam_predictive_linear": "-.",
    "pairwise_ranking_policy": "-",
    "g15_clean148": "-",
    "dagger05_vote3": "--",
    "dagger05_vote3_guarded": "-",
}
FAMILIES = (
    ("Fixed reducer", ("girard", "scott", "pca", "combastel")),
    ("Offline oracle", ("mpc_terminal_beam", "mpc_terminal_full_width")),
    ("Online", ("mpc_terminal_beam_predictive_linear", "pairwise_ranking_policy",
                *PENDING_CANONICAL_METHODS)),
)


def _configure_matplotlib() -> None:
    plt.style.use(STYLE)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_csv(path: Path) -> pd.DataFrame:
    _require(path.is_file(), f"missing source artifact: {path}")
    return pd.read_csv(path)


def _completed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["status"].astype(str) == "completed"].copy()


def _throughput(frame: pd.DataFrame) -> pd.Series:
    return frame["event_count"].astype(float) * 1000.0 / frame["event_loop_time_ms"].astype(float)


def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(
        OUTPUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(OUTPUT / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _source_hashes(paths: list[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _style(method: str) -> dict:
    return {
        "color": COLORS[method],
        "marker": MARKERS[method],
        "linestyle": LINESTYLES[method],
    }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _canonical_nominal() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = _read_csv(CANONICAL / "generalization/summary.csv")
    # The cell count is derived rather than fixed at 1,120 so that a rerun which
    # adds the pending voting policies passes without editing this assertion --
    # while a rerun that silently drops a seed or a bound still fails it.
    observed = set(summary["method"])
    _require(set(CANONICAL_METHODS) <= observed, "canonical method set is missing a method")
    _require(observed - set(CANONICAL_METHODS) <= set(PENDING_CANONICAL_METHODS),
             "canonical cohort carries an unexpected method")
    _require(len(summary) == len(observed) * 20 * len(BUDGETS),
             "canonical nominal summary is not a full method x seed x bound grid")
    _require(set(summary["seed"].astype(int)) == set(range(100, 120)), "canonical seed set differs")
    _require(set(summary["budget"].astype(int)) == set(BUDGETS), "canonical budget set differs")
    _require((summary["event_count"].astype(int) == 500).all(), "canonical nominal trace length differs")
    aggregate = _read_csv(
        CANONICAL / "science-report/artifacts/nominal_generalization_aggregates.csv"
    )
    _require(len(aggregate) == len(observed) * len(BUDGETS),
             "nominal aggregate does not cover every method-budget point")
    return summary, aggregate


def _confirmation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nominal = _read_csv(GUARDED / "evaluate-nominal/summary.csv")
    fixed = _read_csv(GUARDED / "evaluate-fixed/summary.csv")
    pairs = _read_csv(GUARDED / "report/artifacts/nominal_guard_benefit.csv")
    cells = _read_csv(GUARDED / "report/artifacts/trace_cells.csv")
    _require(len(nominal) == 560, "confirmation nominal summary must contain 560 cells")
    _require(len(fixed) == 112, "confirmation fixed summary must contain 112 cells")
    _require(set(nominal["method"]) == set(CONFIRMATION_METHODS), "confirmation method set differs")
    _require(set(nominal["seed"].astype(int)) == set(range(328, 348)), "confirmation seed set differs")
    _require(set(nominal["budget"].astype(int)) == set(BUDGETS), "confirmation budget set differs")
    _require((nominal["status"].astype(str) == "completed").all(), "nominal confirmation has unavailable cells")
    _require((fixed["status"].astype(str) == "completed").all(), "fixed confirmation has unavailable cells")
    _require(len(pairs) == 140, "guard-benefit table must contain 140 paired cells")
    return nominal, fixed, pairs, cells


def _search_ablation() -> pd.DataFrame:
    frame = _read_csv(CANONICAL / "ablation/summary.csv")
    _require(len(frame) == 80, "H/W ablation must contain 80 cells")
    _require((frame["status"].astype(str) == "completed").all(), "H/W ablation has unavailable cells")
    _require(set(frame["budget"].astype(int)) == {150}, "H/W ablation must fix budget 150")
    frame = frame.copy()
    frame["H"] = frame["method"].str.extract(r"_h(\d+)_").astype(int)
    frame["W"] = frame["method"].str.extract(r"_w(\d+)$").astype(int)
    frame["throughput"] = _throughput(frame)
    return frame


def _objective_comparison() -> pd.DataFrame:
    frame = _read_csv(CANONICAL / "objective-comparison/summary.csv")
    _require(len(frame) == 56, "objective comparison must contain 56 cells")
    _require((frame["status"].astype(str) == "completed").all(), "objective comparison has unavailable cells")
    frame = frame.copy()
    frame["throughput"] = _throughput(frame)
    return frame


def _aggregate_index(aggregate: pd.DataFrame) -> pd.DataFrame:
    frame = aggregate.set_index(["method", "budget"]).sort_index()
    return frame


def _scott_failures() -> pd.DataFrame:
    """Per-cell records for the fixed Scott runs that never complete.

    The aggregated ``summary.csv`` reports these cells only as
    ``fallback_failed`` with a null loss, but each cell directory keeps the
    partial record: where the interval fallback fired, how much of the trace had
    been consumed, and how loose the set already was.  Those three numbers are
    the whole reason Scott is worth a table instead of an omission note.
    """
    frames = [
        pd.read_csv(path)
        for path in sorted(CANONICAL.glob("headline/cells/*/*/seed-*/budget-*/scott/summary.csv"))
    ]
    _require(len(frames) == 28, "fixed Scott must expose 28 per-cell records")
    frame = pd.concat(frames, ignore_index=True)
    _require((frame["status"].astype(str) == "fallback_failed").all(),
             "every fixed Scott cell is expected to fail")
    _require((frame["failure_type"].astype(str) == "IntervalFallback").all(),
             "fixed Scott is expected to fail through the interval fallback")
    _require(set(frame["budget"].astype(int)) == set(BUDGETS), "Scott failure budgets differ")
    return frame


def _availability(summary: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """Completed cells per method over the whole sweep.

    For a runtime monitor "does it run at all" is a headline property, not a
    footnote, so the headline table carries it as a column beside loss and FPR.
    """
    result = {}
    for method, rows in summary.groupby("method"):
        result[str(method)] = (int((rows["status"].astype(str) == "completed").sum()), len(rows))
    return result


# ---------------------------------------------------------------------------
# Figure 1: accuracy-throughput Pareto
# ---------------------------------------------------------------------------


def _plot_tradeoff(summary: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    """The accuracy-cost trade-off, plus the one axis it cannot encode.

    Panel (a) is the paper's central claim in a plane: fixed reducers are fast
    and loose, predictive MPC is tight but misses the 10 Hz deadline at every
    bound, and the distilled policy reaches MPC accuracy inside the fixed-reducer
    speed class.  Its trails already carry the bound sweep, so a separate
    sweep figure would restate the same data; the one quantity the plane has no
    axis for is the false-positive rate, which panel (b) supplies.
    """
    index = _aggregate_index(aggregate)
    # Side by side across the text width rather than stacked in one column: the
    # plane holds eight labelled trails and the sweep six labelled curves, and
    # neither can be labelled in place at 3.4 in without collisions.  Panel (a)
    # gets the wider share because its labels sit inside the axes.
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.6),
                             width_ratios=(1.12, 1.0), constrained_layout=True)
    axis = axes[0]

    # The 10 Hz deadline carries no rule or shading here.  Where each family
    # falls relative to it is a claim the text and the real-time column of the
    # headline table both make; drawing it as well split the plane in two and
    # competed with the trails for the reader's attention.
    #
    # Label offsets are in axes-fraction points and tuned against the rendered
    # figure so trails stay readable at 3.4 in.
    # (offset in points, which end of the budget trail to anchor to).  Labels go
    # wherever they stay legible rather than always at the same end.
    offsets = {
        "girard": ((0, 8), -1),
        "scott": ((0, 8), 0),
        "pca": ((2, 7), 0),
        "combastel": ((-2, 8), 0),
        "mpc_terminal_beam": ((7, -12), 0),
        "mpc_terminal_full_width": ((7, 4), 0),
        "mpc_terminal_beam_predictive_linear": ((-2, 7), 0),
        "pairwise_ranking_policy": ((-3, -12), 0),
        # Reserved.  Both voting policies should land in G15's neighbourhood, so
        # they are anchored to opposite ends of their trails to keep three labels
        # off one another.  Provisional: re-tune against the real cells.
        "dagger05_vote3": ((-3, 8), 0),
        "dagger05_vote3_guarded": ((0, -11), -1),
    }
    drawn = [method for method in (*CANONICAL_METHODS, *PENDING_CANONICAL_METHODS)
             if (method, HEADLINE_BUDGET) in index.index]
    pending = [method for method in PENDING_CANONICAL_METHODS if method not in drawn]
    for method in drawn:
        points = []
        for budget in BUDGETS:
            row = index.loc[(method, budget)]
            if str(row["available"]) != "True":
                continue
            points.append((float(row["median_throughput_events_per_second"]),
                           float(row["median_mean_approx_loss"])))
        values = np.asarray(points, dtype=float)
        style = _style(method)
        axis.plot(
            values[:, 0], values[:, 1], alpha=0.85, markersize=3.4,
            markerfacecolor="none", markeredgewidth=0.8, zorder=3, **style,
        )
        # Solid marker at the largest budget so the trail direction is legible.
        axis.plot(
            values[-1, 0], values[-1, 1], marker=style["marker"], color=style["color"],
            markersize=4.4, linestyle="none", zorder=4,
        )
        offset, anchor = offsets[method]
        axis.annotate(
            METHOD_LABELS[method], xy=values[anchor], xytext=offset,
            textcoords="offset points", fontsize=6.4, color=style["color"],
            ha="center", zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.82},
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(0.35, 900)
    # The good corner of this plane is the lower right, which is not the
    # convention a reader arrives with, so each axis states its own direction.
    axis.set_xlabel("Event-loop throughput (events/s)\n" r"(faster $\rightarrow$)")
    axis.set_ylabel(r"Event-mean approximation loss $\overline{L}$"
                    "\n" r"(tighter $\downarrow$)")
    axis.grid(True, which="major", linewidth=0.45)
    axis.set_title("(a)", loc="left", fontsize=8)
    # A standing note, not decoration: the policy the paper promotes is missing
    # from this plane, and the note disappears by itself once the cells exist.
    if pending:
        axis.annotate(
            f"pending canonical run: {', '.join(METHOD_LABELS[m] for m in pending)}",
            xy=(0.025, 0.955), xycoords="axes fraction", fontsize=6.0, color="#CC0000",
            ha="left", va="top",
        )

    # (b) Macro FPR against the bound, on a log axis: the range spans 0.8-87 %,
    # so a linear axis flattens the entire dynamic-selector region onto the
    # baseline and hides the thirty-fold separation between the families.
    lower = axes[1]
    positions = _budget_positions()
    oracles = ("mpc_terminal_beam", "mpc_terminal_full_width")
    low, high = [], []
    for budget in BUDGETS:
        values = [100.0 * float(index.loc[(method, budget)]["macro_fpr"]) for method in oracles]
        low.append(min(values))
        high.append(max(values))
    lower.fill_between(positions, low, high, color=COLORS["mpc_terminal_beam"],
                       alpha=0.28, linewidth=0, zorder=2)
    # Anchor: which budget index to label at, and the offset in points.  The two
    # low curves cross near both ends of the sweep and can only be told apart in
    # the middle, so they are labelled there rather than at a common edge.
    sweep_labels = {
        # PCA crosses below Scott between b=250 and b=500, so it is labelled
        # before the crossing rather than in the pile-up at the right edge.
        "pca": (-2, (0, 7)),
        "scott": (-1, (5, -1)),
        "girard": (-1, (5, 1)),
        "combastel": (-1, (5, -4)),
        "pairwise_ranking_policy": (1, (0, 6)),
        "mpc_terminal_beam_predictive_linear": (3, (0, -11)),
        # Reserved for the pending canonical run; see PENDING_CANONICAL_METHODS.
        "dagger05_vote3": (-1, (5, 3)),
        "dagger05_vote3_guarded": (-1, (5, -3)),
    }
    for method, (anchor, offset) in sweep_labels.items():
        if (method, HEADLINE_BUDGET) not in index.index:
            continue
        x, y = _sweep_series(index, method, "macro_fpr")
        band_low = _sweep_series(index, method, "macro_fpr_ci_low")[1] * 100.0
        band_high = _sweep_series(index, method, "macro_fpr_ci_high")[1] * 100.0
        style = _style(method)
        lower.fill_between(x, band_low, band_high, color=style["color"], alpha=0.16,
                           linewidth=0, zorder=2)
        lower.plot(x, y * 100.0, markersize=3.2, markerfacecolor="none",
                   markeredgewidth=0.8, zorder=3, **style)
        lower.annotate(
            METHOD_LABELS[method], xy=(x[anchor], y[anchor] * 100.0), xytext=offset,
            textcoords="offset points", fontsize=6.4, color=style["color"],
            ha="left" if anchor == -1 else "center", va="center", zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.82},
        )
    # The oracle envelope is only 0.2 pp wide and MPC-L runs inside it, so it
    # reads as a thickened line rather than a band and cannot carry a label in
    # place: a leader points at it from the empty region under G15's peak.
    lower.annotate(
        f"{METHOD_LABELS['mpc_terminal_beam']}, {METHOD_LABELS['mpc_terminal_full_width']}",
        xy=(positions[5], high[5]), xytext=(positions[3] + 0.15, 4.0),
        fontsize=6.4, color=COLORS["mpc_terminal_beam"], ha="left", va="center", zorder=5,
        arrowprops={"arrowstyle": "-", "linewidth": 0.5,
                    "color": COLORS["mpc_terminal_beam"], "shrinkB": 1.0},
    )
    lower.annotate("band: paired-seed bootstrap CI",
                   xy=(0.02, 0.04), xycoords="axes fraction", fontsize=6.0, color="0.35")
    lower.annotate("Scott: no $b{=}40$ cell", xy=(0.02, 0.15), xycoords="axes fraction",
                   fontsize=6.0, color=COLORS["scott"])
    lower.set_yscale("log")
    lower.set_xticks(positions, [str(budget) for budget in BUDGETS])
    # Room on the right for the four end-labels.
    lower.set_xlim(-0.35, len(BUDGETS) + 0.55)
    lower.set_xlabel("Transform bound $b$")
    lower.set_ylabel("Macro FPR (%)")
    lower.grid(True, which="major", axis="y", linewidth=0.45)
    lower.set_title("(b)", loc="left", fontsize=8)
    _save(fig, "accuracy_cost_tradeoff")


# ---------------------------------------------------------------------------
# Budget sweep helpers
# ---------------------------------------------------------------------------


def _budget_positions() -> np.ndarray:
    return np.arange(len(BUDGETS), dtype=float)


def _sweep_series(index: pd.DataFrame, method: str, column: str) -> tuple[np.ndarray, np.ndarray]:
    positions, values = [], []
    for position, budget in enumerate(BUDGETS):
        row = index.loc[(method, budget)]
        if str(row["available"]) != "True":
            continue
        positions.append(float(position))
        values.append(float(row[column]))
    return np.asarray(positions), np.asarray(values)




# ---------------------------------------------------------------------------
# Figure 3: guarded voting
# ---------------------------------------------------------------------------


def _cross_check_guard(cells: pd.DataFrame, pairs: pd.DataFrame) -> None:
    """Recompute the guard-versus-vote win counts from the raw cells.

    The section quotes these counts, and the frozen report artifact derives them
    independently, so disagreeing here means one of the two is stale.
    """
    nominal = cells.loc[cells["scope"].astype(str) == "nominal"]
    _require(len(nominal) == 560, "report trace-cell nominal count differs")
    loss = nominal.pivot_table(index=["seed", "budget"], columns="method",
                               values="mean_approx_loss")
    guarded = loss["dagger05_vote3_guarded"].astype(float)
    pure = loss["dagger05_vote3"].astype(float)
    reported = pairs["mean_loss_ratio_guarded_vs_pure"].astype(float)
    _require(int((guarded < pure).sum()) == int((reported < 1.0).sum()),
             "guard win count differs from report artifact")
    _require(int((guarded > pure).sum()) == int((reported > 1.0).sum()),
             "guard loss count differs from report artifact")


def _guard_flow(decisions: pd.DataFrame) -> dict:
    """Vote winner to final action over the decisions where the guard engages."""
    guarded = decisions.loc[
        (decisions["method"].astype(str) == "dagger05_vote3_guarded")
        & (decisions["scope"].astype(str) == "nominal")
        & (decisions["over_bound"].astype(str) == "True")
    ].copy()
    _require(len(guarded) == 69120, "guarded nominal over-bound decision count differs")
    guarded["winner"] = guarded["plurality_order"].map(lambda value: json.loads(value)[0])
    invoked = guarded.loc[guarded["guard_invoked"].astype(str) == "True"]
    flow = Counter(zip(invoked["winner"], invoked["selected_action"].astype(str), strict=True))
    _require(sum(flow.values()) == len(invoked), "flow counts must partition the activations")
    return {
        "decisions": int(len(guarded)),
        "unanimous": int((guarded["winner_margin"].astype(int) == 3).sum()),
        "invoked": int(len(invoked)),
        "overrides": int((invoked["guard_override"].astype(str) == "True").sum()),
        "flow": flow,
    }


def _fixed_scott_usage(decisions: pd.DataFrame) -> dict:
    """How often a controller applies Scott on the traces where fixed Scott dies.

    This is the counterpart to the Scott failure profile: the reducer that cannot
    survive being scheduled unconditionally is the one the controller picks
    almost always, and it never meets an infeasible candidate while doing so.
    """
    frame = decisions.loc[
        (decisions["method"].astype(str) == "dagger05_vote3_guarded")
        & (decisions["scope"].astype(str) == "fixed")
        & (decisions["over_bound"].astype(str) == "True")
    ]
    selected = frame["selected_action"].astype(str)
    return {
        "decisions": int(len(frame)),
        "scott_share_percent": float(100.0 * (selected == "scott").mean()),
        "min_budget_scott_share_percent": float(
            100.0 * frame.assign(hit=selected == "scott")
            .groupby("budget")["hit"].mean().min()),
        "infeasible_candidates": int(frame["infeasible_candidate_count"].astype(int).sum()),
    }


def _ribbon(axis, x0, x1, top0, top1, height, **kwargs) -> None:
    """One flow band, drawn as a pair of mirrored cubic Beziers."""
    middle = 0.5 * (x0 + x1)
    vertices = [
        (x0, top0), (middle, top0), (middle, top1), (x1, top1),
        (x1, top1 - height),
        (middle, top1 - height), (middle, top0 - height), (x0, top0 - height),
        (x0, top0),
    ]
    codes = [
        MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    axis.add_patch(PathPatch(MplPath(vertices, codes), **kwargs))


def _plot_guard_flow(decisions: pd.DataFrame) -> dict:
    """What the symbolic override actually does to the ensemble's choice.

    A rate against the transform bound answers "how often could the guard fire",
    which is not the question a reader has: it conflates opportunity with effect
    and reads as though the guard grows busy at loose bounds.  The flow answers
    "what does it change", and the answer is a mechanism -- the guard keeps Scott
    three times in four but overrides Girard three times in five, moving
    selections toward Scott and Combastel.  That is the same shift the
    composition table records, arrived at independently.
    """
    summary = _guard_flow(decisions)
    flow = summary["flow"]
    order = ("girard", "scott", "combastel", "pca")
    outgoing = {reducer: sum(count for (src, _), count in flow.items() if src == reducer)
                for reducer in order}
    incoming = {reducer: sum(count for (_, dst), count in flow.items() if dst == reducer)
                for reducer in order}
    total = sum(outgoing.values())

    fig, axes = plt.subplots(
        2, 1, figsize=(COLUMN_WIDTH, 2.05), height_ratios=(1, 8),
        constrained_layout=True,
    )

    # The scale strip: the flow below covers 6 % of decisions, and a reader who
    # sees only the alluvial would have no way to know that.
    strip, axis = axes
    unanimous = summary["unanimous"]
    split = summary["decisions"] - unanimous
    strip.barh(0, unanimous, color="0.86", height=0.55)
    strip.barh(0, split, left=unanimous, color=COLORS["dagger05_vote3_guarded"], height=0.55)
    strip.annotate(f"unanimous 3\u20130 vote, guard cannot fire: {unanimous:,} "
                   f"({100.0 * unanimous / summary['decisions']:.0f}%)",
                   xy=(unanimous * 0.5, 0), fontsize=5.8, color="0.3", ha="center", va="center")
    strip.annotate(f"split: {split:,} ({100.0 * split / summary['decisions']:.1f}%)",
                   xy=(summary["decisions"], 0), xytext=(-2, 9), textcoords="offset points",
                   fontsize=5.8, color=COLORS["dagger05_vote3_guarded"], ha="right", va="center")
    strip.set_xlim(0, summary["decisions"] * 1.005)
    strip.set_ylim(-0.6, 0.6)
    strip.set_axis_off()

    gap = 0.055 * total
    span = total + gap * (len(order) - 1)
    # Gutters are sized to the label text and nothing more.  The earlier layout
    # reserved a third of the panel on each side, which left the bands running
    # almost vertically over a 1.3 in span and made a flow read as a stack.
    left_x, right_x = 0.176, 0.812

    tops, cursor = {}, span
    for reducer in order:
        tops[reducer] = cursor
        cursor -= outgoing[reducer] + gap
    in_tops, cursor = {}, span
    for reducer in order:
        in_tops[reducer] = cursor
        cursor -= incoming[reducer] + gap

    # Ribbons are drawn thinnest-last so a narrow override stays visible where it
    # runs alongside a wide one.
    out_cursor = dict(tops)
    in_cursor = dict(in_tops)
    for source in order:
        for target in sorted(order, key=lambda name: -flow.get((source, name), 0)):
            count = flow.get((source, target), 0)
            if count == 0:
                continue
            kept = source == target
            # Overrides are drawn saturated and outlined, votes the guard leaves
            # alone are pale and unoutlined.  That contrast, not the hue, is what
            # separates the two populations under grayscale reduction; the outline
            # keeps crossing bands apart.  A per-source hatch was tried here and
            # dominates the panel at 3.4 in.
            _ribbon(
                axis, left_x, right_x, out_cursor[source], in_cursor[target], count,
                facecolor=COLORS[source],
                edgecolor="white" if kept else COLORS[source],
                linewidth=0.0 if kept else 0.35,
                alpha=0.22 if kept else 0.66, zorder=2 if kept else 3,
            )
            out_cursor[source] -= count
            in_cursor[target] -= count

    node_width = 0.018
    for reducer in order:
        # Nodes sit just outside the ribbon span so no band is hidden behind one.
        for x, top, height in ((left_x - node_width, tops[reducer], outgoing[reducer]),
                               (right_x, in_tops[reducer], incoming[reducer])):
            axis.add_patch(plt.Rectangle((x, top - height), node_width, height,
                                         facecolor=COLORS[reducer], edgecolor="none",
                                         zorder=4))
        # Only the name goes on the left: the outgoing count is the incoming one
        # less the net change, so printing it costs a third of the flow's run to
        # restate arithmetic the right-hand label already carries.
        axis.annotate(METHOD_LABELS[reducer],
                      xy=(left_x - 0.022, tops[reducer] - outgoing[reducer] / 2),
                      fontsize=6.0, color=COLORS[reducer], ha="right", va="center")
        change = incoming[reducer] - outgoing[reducer]
        # A hyphen would read as a dash between two numbers; use a real minus.
        signed = f"+{change:,}" if change >= 0 else f"\u2212{abs(change):,}"
        axis.annotate(f"{incoming[reducer]:,} ({signed})",
                      xy=(right_x + 0.022, in_tops[reducer] - incoming[reducer] / 2),
                      fontsize=6.0, color=COLORS[reducer], ha="left", va="center")

    axis.annotate("vote winner", xy=(left_x + 0.009, span * 1.04), fontsize=6.2,
                  color="0.25", ha="center", va="bottom")
    axis.annotate("after guard (net change)", xy=(right_x - 0.009, span * 1.04),
                  fontsize=6.2, color="0.25", ha="center", va="bottom")
    axis.annotate(
        f"{summary['invoked']:,} activations, {summary['overrides']:,} overrides "
        f"({100.0 * summary['overrides'] / summary['invoked']:.0f}%)",
        xy=(0.5, -0.105 * span), fontsize=6.0, color="0.3", ha="center", va="center",
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.155 * span, span * 1.14)
    axis.set_axis_off()
    _save(fig, "guard_flow")

    invoked = decisions.loc[
        (decisions["method"].astype(str) == "dagger05_vote3_guarded")
        & (decisions["scope"].astype(str) == "nominal")
        & (decisions["over_bound"].astype(str) == "True")
        & (decisions["guard_invoked"].astype(str) == "True")
    ]
    _require(len(invoked) == summary["invoked"],
             "guard activations disagree with the decision log")
    added = float(invoked["guard_added_decision_time_ms"].astype(float).sum())
    return {
        "decisions": summary["decisions"],
        "unanimous": summary["unanimous"],
        "invoked": summary["invoked"],
        "overrides": summary["overrides"],
        "ms_per_activation": added / float(len(invoked)),
        "ms_amortised_per_decision": added / float(summary["decisions"]),
        "outgoing": outgoing,
        "incoming": incoming,
        "flow": {f"{source}->{target}": count for (source, target), count in sorted(flow.items())},
    }


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------


def _num(value: float, digits: int = 2) -> str:
    """siunitx-formatted scientific value, or an em dash when unavailable."""
    if value is None or not np.isfinite(value):
        return "--"
    return f"\\num{{{value:.{digits - 1}e}}}"


def _fixed(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def _format_fpr(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{100.0 * value:.2f}"


def _write(name: str, lines: list[str]) -> None:
    (OUTPUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Best-value marking
# ---------------------------------------------------------------------------


def _sig(value: float, digits: int = 2) -> float:
    """The value as the reader sees it after significant-digit rounding."""
    return float(f"{value:.{digits - 1}e}")


def _best_mask(values, mode: str = "min", key=float) -> list[bool]:
    """Mark the best entry of a group, comparing displayed values.

    ``key`` rounds to whatever precision the cell is typeset at, so two cells
    that print identically are either both marked or both left plain.  A group
    whose members all tie -- the figure-eight FPR rows where every causal policy
    sits at 0.00 -- carries no ranking, so it is left unmarked rather than set
    entirely in bold, which would spend the emphasis on nothing.
    """
    shown = [None if value is None or not np.isfinite(value) else key(float(value))
             for value in values]
    finite = [value for value in shown if value is not None]
    if len(finite) < 2 or min(finite) == max(finite):
        return [False] * len(values)
    target = min(finite) if mode == "min" else max(finite)
    return [value is not None and value == target for value in shown]


def _bold(cell: str, marked: bool) -> str:
    """Bold a cell in place.  Works in both plain and siunitx ``S`` columns."""
    return f"\\bfseries {cell}" if marked else cell


# ---------------------------------------------------------------------------
# Table I: headline
# ---------------------------------------------------------------------------


def _headline_table(aggregate: pd.DataFrame,
                    availability: dict[str, tuple[int, int]]) -> tuple[str, ...]:
    index = _aggregate_index(aggregate)
    lines = [
        "% Generated from frozen result artifacts; do not edit.",
        "\\begin{tabular}{l r S[table-format=2.2] S[table-format=3.1] c c}",
        "\\toprule",
        "Method & {$\\overline{L}\\downarrow$} & {FPR (\\%)\\,$\\downarrow$}"
        " & {Ev./s\\,$\\uparrow$} & {RT} & {Cells} \\\\",
        "\\midrule",
    ]
    # Only the causal methods compete for bold: the offline oracles read the
    # remainder of the trace before choosing, so they set a reference, not a bar.
    deployable = tuple(
        method for method in ("girard", "scott", "pca", "combastel",
                              "mpc_terminal_beam_predictive_linear", "pairwise_ranking_policy")
        if str(index.loc[(method, HEADLINE_BUDGET)]["available"]) == "True"
    )

    def column_best(column, mode, key):
        values = [float(index.loc[(method, HEADLINE_BUDGET)][column]) for method in deployable]
        return {method: mark for method, mark
                in zip(deployable, _best_mask(values, mode=mode, key=key), strict=True)}

    best = {
        "loss": column_best("median_mean_approx_loss", "min", _sig),
        "fpr": column_best("macro_fpr", "min", lambda value: round(100.0 * value, 2)),
        "rate": column_best("median_throughput_events_per_second", "max",
                            lambda value: round(value, 1)),
    }
    pending = []
    for family, methods in FAMILIES:
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{family}}}}} \\\\")
        for method in methods:
            label = METHOD_LABELS[method]
            # A reserved row rather than a missing one: the promoted policy has
            # no canonical cells yet, and an absent row would read as a decision
            # to leave it out.  \placeholder is the draft's red XX marker.
            if (method, HEADLINE_BUDGET) not in index.index:
                pending.append(method)
                cells = " & ".join(["{\\placeholder}"] * 5)
                lines.append(f"\\quad {label} & {cells} \\\\")
                continue
            row = index.loc[(method, HEADLINE_BUDGET)]
            # Availability spans the whole sweep, not just the headline bound:
            # Scott completes at b=150 and the reader would otherwise never learn
            # that a fifth of its cells do not run.
            done, total = availability[method]
            cells_cell = f"{done}/{total}" if done == total else f"\\textbf{{{done}}}/{total}"
            if str(row["available"]) != "True":
                lines.append(
                    f"\\quad {label} & {{--}} & {{--}} & {{--}} & {{--}} & {cells_cell} \\\\")
                continue
            loss = float(row["median_mean_approx_loss"])
            fpr = float(row["macro_fpr"])
            events = float(row["median_throughput_events_per_second"])
            loss_cell = _bold(_num(loss), best["loss"].get(method, False))
            fpr_cell = _bold(f"{100.0 * fpr:.2f}", best["fpr"].get(method, False))
            rate_cell = _bold(f"{events:.1f}", best["rate"].get(method, False))
            real_time = "\\checkmark" if events >= EVENT_RATE_HZ else "$\\times$"
            lines.append(
                f"\\quad {label} & {loss_cell} & {fpr_cell} & {rate_cell} & {real_time}"
                f" & {cells_cell} \\\\"
            )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    _write("headline_results.tex", lines)
    return tuple(pending)


# ---------------------------------------------------------------------------
# Where the fixed Scott run terminates
# ---------------------------------------------------------------------------


def _scott_failure_profile(failures: pd.DataFrame) -> dict:
    """Scott's failure profile on the figure-eight traces.

    Scott is the reducer every controller selects most of the time and the only
    one that cannot be run unconditionally, so reporting *where* it dies is more
    informative than dropping its column.  The four fault variants agree to
    within a few events, so this reports the median across them.

    These numbers reach the paper as prose rather than a float.  Laid out as a
    table they are seven rows of which six are the same row -- every bound above
    the smallest dies within ninety events of the others -- so the shape a reader
    scans a table for is not there.  The manifest keeps the full grid so the two
    sentences in the text stay checkable against the artifacts.
    """
    grouped = failures.groupby("budget")
    events = int(failures["event_count"].astype(int).iloc[0])
    _require((failures["event_count"].astype(int) == events).all(),
             "figure-eight traces must share a length")

    profile = {}
    for budget in BUDGETS:
        rows = grouped.get_group(budget)
        profile[budget] = {
            "first_fallback_event": float(rows["first_fallback_event"].median()),
            "completed_percent": 100.0 * float(rows["completed_fraction"].median()),
            # A zero here is not a tight set: at b=40 the run dies on its second
            # event, before any approximation error has accumulated.
            "pre_fallback_mean_loss": float(rows["pre_fallback_mean_loss"].median()),
        }
    return {"trace_events": events, "by_budget": profile}


# ---------------------------------------------------------------------------
# Table II: cost of predictive search
# ---------------------------------------------------------------------------


def _search_cost_table(ablation: pd.DataFrame, objective: pd.DataFrame) -> None:
    grid = ablation.groupby(["H", "W"]).agg(
        loss=("mean_approx_loss", "median"),
        fpr=("fpr", "mean"),
        throughput=("throughput", "median"),
    )
    widths = (1, 2, 4, 8)
    horizons = (1, 2, 4, 8)
    lines = [
        "% Generated from frozen result artifacts; do not edit.",
        "\\begin{tabular}{l *{4}{r} c *{4}{S[table-format=2.1]}}",
        "\\toprule",
        " & \\multicolumn{4}{c}{Median $\\overline{L}$ ($\\times 10^{-7}$)} & "
        "& \\multicolumn{4}{c}{Median events/s} \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){7-10}",
        "$H$ & {$W{=}1$} & {$W{=}2$} & {$W{=}4$} & {$W{=}8$} & "
        "& {$W{=}1$} & {$W{=}2$} & {$W{=}4$} & {$W{=}8$} \\\\",
        "\\midrule",
    ]
    # The trade-off is the point of this grid, so bold marks the extreme of each
    # half over the whole 4x4 block rather than row by row.
    cells = [(horizon, width) for horizon in horizons for width in widths]
    tightest = _best_mask([grid.loc[key, "loss"] * 1e7 for key in cells],
                          mode="min", key=lambda value: round(value, 2))
    fastest = _best_mask([grid.loc[key, "throughput"] for key in cells],
                         mode="max", key=lambda value: round(value, 1))
    tightest = dict(zip(cells, tightest, strict=True))
    fastest = dict(zip(cells, fastest, strict=True))
    for horizon in horizons:
        losses = " & ".join(
            _bold(f"{grid.loc[(horizon, width), 'loss'] * 1e7:.2f}", tightest[(horizon, width)])
            for width in widths
        )
        rates = " & ".join(
            _bold(f"{grid.loc[(horizon, width), 'throughput']:.1f}", fastest[(horizon, width)])
            for width in widths
        )
        lines.append(f"{horizon} & {losses} & & {rates} \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    _write("search_cost.tex", lines)

    pivot = objective.pivot_table(index=["condition", "budget"], columns="method",
                                  values="mean_approx_loss")
    ratio = (pivot["mpc_cumulative_beam"] / pivot["mpc_terminal_beam"]).to_numpy()
    summary = {
        "terminal_median_loss": float(objective.loc[
            objective["method"] == "mpc_terminal_beam", "mean_approx_loss"].median()),
        "cumulative_median_loss": float(objective.loc[
            objective["method"] == "mpc_cumulative_beam", "mean_approx_loss"].median()),
        "cumulative_over_terminal_median_ratio": float(np.median(ratio)),
        "cumulative_wins": int((ratio < 1.0).sum()),
        "paired_cells": int(len(ratio)),
    }
    # The terminal-versus-cumulative comparison is a null result defending a
    # design choice, not a finding, so it is reported in one sentence of prose.
    # The summary stays in the manifest so those numbers remain derived rather
    # than typed.
    return summary


# ---------------------------------------------------------------------------
# Tables III: figure-eight case studies
# ---------------------------------------------------------------------------


CONDITION_LABELS = {
    "figure8": "Figure-8",
    "figure8_drift": "Figure-8 drift",
    "figure8_geofence": "Figure-8 geofence",
    "figure8_drift_geofence": "Figure-8 drift + geofence",
}


def _fixed_maps(canonical: pd.DataFrame, confirmation: pd.DataFrame):
    canonical = canonical.loc[canonical["method"].isin(CANONICAL_METHODS)].copy()
    confirmation = confirmation.loc[confirmation["method"].isin(CONFIRMATION_METHODS)].copy()
    _require(len(canonical) == 224, "canonical fixed matrix must contain 224 cells")
    _require(len(confirmation) == 112, "confirmation fixed matrix must contain 112 cells")
    canonical_map = canonical.set_index(["condition", "budget", "method"])
    confirmation_map = confirmation.set_index(["condition", "budget", "method"])
    for condition in CONDITION_LABELS:
        for budget in BUDGETS:
            for old, new in (("pairwise_ranking_policy", "g15_clean148"),
                             ("mpc_terminal_beam_predictive_linear",
                              "mpc_terminal_beam_predictive_linear")):
                a = canonical_map.loc[(condition, budget, old)]
                b = confirmation_map.loc[(condition, budget, new)]
                _require(np.isclose(float(a["mean_approx_loss"]), float(b["mean_approx_loss"]),
                                    rtol=0.0, atol=1e-18), "reused fixed loss differs")
                _require(np.isclose(float(a["fpr"]), float(b["fpr"]), rtol=0.0, atol=1e-18),
                         "reused fixed FPR differs")
    return canonical_map, confirmation_map


def _scott_partial_map(failures: pd.DataFrame) -> dict:
    """Pre-fallback loss for every fixed Scott cell, keyed like the other maps."""
    return {
        (str(row["condition"]), int(row["budget"])): float(row["pre_fallback_mean_loss"])
        for _, row in failures.iterrows()
    }


def _lookup(canonical_map, confirmation_map, scott_partial, condition, budget, method):
    """Return ``(fpr, loss, partial)`` for one cell.

    ``partial`` marks a value measured before an interval fallback terminated the
    run.  Scott never completes a figure-eight cell, so reporting the loss it had
    already accumulated is the only way to keep it in the table with a number
    rather than a dash -- but that number is not comparable with a completed run
    and never competes for the best-in-block mark.
    """
    source = confirmation_map if method in {"dagger05_vote3", "dagger05_vote3_guarded"} else canonical_map
    row = source.loc[(condition, budget, method)]
    if str(row["status"]) == "completed":
        return float(row["fpr"]), float(row["mean_approx_loss"]), False
    if method == "scott":
        loss = scott_partial.get((condition, budget))
        # A zero mean loss at b=40 records a run that died before accumulating
        # any error, which is not a tight set; report it as unavailable.
        if loss is not None and loss > 0.0:
            return None, loss, True
    return None, None, False


# Both figure-eight tables share this grouping so a reader who learns the
# column blocks once can read either.  Bold marks the best value *within a
# block*: comparing a causal policy against an oracle that has already seen the
# rest of the trace would not be a like-for-like ranking.
FIXED_GROUPS = (
    ("Fixed", ("girard", "scott", "pca", "combastel")),
    ("Offline oracle", ("mpc_terminal_beam", "mpc_terminal_full_width")),
    ("Online", ("mpc_terminal_beam_predictive_linear", "pairwise_ranking_policy",
                "dagger05_vote3", "dagger05_vote3_guarded")),
)


def _fixed_header(column_spec: str) -> tuple[tuple[str, ...], list[str]]:
    method_order = tuple(method for _, methods in FIXED_GROUPS for method in methods)
    spans = []
    start = 3
    for name, methods in FIXED_GROUPS:
        stop = start + len(methods) - 1
        spans.append((name, start, stop))
        start = stop + 1
    lines = [
        "% Generated from frozen result artifacts; do not edit.",
        f"\\begin{{tabular}}{{l l{column_spec * len(method_order)}}}",
        "\\toprule",
        "Trace & $b$ & " + " & ".join(
            f"\\multicolumn{{{stop - begin + 1}}}{{c}}{{{name}}}" for name, begin, stop in spans
        ) + " \\\\",
        "".join(f"\\cmidrule(lr){{{begin}-{stop}}}" for _, begin, stop in spans),
        " & & " + " & ".join(f"{{{SHORT_LABELS[method]}}}" for method in method_order) + " \\\\",
        "\\midrule",
    ]
    return method_order, lines


def _fixed_body(lines, method_order, cell, mode, key, render, missing) -> None:
    """``cell`` returns ``(value, partial)``; partial values never take bold."""
    for condition_index, condition in enumerate(CONDITION_LABELS):
        for row_index, budget in enumerate(BUDGETS):
            raw, partial = {}, {}
            for method in method_order:
                raw[method], partial[method] = cell(condition, budget, method)
            marked = {}
            for _, methods in FIXED_GROUPS:
                mask = _best_mask(
                    [None if partial[method] else raw[method] for method in methods],
                    mode=mode, key=key,
                )
                marked.update(dict(zip(methods, mask, strict=True)))
            values = [CONDITION_LABELS[condition] if row_index == 0 else "", str(budget)]
            for method in method_order:
                if raw[method] is None:
                    values.append(missing)
                elif partial[method]:
                    values.append(f"{render(raw[method])}\\rlap{{$^{{\\dagger}}$}}")
                else:
                    values.append(_bold(render(raw[method]), marked[method]))
            lines.append(" & ".join(values) + " \\\\")
        if condition_index != len(CONDITION_LABELS) - 1:
            lines.append("\\midrule")
    lines.extend(("\\bottomrule", "\\end{tabular}"))


def _fixed_table_main(canonical_map, confirmation_map, scott_partial) -> None:
    """Loss over every method, including Scott's pre-fallback values."""
    method_order, lines = _fixed_header(" r")
    _fixed_body(
        lines, method_order,
        cell=lambda c, b, m: _lookup(canonical_map, confirmation_map, scott_partial, c, b, m)[1:],
        mode="min", key=_sig, render=_num, missing="--",
    )
    _write("figure8_loss.tex", lines)


def _fixed_table_fpr(canonical_map, confirmation_map, scott_partial) -> None:
    """Companion FPR table: the second metric, kept out of the loss table."""
    method_order, lines = _fixed_header(" S[table-format=2.2]")

    def cell(condition, budget, method):
        fpr, _, _ = _lookup(canonical_map, confirmation_map, scott_partial,
                            condition, budget, method)
        return fpr, False

    _fixed_body(
        lines, method_order, cell=cell,
        mode="min", key=lambda value: round(100.0 * value, 2),
        render=_format_fpr, missing="{--}",
    )
    _write("figure8_fpr.tex", lines)


# The main paper carries one compact digest of the four fault variants; the
# comprehensive per-bound matrices live in the appendix.  Methods are rows here
# so the orientation matches the headline table, and the two offline oracles are
# dropped because Table I and Fig. 1 already place them.
SUMMARY_METHODS = ("girard", "scott", "pca", "combastel",
                   "mpc_terminal_beam_predictive_linear", "pairwise_ranking_policy",
                   "dagger05_vote3", "dagger05_vote3_guarded")


def _fixed_summary_table(canonical_map, confirmation_map, scott_partial) -> dict:
    """Median over the seven bounds, one column per fault variant."""
    stats, availability = {}, {}
    for method in SUMMARY_METHODS:
        stats[method] = {}
        completed = 0
        for condition in CONDITION_LABELS:
            losses, partials, fprs = [], [], []
            for budget in BUDGETS:
                fpr, loss, partial = _lookup(canonical_map, confirmation_map,
                                             scott_partial, condition, budget, method)
                if loss is None:
                    continue
                if partial:
                    partials.append(loss)
                else:
                    losses.append(loss)
                    completed += 1
                if fpr is not None:
                    fprs.append(fpr)
            # A method that never completes still reports the looseness it had
            # reached when the run died; the dagger keeps that from reading as a
            # like-for-like result.
            stats[method][condition] = {
                "loss": float(np.median(losses or partials)) if (losses or partials) else None,
                "partial": not losses and bool(partials),
                "fpr": float(np.mean(fprs)) if fprs else None,
            }
        availability[method] = completed
    total = len(CONDITION_LABELS) * len(BUDGETS)

    lines = [
        "% Generated from frozen result artifacts; do not edit.",
        "\\begin{tabular}{l" + " r" * len(CONDITION_LABELS) + " c}",
        "\\toprule",
        " & \\multicolumn{4}{c}{Fault variant} & \\\\",
        f"\\cmidrule(lr){{2-{1 + len(CONDITION_LABELS)}}}",
        "Method & {Nom.} & {Drift} & {Geo.} & {D+G} & {Cells} \\\\",
        "\\midrule",
    ]
    for group, methods in (("Fixed", SUMMARY_METHODS[:4]), ("Online", SUMMARY_METHODS[4:])):
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{group}}}}} \\\\")
        for condition in CONDITION_LABELS:
            values = [None if stats[method][condition]["partial"]
                      else stats[method][condition]["loss"] for method in methods]
            mask = _best_mask(values, mode="min", key=_sig)
            for method, mark in zip(methods, mask, strict=True):
                stats[method][condition]["best"] = mark
        for method in methods:
            cells_out = []
            for condition in CONDITION_LABELS:
                entry = stats[method][condition]
                if entry["loss"] is None:
                    cells_out.append("--")
                elif entry["partial"]:
                    cells_out.append(f"{_num(entry['loss'])}\\rlap{{$^{{\\dagger}}$}}")
                else:
                    cells_out.append(_bold(_num(entry["loss"]), entry.get("best", False)))
            done = availability[method]
            count = f"{done}/{total}" if done == total else f"\\textbf{{{done}}}/{total}"
            lines.append(f"\\quad {METHOD_LABELS[method]} & " + " & ".join(cells_out)
                         + f" & {count} \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    _write("figure8_summary.tex", lines)
    return {"availability": availability, "cells_per_method": total}


# ---------------------------------------------------------------------------
# Table IV: reducer composition
# ---------------------------------------------------------------------------


def _selection_counts_from_csv(path: Path, methods: tuple[str, ...]) -> dict[str, Counter[str]]:
    counts = {method: Counter() for method in methods}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            reducer = row["reducer_used"]
            if method not in counts or reducer not in REDUCERS:
                continue
            if row["fallback_used"].lower() in {"true", "1"} or int(float(row["infeasible_candidate_count"])) != 0:
                continue
            counts[method][reducer] += 1
    return counts


def _selection_counts_from_cell_files(methods: tuple[str, ...]) -> dict[str, Counter[str]]:
    counts = {method: Counter() for method in methods}
    root = GUARDED / "evaluate/nominal/cells"
    files = tuple(root.glob("*/*/*/*/timeseries.csv"))
    _require(len(files) == 560, "confirmation must expose 560 time-series files")
    for path in files:
        method = path.parent.name
        if method not in counts:
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                reducer = row["reducer_used"]
                if reducer not in REDUCERS:
                    continue
                if row["fallback_used"].lower() in {"true", "1"} or int(float(row["infeasible_candidate_count"])) != 0:
                    continue
                counts[method][reducer] += 1
    return counts


def _composition_table() -> dict:
    canonical_methods = (
        "mpc_terminal_beam",
        "mpc_terminal_beam_predictive_linear",
        "mpc_terminal_full_width",
        "pairwise_ranking_policy",
    )
    confirmation_methods = (
        "mpc_terminal_beam_predictive_linear",
        "g15_clean148",
        "dagger05_vote3",
        "dagger05_vote3_guarded",
    )
    canonical = _selection_counts_from_csv(
        CANONICAL / "generalization/timeseries.csv", canonical_methods)
    confirmation = _selection_counts_from_cell_files(confirmation_methods)
    lines = [
        "% Generated from frozen result artifacts; do not edit.",
        "\\begin{tabular}{l *{4}{S[table-format=2.2]}}",
        "\\toprule",
        "Controller & {Girard} & {Scott} & {PCA} & {Combastel} \\\\",
        "\\midrule",
        "\\multicolumn{5}{l}{\\emph{Canonical nominal seeds 100--119}} \\\\",
    ]
    totals = {}
    for method in canonical_methods:
        count = sum(canonical[method].values())
        totals[f"canonical/{method}"] = count
        values = [100.0 * canonical[method][reducer] / count for reducer in REDUCERS]
        modal = _best_mask(values, mode="max", key=lambda value: round(value, 2))
        lines.append(f"\\quad {METHOD_LABELS[method]} & " +
                     " & ".join(_bold(f"{value:.2f}", mark)
                                for value, mark in zip(values, modal, strict=True)) + " \\\\")
    lines.extend((
        "\\midrule",
        "\\multicolumn{5}{l}{\\emph{Vote3 confirmation seeds 328--347}} \\\\",
    ))
    for method in confirmation_methods:
        count = sum(confirmation[method].values())
        totals[f"confirmation/{method}"] = count
        values = [100.0 * confirmation[method][reducer] / count for reducer in REDUCERS]
        modal = _best_mask(values, mode="max", key=lambda value: round(value, 2))
        lines.append(f"\\quad {METHOD_LABELS[method]} & " +
                     " & ".join(_bold(f"{value:.2f}", mark)
                                for value, mark in zip(values, modal, strict=True)) + " \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    _write("reducer_composition.tex", lines)
    return totals


# ---------------------------------------------------------------------------
# Learned-policy comparison on the confirmation cohort
# ---------------------------------------------------------------------------


LEARNED_ORDER = ("mpc_terminal_beam_predictive_linear", "g15_clean148",
                 "dagger05_vote3", "dagger05_vote3_guarded")


def _learned_comparison(cells: pd.DataFrame) -> dict:
    """Absolute standing of the three learned policies against the teacher.

    Table~\\ref{tab:headline} cannot carry these: it is built on canonical seeds
    100--119, whereas the voting policies exist only on the untouched
    confirmation cohort 328--347.  Without this table the paper reports the
    voting policies only as ratios and never states where they actually sit.
    """
    nominal = cells.loc[cells["scope"].astype(str) == "nominal"]
    pivot = nominal.pivot_table(index=["seed", "budget"], columns="method",
                                values=["mean_approx_loss", "fpr",
                                        "event_count", "event_loop_time_ms"])
    loss, fpr = pivot["mean_approx_loss"], pivot["fpr"]
    rate = (pivot["event_count"].astype(float) * 1000.0
            / pivot["event_loop_time_ms"].astype(float))
    reference = loss["mpc_terminal_beam_predictive_linear"].astype(float)
    _require(len(loss) == 140, "confirmation cohort must hold 140 paired cells")

    stats = {}
    for method in LEARNED_ORDER:
        ratio = loss[method].astype(float) / reference
        stats[method] = {
            "median_loss": float(loss[method].astype(float).median()),
            "p50": float(ratio.quantile(0.50)),
            "p90": float(ratio.quantile(0.90)),
            "p95": float(ratio.quantile(0.95)),
            "severe": int((ratio >= 1e3).sum()),
            "macro_fpr": float(fpr[method].astype(float).mean()),
            "throughput": float(rate[method].median()),
        }

    # The teacher is not real-time, so it sets a reference rather than a bar and
    # is excluded from the bold comparison.
    deployable = LEARNED_ORDER[1:]
    marks = {}
    for field, mode, key in (("median_loss", "min", _sig),
                             ("p50", "min", lambda v: round(v, 2)),
                             ("p90", "min", lambda v: round(v, 2)),
                             ("p95", "min", lambda v: _sig(v, 3)),
                             ("severe", "min", float),
                             ("macro_fpr", "min", lambda v: round(100.0 * v, 2)),
                             ("throughput", "max", lambda v: round(v, 1))):
        mask = _best_mask([stats[m][field] for m in deployable], mode=mode, key=key)
        marks[field] = dict(zip(deployable, mask, strict=True))

    def compact(value: float) -> str:
        """Plain below the severe threshold, scientific above it."""
        return f"{value:.2f}" if value < 1e3 else _num(value, 3)

    # A mechanism column would push the table past \columnwidth and force it to
    # \textwidth, where it competes for the same top-of-page slot as Fig. 3 and
    # defers past the bibliography.  The tiers are named in the caption instead.
    lines = [
        "% Generated from frozen result artifacts; do not edit.",
        "\\begin{tabular}{l r S[table-format=1.2] S[table-format=1.2] r"
        " S[table-format=1.0] S[table-format=1.2] S[table-format=3.1]}",
        "\\toprule",
        " & & \\multicolumn{3}{c}{Loss ratio to MPC-L} & & \\\\",
        "\\cmidrule(lr){3-5}",
        "Policy & {$\\overline{L}$} & {p50} & {p90} & {p95}"
        " & {Sev.} & {FPR (\\%)} & {Ev./s} \\\\",
        "\\midrule",
    ]
    for method in LEARNED_ORDER:
        row = stats[method]
        cells_out = (
            _bold(_num(row["median_loss"]), marks["median_loss"].get(method, False)),
            _bold(f"{row['p50']:.2f}", marks["p50"].get(method, False)),
            _bold(f"{row['p90']:.2f}", marks["p90"].get(method, False)),
            _bold(compact(row["p95"]), marks["p95"].get(method, False)),
            _bold(str(row["severe"]), marks["severe"].get(method, False)),
            _bold(f"{100.0 * row['macro_fpr']:.2f}", marks["macro_fpr"].get(method, False)),
            _bold(f"{row['throughput']:.1f}", marks["throughput"].get(method, False)),
        )
        lines.append(f"{METHOD_LABELS[method]} & " + " & ".join(cells_out) + " \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    _write("learned_comparison.tex", lines)
    return stats


# ---------------------------------------------------------------------------
# Table V: confirmation gate scorecard
# ---------------------------------------------------------------------------


GATES = (
    ("zero_severe_tail", "No severe cell", "severe_tail_count", "{:.0f}", None),
    ("all_budget_medians_within_1_25_g15", "Budget medians $\\leq 1.25\\times$",
     "max_budget_median_loss_ratio_vs_g15", "{:.2f}", None),
    ("throughput_at_least_half_g15", "Throughput $\\geq 0.5\\times$",
     "median_paired_throughput_retention", "{:.2f}", None),
    ("mean_fpr_regression_at_most_0_005", "Mean FPR $\\leq 0.5$\\,pp",
     "mean_fpr_difference_vs_g15", "{:+.2f}", 100.0),
    ("individual_fpr_regression_at_most_0_05", "Worst-trace FPR $\\leq 5$\\,pp",
     "max_fpr_difference_vs_g15", "{:+.2f}", 100.0),
    ("zero_failures", "No cell failure", "failure_count", "{:.0f}", None),
    ("all_cells_available", "All cells available", "valid_count", "{:.0f}", None),
    ("zero_fallbacks", "No interval fallback", "fallback_count", "{:.0f}", None),
)


# Four gates are absolute properties of a policy and four are defined relative
# to G15/Clean148.  Scoring the deployed policy on the absolute four is the only
# way to see that the severe-tail gate the challengers fail is one the fallback
# fails harder; the relative four are vacuous for it and are dashed.
ABSOLUTE_GATES = frozenset({"zero_severe_tail", "zero_failures",
                            "all_cells_available", "zero_fallbacks"})


def _gate_summary(cells: pd.DataFrame, learned: dict) -> dict:
    """Preregistered gate outcomes, kept as derived numbers rather than a table.

    The scorecard was internal bookkeeping: the section states in two sentences
    which gates the challengers clear and which they miss, and a float repeating
    that costs more space than the prose it duplicates.  The values stay here so
    the prose cites derived numbers instead of typed ones, and so the deployed
    policy is still scored on the four gates that are absolute properties of a
    policy rather than comparisons against itself -- that is the only way to see
    that the severe-tail gate the challengers miss is one the incumbent misses
    by more.
    """
    frame = _read_csv(GUARDED / "report/artifacts/confirmation_eligibility.csv")
    frame = frame.loc[frame["scope"].astype(str) == "nominal"].set_index("method")
    _require(len(frame) == 2, "eligibility table must score two challengers")
    methods = ("dagger05_vote3", "dagger05_vote3_guarded")

    nominal = cells.loc[(cells["scope"].astype(str) == "nominal")
                        & (cells["method"].astype(str) == "g15_clean148")]
    _require(len(nominal) == 140, "deployed policy must expose 140 nominal cells")
    baseline = {
        "severe_tail_count": float(learned["g15_clean148"]["severe"]),
        "failure_count": float(nominal["reducer_failure_count"].astype(float).sum()),
        "valid_count": float((nominal["status"].astype(str) == "completed").sum()),
        "fallback_count": float(nominal["fallback_count"].astype(float).sum()),
    }
    result = {"g15_clean148": baseline}
    for method in methods:
        result[method] = {
            "eligible": str(frame.loc[method, "eligible"]) == "True",
            "gates_passed": int(sum(str(frame.loc[method, flag]) == "True"
                                    for flag, *_ in GATES)),
            "gates_total": len(GATES),
            "failed_gates": [label for flag, label, *_ in GATES
                             if str(frame.loc[method, flag]) != "True"],
            "severe_tail_count": int(frame.loc[method, "severe_tail_count"]),
            "median_loss_ratio_vs_g15": float(frame.loc[method, "median_loss_ratio_vs_g15"]),
            "mean_fpr_difference_vs_g15_pp": 100.0 * float(frame.loc[method, "mean_fpr_difference_vs_g15"]),
            "max_fpr_difference_vs_g15_pp": 100.0 * float(frame.loc[method, "max_fpr_difference_vs_g15"]),
            "median_paired_throughput_retention": float(frame.loc[method, "median_paired_throughput_retention"]),
        }
    return result


# ---------------------------------------------------------------------------
# V4 final-cohort tables (runtime is supplied by the separate RPi experiment)
# ---------------------------------------------------------------------------


V4_METHODS = (
    "girard",
    "scott",
    "pca",
    "combastel",
    "mpc_terminal_beam",
    "mpc_terminal_full_width",
    "mpc_terminal_beam_predictive_linear",
    "pairwise_ranking_policy",
    "dagger05_vote3",
    "dagger05_vote3_guarded",
)
V4_LEARNED = (
    "mpc_terminal_beam_predictive_linear",
    "pairwise_ranking_policy",
    "dagger05_vote3",
    "dagger05_vote3_guarded",
)


def _v4_nominal() -> pd.DataFrame:
    frame = _read_csv(V4 / "nominal/summary.csv")
    _require(len(frame) == 1_400, "v4 nominal summary must contain 1,400 cells")
    _require(set(frame["method"].astype(str)) == set(V4_METHODS),
             "v4 nominal method matrix differs")
    _require(set(frame["seed"].astype(int)) == set(range(348, 368)),
             "v4 nominal seed cohort differs")
    _require(set(frame["budget"].astype(int)) == set(BUDGETS),
             "v4 nominal budget matrix differs")
    _require((frame["event_count"].astype(int) == 500).all(),
             "v4 nominal trace length differs")
    completed = frame.loc[frame["status"].astype(str) == "completed"]
    _require(int(completed["false_negative_count"].astype(int).sum()) == 0,
             "v4 nominal completed cells contain a false negative")
    return frame


def _v4_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    """Failure-aware budget aggregates with deterministic seed bootstrap CIs."""
    rows = []
    rng = np.random.default_rng(20260721)
    for (budget, method), group in frame.groupby(["budget", "method"], sort=True):
        group = group.sort_values("seed")
        valid = group.loc[group["status"].astype(str) == "completed"].copy()
        failed = len(group) - len(valid)
        fpr = valid["fpr"].astype(float).to_numpy()
        ci_low = ci_high = float("nan")
        if failed == 0:
            indices = rng.integers(0, len(fpr), size=(10_000, len(fpr)))
            ci_low, ci_high = np.quantile(fpr[indices].mean(axis=1), (0.025, 0.975))
        rates = (valid["event_count"].astype(float) * 1000.0
                 / valid["event_loop_time_ms"].astype(float))
        rows.append({
            "method": str(method),
            "budget": int(budget),
            "available": failed == 0,
            "valid_count": len(valid),
            "failed_count": failed,
            "macro_fpr": float(fpr.mean()) if failed == 0 else float("nan"),
            "macro_fpr_ci_low": float(ci_low),
            "macro_fpr_ci_high": float(ci_high),
            "valid_only_macro_fpr": float(fpr.mean()) if len(fpr) else float("nan"),
            "median_mean_approx_loss": (
                float(valid["mean_approx_loss"].astype(float).median())
                if failed == 0 else float("nan")
            ),
            "valid_only_median_mean_approx_loss": (
                float(valid["mean_approx_loss"].astype(float).median())
                if len(valid) else float("nan")
            ),
            # Diagnostic workstation timing only. It is deliberately not used
            # in v4 paper tables; the RPi timing artifact supplies deployment latency.
            "median_throughput_events_per_second": (
                float(rates.median()) if len(rates) else float("nan")
            ),
        })
    return pd.DataFrame(rows)


def _headline_table_v4(aggregate: pd.DataFrame,
                       availability: dict[str, tuple[int, int]]) -> None:
    index = _aggregate_index(aggregate)
    causal = (
        "girard", "scott", "pca", "combastel",
        "mpc_terminal_beam_predictive_linear", "pairwise_ranking_policy",
        "dagger05_vote3", "dagger05_vote3_guarded",
    )
    causal = tuple(method for method in causal
                   if str(index.loc[(method, HEADLINE_BUDGET), "available"]) == "True")
    loss_best = dict(zip(
        causal,
        _best_mask(
            [float(index.loc[(m, HEADLINE_BUDGET), "median_mean_approx_loss"])
             for m in causal],
            mode="min", key=_sig,
        ),
        strict=True,
    ))
    fpr_best = dict(zip(
        causal,
        _best_mask(
            [float(index.loc[(m, HEADLINE_BUDGET), "macro_fpr"]) for m in causal],
            mode="min", key=lambda value: round(100.0 * value, 2),
        ),
        strict=True,
    ))
    lines = [
        "% Generated from the paired v4 final cohort; do not edit.",
        "\\begin{tabular}{l r S[table-format=2.2] c}",
        "\\toprule",
        "Method & {$\\overline{L}\\downarrow$} & {FPR (\\%)\\,$\\downarrow$}"
        " & {Cells} \\\\",
        "\\midrule",
    ]
    for family, methods in (
        ("Fixed reducer", V4_METHODS[:4]),
        ("Offline oracle", V4_METHODS[4:6]),
        ("Online", V4_METHODS[6:]),
    ):
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{family}}}}} \\\\")
        for method in methods:
            row = index.loc[(method, HEADLINE_BUDGET)]
            done, total = availability[method]
            count = f"{done}/{total}" if done == total else f"\\textbf{{{done}}}/{total}"
            if str(row["available"]) != "True":
                lines.append(f"\\quad {METHOD_LABELS[method]} & -- & {{--}} & {count} \\\\")
                continue
            loss = float(row["median_mean_approx_loss"])
            fpr = float(row["macro_fpr"])
            lines.append(
                f"\\quad {METHOD_LABELS[method]} & "
                f"{_bold(_num(loss), loss_best.get(method, False))} & "
                f"{_bold(f'{100.0 * fpr:.2f}', fpr_best.get(method, False))} & "
                f"{count} \\\\"
            )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    _write("headline_results.tex", lines)


def _learned_comparison_v4(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    complete = frame.loc[frame["status"].astype(str) == "completed"]
    pivot = complete.pivot(index=["seed", "budget"], columns="method",
                           values=["mean_approx_loss", "fpr"])
    loss = pivot["mean_approx_loss"]
    fpr = pivot["fpr"]
    reference = loss["mpc_terminal_beam_predictive_linear"].astype(float)
    _require(len(reference) == 140, "v4 learned comparison must have 140 paired cells")
    stats: dict[str, dict[str, float]] = {}
    for method in V4_LEARNED:
        ratio = loss[method].astype(float) / reference
        stats[method] = {
            "median_loss": float(loss[method].astype(float).median()),
            "p50": float(ratio.quantile(0.50)),
            "p90": float(ratio.quantile(0.90)),
            "p95": float(ratio.quantile(0.95)),
            "severe": int((ratio >= 1e3).sum()),
            "macro_fpr": float(fpr[method].astype(float).mean()),
        }
    deployable = V4_LEARNED[1:]
    marks = {}
    for field, key in (
        ("median_loss", _sig),
        ("p50", lambda value: round(value, 2)),
        ("p90", lambda value: _sig(value, 3)),
        ("p95", lambda value: _sig(value, 3)),
        ("severe", float),
        ("macro_fpr", lambda value: round(100.0 * value, 2)),
    ):
        marks[field] = dict(zip(
            deployable,
            _best_mask([stats[m][field] for m in deployable], mode="min", key=key),
            strict=True,
        ))

    def compact(value: float) -> str:
        return f"{value:.2f}" if value < 1e3 else _num(value, 3)

    lines = [
        "% Generated from the paired v4 final cohort; do not edit.",
        "\\begin{tabular}{l r S[table-format=1.2] r r S[table-format=2.0] "
        "S[table-format=1.2]}",
        "\\toprule",
        " & & \\multicolumn{3}{c}{Loss ratio to MPC-L} & & \\\\",
        "\\cmidrule(lr){3-5}",
        "Policy & {$\\overline{L}$} & {p50} & {p90} & {p95}"
        " & {Sev.} & {FPR (\\%)} \\\\",
        "\\midrule",
    ]
    for method in V4_LEARNED:
        row = stats[method]
        cells = (
            _bold(_num(row["median_loss"]), marks["median_loss"].get(method, False)),
            _bold(f"{row['p50']:.2f}", marks["p50"].get(method, False)),
            _bold(compact(row["p90"]), marks["p90"].get(method, False)),
            _bold(compact(row["p95"]), marks["p95"].get(method, False)),
            _bold(str(int(row["severe"])), marks["severe"].get(method, False)),
            _bold(f"{100.0 * row['macro_fpr']:.2f}", marks["macro_fpr"].get(method, False)),
        )
        label = SHORT_LABELS.get(method, METHOD_LABELS[method])
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    _write("learned_comparison.tex", lines)
    return stats


def _predictor_table_v4() -> dict[str, dict[str, float]]:
    frame = _read_csv(V4 / "prediction-ablation/summary.csv")
    _require(len(frame) == 15, "v4 predictor ablation must contain 15 cells")
    _require((frame["status"].astype(str) == "completed").all(),
             "v4 predictor ablation has unavailable cells")
    _require(int(frame["false_negative_count"].astype(int).sum()) == 0,
             "v4 predictor ablation contains a false negative")
    pivot = frame.pivot(index="seed", columns="predictor", values="mean_approx_loss")
    stats = {}
    for predictor in ("hold", "linear", "quadratic"):
        rows = frame.loc[frame["predictor"].astype(str) == predictor]
        stats[predictor] = {
            "median_loss": float(rows["mean_approx_loss"].astype(float).median()),
            "paired_ratio": float((pivot[predictor] / pivot["linear"]).median()),
            "macro_fpr": float(rows["fpr"].astype(float).mean()),
            "false_negatives": int(rows["false_negative_count"].astype(int).sum()),
        }
    lines = [
        "% Generated from the v4 predictor ablation; do not edit.",
        "\\begin{tabular}{l r S[table-format=1.2] S[table-format=1.2] S[table-format=1.0]}",
        "\\toprule",
        "Predictor & {$\\overline{L}$} & {Ratio to linear} & {FPR (\\%)} & {FN} \\\\",
        "\\midrule",
    ]
    loss_marks = _best_mask([stats[p]["median_loss"] for p in stats], key=_sig)
    ratio_marks = _best_mask([stats[p]["paired_ratio"] for p in stats],
                             key=lambda value: round(value, 2))
    labels = {"hold": "Hold", "linear": "Linear", "quadratic": "Quadratic"}
    for predictor, loss_mark, ratio_mark in zip(stats, loss_marks, ratio_marks, strict=True):
        row = stats[predictor]
        ratio_cell = _bold(f"{row['paired_ratio']:.2f}", ratio_mark)
        lines.append(
            f"{labels[predictor]} & {_bold(_num(row['median_loss']), loss_mark)} & "
            f"{ratio_cell} & "
            f"{100.0 * row['macro_fpr']:.2f} & {row['false_negatives']} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    _write("predictor_ablation.tex", lines)
    return stats


def _composition_table_v4() -> dict[str, int]:
    methods = V4_METHODS[4:]
    counts = {method: Counter() for method in methods}
    files = tuple(
        path for path in sorted(
            (V4 / "cells").glob("random_waypoint:seed-*/budget-*/*/timeseries.csv")
        )
        if 348 <= int(path.parts[-4].rsplit("-", 1)[1]) <= 367
    )
    _require(len(files) == 1_400, "v4 nominal time-series matrix is incomplete")
    for path in files:
        method = path.parent.name
        if method not in counts:
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                reducer = row["reducer_used"]
                if reducer not in REDUCERS:
                    continue
                if row["fallback_used"].lower() in {"true", "1"}:
                    continue
                if int(float(row["infeasible_candidate_count"])) != 0:
                    continue
                counts[method][reducer] += 1
    lines = [
        "% Generated from the paired v4 final cohort; do not edit.",
        "\\begin{tabular}{l *{4}{S[table-format=2.2]}}",
        "\\toprule",
        "Controller & {Girard} & {Scott} & {PCA} & {Combastel} \\\\",
        "\\midrule",
    ]
    totals = {}
    for method in methods:
        total = sum(counts[method].values())
        _require(total > 0, f"v4 composition has no decisions for {method}")
        totals[method] = total
        values = [100.0 * counts[method][reducer] / total for reducer in REDUCERS]
        marks = _best_mask(values, mode="max", key=lambda value: round(value, 2))
        lines.append(
            f"\\quad {METHOD_LABELS[method]} & "
            + " & ".join(_bold(f"{value:.2f}", mark)
                           for value, mark in zip(values, marks, strict=True))
            + " \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    _write("reducer_composition.tex", lines)
    return totals


def _plot_accuracy_v4(aggregate: pd.DataFrame) -> None:
    """Plot the complete v4 accuracy sweep while deployment timing is pending.

    The independent unit is one 500-event nominal seed trace (n=20 per point).
    Panel (a) reports the median trace-level loss. Panel (b) reports macro FPR
    and a deterministic percentile-bootstrap 95% CI over trace seeds.
    """
    index = _aggregate_index(aggregate)
    positions = _budget_positions()
    fig, axes = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH, 2.75), constrained_layout=True,
    )
    loss_axis, fpr_axis = axes
    handles = []
    for method in V4_METHODS:
        x_values = []
        losses = []
        fprs = []
        fpr_low = []
        fpr_high = []
        for position, budget in enumerate(BUDGETS):
            row = index.loc[(method, budget)]
            if str(row["available"]) != "True":
                continue
            x_values.append(float(position))
            losses.append(float(row["median_mean_approx_loss"]))
            fprs.append(100.0 * float(row["macro_fpr"]))
            fpr_low.append(100.0 * float(row["macro_fpr_ci_low"]))
            fpr_high.append(100.0 * float(row["macro_fpr_ci_high"]))
        x = np.asarray(x_values)
        style = _style(method)
        (line,) = loss_axis.plot(
            x, losses, label=METHOD_LABELS[method], markersize=3.7,
            markerfacecolor="white", markeredgewidth=0.8, linewidth=1.0,
            **style,
        )
        handles.append(line)
        fpr_axis.fill_between(
            x, fpr_low, fpr_high, color=style["color"], alpha=0.09,
            linewidth=0, zorder=1,
        )
        fpr_axis.plot(
            x, fprs, markersize=3.7, markerfacecolor="white",
            markeredgewidth=0.8, linewidth=1.0, zorder=2, **style,
        )

    for axis in axes:
        axis.set_xticks(positions, [str(budget) for budget in BUDGETS])
        axis.set_xlabel("Transform bound $b$")
        axis.grid(True, which="major", axis="y", linewidth=0.45)
        axis.set_xlim(-0.25, len(BUDGETS) - 0.75)
    loss_axis.set_yscale("log")
    loss_axis.set_ylabel(r"Median event-mean loss $\overline{L}$")
    loss_axis.set_title("(a)", loc="left", fontsize=8)
    loss_axis.annotate(
        "Scott unavailable at $b{=}40$", xy=(0.02, 0.04),
        xycoords="axes fraction", fontsize=6.0, color=COLORS["scott"],
    )
    fpr_axis.set_yscale("log")
    fpr_axis.set_ylabel("Macro FPR (%)")
    fpr_axis.set_title("(b)", loc="left", fontsize=8)
    fpr_axis.annotate(
        "bands: paired-seed bootstrap 95% CI", xy=(0.02, 0.04),
        xycoords="axes fraction", fontsize=6.0, color="0.35",
    )
    fig.legend(
        handles=handles,
        labels=[METHOD_LABELS[method] for method in V4_METHODS],
        loc="outside upper center", ncols=5, frameon=False,
        fontsize=6.4, columnspacing=1.0, handlelength=2.1,
    )
    _save(fig, "accuracy_cost_tradeoff")


def _v4_guard_decisions() -> tuple[pd.DataFrame, str]:
    paths = tuple(sorted(
        (V4 / "cells").glob(
            "random_waypoint:seed-*/budget-*/dagger05_vote3_guarded/decisions.csv"
        )
    ))
    selected_paths = []
    frames = []
    combined_hash = hashlib.sha256()
    for path in paths:
        seed = int(path.parts[-4].rsplit("-", 1)[1])
        if seed not in range(348, 368):
            continue
        budget = int(path.parts[-3].split("-", 1)[1])
        frame = pd.read_csv(path)
        frame["scope"] = "nominal"
        frame["method"] = "dagger05_vote3_guarded"
        frame["trace_kind"] = "random_waypoint"
        frame["seed"] = seed
        frame["budget"] = budget
        frames.append(frame)
        selected_paths.append(path)
        combined_hash.update(str(path.relative_to(ROOT)).encode("utf-8"))
        combined_hash.update(path.read_bytes())
    _require(len(selected_paths) == 140, "v4 guard-decision matrix is incomplete")
    decisions = pd.concat(frames, ignore_index=True)
    _require(len(decisions) == 70_000, "v4 guard decisions must cover 140 x 500 events")
    return decisions, combined_hash.hexdigest()


def main_v4_tables() -> None:
    """Regenerate v4 paper artifacts while deployment timing remains pending."""
    OUTPUT.mkdir(exist_ok=True)
    _configure_matplotlib()
    nominal = _v4_nominal()
    aggregate = _v4_aggregate(nominal)
    # Preserve the paper's established accuracy/throughput Pareto composition;
    # v4 only fills the two previously pending voting-policy trails.
    _plot_tradeoff(nominal, aggregate)
    guard_decisions, guard_decision_hash = _v4_guard_decisions()
    guard_flow = _plot_guard_flow(guard_decisions)
    _headline_table_v4(aggregate, _availability(nominal))
    learned = _learned_comparison_v4(nominal)
    predictor = _predictor_table_v4()
    composition = _composition_table_v4()
    sources = [
        V4 / "nominal/manifest.json",
        V4 / "nominal/summary.csv",
        V4 / "prediction-ablation/manifest.json",
        V4 / "prediction-ablation/summary.csv",
    ]
    (OUTPUT / "v4_table_manifest.json").write_text(json.dumps({
        "schema": "pzr.paper-table-generation.v4",
        "nominal_cells": len(nominal),
        "completed_nominal_cells": int((nominal["status"] == "completed").sum()),
        "unavailable_nominal_cells": int((nominal["status"] != "completed").sum()),
        "false_negative_count": int(
            nominal.loc[nominal["status"] == "completed", "false_negative_count"].sum()
        ),
        "nominal_seeds": list(range(348, 368)),
        "runtime_status": "pending Raspberry Pi timing artifact",
        "figure1": {
            "panel_a": "median event-mean approximation loss over 20 seeds",
            "panel_b": "macro FPR with paired-seed percentile-bootstrap 95% CI",
            "deployment_timing_included": False,
        },
        "guard_flow": guard_flow,
        "guard_decisions_combined_sha256": guard_decision_hash,
        "learned_comparison": learned,
        "predictor_ablation": predictor,
        "reducer_decision_totals": composition,
        "source_sha256": _source_hashes(sources),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main_v4_figures() -> None:
    """Update only figures whose canonical Vote3 cells were previously absent."""
    OUTPUT.mkdir(exist_ok=True)
    _configure_matplotlib()
    nominal = _v4_nominal()
    aggregate = _v4_aggregate(nominal)
    _plot_tradeoff(nominal, aggregate)
    guard_decisions, guard_decision_hash = _v4_guard_decisions()
    guard_flow = _plot_guard_flow(guard_decisions)
    sources = [
        V4 / "nominal/manifest.json",
        V4 / "nominal/summary.csv",
    ]
    (OUTPUT / "v4_figure_manifest.json").write_text(json.dumps({
        "schema": "pzr.paper-figure-generation.v4",
        "nominal_cells": len(nominal),
        "nominal_seeds": list(range(348, 368)),
        "figures": {
            "accuracy_cost_tradeoff": {
                "composition": "existing Pareto plus FPR panels",
                "methods": list(V4_METHODS),
                "deployment_timing_status": "pending Raspberry Pi replacement",
            },
            "guard_flow": guard_flow,
        },
        "guard_decisions_combined_sha256": guard_decision_hash,
        "source_sha256": _source_hashes(sources),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    _configure_matplotlib()

    canonical_nominal, aggregate = _canonical_nominal()
    confirmation_nominal, confirmation_fixed, pairs, cells = _confirmation()
    canonical_fixed = _read_csv(CANONICAL / "headline/summary.csv")
    ablation = _search_ablation()
    objective = _objective_comparison()
    scott_failures = _scott_failures()
    decisions = _read_csv(GUARDED / "report/artifacts/vote_decisions.csv")

    _plot_tradeoff(canonical_nominal, aggregate)
    guard_flow = _plot_guard_flow(decisions)
    _cross_check_guard(cells, pairs)

    pending_canonical = _headline_table(aggregate, _availability(canonical_nominal))
    scott_profile = _scott_failure_profile(scott_failures)
    objective_summary = _search_cost_table(ablation, objective)
    canonical_map, confirmation_map = _fixed_maps(canonical_fixed, confirmation_fixed)
    scott_partial = _scott_partial_map(scott_failures)
    _fixed_table_main(canonical_map, confirmation_map, scott_partial)
    _fixed_table_fpr(canonical_map, confirmation_map, scott_partial)
    fixed_summary = _fixed_summary_table(canonical_map, confirmation_map, scott_partial)
    composition_totals = _composition_table()
    learned_summary = _learned_comparison(cells)
    gate_summary = _gate_summary(cells, learned_summary)

    sources = [
        CANONICAL / "generalization/summary.csv",
        CANONICAL / "generalization/timeseries.csv",
        CANONICAL / "headline/summary.csv",
        CANONICAL / "ablation/summary.csv",
        CANONICAL / "objective-comparison/summary.csv",
        CANONICAL / "science-report/artifacts/nominal_generalization_aggregates.csv",
        GUARDED / "evaluate-nominal/summary.csv",
        GUARDED / "evaluate-fixed/summary.csv",
        GUARDED / "report/artifacts/confirmation_eligibility.csv",
        GUARDED / "report/artifacts/nominal_guard_benefit.csv",
        GUARDED / "report/artifacts/trace_cells.csv",
        GUARDED / "report/artifacts/vote_decisions.csv",
    ]
    (OUTPUT / "generation_manifest.json").write_text(json.dumps({
        "canonical_nominal_cells": len(canonical_nominal),
        "confirmation_nominal_cells": len(confirmation_nominal),
        "confirmation_fixed_cells": len(confirmation_fixed),
        "ablation_cells": len(ablation),
        "objective_cells": len(objective),
        "budgets": list(BUDGETS),
        "headline_budget": HEADLINE_BUDGET,
        "event_rate_hz": EVENT_RATE_HZ,
        "objective_comparison": objective_summary,
        "reducer_decision_totals": composition_totals,
        "learned_comparison": learned_summary,
        "confirmation_gates": gate_summary,
        "guard_flow": guard_flow,
        # Empty once the canonical cohort covers the voting policies; until then
        # this is the machine-readable form of the placeholder rows in Table I.
        "pending_canonical_methods": list(pending_canonical),
        "scott_failure_profile": scott_profile,
        "fixed_scott_usage": _fixed_scott_usage(decisions),
        "fixed_summary": fixed_summary,
        "source_sha256": _source_hashes(sources),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if "--v4-figures-only" in sys.argv[1:]:
        main_v4_figures()
    elif "--v4-tables-only" in sys.argv[1:]:
        main_v4_tables()
    else:
        main()
