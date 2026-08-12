from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


TOOLS = Path(__file__).parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from dart_rescue_experiment import (  # noqa: E402
    METHODS,
    _method_config,
    load_config,
    paired_effects,
    rescue_cell_identity,
)
from pzr.rtlola.paper_experiment import TraceSource  # noqa: E402
from pzr.rtlola.paper_pipeline import EvaluationTrace  # noqa: E402


CONFIG = Path(__file__).parents[1] / "experiments" / "dart_rescue_v1.yaml"


def test_canonical_dart_rescue_scope_and_exact_counts():
    config = load_config(CONFIG)

    assert config.budgets == (40, 80, 120, 150, 200, 250, 500)
    assert config.extra_training_seeds == tuple(range(26, 42))
    assert config.replay_seeds == tuple(range(100, 120))
    assert config.confirmation_seeds == tuple(range(120, 140))
    assert config.expected_clean_shards == 112
    assert config.expected_dart_shards == 112
    assert config.expected_new_models == 14
    assert config.reported_cells("replay") == 420
    assert config.reported_cells("confirmation") == 420
    assert config.reported_cells("fixed") == 84
    assert config.new_cells("replay") == 280
    assert config.new_cells("confirmation") == 420
    assert config.new_cells("fixed") == 56
    assert sum(config.reported_cells(scope) for scope in (
        "replay", "confirmation", "fixed",
    )) == 924
    assert sum(config.new_cells(scope) for scope in (
        "replay", "confirmation", "fixed",
    )) == 756
    assert config.expected_pzr_source_sha256 == (
        "f230d481022de2c69c610c917deae901e7a87e4322c979c248f2cf8f4fa1e5ca"
    )


def test_smoke_scope_executes_every_model_without_importing_paper_cells():
    config = load_config(CONFIG, smoke=True)

    assert config.output_root == Path("/tmp/pzr-dart-rescue-v1-smoke")
    assert config.event_count == 100
    assert config.budgets == (40,)
    assert config.extra_training_seeds == (26,)
    assert config.reported_cells("replay") == len(METHODS)
    assert config.new_cells("replay") == len(METHODS)
    assert sum(config.new_cells(scope) for scope in (
        "replay", "confirmation", "fixed",
    )) == 9


def _summary(*, failed_method: str | None = None) -> pd.DataFrame:
    rows = []
    losses = {
        "clean20": (10.0, 20.0),
        "clean36": (8.0, 16.0),
        "dart36": (4.0, 8.0),
    }
    false_positives = {
        "clean20": (10, 20),
        "clean36": (8, 16),
        "dart36": (5, 10),
    }
    for method in METHODS:
        for seed in (100, 101):
            index = seed - 100
            failed = method == failed_method and seed == 101
            rows.append({
                "trace_id": f"random_waypoint:seed-{seed}",
                "trace_sha256": f"hash-{seed}",
                "trace_source": TraceSource.GENERATED_NOMINAL.value,
                "trace_kind": "random_waypoint",
                "condition": "random_waypoint",
                "seed": seed,
                "budget": 40,
                "method": method,
                "status": "fallback_failed" if failed else "completed",
                "false_positive_count": 0 if failed else false_positives[method][index],
                "false_negative_count": 0,
                "reference_negative_count": 0 if failed else 100,
                "reference_positive_count": 0 if failed else 20,
                "mean_approx_loss": np.nan if failed else losses[method][index],
                "total_time_ms": np.nan if failed else 100.0,
                "event_count": 500,
            })
    return pd.DataFrame(rows)


def test_paired_dart_effects_are_seed_aligned_and_deterministic():
    first, trace = paired_effects(
        _summary(),
        scope="replay",
        bootstrap_replicates=1000,
        bootstrap_seed=7,
    )
    second, _ = paired_effects(
        _summary(),
        scope="replay",
        bootstrap_replicates=1000,
        bootstrap_seed=7,
    )
    pd.testing.assert_frame_equal(first, second)

    dart = first[first["comparison"] == "dart_effect"].iloc[0]
    assert dart["valid_pair_count"] == 2
    assert dart["main_available"]
    assert np.isclose(dart["mean_fpr_difference"], -0.045)
    assert np.isclose(dart["geometric_mean_loss_ratio"], 0.5)
    assert np.isclose(dart["loss_win_fraction"], 1.0)
    assert len(trace) == 3 * 2


def test_failed_pair_makes_main_effect_unavailable():
    effects, _ = paired_effects(
        _summary(failed_method="dart36"),
        scope="confirmation",
        bootstrap_replicates=100,
        bootstrap_seed=11,
    )

    dart = effects[effects["comparison"] == "dart_effect"].iloc[0]
    assert dart["valid_pair_count"] == 1
    assert not dart["main_available"]
    assert np.isnan(dart["mean_fpr_difference"])
    assert np.isnan(dart["geometric_mean_loss_ratio"])


def test_rescue_cell_identity_records_model_budget_trace_and_tool(tmp_path):
    config = load_config(CONFIG, smoke=True)
    reference = tmp_path / "reference.json"
    reference.write_text("{}")
    trace = EvaluationTrace(
        trace_id="random_waypoint:seed-120",
        condition="random_waypoint",
        seed=120,
        events=(object(),) * 100,
        trace_sha256="trace-hash",
        trace_source=TraceSource.GENERATED_NOMINAL,
        trace_kind="random_waypoint",
        provenance={"generator_config_sha256": "generator-hash"},
    )

    identity = rescue_cell_identity(
        config,
        scope="confirmation",
        trace=trace,
        budget=40,
        method=_method_config(config, "dart36"),
        reference_path=reference,
        model_hash="model-hash",
    )

    assert identity["scope"] == "confirmation"
    assert identity["seed"] == 120
    assert identity["budget"] == 40
    assert identity["model_training_budget"] == 40
    assert identity["method"]["name"] == "dart36"
    assert identity["model_sha256"] == "model-hash"
    assert identity["tool_sha256"]
    assert identity["fingerprint"]
