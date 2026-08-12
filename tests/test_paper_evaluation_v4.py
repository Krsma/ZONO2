from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/paper_evaluation_v4.py"
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location("paper_evaluation_v4", TOOL)
assert SPEC is not None and SPEC.loader is not None
v4 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = v4
SPEC.loader.exec_module(v4)


def test_v4_canonical_matrix_and_seed_contract() -> None:
    config = v4.load_config()
    assert config.nominal_seeds == tuple(range(348, 368))
    assert config.expected_nominal_cells == 1_400
    assert config.expected_prediction_cells == 15
    assert config.expected_timing_cells == 350
    assert tuple(method.name for method in config.methods) == v4.METHOD_NAMES
    assert config.prediction_seeds == tuple(range(60, 65))
    assert not set(config.nominal_seeds) & set(range(0, 348))


def test_v4_mpc_f_and_learning_teacher_identities() -> None:
    config = v4.load_config()
    methods = {method.name: method for method in config.methods}
    full = methods["mpc_terminal_full_width"]
    assert (full.horizon, full.beam_width, full.runtime_method) == (
        1,
        1,
        "mpc_terminal_full_width",
    )
    assert methods["pairwise_ranking_policy"].kind == "g15"
    assert methods["mpc_terminal_beam_predictive_linear"].kind == "mpc"


def test_v4_predictor_and_timing_contracts() -> None:
    config = v4.load_config()
    assert config.predictors == ("hold", "linear", "quadratic")
    assert set(v4.PREDICTOR_METHODS.values()) == {
        "mpc_terminal_beam_predictive_hold",
        "mpc_terminal_beam_predictive_linear",
        "mpc_terminal_beam_predictive_quadratic",
    }
    warmup, measured = v4.timing_window_indices(config)
    assert (warmup.start, warmup.stop) == (0, 100)
    assert (measured.start, measured.stop) == (100, 300)


def test_v4_method_order_rotation_is_balanced_and_deterministic() -> None:
    methods = v4.METHOD_NAMES
    rotations = [v4.rotate_method_order(methods, seed, 0) for seed in range(5)]
    assert rotations[0] == methods
    assert rotations[1] == methods[1:] + methods[:1]
    assert all(set(order) == set(methods) for order in rotations)
    for position in range(len(methods)):
        assert len({order[position] for order in rotations}) == 5


def test_v4_import_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps({"ok": True}))
    with pytest.raises(ValueError, match="hash mismatch"):
        v4._verify_file(path, "0" * 64, "test artifact")


def test_v4_all_pinned_imports_and_models_are_verifiable() -> None:
    config = v4.load_config()
    for item in config.imports:
        v4._verify_file(item.manifest, item.manifest_sha256, item.name)
        v4._verify_file(item.summary, item.summary_sha256, item.name)
    verified = v4._verify_models(config)
    assert verified["verified_model_count"] == 28


def test_v4_smoke_contract_covers_all_method_and_predictor_types(tmp_path: Path) -> None:
    config = v4.load_config(output=tmp_path, smoke=True)
    assert config.expected_nominal_cells == 10
    assert config.expected_prediction_cells == 3
    assert config.expected_timing_cells == 10
    assert {method.kind for method in config.methods} == {
        "static",
        "mpc",
        "g15",
        "vote3",
        "vote3_guarded",
    }
    assert v4.timing_window_indices(config) == (range(0, 5), range(5, 15))


def test_v4_imports_hw_without_scheduling_hw_execution() -> None:
    config = v4.load_config()
    assert "v3_ablation" in {item.name for item in config.imports}
    assert "ablation" not in v4.STAGES


def test_v4_default_run_stops_before_workstation_timing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = v4.load_config(output=tmp_path, smoke=True)
    called: list[str] = []

    def record_stage(_config: v4.V4Config, stage: str) -> Path:
        called.append(stage)
        return v4._stage_path(_config, stage)

    monkeypatch.setattr(v4, "run_stage", record_stage)
    result = v4.run_all(config)

    assert tuple(called) == v4.RUN_STAGES
    assert v4.RUN_STAGES == v4.STAGES[:5]
    assert "runtime" not in called
    assert "report" not in called
    assert "validate" not in called
    assert result == tmp_path / "prediction-ablation/manifest.json"


def test_v4_severe_tail_table_tolerates_an_unavailable_method() -> None:
    import pandas as pd

    rows = [
        {"seed": 348, "budget": 40, "method": "mpc_terminal_beam_predictive_linear", "status": "completed", "mean_approx_loss": 1.0},
        {"seed": 348, "budget": 40, "method": "girard", "status": "completed", "mean_approx_loss": 2_000.0},
        {"seed": 348, "budget": 40, "method": "scott", "status": "fallback_failed", "mean_approx_loss": float("nan")},
    ]
    result = v4._severe_tails(pd.DataFrame(rows))
    assert result[["method", "loss_ratio_vs_mpc_l"]].to_dict("records") == [
        {"method": "girard", "loss_ratio_vs_mpc_l": 2_000.0}
    ]
