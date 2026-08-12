from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/paper_evaluation_v4_pi_timing_v1.py"
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location(
    "paper_evaluation_v4_pi_timing_v1", TOOL
)
assert SPEC is not None and SPEC.loader is not None
pi = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pi
SPEC.loader.exec_module(pi)


def test_pi_timing_matrix_and_parent_pins_are_exact() -> None:
    config = pi.load_config()
    assert config.seeds == tuple(range(348, 353))
    assert config.budgets == (40, 80, 120, 150, 200, 250, 500)
    assert config.methods == pi.METHOD_NAMES
    assert config.expected_cells == 350
    assert config.measured_event_count == 200
    assert pi.verify_parent_pins(config)


def test_pi_smoke_contract_keeps_all_method_types() -> None:
    config = pi.load_config(smoke=True)
    assert config.seeds == (348,)
    assert config.budgets == (40,)
    assert config.methods == pi.METHOD_NAMES
    assert config.expected_cells == 10
    assert (config.warmup_start, config.warmup_stop) == (0, 5)
    assert (config.measured_start, config.measured_stop) == (5, 15)


def test_pi_outputs_cannot_target_parent_result_roots(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="isolated versioned roots"):
        pi._assert_isolated_output(ROOT / "results/paper-evaluation-v4")
    with pytest.raises(ValueError, match="isolated versioned roots"):
        pi._assert_isolated_output(tmp_path / "paper-evaluation-v4-pi-timing-v1")
    pi._assert_isolated_output(ROOT / "results/paper-evaluation-v4-pi-timing-v1")
    pi._assert_isolated_output(ROOT / "results/paper-evaluation-v4-final-report-v1")
    with pytest.raises(ValueError, match="isolated add-on directory"):
        pi._assert_runtime_output(ROOT / "results/paper-evaluation-v4")
    pi._assert_runtime_output(ROOT / "results/paper-evaluation-v4-pi-timing-v1")
    pi._assert_runtime_output(
        tmp_path / "output/paper-evaluation-v4-pi-timing-v1"
    )


def test_latency_statistics_known_constant_case() -> None:
    result = pi.latency_statistics([10.0] * 200)
    assert result["p50_selection_commit_latency_ms"] == pytest.approx(10.0)
    assert result["p99_selection_commit_latency_ms"] == pytest.approx(10.0)
    assert result["mad_selection_commit_latency_ms"] == pytest.approx(0.0)
    assert result["iqr_selection_commit_latency_ms"] == pytest.approx(0.0)
    assert result["p99_p50_tail_ratio"] == pytest.approx(1.0)
    assert result["saturation_throughput_events_per_second"] == pytest.approx(100.0)


def test_rate_capacity_constant_and_bursty_cases() -> None:
    constant = [10.0] * 200
    share, backlog = pi.replay_service_times(constant, 100.0)
    assert share == pytest.approx(1.0)
    assert backlog == pytest.approx(0.0, abs=1e-12)
    assert pi.maximum_empirical_rate(constant, 1.0) == pytest.approx(100.0)

    bursty = [5.0] * 95 + [30.0] * 5
    rates = [pi.maximum_empirical_rate(bursty, target) for target in (0.95, 0.99, 1.0)]
    saturation = pi.latency_statistics(bursty)["saturation_throughput_events_per_second"]
    assert rates[2] <= rates[1] <= rates[0] <= saturation * (1.0 + 1e-10)


def test_rate_metrics_reject_invalid_samples() -> None:
    with pytest.raises(ValueError, match="positive"):
        pi.latency_statistics([0.0, 1.0])
    with pytest.raises(ValueError, match="positive"):
        pi.replay_service_times([1.0], 0.0)
    with pytest.raises(ValueError, match="target"):
        pi.maximum_empirical_rate([1.0], 0.0)


def test_method_rotation_matches_v4_contract() -> None:
    methods = pi.METHOD_NAMES
    assert pi.rotate_method_order(methods, 0, 0) == methods
    assert pi.rotate_method_order(methods, 1, 0) == methods[1:] + methods[:1]
    assert pi.rotate_method_order(methods, 4, 6) == methods


def test_bundle_verification_rejects_missing_extra_and_changed_files(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    payload = bundle / "payload.txt"
    payload.write_text("frozen")
    nested_manifest = bundle / "nested/manifest.json"
    nested_manifest.parent.mkdir()
    nested_manifest.write_text("nested parent manifest")
    manifest = {
        "schema": pi.BUNDLE_SCHEMA,
        "files": {
            "payload.txt": pi.raw_sha256(payload),
            "nested/manifest.json": pi.raw_sha256(nested_manifest),
        },
    }
    manifest["bundle_identity_sha256"] = pi.payload_sha256(manifest)
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    assert pi.verify_bundle(bundle)["bundle_identity_sha256"]

    payload.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        pi.verify_bundle(bundle)
    payload.write_text("frozen")
    (bundle / "extra.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="file set differs"):
        pi.verify_bundle(bundle)


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    source = tmp_path / "payload.txt"
    source.write_text("unsafe")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source, arcname="../escape.txt")
    with pytest.raises(ValueError, match="unsafe archive path"):
        pi._safe_extract(archive, tmp_path / "extract")


def test_pareto_flags_known_points() -> None:
    frame = pd.DataFrame({
        "latency": [1.0, 2.0, 3.0, 2.0],
        "loss": [4.0, 2.0, 1.0, 5.0],
    })
    assert pi._pareto_flags(frame, "latency", "loss").tolist() == [True, True, True, False]


def test_memory_snapshot_has_linux_process_fields() -> None:
    result = pi._memory_snapshot()
    assert set(result) == {"rss_kib", "pss_kib", "uss_kib", "peak_rss_kib"}
    assert all(isinstance(value, int) for value in result.values())
    assert result["rss_kib"] > 0


def test_semantic_frame_removes_only_timing_columns() -> None:
    class Result:
        timeseries = pd.DataFrame({
            "step": [0, 1],
            "decision_time_ms": [1.0, 2.0],
            "binding_runtime_ns": [3.0, 4.0],
            "reducer_used": ["girard", "scott"],
        })

    frame = pi._semantic_frame(Result())
    assert frame.to_dict("list") == {
        "step": [0, 1],
        "reducer_used": ["girard", "scott"],
    }


def test_artifact_identity_seal_rejects_changed_payload() -> None:
    payload = {"schema": "test", "value": 3}
    sealed = pi._seal_identity(payload, "identity_sha256")
    pi._verify_identity(sealed, "identity_sha256")
    sealed["value"] = 4
    with pytest.raises(ValueError, match="identity hash differs"):
        pi._verify_identity(sealed, "identity_sha256")
