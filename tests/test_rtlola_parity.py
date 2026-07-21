import json

import pytest

from pzr.rtlola.parity import (
    _Series,
    _compare_cell,
    _load_or_run_cell,
    _parse_csv_ints,
)


def test_parity_cell_checks_losses_triggers_states_and_reduction_choices():
    oracle = _Series(
        failed=None,
        losses=[0.0, 2.0],
        triggers=[False, True],
        choices=["girard", "scott"],
        reduction_required=[False, True],
        state_hashes=["same-0", "same-1"],
        elapsed=1.0,
    )
    production = _Series(
        failed=None,
        losses=[0.0, 2.0],
        triggers=[False, True],
        choices=["none", "scott"],
        reduction_required=[False, True],
        state_hashes=["same-0", "same-1"],
        elapsed=2.0,
    )
    golden = {
        "failed": None,
        "acc_loss": 2.0,
        "avg_loss": 1.0,
        "final_loss": 2.0,
        "ev_per_sec": 123.0,
        "triggers": [False, True],
        "ref_triggers": [False, True],
    }

    result = _compare_cell(
        oracle,
        production,
        [False, True],
        golden,
        compare_choices=True,
    )

    assert result["correctness_passed"] is True
    assert result["choice_mismatch_count"] == 0
    assert result["state_mismatch_count"] == 0


def test_parity_cells_resume_only_with_the_same_fingerprint(tmp_path):
    path = tmp_path / "cell.json"
    calls = []

    first = _load_or_run_cell(
        path,
        fingerprint="same",
        run=lambda: calls.append("run") or {"correctness_passed": True},
    )
    second = _load_or_run_cell(
        path,
        fingerprint="same",
        run=lambda: calls.append("rerun") or {},
    )

    assert first == second
    assert calls == ["run"]
    assert json.loads(path.read_text())["fingerprint"] == "same"
    with pytest.raises(ValueError, match="stale parity cell"):
        _load_or_run_cell(path, fingerprint="different", run=lambda: {})


def test_parity_treats_unpinned_archived_golden_as_informational():
    series = _Series(
        failed=None,
        losses=[3.0],
        triggers=[False],
        choices=["pca"],
        reduction_required=[True],
        state_hashes=["same"],
        elapsed=1.0,
    )
    stale_golden = {
        "failed": None,
        "acc_loss": 9.0,
        "avg_loss": 9.0,
        "final_loss": 9.0,
        "ev_per_sec": 123.0,
        "triggers": [False],
        "ref_triggers": [False],
    }

    result = _compare_cell(
        series,
        series,
        [False],
        stale_golden,
        compare_choices=True,
    )

    assert result["correctness_passed"] is True
    assert result["implementation_matches_oracle"] is True
    assert result["production_matches_golden"] is False
    assert "archived upstream JSON differs" in result["message"]


def test_parity_bound_parser_uses_comma_separated_integers():
    assert _parse_csv_ints("15, 50,500") == (15, 50, 500)
