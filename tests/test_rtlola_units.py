import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pzr.rtlola.actions import (
    EXPLICIT_ACTION_METHOD_NAMES,
    RtlolaAction,
    STATIC_ACTION_METHOD_NAMES,
)
from pzr.rtlola.benchmark import (
    CORE_METHODS,
    METHOD_SET_CHOICES,
    MPC_METHODS,
    RtlolaBenchmarkConfig,
    RtlolaExecutedStep,
    RtlolaReferenceStep,
    make_step_record,
    methods_for_config,
    trigger_confusion,
)
from pzr.rtlola.cli import main as cli_main
from pzr.rtlola.engine import RtlolaEngine, RtlolaEvent, RtlolaStateRef
from pzr.rtlola.metrics import (
    active_generator_count,
    generator_count,
    matrix_metrics,
)
from pzr.rtlola.omni import (
    OMNI_DEFAULT_TRACE_KIND,
    OMNI_PUBLIC_STREAM_KEYS,
    OMNI_SPEC,
    OMNI_TRACE_KINDS,
    generate_omni_events,
)
from pzr.rtlola.robot_arm import (
    ARM_PUBLIC_STREAM_KEYS,
    ARM_SPEC,
    ARM_TRIGGER_KEYS,
    DEFAULT_TRACE_KIND,
    ROBOT_ARM_SPEC_SHA256,
    ROBOT_ARM_TRACE_ROWS,
    ROBOT_ARM_TRACE_SHA256,
    RLOLAEVAL_REVISION,
    TRACE_KINDS,
    generate_robot_arm_events,
    load_robot_arm_trace,
    trace_path,
    validate_trace_tcp_against_fk,
)
from pzr.rtlola.scenarios import scenario_by_name
from pzr.rtlola.search import (
    MPC_VARIANTS,
    RtlolaSearchResult,
    beam_search,
    search_mpc_variant,
)
from pzr.rtlola.sweep_report import consolidate_sweep


def test_trigger_confusion_uses_reference_class_denominators():
    timeseries = pd.DataFrame({
        "method": ["girard"] * 4,
        "trigger_positive": [True, False, False, True],
        "exact_trigger_positive": [False, False, True, True],
        "alarm": [True, False, False, True],
        "exact_alarm": [False, False, True, True],
    })

    confusion = trigger_confusion(timeseries, ("alarm",))

    assert list(confusion["trigger_key"]) == ["__any__", "alarm"]
    assert (confusion["false_positive_steps"] == 1).all()
    assert (confusion["false_negative_steps"] == 1).all()
    assert (confusion["reference_negative_steps"] == 2).all()
    assert (confusion["reference_positive_steps"] == 2).all()
    assert (confusion["fpr"] == 0.5).all()
    assert (confusion["fnr"] == 0.5).all()


def test_sweep_report_compares_mpc_with_best_static(tmp_path):
    scenario_dir = tmp_path / "runs" / "figure8_drift" / "budget_40" / "robot_arm"
    scenario_dir.mkdir(parents=True)
    pd.DataFrame([
        {
            "method": "girard",
            "seed": 0,
            "budget": 40,
            "trace_kind": "figure8_drift",
            "false_positive_count": 4,
            "false_negative_count": 1,
            "reference_positive_count": 10,
            "reference_negative_count": 10,
            "fpr": 0.4,
            "fnr": 0.1,
            "mean_approx_loss": 2.0,
            "final_approx_loss": 4.0,
            "max_approx_loss": 4.0,
            "sum_approx_loss": 8.0,
            "mean_state_width": 8.0,
            "max_state_width": 9.0,
            "total_time_ms": 10.0,
            "fallback_count": 0,
            "reducer_failure_count": 0,
        },
        {
            "method": "scott",
            "seed": 0,
            "budget": 40,
            "trace_kind": "figure8_drift",
            "false_positive_count": 5,
            "false_negative_count": 0,
            "reference_positive_count": 10,
            "reference_negative_count": 10,
            "fpr": 0.5,
            "fnr": 0.0,
            "mean_approx_loss": 3.0,
            "final_approx_loss": 3.5,
            "max_approx_loss": 3.5,
            "sum_approx_loss": 12.0,
            "mean_state_width": 6.0,
            "max_state_width": 7.0,
            "total_time_ms": 12.0,
            "fallback_count": 0,
            "reducer_failure_count": 0,
        },
        {
            "method": "mpc_terminal_beam",
            "seed": 0,
            "budget": 40,
            "trace_kind": "figure8_drift",
            "false_positive_count": 3,
            "false_negative_count": 2,
            "reference_positive_count": 10,
            "reference_negative_count": 10,
            "fpr": 0.3,
            "fnr": 0.2,
            "mean_approx_loss": 1.0,
            "final_approx_loss": 0.25,
            "max_approx_loss": 0.75,
            "sum_approx_loss": 1.0,
            "mean_state_width": 7.0,
            "max_state_width": 8.0,
            "total_time_ms": 30.0,
            "fallback_count": 1,
            "reducer_failure_count": 2,
        },
    ]).to_csv(scenario_dir / "summary.csv", index=False)
    pd.DataFrame([
        {
            "method": "girard",
            "seed": 0,
            "budget": 40,
            "trace_kind": "figure8_drift",
            "step": 0,
            "reducer_used": "girard",
            "predicted_sequence": "girard",
            "approx_loss": 2.0,
            "state_width": 8.0,
        },
        {
            "method": "mpc_terminal_beam",
            "seed": 0,
            "budget": 40,
            "trace_kind": "figure8_drift",
            "step": 0,
            "reducer_used": "scott",
            "predicted_sequence": "scott,girard",
            "approx_loss": 0.75,
            "state_width": 8.0,
        },
        {
            "method": "mpc_terminal_beam",
            "seed": 0,
            "budget": 40,
            "trace_kind": "figure8_drift",
            "step": 1,
            "reducer_used": "scott",
            "predicted_sequence": "scott,girard",
            "approx_loss": 0.25,
            "state_width": 6.0,
        },
    ]).to_csv(scenario_dir / "timeseries.csv", index=False)
    pd.DataFrame([{
        "scenario": "robot_arm",
        "trace_kind": "figure8_drift",
        "method": "interval_hull",
        "seed": 0,
        "budget": 40,
        "step": 100,
        "time": 10.0,
        "phase": "select",
        "failure_type": "RtlolaNoFeasibleAction",
        "message": "no sound action",
    }]).to_csv(scenario_dir / "run_failures.csv", index=False)

    consolidate_sweep(tmp_path)

    comparison = pd.read_csv(tmp_path / "mpc_vs_static_metrics.csv")
    fpr = comparison[comparison["metric"] == "fpr"].iloc[0]
    assert fpr["best_static_method"] == "girard"
    assert fpr["absolute_improvement"] == pytest.approx(0.1)
    assert fpr["relative_improvement"] == pytest.approx(0.25)
    fnr = comparison[comparison["metric"] == "fnr"].iloc[0]
    assert fnr["best_static_method"] == "scott"
    width = comparison[comparison["metric"] == "mean_state_width"].iloc[0]
    assert width["best_static_method"] == "scott"
    loss = comparison[comparison["metric"] == "mean_approx_loss"].iloc[0]
    assert loss["best_static_method"] == "girard"
    assert loss["relative_improvement"] == pytest.approx(0.5)
    final_loss = comparison[
        comparison["metric"] == "final_approx_loss"
    ].iloc[0]
    assert final_loss["best_static_method"] == "scott"
    assert final_loss["best_static_value"] == pytest.approx(3.5)
    assert final_loss["mpc_value"] == pytest.approx(0.25)
    sum_loss = comparison[comparison["metric"] == "sum_approx_loss"].iloc[0]
    assert sum_loss["best_static_method"] == "girard"
    assert sum_loss["best_static_value"] == pytest.approx(8.0)
    assert sum_loss["mpc_value"] == pytest.approx(1.0)
    combined = pd.read_csv(tmp_path / "combined_summary.csv")
    mpc_summary = combined[combined["method"] == "mpc_terminal_beam"].iloc[0]
    assert mpc_summary["final_approx_loss"] == pytest.approx(0.25)
    assert mpc_summary["sum_approx_loss"] == pytest.approx(1.0)
    primary = pd.read_csv(tmp_path / "primary_metrics.csv")
    assert list(primary.columns) == [
        "trace_kind",
        "budget",
        "method",
        "seed",
        "false_positive_count",
        "false_negative_count",
        "reference_positive_count",
        "reference_negative_count",
        "fpr",
        "fnr",
        "mean_approx_loss",
        "final_approx_loss",
        "sum_approx_loss",
        "mean_state_width",
        "max_state_width",
        "total_time_ms",
    ]
    primary_mpc = primary[primary["method"] == "mpc_terminal_beam"].iloc[0]
    assert primary_mpc["final_approx_loss"] == pytest.approx(0.25)
    assert primary_mpc["sum_approx_loss"] == pytest.approx(1.0)
    assert (tmp_path / "combined_reducer_counts.csv").stat().st_size > 0
    methods = pd.read_csv(tmp_path / "method_comparison.csv")
    assert set(methods["method"]) == {"girard", "scott", "mpc_terminal_beam"}
    composition = pd.read_csv(tmp_path / "mpc_action_composition.csv")
    assert composition.loc[0, "reducer_used"] == "scott"
    assert composition.loc[0, "step_share"] == pytest.approx(1.0)
    assert composition.loc[0, "reduction_share"] == pytest.approx(1.0)
    followthrough = pd.read_csv(tmp_path / "mpc_plan_followthrough.csv")
    terminal_girard = followthrough[
        (followthrough["position"] == 1)
        & (followthrough["predicted_action"] == "girard")
    ].iloc[0]
    assert terminal_girard["realization_rate"] == pytest.approx(0.0)
    deferral = pd.read_csv(tmp_path / "mpc_girard_deferral.csv")
    assert deferral.loc[0, "girard_deferral_rate"] == pytest.approx(1.0)
    failures = pd.read_csv(tmp_path / "combined_run_failures.csv")
    assert failures.loc[0, "method"] == "interval_hull"


def test_matrix_metrics_distinguish_dense_active_and_constant_generators():
    dynamic = np.array([
        [1.0, 0.5, -0.25, 0.0],
        [2.0, 0.0, 0.75, 0.0],
    ])
    total = np.column_stack([dynamic, np.array([0.1, 0.2])])

    metrics = matrix_metrics(dynamic, total)

    assert generator_count(dynamic) == 3
    assert active_generator_count(dynamic) == 2
    assert metrics.dynamic_generator_count == 3
    assert metrics.active_dynamic_generator_count == 2
    assert metrics.zero_dynamic_generator_count == 1
    assert metrics.total_generator_count == 4
    assert metrics.state_width == pytest.approx(3.0)


def test_packaged_specs_and_registered_scenarios_are_authoritative():
    assert "constant delta: Variable" in OMNI_SPEC
    assert "#[public]\noutput position_x :=" in OMNI_SPEC
    assert "constant a5H: Variable" in ARM_SPEC
    omni = scenario_by_name("omni_robot")
    assert omni.spec == OMNI_SPEC
    assert omni.trace_kinds == OMNI_TRACE_KINDS
    assert omni.default_trace_kind == OMNI_DEFAULT_TRACE_KIND
    assert omni.public_stream_keys == OMNI_PUBLIC_STREAM_KEYS
    arm = scenario_by_name("robot_arm")
    assert arm.spec == ARM_SPEC
    assert arm.event_arity == 9
    assert arm.trace_kinds == TRACE_KINDS
    assert arm.default_trace_kind == "figure8_drift"
    assert arm.public_stream_keys == ARM_PUBLIC_STREAM_KEYS
    assert arm.trigger_keys == ARM_TRIGGER_KEYS
    assert arm.source_revision == RLOLAEVAL_REVISION


def test_mpc_objective_is_fixed_and_not_a_cli_option(capsys):
    with pytest.raises(TypeError, match="mpc_objective"):
        RtlolaBenchmarkConfig(mpc_objective="python_proxy")  # type: ignore[call-arg]
    with pytest.raises(SystemExit):
        cli_main(["--mpc-objective", "python_proxy"])
    assert "unrecognized arguments: --mpc-objective" in capsys.readouterr().err


def test_mpc_candidates_can_be_restricted_per_run():
    config = RtlolaBenchmarkConfig(
        mpc_candidate_names=["girard", "scott", "combastel", "pca"],
    )
    assert config.mpc_candidate_names == [
        "girard",
        "scott",
        "combastel",
        "pca",
    ]


def test_method_set_expansion_preserves_public_contract():
    assert METHOD_SET_CHOICES == ("core", "static", "mpc", "all")
    assert CORE_METHODS == (
        "none",
        "girard",
        "scott",
        "interval_hull",
        "pca",
        "mpc_terminal_beam",
    )
    assert methods_for_config(RtlolaBenchmarkConfig(method_set="core")) == CORE_METHODS
    assert methods_for_config(RtlolaBenchmarkConfig(method_set="static")) == (
        "none",
        "girard",
        "scott",
        "interval_hull",
        "pca",
        "althoff_a",
        "clustering",
        "combastel",
        "colinear_scale",
    )
    assert methods_for_config(RtlolaBenchmarkConfig(method_set="mpc")) == MPC_METHODS
    assert methods_for_config(RtlolaBenchmarkConfig(method_set="all")) == (
        *STATIC_ACTION_METHOD_NAMES,
        *MPC_METHODS,
    )


def test_explicit_method_override_accepts_static_mpc_and_fallback_names():
    methods = ["colinear", "interval", "mpc_terminal_beam", "girard"]

    assert methods_for_config(RtlolaBenchmarkConfig(methods=methods)) == tuple(methods)
    assert set(EXPLICIT_ACTION_METHOD_NAMES) >= {
        "none",
        "girard",
        "colinear",
        "interval",
    }


def test_method_config_rejects_unknown_method_names():
    with pytest.raises(ValueError, match="unknown RTLola method"):
        methods_for_config(RtlolaBenchmarkConfig(methods=["girard", "not_real"]))

    with pytest.raises(ValueError, match="method_set must be one of"):
        methods_for_config(RtlolaBenchmarkConfig(method_set="not_real"))


def test_make_step_record_uses_executed_step_boundary():
    metrics = SimpleNamespace(
        dynamic_generator_count=12,
        total_generator_count=13,
        active_dynamic_generator_count=10,
        active_total_generator_count=11,
        zero_dynamic_generator_count=2,
        zero_total_generator_count=2,
        state_width=4.5,
    )
    committed = SimpleNamespace(
        state=object(),
        verdict={"alarm": True, "runtime_ns": 123},
        metrics=metrics,
    )
    action = RtlolaAction("girard", lambda _budget: object())
    decision = RtlolaSearchResult(
        first_action=action,
        first_action_budget=7,
        first_step=committed,
        predicted_cost=1.25,
        predicted_sequence=("girard", "scott"),
        evaluated_leaves=3,
        pruned_branches=1,
        fallback_used=True,
        reducer_failure_count=2,
        infeasible_candidate_count=2,
    )
    executed = RtlolaExecutedStep(
        pre_generator_count=9,
        committed=committed,
        decision=decision,
        decision_time_ms=0.5,
    )
    engine = SimpleNamespace(
        approx_loss_reference=lambda approximation, state: 8.0,
    )
    scenario = SimpleNamespace(trigger_keys=("alarm",))

    record = make_step_record(
        engine=engine,
        scenario=scenario,
        seed=4,
        method="girard",
        step=2,
        budget=7,
        executed=executed,
        reference=RtlolaReferenceStep(
            verdicts={"alarm": False},
            approximation=object(),  # type: ignore[arg-type]
        ),
    )

    assert record.pre_generator_count == 9
    assert record.reducer_used == "girard"
    assert record.reduced is True
    assert record.approx_loss == pytest.approx(8.0)
    assert record.false_positive is True
    assert record.post_event_over_bound is True
    assert record.fallback_used is True
    assert record.infeasible_candidate_count == 2
    assert record.predicted_sequence == ("girard", "scott")


def test_omni_trace_is_seeded_and_deterministic():
    left = generate_omni_events(5, seed=42)
    right = generate_omni_events(5, seed=42)

    assert left == right
    np.testing.assert_allclose(
        [event.values for event in left[:3]],
        [
            (0.0, 0.05484907435579764, -0.041599364249619825),
            (1.0, 0.18993028960095995, 0.07338306819875957),
            (2.0, -0.1612560443567306, 0.018008121341064366),
        ],
    )
    assert left[0].time == 0.0
    assert all(a.time < b.time for a, b in zip(left, left[1:]))


@pytest.mark.parametrize("trace_kind", OMNI_TRACE_KINDS)
def test_all_omni_trace_kinds_are_seeded_and_resolve_in_scenario(trace_kind):
    left = generate_omni_events(8, seed=7, trace_kind=trace_kind)
    right = scenario_by_name("omni_robot").generate_trace(
        8,
        7,
        trace_kind,
    )

    assert left == right.events
    assert right.trace_kind == trace_kind
    assert all(a.time < b.time for a, b in zip(left, left[1:]))


def test_robot_arm_trace_matches_packaged_forward_kinematics():
    events = generate_robot_arm_events(4, trace_kind=DEFAULT_TRACE_KIND)

    assert len(events) == 4
    assert len(events[0].values) == 9
    assert all(value is not None for value in events[0].values[-3:])
    assert events[1].values[-3:] == (None, None, None)
    assert validate_trace_tcp_against_fk(
        DEFAULT_TRACE_KIND,
        max_rows=4,
    ) < 3e-4


def test_robot_arm_assets_match_rlolaeval_revision():
    assert hashlib.sha256(ARM_SPEC.encode()).hexdigest() == ROBOT_ARM_SPEC_SHA256
    assert set(TRACE_KINDS) == set(ROBOT_ARM_TRACE_SHA256)
    for trace_kind in TRACE_KINDS:
        assert hashlib.sha256(trace_path(trace_kind).read_bytes()).hexdigest() == (
            ROBOT_ARM_TRACE_SHA256[trace_kind]
        )
        assert len(load_robot_arm_trace(trace_kind)) == ROBOT_ARM_TRACE_ROWS[trace_kind]
        assert validate_trace_tcp_against_fk(trace_kind, max_rows=None) < 3e-4


def test_beam_search_supports_forced_root_with_full_continuation_pool():
    first = RtlolaAction("first", lambda _budget: object(), explicit_budget=False)
    future = RtlolaAction("future", lambda _budget: object(), explicit_budget=False)

    class FakeEngine:
        def metrics(self, state):
            return SimpleNamespace(dynamic_generator_count=99, dimension=1)

        def branch_step(self, state, event, action, config_budget):
            del event, config_budget
            depth = state.depth + 1
            cost = 5.0 if action.name == "first" else 1.0
            return SimpleNamespace(
                verdict={},
                state=SimpleNamespace(depth=depth),
                action_name=action.name,
                metrics=SimpleNamespace(state_width=cost),
            )

    result = beam_search(
        FakeEngine(),
        SimpleNamespace(depth=0),
        object(),
        (object(),),
        (first, future),
        budget=10,
        beam_width=2,
        fallback=future,
        forced_first_action=first,
        cost_fn=lambda _engine, step: step.metrics.state_width,
    )

    assert result.predicted_sequence == ("first", "future")
    assert result.predicted_cost == pytest.approx(1.0)


def test_beam_search_uses_binding_reference_loss_at_terminal_horizon():
    none = RtlolaAction("none", lambda _budget: object(), explicit_budget=False)
    first = RtlolaAction("first", lambda _budget: object(), explicit_budget=False)
    second = RtlolaAction("second", lambda _budget: object(), explicit_budget=False)

    class FakeEngine:
        loss_calls = []

        def metrics(self, state):
            return SimpleNamespace(dynamic_generator_count=99, dimension=1)

        def branch_step(self, state, event, action, config_budget):
            del event, config_budget
            next_state = SimpleNamespace(
                depth=state.depth + 1,
                path=(*state.path, action.name),
            )
            return SimpleNamespace(
                verdict={},
                state=next_state,
                action_name=action.name,
                metrics=SimpleNamespace(state_width=1000.0),
            )

        def approx_loss(self, reference, candidate):
            self.loss_calls.append((reference.path, candidate.path))
            if candidate.depth == 1:
                return 0.0 if candidate.path[0] == "first" else 1.0
            return 5.0 if candidate.path[0] == "first" else 2.0

    engine = FakeEngine()
    result = beam_search(
        engine,
        SimpleNamespace(depth=0, path=()),
        object(),
        (object(),),
        (first, second),
        budget=10,
        beam_width=2,
        fallback=first,
        none_action=none,
        cost_fn=lambda _engine, _step: pytest.fail("Python cost must not run"),
        use_reference_loss=True,
    )

    assert result.first_action.name == "second"
    assert result.predicted_cost == pytest.approx(2.0)
    assert {reference for reference, _candidate in engine.loss_calls} == {
        ("none",),
        ("none", "none"),
    }


def test_tail_variants_preserve_roots_and_score_distinct_objectives():
    none = RtlolaAction("none", lambda _budget: object(), explicit_budget=False)
    girard = RtlolaAction("girard", lambda _budget: object(), explicit_budget=False)
    fallback = RtlolaAction("interval", lambda _budget: object(), explicit_budget=False)
    alpha = RtlolaAction("alpha", lambda _budget: object(), explicit_budget=False)
    beta = RtlolaAction("beta", lambda _budget: object(), explicit_budget=False)

    class FakeEngine:
        def metrics(self, state):
            return SimpleNamespace(dynamic_generator_count=99, dimension=1)

        def branch_step(self, state, event, action, config_budget):
            del event, config_budget
            next_state = SimpleNamespace(
                depth=state.depth + 1,
                path=(*state.path, action.name),
            )
            return SimpleNamespace(
                verdict={},
                state=next_state,
                action_name=action.name,
                metrics=SimpleNamespace(state_width=0.0),
            )

        def approx_loss(self, reference, candidate):
            del reference
            root = candidate.path[0]
            if root == "alpha":
                return 10.0 if candidate.depth == 3 else 0.0
            if root == "beta":
                return 4.0
            return 100.0

    endpoint = search_mpc_variant(
        FakeEngine(),
        SimpleNamespace(depth=0, path=()),
        object(),
        (object(),),
        (object(),),
        (alpha, beta),
        budget=10,
        beam_width=1,
        variant=MPC_VARIANTS["mpc_terminal_girard_tail"],
        root_beam_width=1,
        fallback=fallback,
        none_action=none,
        tail_action=girard,
    )
    integrated = search_mpc_variant(
        FakeEngine(),
        SimpleNamespace(depth=0, path=()),
        object(),
        (object(),),
        (object(),),
        (alpha, beta),
        budget=10,
        beam_width=1,
        variant=MPC_VARIANTS["mpc_cumulative_girard_tail"],
        root_beam_width=1,
        fallback=fallback,
        none_action=none,
        tail_action=girard,
    )

    assert endpoint.first_action.name == "beta"
    assert endpoint.predicted_cost == pytest.approx(4.0)
    assert integrated.first_action.name == "alpha"
    assert integrated.predicted_cost == pytest.approx(10.0)
    assert {row.root_action for row in endpoint.root_evaluations} == {"alpha", "beta"}
    assert all(row.feasible for row in endpoint.root_evaluations)
    assert endpoint.realized_tail_steps == 1
    assert len(endpoint.predicted_sequence) == 2
    assert "girard" not in endpoint.predicted_sequence


def test_root_tail_branches_only_at_current_event():
    none = RtlolaAction("none", lambda _budget: object(), explicit_budget=False)
    girard = RtlolaAction("girard", lambda _budget: object(), explicit_budget=False)
    fallback = RtlolaAction("interval", lambda _budget: object(), explicit_budget=False)
    alpha = RtlolaAction("alpha", lambda _budget: object(), explicit_budget=False)
    beta = RtlolaAction("beta", lambda _budget: object(), explicit_budget=False)

    class FakeEngine:
        def metrics(self, state):
            return SimpleNamespace(dynamic_generator_count=99, dimension=1)

        def branch_step(self, state, event, action, config_budget):
            del event, config_budget
            next_state = SimpleNamespace(
                depth=state.depth + 1,
                path=(*state.path, action.name),
            )
            return SimpleNamespace(
                verdict={},
                state=next_state,
                action_name=action.name,
                metrics=SimpleNamespace(state_width=0.0),
            )

        def approx_loss(self, reference, candidate):
            del reference
            return 1.0 if candidate.path[0] == "alpha" else 2.0

    result = search_mpc_variant(
        FakeEngine(),
        SimpleNamespace(depth=0, path=()),
        object(),
        (object(), object(), object()),
        (object(), object()),
        (alpha, beta),
        budget=10,
        beam_width=1,
        variant=MPC_VARIANTS["mpc_one_step_girard_rollout"],
        root_beam_width=1,
        fallback=fallback,
        none_action=none,
        tail_action=girard,
    )

    assert result.first_action.name == "alpha"
    assert result.predicted_sequence == ("alpha",)
    assert result.optimized_horizon == 0
    assert result.realized_tail_steps == 2
    assert result.predicted_cost == pytest.approx(3.0)
    assert len(result.root_evaluations) == 2


def test_engine_wraps_binding_panics_so_search_can_fallback():
    class BindingPanic(BaseException):
        pass

    class PanickingPlanner:
        def accept_event_from_state(self, *args):
            raise BindingPanic("native transform panic")

    engine = RtlolaEngine.__new__(RtlolaEngine)
    engine.spec_id = "spec"
    engine.event_arity = 1
    engine.planner = PanickingPlanner()
    state = RtlolaStateRef(object(), "spec", 0, 0.0)
    action = RtlolaAction("panic", lambda _budget: object(), explicit_budget=False)

    with pytest.raises(RuntimeError, match="planner branch failed") as captured:
        engine.branch_step(
            state,
            RtlolaEvent(1.0, (1.0,)),
            action,
            budget=0,
        )

    assert isinstance(captured.value.__cause__, BindingPanic)
