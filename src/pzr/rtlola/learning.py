"""Scenario-neutral regret distillation for RTLola reducer actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import pandas as pd

from pzr.learning.ranking import (
    RegretDataset,
    RegretRankingPolicy,
    train_regret_policy,
)
from pzr.rtlola.actions import RtlolaAction, RtlolaActionCatalog, default_action_catalog
from pzr.rtlola.benchmark import (
    RtlolaBenchmarkConfig,
    RtlolaRunResult,
    RtlolaTriggerReferenceStep,
    load_or_compute_trigger_reference,
    make_step_record,
    results_to_dataframe,
    summarize_results,
)
from pzr.rtlola.binding import require_binding
from pzr.rtlola.engine import RtlolaEngine, RtlolaEvent, RtlolaStateRef
from pzr.rtlola.metrics import RtlolaMatrixMetrics
from pzr.rtlola.scenarios import RtlolaScenario, scenario_by_name
from pzr.rtlola.search import (
    RtlolaSearchResult,
    beam_search,
    normalized_trigger_width_cost,
)


RTL_FEATURE_NAMES = (
    "dynamic_generator_count",
    "total_generator_count",
    "dimension",
    "budget_headroom",
    "full_width_sum",
    "width_mean",
    "width_max",
    "center_l2",
    "center_linf",
    "gen_norm_mean",
    "gen_norm_max",
    "gen_norm_std",
    "gen_sparsity",
    "gen_coupling",
    "gen_pca_explained",
    "future_width_mean",
    "future_width_max",
    "future_width_final",
    "future_width_growth",
    "future_overflow_count",
    "future_generator_count_mean",
)


@dataclass(frozen=True)
class RtlolaCandidateCost:
    name: str
    cost: float
    sequence: tuple[str, ...]


@dataclass(frozen=True)
class RtlolaRegretTrace:
    features: np.ndarray
    candidate_names: tuple[str, ...]
    costs: tuple[float, ...]
    regrets: tuple[float, ...]
    best_action: str
    best_cost: float
    second_best_margin: float
    tie_count: int
    step: int
    seed: int
    iteration: int
    budget: int
    trace_kind: str


@dataclass(frozen=True)
class RtlolaRegretResult:
    policy: RegretRankingPolicy
    traces: tuple[RtlolaRegretTrace, ...]
    eval_results: tuple[RtlolaRunResult, ...]
    training_frames: tuple[pd.DataFrame, ...]


class RtlolaLearnedPolicy:
    """Choose a certified binding transform directly from a learned ranking."""

    def __init__(
        self,
        policy: RegretRankingPolicy,
        catalog: RtlolaActionCatalog,
    ) -> None:
        expected = tuple(catalog.mpc_candidate_names)
        if tuple(policy.candidate_names) != expected:
            raise ValueError(
                "learned policy candidate catalog does not match RTLola MPC "
                f"candidates: policy={policy.candidate_names}, expected={expected}"
            )
        self.policy = policy
        self.catalog = catalog

    def choose(
        self,
        engine: RtlolaEngine,
        state: RtlolaStateRef,
        event: RtlolaEvent,
        future_events: Sequence[RtlolaEvent],
        budget: int,
    ) -> RtlolaSearchResult:
        pre_metrics = engine.metrics(state)
        if pre_metrics.dynamic_generator_count <= budget:
            step = engine.branch_step(
                state,
                event,
                self.catalog.no_op,
                budget,
            )
            return RtlolaSearchResult(
                first_action=self.catalog.no_op,
                first_action_budget=budget,
                first_step=step,
                predicted_cost=0.0,
                predicted_sequence=(self.catalog.no_op.name,),
                evaluated_leaves=1,
                pruned_branches=0,
            )

        features = extract_features(engine, state, budget, future_events)
        predictions = self.policy.predict_regret(features)
        predicted_by_name = dict(zip(self.policy.candidate_names, predictions))
        failures = 0
        for name in self.policy.rank_reducers(features):
            action = self.catalog.by_name[name]
            try:
                step = engine.branch_step(state, event, action, budget)
            except (RuntimeError, ValueError):
                failures += 1
                continue
            return RtlolaSearchResult(
                first_action=action,
                first_action_budget=budget,
                first_step=step,
                predicted_cost=float(predicted_by_name[name]),
                predicted_sequence=(action.name,),
                evaluated_leaves=1,
                pruned_branches=0,
                reducer_failure_count=failures,
                infeasible_candidate_count=failures,
            )

        try:
            step = engine.branch_step(
                state,
                event,
                self.catalog.fallback,
                budget,
            )
        except (RuntimeError, ValueError) as exc:
            raise ValueError("learned RTLola policy and fallback were infeasible") from exc
        return RtlolaSearchResult(
            first_action=self.catalog.fallback,
            first_action_budget=budget,
            first_step=step,
            predicted_cost=float("nan"),
            predicted_sequence=(self.catalog.fallback.name,),
            evaluated_leaves=1,
            pruned_branches=0,
            fallback_used=True,
            reducer_failure_count=failures,
            infeasible_candidate_count=failures,
        )


def train_and_evaluate_regret(
    config: RtlolaBenchmarkConfig,
    *,
    show_progress: bool = False,
    reference_cache_dir: Path | None = None,
) -> RtlolaRegretResult:
    scenario = scenario_by_name(config.scenario)
    catalog = default_action_catalog()
    candidate_names = tuple(catalog.mpc_candidate_names)
    budgets = tuple(config.regret_budgets or [config.budget])
    train_trace_kinds = tuple(
        config.regret_train_trace_kinds or [config.trace_kind]
    )
    eval_trace_kinds = tuple(
        config.regret_eval_trace_kinds or [config.trace_kind]
    )
    if not budgets or any(budget < 0 for budget in budgets):
        raise ValueError("regret budgets must be non-empty and non-negative")
    all_traces: list[RtlolaRegretTrace] = []
    policy: RegretRankingPolicy | None = None
    training_frames: list[pd.DataFrame] = []
    train_seed_count = config.regret_train_seeds or config.seeds
    eval_seed_count = config.regret_eval_seeds or max(1, config.seeds // 3)

    for iteration in range(config.regret_iterations):
        for trace_kind in train_trace_kinds:
            for budget in budgets:
                for seed in range(train_seed_count):
                    trace = scenario.generate_trace(
                        config.length,
                        seed + iteration * 1000,
                        trace_kind,
                    )
                    learned = (
                        RtlolaLearnedPolicy(policy, catalog)
                        if policy is not None else None
                    )
                    all_traces.extend(_collect_episode(
                        scenario=scenario,
                        trace=trace.events,
                        trace_kind=trace.trace_kind,
                        budget=budget,
                        seed=seed,
                        iteration=iteration,
                        config=config,
                        catalog=catalog,
                        learned_policy=learned,
                    ))
        dataset = _build_dataset(all_traces, candidate_names)
        policy, training = train_regret_policy(
            dataset,
            epochs=config.regret_epochs,
            seed=42 + iteration,
            show_progress=show_progress,
            loss=config.regret_loss,  # type: ignore[arg-type]
        )
        training_frames.append(pd.DataFrame([{
            "iteration": iteration,
            **asdict(training),
        }]))

    if policy is None:
        raise ValueError("RTLola regret distillation produced no policy")

    learned_policy = RtlolaLearnedPolicy(policy, catalog)
    eval_results: list[RtlolaRunResult] = []
    for trace_kind in eval_trace_kinds:
        for seed in range(eval_seed_count):
            trace = scenario.generate_trace(config.length, seed, trace_kind)
            cache_path = (
                reference_cache_dir / f"{trace.trace_kind}.seed_{seed}.json"
                if reference_cache_dir is not None else None
            )
            trigger_reference = load_or_compute_trigger_reference(
                trace.events,
                scenario=scenario,
                trace_kind=trace.trace_kind,
                seed=seed,
                cache_path=cache_path,
            )
            for budget in budgets:
                eval_results.append(_evaluate_learned_episode(
                    scenario=scenario,
                    trace=trace.events,
                    trace_kind=trace.trace_kind,
                    trigger_reference=trigger_reference,
                    budget=budget,
                    seed=seed,
                    config=config,
                    learned_policy=learned_policy,
                ))
    return RtlolaRegretResult(
        policy=policy,
        traces=tuple(all_traces),
        eval_results=tuple(eval_results),
        training_frames=tuple(training_frames),
    )


def extract_features(
    engine: RtlolaEngine,
    state: RtlolaStateRef,
    budget: int,
    future_events: Sequence[RtlolaEvent] = (),
) -> np.ndarray:
    metrics = engine.metrics(state)
    base = _features_from_metrics(metrics, budget)
    if not future_events:
        return np.concatenate([base, np.zeros(6, dtype=np.float64)])
    widths: list[float] = []
    counts: list[float] = []
    overflow = 0.0
    rollout_state = state
    _, _, ZonotopeConfig = require_binding()
    for event in future_events:
        _, child = engine.planner.accept_event_from_state(
            rollout_state.state,
            list(event.values),
            event.time,
            ZonotopeConfig.none(),
        )
        rollout_state = RtlolaStateRef(
            child,
            engine.spec_id,
            rollout_state.step + 1,
            event.time,
        )
        future_metrics = engine.metrics(rollout_state)
        widths.append(future_metrics.full_width_sum)
        counts.append(float(future_metrics.dynamic_generator_count))
        overflow += float(future_metrics.dynamic_generator_count > budget)
    future = np.asarray([
        np.mean(widths),
        np.max(widths),
        widths[-1],
        widths[-1] - metrics.full_width_sum,
        overflow,
        np.mean(counts),
    ], dtype=np.float64)
    future[~np.isfinite(future)] = 0.0
    return np.concatenate([base, future])


def write_regret_artifacts(
    result: RtlolaRegretResult,
    output_dir: Path,
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.policy.save(output_dir / "learned_direct_ranker.npz")
    regret_traces_to_dataframe(result.traces).to_csv(
        output_dir / "regret_candidate_costs.csv",
        index=False,
    )
    regret_trace_summary(result.traces).to_csv(
        output_dir / "regret_trace_summary.csv",
        index=False,
    )
    pd.concat(result.training_frames, ignore_index=True).to_csv(
        output_dir / "regret_training.csv",
        index=False,
    )
    regret_ranking_metrics(result.traces).to_csv(
        output_dir / "regret_ranking_metrics.csv",
        index=False,
    )
    summarize_results(result.eval_results).to_csv(
        output_dir / "regret_eval_summary.csv",
        index=False,
    )
    results_to_dataframe(result.eval_results).to_csv(
        output_dir / "regret_eval_timeseries.csv",
        index=False,
    )
    payload = {
        **metadata,
        "policy_behavior": "direct_ranked_action",
        "candidate_names": list(result.policy.candidate_names),
        "feature_names": list(result.policy.feature_names),
        "total_traces": len(result.traces),
    }
    (output_dir / "regret_metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
    )


def regret_traces_to_dataframe(
    traces: Sequence[RtlolaRegretTrace],
) -> pd.DataFrame:
    rows = []
    for decision_id, trace in enumerate(traces):
        for name, cost, regret in zip(
            trace.candidate_names,
            trace.costs,
            trace.regrets,
        ):
            rows.append({
                "decision_id": decision_id,
                "seed": trace.seed,
                "iteration": trace.iteration,
                "step": trace.step,
                "budget": trace.budget,
                "trace_kind": trace.trace_kind,
                "candidate": name,
                "cost": cost,
                "regret": regret,
                "best_action": trace.best_action,
                "best_cost": trace.best_cost,
                "second_best_margin": trace.second_best_margin,
                "tie_count": trace.tie_count,
            })
    return pd.DataFrame(rows)


def regret_trace_summary(
    traces: Sequence[RtlolaRegretTrace],
) -> pd.DataFrame:
    if not traces:
        return pd.DataFrame()
    best = [trace.best_action for trace in traces]
    margins = np.asarray(
        [trace.second_best_margin for trace in traces],
        dtype=np.float64,
    )
    return pd.DataFrame([{
        "decisions": len(traces),
        "mean_second_best_margin": float(np.mean(margins)),
        "median_second_best_margin": float(np.median(margins)),
        **{
            f"best_{name}_count": best.count(name)
            for name in sorted(set(best))
        },
    }])


def regret_ranking_metrics(
    traces: Sequence[RtlolaRegretTrace],
) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "seed": trace.seed,
            "iteration": trace.iteration,
            "step": trace.step,
            "budget": trace.budget,
            "trace_kind": trace.trace_kind,
            "best_action": trace.best_action,
            "second_best_margin": trace.second_best_margin,
            "max_regret": float(np.max(trace.regrets)),
            "mean_regret": float(np.mean(trace.regrets)),
            "tie_count": trace.tie_count,
        }
        for trace in traces
    ])


def _collect_episode(
    *,
    scenario: RtlolaScenario,
    trace: Sequence[RtlolaEvent],
    trace_kind: str,
    budget: int,
    seed: int,
    iteration: int,
    config: RtlolaBenchmarkConfig,
    catalog: RtlolaActionCatalog,
    learned_policy: RtlolaLearnedPolicy | None,
) -> list[RtlolaRegretTrace]:
    engine = _engine_for_scenario(scenario)
    traces: list[RtlolaRegretTrace] = []
    for step, event in enumerate(trace):
        state = engine.snapshot(step=step, time=event.time)
        future = tuple(trace[step + 1:step + 1 + config.horizon])
        if engine.metrics(state).dynamic_generator_count <= budget:
            action = catalog.no_op
        else:
            rows = _evaluate_candidates(
                engine,
                state,
                event,
                future,
                scenario,
                catalog,
                config,
                budget=budget,
            )
            features = extract_features(engine, state, budget, future)
            traces.append(_trace_from_rows(
                rows=rows,
                features=features,
                seed=seed,
                step=step,
                iteration=iteration,
                budget=budget,
                trace_kind=trace_kind,
            ))
            if learned_policy is None:
                best = min(rows, key=lambda row: (row.cost, row.sequence))
                action = catalog.by_name[best.name]
            else:
                action = learned_policy.choose(
                    engine,
                    state,
                    event,
                    future,
                    budget,
                ).first_action
        engine.live_step(event, action, budget, step=step + 1)
    return traces


def _evaluate_learned_episode(
    *,
    scenario: RtlolaScenario,
    trace: Sequence[RtlolaEvent],
    trace_kind: str,
    trigger_reference: Sequence[RtlolaTriggerReferenceStep],
    budget: int,
    seed: int,
    config: RtlolaBenchmarkConfig,
    learned_policy: RtlolaLearnedPolicy,
) -> RtlolaRunResult:
    engine = _engine_for_scenario(scenario)
    records = []
    for step, event in enumerate(trace):
        state = engine.snapshot(step=step, time=event.time)
        pre_count = engine.metrics(state).dynamic_generator_count
        future = tuple(trace[step + 1:step + 1 + config.horizon])
        start = time.perf_counter()
        decision = learned_policy.choose(
            engine,
            state,
            event,
            future,
            budget,
        )
        committed = engine.live_step(
            event,
            decision.first_action,
            decision.first_action_budget,
            step=step + 1,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        records.append(make_step_record(
            engine=engine,
            scenario=scenario,
            seed=seed,
            method="learned_direct",
            step=step,
            budget=budget,
            pre_generator_count=pre_count,
            committed=committed,
            decision=decision,
            decision_time_ms=elapsed_ms,
            ground_truth=None,
            trigger_reference=trigger_reference[step],
        ))
    return RtlolaRunResult(
        "learned_direct",
        seed,
        tuple(records),
        budget=budget,
        trace_kind=trace_kind,
    )


def _evaluate_candidates(
    engine: RtlolaEngine,
    state: RtlolaStateRef,
    event: RtlolaEvent,
    future: Sequence[RtlolaEvent],
    scenario: RtlolaScenario,
    catalog: RtlolaActionCatalog,
    config: RtlolaBenchmarkConfig,
    *,
    budget: int | None = None,
) -> list[RtlolaCandidateCost]:
    """Score forced roots, then continue with the full MPC candidate pool."""
    selected_budget = config.budget if budget is None else budget
    rows: list[RtlolaCandidateCost] = []
    for first in catalog.mpc_candidates:
        try:
            search = beam_search(
                engine,
                state,
                event,
                future,
                catalog.mpc_candidates,
                selected_budget,
                config.beam_width,
                fallback=catalog.fallback,
                none_action=catalog.no_op,
                cost_fn=normalized_trigger_width_cost(scenario.trigger_values),
                forced_first_action=first,
            )
        except ValueError:
            continue
        if search.first_action.name != first.name:
            continue
        rows.append(RtlolaCandidateCost(
            first.name,
            search.predicted_cost,
            search.predicted_sequence,
        ))
    if not rows:
        raise ValueError("RTLola regret oracle found no feasible candidate")
    rows.sort(key=lambda row: (row.cost, row.sequence))
    return rows


def _trace_from_rows(
    *,
    rows: Sequence[RtlolaCandidateCost],
    features: np.ndarray,
    seed: int,
    step: int,
    iteration: int,
    budget: int,
    trace_kind: str,
) -> RtlolaRegretTrace:
    ordered = sorted(rows, key=lambda row: row.name)
    best = min(rows, key=lambda row: (row.cost, row.sequence))
    costs = np.asarray([row.cost for row in ordered], dtype=np.float64)
    scale = max(abs(best.cost), 1.0)
    regrets = np.maximum((costs - best.cost) / scale, 0.0)
    sorted_costs = sorted(float(row.cost) for row in rows)
    margin = sorted_costs[1] - sorted_costs[0] if len(sorted_costs) > 1 else 0.0
    tie_tol = max(1e-9, abs(best.cost) * 1e-9)
    return RtlolaRegretTrace(
        features=features,
        candidate_names=tuple(row.name for row in ordered),
        costs=tuple(float(value) for value in costs),
        regrets=tuple(float(value) for value in regrets),
        best_action=best.name,
        best_cost=float(best.cost),
        second_best_margin=float(margin),
        tie_count=sum(
            abs(row.cost - best.cost) <= tie_tol
            for row in rows
        ),
        step=step,
        seed=seed,
        iteration=iteration,
        budget=budget,
        trace_kind=trace_kind,
    )


def _build_dataset(
    traces: Sequence[RtlolaRegretTrace],
    candidate_names: tuple[str, ...],
) -> RegretDataset:
    if not traces:
        raise ValueError(
            "no RTLola reduction decisions were observed; lower the budget "
            "or increase the training trace length"
        )
    features = np.stack([trace.features for trace in traces])
    regrets = []
    for trace in traces:
        by_name = dict(zip(trace.candidate_names, trace.regrets))
        regrets.append([
            float(by_name.get(name, 1.0))
            for name in candidate_names
        ])
    return RegretDataset(
        features=features,
        regrets=np.asarray(regrets, dtype=np.float64),
        candidate_names=candidate_names,
        feature_names=RTL_FEATURE_NAMES,
    )


def _features_from_metrics(
    metrics: RtlolaMatrixMetrics,
    budget: int,
) -> np.ndarray:
    values = np.asarray([
        metrics.dynamic_generator_count,
        metrics.total_generator_count,
        metrics.dimension,
        budget - metrics.dynamic_generator_count,
        metrics.full_width_sum,
        metrics.width_mean,
        metrics.width_max,
        metrics.center_l2,
        metrics.center_linf,
        metrics.gen_norm_mean,
        metrics.gen_norm_max,
        metrics.gen_norm_std,
        metrics.gen_sparsity,
        metrics.gen_coupling,
        metrics.gen_pca_explained,
    ], dtype=np.float64)
    values[~np.isfinite(values)] = 0.0
    return values


def _engine_for_scenario(scenario: RtlolaScenario) -> RtlolaEngine:
    return RtlolaEngine(
        scenario.spec,
        event_arity=scenario.event_arity,
        expected_verdict_keys=(
            *scenario.expected_verdict_keys,
            *scenario.public_stream_keys,
        ),
    )
