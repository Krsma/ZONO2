"""Regret/ranking distillation for RTLola actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from pzr.imitation.regret import RegretDataset, RegretRankingPolicy, train_regret_policy
from pzr.rtlola.actions import RtlolaAction, action_by_name, default_actions
from pzr.rtlola.binding import require_binding
from pzr.rtlola.engine import RtlolaEngine, RtlolaEvent, RtlolaStateRef
from pzr.rtlola.metrics import RtlolaMatrixMetrics
from pzr.rtlola.omni import OMNI_EXPECTED_VERDICT_KEYS, OMNI_SPEC, generate_omni_events
from pzr.rtlola.runner import (
    RtlolaBenchmarkConfig,
    RtlolaRunResult,
    RtlolaStepRecord,
    _public_bounds,
    _selected_interval_error,
    _false_positive,
    _false_negative,
    _trigger_positive,
    _state_interval_bounds,
    compute_ground_truth,
    results_to_dataframe,
    summarize_results,
)
from pzr.rtlola.search import beam_search
from pzr.utils.serialization import save_json

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


@dataclass(frozen=True)
class RtlolaRegretResult:
    policy: RegretRankingPolicy
    traces: tuple[RtlolaRegretTrace, ...]
    eval_results: tuple[RtlolaRunResult, ...]
    training_frames: tuple[pd.DataFrame, ...]


class RtlolaLearnedPolicy:
    """Rank RTLola actions with a trained regret policy."""

    def __init__(
        self,
        policy: RegretRankingPolicy,
        actions: tuple[RtlolaAction, ...],
        fallback: RtlolaAction,
    ) -> None:
        self.policy = policy
        self.actions = action_by_name(actions)
        self.fallback = fallback

    def choose(
        self,
        engine: RtlolaEngine,
        state: RtlolaStateRef,
        event: RtlolaEvent,
        future_events: Sequence[RtlolaEvent],
        budget: int,
        beam_width: int,
    ):
        # This is intentionally a ranked-beam policy for now: the learned model
        # ranks actions, then RTLola beam search evaluates the ranked set.
        # A top-1 distilled classifier policy is a separate follow-up.
        features = extract_features(engine, state, budget, future_events)
        ranked = self.policy.rank_reducers(features)
        candidate_actions = tuple(
            self.actions[name] for name in ranked if name in self.actions
        )
        if not candidate_actions:
            candidate_actions = (self.fallback,)
        none_action = self.actions.get("none")
        try:
            return beam_search(
                engine,
                state,
                event,
                future_events,
                candidate_actions,
                budget,
                beam_width,
                fallback=self.fallback,
                none_action=none_action,
            )
        except ValueError:
            return beam_search(
                engine,
                state,
                event,
                future_events,
                (self.fallback,),
                budget,
                beam_width,
                fallback=self.fallback,
                none_action=none_action,
            )


def train_and_evaluate_regret(
    config: RtlolaBenchmarkConfig,
    *,
    show_progress: bool = False,
) -> RtlolaRegretResult:
    if config.scenario != "omni_robot":
        raise ValueError("RTLola regret distillation is currently implemented for omni_robot only")
    actions = default_actions()
    by_name = action_by_name(actions)
    fallback = by_name["interval"]
    candidate_names = tuple(action.name for action in actions)
    all_traces: list[RtlolaRegretTrace] = []
    policy: RegretRankingPolicy | None = None
    training_frames: list[pd.DataFrame] = []
    train_seeds = range(config.regret_train_seeds or config.seeds)
    eval_seeds = range(config.regret_eval_seeds or max(1, config.seeds // 3))

    for iteration in range(config.regret_iterations):
        for seed in train_seeds:
            trace = generate_omni_events(config.length, seed=seed + iteration * 1000)
            learned = (
                RtlolaLearnedPolicy(policy, actions, fallback)
                if policy is not None else None
            )
            all_traces.extend(_collect_episode(
                trace=trace,
                seed=seed,
                iteration=iteration,
                config=config,
                actions=actions,
                fallback=fallback,
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
        training_frames.append(pd.DataFrame([{"iteration": iteration, **asdict(training)}]))

    if policy is None:
        raise ValueError("RTLola regret distillation produced no policy")

    learned_policy = RtlolaLearnedPolicy(policy, actions, fallback)
    eval_results = tuple(
        _evaluate_learned_episode(
            trace=generate_omni_events(config.length, seed=seed),
            seed=seed,
            config=config,
            learned_policy=learned_policy,
        )
        for seed in eval_seeds
    )
    return RtlolaRegretResult(
        policy=policy,
        traces=tuple(all_traces),
        eval_results=eval_results,
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
    # Exact future preview for features only. It may overflow; that is a signal.
    for event in future_events:
        verdict, child = engine.planner.accept_event_from_state(
            rollout_state.state,
            list(event.values),
            event.time,
            ZonotopeConfig.none(),
        )
        _ = verdict
        rollout_state = RtlolaStateRef(child, engine.spec_id, rollout_state.step + 1, event.time)
        m = engine.metrics(rollout_state)
        widths.append(m.full_width_sum)
        counts.append(float(m.dynamic_generator_count))
        if m.dynamic_generator_count > budget:
            overflow += 1.0
    future = np.array([
        float(np.mean(widths)),
        float(np.max(widths)),
        float(widths[-1]),
        float(widths[-1] - metrics.full_width_sum),
        overflow,
        float(np.mean(counts)),
    ], dtype=np.float64)
    future[~np.isfinite(future)] = 0.0
    return np.concatenate([base, future])


def write_regret_artifacts(
    result: RtlolaRegretResult,
    output_dir: Path,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.policy.save(output_dir / "learned_ranked_beam.npz")
    regret_traces_to_dataframe(result.traces).to_csv(
        output_dir / "regret_candidate_costs.csv", index=False,
    )
    regret_trace_summary(result.traces).to_csv(
        output_dir / "regret_trace_summary.csv", index=False,
    )
    pd.concat(result.training_frames, ignore_index=True).to_csv(
        output_dir / "regret_training.csv", index=False,
    )
    regret_ranking_metrics(result.traces).to_csv(
        output_dir / "regret_ranking_metrics.csv", index=False,
    )
    summarize_results(result.eval_results).to_csv(
        output_dir / "regret_eval_summary.csv", index=False,
    )
    results_to_dataframe(result.eval_results).to_csv(
        output_dir / "regret_eval_timeseries.csv", index=False,
    )
    save_json(
        {
            **metadata,
            "policy_behavior": "ranked_beam",
            "total_traces": len(result.traces),
        },
        output_dir / "regret_metadata.json",
    )


def regret_traces_to_dataframe(traces: Sequence[RtlolaRegretTrace]) -> pd.DataFrame:
    rows = []
    for idx, trace in enumerate(traces):
        for name, cost, regret in zip(trace.candidate_names, trace.costs, trace.regrets):
            rows.append({
                "decision_id": idx,
                "seed": trace.seed,
                "iteration": trace.iteration,
                "step": trace.step,
                "candidate": name,
                "cost": cost,
                "regret": regret,
                "best_action": trace.best_action,
                "best_cost": trace.best_cost,
                "second_best_margin": trace.second_best_margin,
                "tie_count": trace.tie_count,
            })
    return pd.DataFrame(rows)


def regret_trace_summary(traces: Sequence[RtlolaRegretTrace]) -> pd.DataFrame:
    if not traces:
        return pd.DataFrame()
    best = [trace.best_action for trace in traces]
    margins = np.asarray([trace.second_best_margin for trace in traces], dtype=np.float64)
    return pd.DataFrame([{
        "decisions": len(traces),
        "mean_second_best_margin": float(np.mean(margins)),
        "median_second_best_margin": float(np.median(margins)),
        **{f"best_{name}_count": best.count(name) for name in sorted(set(best))},
    }])


def regret_ranking_metrics(traces: Sequence[RtlolaRegretTrace]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "seed": trace.seed,
            "iteration": trace.iteration,
            "step": trace.step,
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
    trace: Sequence[RtlolaEvent],
    seed: int,
    iteration: int,
    config: RtlolaBenchmarkConfig,
    actions: tuple[RtlolaAction, ...],
    fallback: RtlolaAction,
    learned_policy: RtlolaLearnedPolicy | None,
) -> list[RtlolaRegretTrace]:
    engine = RtlolaEngine(OMNI_SPEC, event_arity=3, expected_verdict_keys=OMNI_EXPECTED_VERDICT_KEYS)
    traces: list[RtlolaRegretTrace] = []
    for step, event in enumerate(trace):
        state = engine.snapshot(step=step, time=event.time)
        future = tuple(trace[step + 1:step + 1 + config.horizon])
        rows = _evaluate_candidates(engine, state, event, future, actions, config, fallback)
        features = extract_features(engine, state, config.budget, future)
        traces.append(_trace_from_rows(
            rows=rows,
            features=features,
            seed=seed,
            step=step,
            iteration=iteration,
        ))
        if learned_policy is None:
            chosen = min(rows, key=lambda row: (row.cost, row.sequence)).name
            action = action_by_name(actions)[chosen]
        else:
            action = learned_policy.choose(
                engine, state, event, future, config.budget, config.beam_width,
            ).first_action
        engine.live_step(event, action, config.budget, step=step + 1)
    return traces


def _evaluate_learned_episode(
    *,
    trace: Sequence[RtlolaEvent],
    seed: int,
    config: RtlolaBenchmarkConfig,
    learned_policy: RtlolaLearnedPolicy,
) -> RtlolaRunResult:
    engine = RtlolaEngine(OMNI_SPEC, event_arity=3, expected_verdict_keys=OMNI_EXPECTED_VERDICT_KEYS)
    ground_truth = compute_ground_truth(trace)
    records: list[RtlolaStepRecord] = []
    for step, event in enumerate(trace):
        state = engine.snapshot(step=step, time=event.time)
        pre_metrics = engine.metrics(state)
        future = tuple(trace[step + 1:step + 1 + config.horizon])
        decision = learned_policy.choose(
            engine, state, event, future, config.budget, config.beam_width,
        )
        committed = engine.live_step(event, decision.first_action, config.budget, step=step + 1)
        gt = ground_truth[step]
        lower, upper = _state_interval_bounds(engine.matrices(committed.state)[0])
        if lower.shape != gt.lower.shape:
            raise RuntimeError(
                "RTLola learned reduced and exact state-zonotope dimensions differ "
                f"(seed={seed}, step={step}, "
                f"reduced_dim={lower.shape[0]}, exact_dim={gt.lower.shape[0]})"
            )
        approx_error = float(np.sum(np.abs(lower - gt.lower) + np.abs(upper - gt.upper)))
        records.append(RtlolaStepRecord(
            seed=seed,
            method="learned_ranked_beam",
            step=step,
            pre_generator_count=pre_metrics.dynamic_generator_count,
            generator_count=committed.metrics.dynamic_generator_count,
            total_generator_count=committed.metrics.total_generator_count,
            active_dynamic_generator_count=committed.metrics.active_dynamic_generator_count,
            active_total_generator_count=committed.metrics.active_total_generator_count,
            zero_dynamic_generator_count=committed.metrics.zero_dynamic_generator_count,
            zero_total_generator_count=committed.metrics.zero_total_generator_count,
            reduced=decision.first_action.name != "none",
            reducer_used=decision.first_action.name,
            state_zonotope_width_sum=committed.metrics.full_width_sum,
            exact_state_zonotope_width_sum=gt.width_sum,
            state_zonotope_approx_error_sum=approx_error,
            relevant_state_width_sum=committed.metrics.full_width_sum,
            exact_relevant_state_width_sum=gt.width_sum,
            relevant_state_approx_error_sum=_selected_interval_error(
                lower, upper, gt.lower, gt.upper, (),
            ),
            approx_loss=approx_error,
            false_positive=_false_positive(
                committed.verdict, gt.verdicts, OMNI_EXPECTED_VERDICT_KEYS,
            ),
            false_negative=_false_negative(
                committed.verdict, gt.verdicts, OMNI_EXPECTED_VERDICT_KEYS,
            ),
            trigger_positive=_trigger_positive(committed.verdict, OMNI_EXPECTED_VERDICT_KEYS),
            verdicts=committed.verdict,
            public_bounds=_public_bounds(committed.verdict, OMNI_EXPECTED_VERDICT_KEYS),
            reduction_time_ms=0.0,
            predicted_cost=decision.predicted_cost,
            predicted_sequence=decision.predicted_sequence,
            evaluated_leaves=decision.evaluated_leaves,
            pruned_branches=decision.pruned_branches,
            post_event_over_bound=committed.metrics.dynamic_generator_count > config.budget,
            budget_violation=False,
            fallback_used=decision.fallback_used,
            reducer_failure_count=decision.reducer_failure_count,
            infeasible_candidate_count=decision.infeasible_candidate_count,
        ))
    return RtlolaRunResult("learned_ranked_beam", seed, tuple(records))


def _evaluate_candidates(
    engine: RtlolaEngine,
    state: RtlolaStateRef,
    event: RtlolaEvent,
    future: Sequence[RtlolaEvent],
    actions: tuple[RtlolaAction, ...],
    config: RtlolaBenchmarkConfig,
    fallback: RtlolaAction,
) -> list[RtlolaCandidateCost]:
    rows: list[RtlolaCandidateCost] = []
    none_action = action_by_name(actions).get("none")
    for first in actions:
        try:
            search = beam_search(
                engine,
                state,
                event,
                future,
                (first,),
                config.budget,
                config.beam_width,
                fallback=fallback,
                none_action=none_action,
            )
        except ValueError:
            continue
        if search.first_action.name != first.name:
            continue
        rows.append(RtlolaCandidateCost(first.name, search.predicted_cost, search.predicted_sequence))
    if not rows:
        raise ValueError("RTLola regret oracle found no candidate that ran with the bound")
    rows.sort(key=lambda row: (row.cost, row.sequence))
    return rows


def _trace_from_rows(
    *,
    rows: Sequence[RtlolaCandidateCost],
    features: np.ndarray,
    seed: int,
    step: int,
    iteration: int,
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
        costs=tuple(float(v) for v in costs),
        regrets=tuple(float(v) for v in regrets),
        best_action=best.name,
        best_cost=float(best.cost),
        second_best_margin=float(margin),
        tie_count=sum(1 for row in rows if abs(row.cost - best.cost) <= tie_tol),
        step=step,
        seed=seed,
        iteration=iteration,
    )


def _build_dataset(
    traces: Sequence[RtlolaRegretTrace],
    candidate_names: tuple[str, ...],
) -> RegretDataset:
    if not traces:
        raise ValueError("no RTLola regret traces collected")
    features = np.stack([trace.features for trace in traces])
    regrets = []
    for trace in traces:
        by_name = dict(zip(trace.candidate_names, trace.regrets))
        regrets.append([float(by_name.get(name, 1.0)) for name in candidate_names])
    return RegretDataset(
        features=features,
        regrets=np.asarray(regrets, dtype=np.float64),
        candidate_names=candidate_names,
        feature_names=RTL_FEATURE_NAMES,
    )


def _features_from_metrics(metrics: RtlolaMatrixMetrics, budget: int) -> np.ndarray:
    values = np.array([
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
