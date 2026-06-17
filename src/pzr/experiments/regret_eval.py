"""Regret-ranking distillation for MPC reducer selection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pzr.experiments.runner import (
    MPCReductionPolicy,
    ReductionPolicy,
    RunResult,
    compute_ground_truth,
    results_to_dataframe,
    run_single,
    summarize_results,
)
from pzr.imitation.features import FEATURE_NAMES, extract_features
from pzr.imitation.regret import (
    RegretDataset,
    RegretRankingPolicy,
    RegretTrainingResult,
    train_regret_policy,
)
from pzr.monitoring.base import MonitorAdapter, MonitorState
from pzr.mpc.objectives import CostWeights, WeightedZonotopeCost
from pzr.mpc.policies import ReductionDecision
from pzr.mpc.prediction import ConstantPredictor, InputPredictor
from pzr.mpc.search import try_certified_reduce
from pzr.utils.serialization import save_json
from pzr.zonotope.protected import ProtectedReducer
from pzr.zonotope.reduction import ALL_REDUCERS, BoxReducer, Reducer, ReductionResult

RegretOracleMode = Literal[
    "beam3",
    "sequence3",
    "pair_rollout3",
    "rollout_wide",
    "sequence_wide",
]

TOP3_CANDIDATES = ("girard", "methA", "scott")
BROAD_CANDIDATES = ("girard", "combastel", "pca", "methA", "scott")
REGRET_ORACLE_MODES = (
    "beam3",
    "sequence3",
    "pair_rollout3",
    "rollout_wide",
    "sequence_wide",
)
REGRET_ROLLOUT_FEATURE_NAMES = (
    "future_trigger_width_mean",
    "future_trigger_width_max",
    "future_trigger_width_final",
    "future_trigger_width_growth",
    "future_straddle_mean",
    "future_straddle_max",
    "future_overflow_count",
    "future_generator_count_mean",
)
REGRET_FEATURE_NAMES = (*FEATURE_NAMES, *REGRET_ROLLOUT_FEATURE_NAMES)


@dataclass(frozen=True)
class CandidateCost:
    """Cost and first-action reduction result for one oracle candidate."""

    name: str
    cost: float
    state: MonitorState
    result: ReductionResult
    sequence: tuple[str, ...]


@dataclass(frozen=True)
class RegretTrace:
    """One reduction-state supervision example with all candidate regrets."""

    features: np.ndarray
    candidate_names: tuple[str, ...]
    costs: tuple[float, ...]
    regrets: tuple[float, ...]
    best_action: str
    best_cost: float
    second_best_margin: float
    tie_count: int
    oracle_mode: str
    step: int
    seed: int
    iteration: int


class RegretTraceCollector:
    """Collect regret traces during oracle and learned rollouts."""

    def __init__(self) -> None:
        self._traces: list[RegretTrace] = []

    def record(self, trace: RegretTrace) -> None:
        self._traces.append(trace)

    @property
    def traces(self) -> list[RegretTrace]:
        return list(self._traces)

    def __len__(self) -> int:
        return len(self._traces)


@dataclass(frozen=True)
class RegretOracleConfig:
    """Configuration for the MPC teacher used by regret distillation."""

    mode: RegretOracleMode = "beam3"
    horizon: int = 4
    beam_width: int = 4
    cost_weights: CostWeights = field(default_factory=CostWeights)
    predictor: InputPredictor | None = None

    @property
    def candidate_names(self) -> tuple[str, ...]:
        if self.mode in {"rollout_wide", "sequence_wide"}:
            return BROAD_CANDIDATES
        return TOP3_CANDIDATES


@dataclass
class RegretEvalResult:
    """Full regret-distillation result and diagnostics."""

    policy: RegretRankingPolicy
    training_results: list[RegretTrainingResult]
    traces: list[RegretTrace]
    eval_results: list[RunResult]
    inference_time_ms: float
    oracle_mode: str

    @property
    def total_traces(self) -> int:
        return len(self.traces)


@dataclass
class RegretReductionPolicy:
    """Wraps a learned regret ranker as a benchmark reduction policy."""

    learned: RegretRankingPolicy
    candidates: dict[str, Reducer | ProtectedReducer]
    oracle_config: RegretOracleConfig | None = None
    _name: str = "learned_regret"

    @property
    def name(self) -> str:
        return self._name

    def decide(
        self,
        monitor: MonitorAdapter,
        state: MonitorState,
        history: Sequence,
        budget: int,
    ) -> ReductionDecision:
        features = _extract_regret_features(
            monitor, state, history, budget, self.oracle_config,
        )
        cal = state.calibration_indices
        selected = self.learned.select_reducer(
            features, self.candidates, state.zonotope, budget,
            protected_indices=cal,
        )
        if selected is None:
            fallback = ProtectedReducer(base=BoxReducer())
            red = fallback.reduce(state.zonotope, budget, protected_indices=cal)
            return ReductionDecision(
                state=state.with_zonotope(
                    red.reduced,
                    calibration_indices=tuple(range(len(cal))) if cal else (),
                ),
                result=red,
                reducer_name="box_fallback",
            )
        name, result = selected
        return ReductionDecision(
            state=state.with_zonotope(
                result.reduced,
                calibration_indices=tuple(range(len(cal))) if cal else (),
            ),
            result=result,
            reducer_name=name,
        )


def evaluate_regret_candidates(
    monitor: MonitorAdapter,
    state: MonitorState,
    history: Sequence,
    budget: int,
    config: RegretOracleConfig,
) -> list[CandidateCost]:
    """Evaluate oracle cost for every first-action candidate."""
    cost_fn = WeightedZonotopeCost(
        weights=config.cost_weights,
        triggers=monitor.triggers,
        trigger_zonotope=monitor.trigger_zonotope,
    )
    predictor = config.predictor or ConstantPredictor()
    predicted_inputs = tuple(predictor.predict(history, config.horizon))[:config.horizon]
    candidates = _candidate_reducers(config.candidate_names)
    fallback = ProtectedReducer(base=BoxReducer())
    rows: list[CandidateCost] = []
    for reducer in candidates:
        first = try_certified_reduce(monitor, state, reducer, budget)
        if first is None:
            continue
        first_state, first_result = first
        if config.mode == "beam3":
            future = _beam_future_cost(
                monitor, first_state, predicted_inputs, candidates, budget,
                cost_fn, config.beam_width, fallback,
            )
        elif config.mode in {"sequence3", "sequence_wide"}:
            future = _sequence_future_cost(
                monitor, first_state, predicted_inputs, candidates, budget,
                cost_fn, fallback,
            )
        elif config.mode == "pair_rollout3":
            future = _best_fixed_base_future_cost(
                monitor, first_state, predicted_inputs, candidates, budget,
                cost_fn, fallback,
            )
        elif config.mode == "rollout_wide":
            future = _fixed_base_future_cost(
                monitor, first_state, predicted_inputs,
                ProtectedReducer(base=ALL_REDUCERS["girard"]), budget,
                cost_fn, fallback,
            )
        else:
            raise ValueError(f"unknown regret oracle mode: {config.mode}")
        if future is None:
            continue
        future_cost, future_sequence = future
        rows.append(CandidateCost(
            name=reducer.name,
            cost=future_cost,
            state=first_state,
            result=first_result,
            sequence=(reducer.name, *future_sequence),
        ))
    if not rows:
        raise ValueError("regret oracle found no certified candidate")
    rows.sort(key=lambda row: (row.cost, row.sequence))
    return rows


def train_and_evaluate_regret(
    monitor: MonitorAdapter,
    trace_fn: Callable[[int, int], Sequence],
    budget: int,
    train_seeds: range,
    eval_seeds: range,
    length: int,
    oracle_config: RegretOracleConfig,
    iterations: int = 3,
    epochs_per_iteration: int = 100,
    hidden_sizes: tuple[int, ...] = (64, 64),
    seed: int = 42,
    show_progress: bool = True,
    regret_loss: str = "pairwise",
) -> RegretEvalResult:
    """Train a regret-ranking policy and evaluate it on held-out seeds."""
    all_collectors: list[RegretTraceCollector] = []
    policy: RegretRankingPolicy | None = None
    training_results: list[RegretTrainingResult] = []

    iter_bar = tqdm(
        range(iterations), desc="regret iters",
        disable=not show_progress, unit="iter", leave=True,
    )
    for iteration in iter_bar:
        collector = RegretTraceCollector()
        learned_policy = None
        if policy is not None:
            learned_policy = RegretReductionPolicy(
                policy, _candidate_reducer_map(oracle_config.candidate_names),
                oracle_config=oracle_config,
                _name="regret_learner",
            )
        seed_bar = tqdm(
            list(train_seeds), desc=f"iter {iteration} · collect",
            disable=not show_progress, unit="seed", leave=False,
        )
        for ep_seed in seed_bar:
            trace = trace_fn(length, ep_seed + iteration * 1000)
            _collect_regret_episode(
                monitor=monitor,
                trace=trace,
                budget=budget,
                seed=ep_seed,
                iteration=iteration,
                oracle_config=oracle_config,
                collector=collector,
                learned_policy=learned_policy,
            )
        all_collectors.append(collector)

        traces = [trace for c in all_collectors for trace in c.traces]
        if not traces:
            continue
        dataset = build_regret_dataset(traces, oracle_config.candidate_names)
        policy, result = train_regret_policy(
            dataset,
            hidden_sizes=hidden_sizes,
            epochs=epochs_per_iteration,
            seed=seed + iteration,
            show_progress=show_progress,
            loss=regret_loss,  # type: ignore[arg-type]
        )
        training_results.append(result)

    if policy is None:
        raise ValueError("regret distillation produced no policy")

    learned = RegretReductionPolicy(
        policy,
        _candidate_reducer_map(oracle_config.candidate_names),
        oracle_config=oracle_config,
        _name=f"learned_regret_{oracle_config.mode}",
    )
    eval_results: list[RunResult] = []
    inference_times: list[float] = []
    eval_bar = tqdm(
        list(eval_seeds), desc="regret eval",
        disable=not show_progress, unit="seed", leave=False,
    )
    for ep_seed in eval_bar:
        trace = trace_fn(length, ep_seed)
        gt = compute_ground_truth(monitor, trace)
        run = run_single(monitor, trace, learned, budget, ep_seed, ground_truth=gt)
        eval_results.append(run)
        if run.total_reductions > 0:
            inference_times.append(run.total_time_ms / run.total_reductions)

    traces = [trace for c in all_collectors for trace in c.traces]
    return RegretEvalResult(
        policy=policy,
        training_results=training_results,
        traces=traces,
        eval_results=eval_results,
        inference_time_ms=float(np.mean(inference_times)) if inference_times else 0.0,
        oracle_mode=oracle_config.mode,
    )


def train_and_evaluate_regret_on_traces(
    monitor: MonitorAdapter,
    train_traces: Sequence[tuple[int, Sequence]],
    eval_traces: Sequence[tuple[int, Sequence]],
    budget: int,
    oracle_config: RegretOracleConfig,
    iterations: int = 3,
    epochs_per_iteration: int = 100,
    hidden_sizes: tuple[int, ...] = (64, 64),
    seed: int = 42,
    show_progress: bool = True,
    regret_loss: str = "pairwise",
) -> RegretEvalResult:
    """Train and evaluate regret ranking from pre-collected traces."""
    all_collectors: list[RegretTraceCollector] = []
    policy: RegretRankingPolicy | None = None
    training_results: list[RegretTrainingResult] = []

    iter_bar = tqdm(
        range(iterations), desc="regret iters",
        disable=not show_progress, unit="iter", leave=True,
    )
    for iteration in iter_bar:
        collector = RegretTraceCollector()
        learned_policy = None
        if policy is not None:
            learned_policy = RegretReductionPolicy(
                policy, _candidate_reducer_map(oracle_config.candidate_names),
                oracle_config=oracle_config,
                _name="regret_learner",
            )
        trace_bar = tqdm(
            list(train_traces), desc=f"iter {iteration} · collect",
            disable=not show_progress, unit="trace", leave=False,
        )
        for ep_seed, trace in trace_bar:
            _collect_regret_episode(
                monitor=monitor,
                trace=trace,
                budget=budget,
                seed=ep_seed,
                iteration=iteration,
                oracle_config=oracle_config,
                collector=collector,
                learned_policy=learned_policy,
            )
        all_collectors.append(collector)

        traces = [trace for c in all_collectors for trace in c.traces]
        if not traces:
            continue
        dataset = build_regret_dataset(traces, oracle_config.candidate_names)
        policy, result = train_regret_policy(
            dataset,
            hidden_sizes=hidden_sizes,
            epochs=epochs_per_iteration,
            seed=seed + iteration,
            show_progress=show_progress,
            loss=regret_loss,  # type: ignore[arg-type]
        )
        training_results.append(result)

    if policy is None:
        raise ValueError("regret distillation produced no policy")

    learned = RegretReductionPolicy(
        policy,
        _candidate_reducer_map(oracle_config.candidate_names),
        oracle_config=oracle_config,
        _name=f"learned_regret_{oracle_config.mode}",
    )
    eval_results: list[RunResult] = []
    inference_times: list[float] = []
    eval_bar = tqdm(
        list(eval_traces), desc="regret eval",
        disable=not show_progress, unit="trace", leave=False,
    )
    for ep_seed, trace in eval_bar:
        gt = compute_ground_truth(monitor, trace)
        run = run_single(monitor, trace, learned, budget, ep_seed, ground_truth=gt)
        eval_results.append(run)
        if run.total_reductions > 0:
            inference_times.append(run.total_time_ms / run.total_reductions)

    traces = [trace for c in all_collectors for trace in c.traces]
    return RegretEvalResult(
        policy=policy,
        training_results=training_results,
        traces=traces,
        eval_results=eval_results,
        inference_time_ms=float(np.mean(inference_times)) if inference_times else 0.0,
        oracle_mode=oracle_config.mode,
    )


def build_regret_dataset(
    traces: Sequence[RegretTrace],
    candidate_names: tuple[str, ...],
) -> RegretDataset:
    """Build a fixed-candidate regret dataset from traces."""
    if not traces:
        raise ValueError("no regret traces to build dataset from")
    features = np.stack([t.features for t in traces])
    regrets = []
    for trace in traces:
        by_name = dict(zip(trace.candidate_names, trace.regrets))
        regrets.append([float(by_name[name]) for name in candidate_names])
    return RegretDataset(
        features=features,
        regrets=np.asarray(regrets, dtype=np.float64),
        candidate_names=candidate_names,
        feature_names=REGRET_FEATURE_NAMES,
    )


def _extract_regret_features(
    monitor: MonitorAdapter,
    state: MonitorState,
    history: Sequence,
    budget: int,
    config: RegretOracleConfig | None,
) -> np.ndarray:
    base = extract_features(
        state, budget, monitor.triggers,
        trigger_zonotope=monitor.trigger_zonotope,
    )
    if config is None:
        return np.concatenate([base, np.zeros(len(REGRET_ROLLOUT_FEATURE_NAMES))])
    predictor = config.predictor or ConstantPredictor()
    predicted_inputs = tuple(predictor.predict(history, config.horizon))[:config.horizon]
    if not predicted_inputs:
        return np.concatenate([base, np.zeros(len(REGRET_ROLLOUT_FEATURE_NAMES))])

    initial_trigger_width = float(np.sum(monitor.trigger_zonotope(state).widths()))
    sim_state = state
    widths: list[float] = []
    straddles: list[float] = []
    generator_counts: list[float] = []
    overflow_count = 0.0
    cost_fn = WeightedZonotopeCost(
        weights=config.cost_weights,
        triggers=monitor.triggers,
        trigger_zonotope=monitor.trigger_zonotope,
    )
    for measurement in predicted_inputs:
        step_result = monitor.step(sim_state, measurement)
        sim_state = step_result.state
        trigger_width = float(np.sum(monitor.trigger_zonotope(sim_state).widths()))
        widths.append(trigger_width)
        generator_counts.append(float(sim_state.zonotope.generator_count))
        if sim_state.zonotope.generator_count > budget:
            overflow_count += 1.0
        # Difference between cost with and without straddling terms isolates
        # the future threshold ambiguity signal without changing the MPC cost.
        width_only = WeightedZonotopeCost(
            weights=CostWeights(
                trigger_width=config.cost_weights.trigger_width,
                straddling=0.0,
                generator_count=config.cost_weights.generator_count,
                total_width=config.cost_weights.total_width,
            ),
            triggers=monitor.triggers,
            trigger_zonotope=monitor.trigger_zonotope,
        )
        straddles.append(max(
            0.0,
            cost_fn(sim_state, step_result.verdicts)
            - width_only(sim_state, step_result.verdicts),
        ))

    future = np.array([
        float(np.mean(widths)),
        float(np.max(widths)),
        float(widths[-1]),
        float(widths[-1] - initial_trigger_width),
        float(np.mean(straddles)),
        float(np.max(straddles)),
        float(overflow_count),
        float(np.mean(generator_counts)),
    ], dtype=np.float64)
    future[~np.isfinite(future)] = 0.0
    return np.concatenate([base, future])


def write_regret_artifacts(
    result: RegretEvalResult,
    output_dir: Path,
    metadata: dict,
) -> None:
    """Persist regret-distillation diagnostics and learned rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result.policy.save(output_dir / f"learned_regret_{result.oracle_mode}.npz")
    candidate_costs = regret_traces_to_dataframe(result.traces)
    candidate_costs.to_csv(output_dir / "regret_candidate_costs.csv", index=False)
    regret_trace_summary(result.traces).to_csv(
        output_dir / "regret_trace_summary.csv", index=False,
    )
    regret_training_dataframe(result.training_results).to_csv(
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
            "oracle_mode": result.oracle_mode,
            "total_traces": len(result.traces),
            "eval_runs": len(result.eval_results),
            "avg_inference_time_ms": result.inference_time_ms,
        },
        output_dir / "regret_metadata.json",
    )


def regret_traces_to_dataframe(traces: Sequence[RegretTrace]) -> pd.DataFrame:
    """Convert traces to one row per candidate per decision."""
    rows = []
    for idx, trace in enumerate(traces):
        for name, cost, regret in zip(trace.candidate_names, trace.costs, trace.regrets):
            rows.append({
                "decision_id": idx,
                "seed": trace.seed,
                "iteration": trace.iteration,
                "step": trace.step,
                "oracle_mode": trace.oracle_mode,
                "candidate": name,
                "cost": cost,
                "regret": regret,
                "best_action": trace.best_action,
                "best_cost": trace.best_cost,
                "second_best_margin": trace.second_best_margin,
                "tie_count": trace.tie_count,
            })
    return pd.DataFrame(rows)


def regret_trace_summary(traces: Sequence[RegretTrace]) -> pd.DataFrame:
    """Aggregate label, margin, and tie diagnostics."""
    if not traces:
        return pd.DataFrame()
    rows = []
    for (mode, iteration), group in _group_traces(traces, ("oracle_mode", "iteration")).items():
        best_actions = [t.best_action for t in group]
        margins = np.asarray([t.second_best_margin for t in group], dtype=np.float64)
        rows.append({
            "oracle_mode": mode,
            "iteration": iteration,
            "decisions": len(group),
            "mean_second_best_margin": float(np.mean(margins)),
            "median_second_best_margin": float(np.median(margins)),
            "near_tie_fraction_1e_3": float(np.mean(margins <= 1e-3)),
            "near_tie_fraction_1e_1": float(np.mean(margins <= 1e-1)),
            **{
                f"best_{name}_count": best_actions.count(name)
                for name in sorted(set(best_actions))
            },
        })
    return pd.DataFrame(rows)


def regret_training_dataframe(results: Sequence[RegretTrainingResult]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "iteration": i,
            **asdict(result),
            "train_loss_history": json.dumps(result.train_loss_history),
        }
        for i, result in enumerate(results)
    ])


def regret_ranking_metrics(traces: Sequence[RegretTrace]) -> pd.DataFrame:
    """Oracle-only diagnostics for the regret target distribution."""
    if not traces:
        return pd.DataFrame()
    rows = []
    for trace in traces:
        best_idx = int(np.argmin(trace.regrets))
        rows.append({
            "oracle_mode": trace.oracle_mode,
            "seed": trace.seed,
            "iteration": trace.iteration,
            "step": trace.step,
            "best_action": trace.candidate_names[best_idx],
            "second_best_margin": trace.second_best_margin,
            "max_regret": float(np.max(trace.regrets)),
            "mean_regret": float(np.mean(trace.regrets)),
            "tie_count": trace.tie_count,
        })
    return pd.DataFrame(rows)


def _collect_regret_episode(
    *,
    monitor: MonitorAdapter,
    trace: Sequence,
    budget: int,
    seed: int,
    iteration: int,
    oracle_config: RegretOracleConfig,
    collector: RegretTraceCollector,
    learned_policy: ReductionPolicy | None,
) -> None:
    state = monitor.initial_state()
    history: list = []
    for step, measurement in enumerate(trace):
        result = monitor.step(state, measurement)
        state = result.state
        history.append(measurement)
        if state.zonotope.generator_count <= budget:
            continue
        rows = evaluate_regret_candidates(monitor, state, history, budget, oracle_config)
        features = _extract_regret_features(
            monitor, state, history, budget, oracle_config,
        )
        collector.record(_trace_from_rows(
            rows=rows,
            features=features,
            oracle_mode=oracle_config.mode,
            seed=seed,
            step=step,
            iteration=iteration,
        ))
        if learned_policy is None:
            state = rows[0].state
        else:
            state = learned_policy.decide(monitor, state, history, budget).state


def _trace_from_rows(
    *,
    rows: Sequence[CandidateCost],
    features: np.ndarray,
    oracle_mode: str,
    seed: int,
    step: int,
    iteration: int,
) -> RegretTrace:
    ordered = sorted(rows, key=lambda row: row.name)
    best = min(rows, key=lambda row: (row.cost, row.sequence))
    costs = np.asarray([row.cost for row in ordered], dtype=np.float64)
    best_cost = float(best.cost)
    scale = max(abs(best_cost), 1.0)
    regrets = (costs - best_cost) / scale
    sorted_costs = sorted(float(row.cost) for row in rows)
    second_margin = (
        sorted_costs[1] - sorted_costs[0]
        if len(sorted_costs) > 1 else 0.0
    )
    tie_tol = max(1e-9, 1e-9 * max(abs(best_cost), 1.0))
    return RegretTrace(
        features=features,
        candidate_names=tuple(row.name for row in ordered),
        costs=tuple(float(v) for v in costs),
        regrets=tuple(float(max(v, 0.0)) for v in regrets),
        best_action=best.name,
        best_cost=best_cost,
        second_best_margin=float(second_margin),
        tie_count=sum(1 for row in rows if abs(row.cost - best_cost) <= tie_tol),
        oracle_mode=oracle_mode,
        step=step,
        seed=seed,
        iteration=iteration,
    )


def _candidate_reducers(names: tuple[str, ...]) -> tuple[ProtectedReducer, ...]:
    return tuple(ProtectedReducer(base=ALL_REDUCERS[name]) for name in names)


def _candidate_reducer_map(names: tuple[str, ...]) -> dict[str, ProtectedReducer]:
    return {name: ProtectedReducer(base=ALL_REDUCERS[name]) for name in names}


def _sequence_future_cost(
    monitor: MonitorAdapter,
    start_state: MonitorState,
    inputs: tuple,
    candidates: tuple[ProtectedReducer, ...],
    budget: int,
    cost_fn: WeightedZonotopeCost,
    fallback: ProtectedReducer,
) -> tuple[float, tuple[str, ...]] | None:
    best: tuple[float, tuple[str, ...]] | None = None

    def rec(index: int, state: MonitorState, total: float, sequence: tuple[str, ...]) -> None:
        nonlocal best
        if best is not None and (total, sequence) >= best:
            return
        if index >= len(inputs):
            if best is None or (total, sequence) < best:
                best = (total, sequence)
            return
        step_result = monitor.step(state, inputs[index])
        next_state = step_result.state
        if next_state.zonotope.generator_count <= budget:
            rec(
                index + 1, next_state,
                total + cost_fn(next_state, step_result.verdicts),
                sequence,
            )
            return
        children = []
        for reducer in candidates:
            reduced = try_certified_reduce(monitor, next_state, reducer, budget)
            if reduced is not None:
                children.append((reducer.name, reduced[0]))
        if not children:
            reduced = try_certified_reduce(monitor, next_state, fallback, budget)
            if reduced is not None:
                children.append((fallback.name, reduced[0]))
        for name, child_state in children:
            rec(
                index + 1, child_state,
                total + cost_fn(child_state, step_result.verdicts),
                (*sequence, name),
            )

    rec(0, start_state, cost_fn(start_state), ())
    return best


def _beam_future_cost(
    monitor: MonitorAdapter,
    start_state: MonitorState,
    inputs: tuple,
    candidates: tuple[ProtectedReducer, ...],
    budget: int,
    cost_fn: WeightedZonotopeCost,
    beam_width: int,
    fallback: ProtectedReducer,
) -> tuple[float, tuple[str, ...]] | None:
    beam = [(cost_fn(start_state), (), start_state)]
    for measurement in inputs:
        expanded: list[tuple[float, tuple[str, ...], MonitorState]] = []
        for total, sequence, state in beam:
            step_result = monitor.step(state, measurement)
            next_state = step_result.state
            if next_state.zonotope.generator_count <= budget:
                expanded.append((
                    total + cost_fn(next_state, step_result.verdicts),
                    sequence,
                    next_state,
                ))
                continue
            children = []
            for reducer in candidates:
                reduced = try_certified_reduce(monitor, next_state, reducer, budget)
                if reduced is not None:
                    children.append((reducer.name, reduced[0]))
            if not children:
                reduced = try_certified_reduce(monitor, next_state, fallback, budget)
                if reduced is not None:
                    children.append((fallback.name, reduced[0]))
            for name, child_state in children:
                expanded.append((
                    total + cost_fn(child_state, step_result.verdicts),
                    (*sequence, name),
                    child_state,
                ))
        if not expanded:
            return None
        expanded.sort(key=lambda item: (item[0], item[1]))
        beam = expanded[:max(int(beam_width), 1)]
    best = min(beam, key=lambda item: (item[0], item[1]))
    return best[0], best[1]


def _fixed_base_future_cost(
    monitor: MonitorAdapter,
    start_state: MonitorState,
    inputs: tuple,
    base_reducer: ProtectedReducer,
    budget: int,
    cost_fn: WeightedZonotopeCost,
    fallback: ProtectedReducer,
) -> tuple[float, tuple[str, ...]] | None:
    total = cost_fn(start_state)
    state = start_state
    sequence: list[str] = []
    for measurement in inputs:
        step_result = monitor.step(state, measurement)
        state = step_result.state
        if state.zonotope.generator_count > budget:
            reduced = try_certified_reduce(monitor, state, base_reducer, budget)
            name = base_reducer.name
            if reduced is None:
                reduced = try_certified_reduce(monitor, state, fallback, budget)
                name = fallback.name
            if reduced is None:
                return None
            state = reduced[0]
            sequence.append(name)
        total += cost_fn(state, step_result.verdicts)
    return total, tuple(sequence)


def _best_fixed_base_future_cost(
    monitor: MonitorAdapter,
    start_state: MonitorState,
    inputs: tuple,
    base_candidates: tuple[ProtectedReducer, ...],
    budget: int,
    cost_fn: WeightedZonotopeCost,
    fallback: ProtectedReducer,
) -> tuple[float, tuple[str, ...]] | None:
    best: tuple[float, tuple[str, ...]] | None = None
    for base in base_candidates:
        result = _fixed_base_future_cost(
            monitor, start_state, inputs, base, budget, cost_fn, fallback,
        )
        if result is not None and (best is None or result < best):
            best = result
    return best


def _group_traces(
    traces: Sequence[RegretTrace],
    keys: tuple[str, ...],
) -> dict[tuple, list[RegretTrace]]:
    grouped: dict[tuple, list[RegretTrace]] = {}
    for trace in traces:
        key = tuple(getattr(trace, name) for name in keys)
        grouped.setdefault(key, []).append(trace)
    return grouped
