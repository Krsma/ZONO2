"""One native execution path shared by learning evaluation and Gym."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from pzr.rtlola.actions import MPC_ACTION_NAMES, default_action_catalog
from pzr.rtlola.engine import RtlolaEngine
from pzr.rtlola.features import RTL_RANKING_FEATURE_SCHEMA, extract_ranking_features
from pzr.rtlola.learned_policy import RtlolaReducerPolicy
from pzr.rtlola.metrics import RtlolaMatrixMetrics
from pzr.rtlola.reference import load_or_compute_reference, trace_sha256
from pzr.rtlola.scenarios import RtlolaTrace, scenario_by_name
from pzr.learning.ranker import ReducerPolicy
from pzr.learning.tasks import TraceSet


@dataclass(frozen=True)
class Zonotope:
    """Z = center + generators @ xi, ||xi||_inf <= 1; columns are generators."""
    center: np.ndarray
    generators: np.ndarray

    def __post_init__(self) -> None:
        c, g = np.array(self.center, dtype=np.float64), np.array(self.generators, dtype=np.float64)
        if c.ndim != 1 or g.ndim != 2 or g.shape[0] != c.size:
            raise ValueError('expected center[d], generators[d,m]')
        if not np.isfinite(c).all() or not np.isfinite(g).all():
            raise ValueError('geometry must be finite')
        c.setflags(write=False)
        g.setflags(write=False)
        object.__setattr__(self, 'center', c)
        object.__setattr__(self, 'generators', g)


@dataclass(frozen=True)
class Observation:
    features: np.ndarray
    budget: int
    step: int
    time: float
    counts: RtlolaMatrixMetrics
    reduction_needed: bool
    action_mask: np.ndarray
    dynamic: Zonotope | None = None
    total: Zonotope | None = None


class _Scores:
    candidate_names = MPC_ACTION_NAMES
    feature_schema = RTL_RANKING_FEATURE_SCHEMA

    def __init__(self, scores: np.ndarray):
        self.scores = scores

    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        return self.scores


class MonitorSession:
    """Pre-event observations; exact references are used only for evaluation.

    Only native ZonotopeConfig transforms execute actions. The action mask
    excludes known bound failures, but is not a guarantee of native feasibility.
    """

    def __init__(self, trace: RtlolaTrace, *, budget: int = 40,
                 reference_cache: str | Path | None = None):
        if not trace.events or not isinstance(budget, int) or budget < 0:
            raise ValueError('nonempty trace and nonnegative integer budget required')
        self.trace, self.budget = trace, budget
        self.scenario = scenario_by_name(trace.scenario)
        self.catalog = default_action_catalog()
        self.engine = RtlolaEngine(self.scenario.spec, event_arity=self.scenario.event_arity,
                                  expected_verdict_keys=self.scenario.expected_verdict_keys)
        self._references = load_or_compute_reference(
            trace.events, scenario=self.scenario, trace_kind=trace.trace_kind, seed=trace.seed,
            cache_path=Path(reference_cache) if reference_cache is not None else None,
            include_approximation=True,
        )
        self.index = 0
        self.records: list[dict[str, object]] = []

    @property
    def done(self) -> bool:
        return self.index == len(self.trace.events)

    def observe(self, *, geometry: bool = False) -> Observation:
        time = self.trace.events[self.index - 1].time if self.index else 0.0
        state = self.engine.snapshot(step=self.index, time=time)
        metrics = self.engine.metrics(state)
        needed = metrics.dynamic_generator_count > self.budget
        mask = np.full(4, not needed or self.budget >= metrics.dimension, dtype=np.int8)
        features = extract_ranking_features(self.engine, state, self.budget)
        features.setflags(write=False)
        mask.setflags(write=False)
        dynamic = total = None
        if geometry:
            dyn, tot = self.engine.matrices(state)
            dynamic, total = Zonotope(dyn[:, 0], dyn[:, 1:]), Zonotope(tot[:, 0], tot[:, 1:])
        return Observation(features, self.budget, self.index, time, metrics, needed, mask, dynamic, total)

    def step(self, action: int | np.ndarray) -> dict[str, object]:
        if self.done:
            raise RuntimeError('episode complete; create/reset a session')
        if isinstance(action, (int, np.integer)):
            if not 0 <= action < 4:
                raise ValueError('action must be an index in [0,4)')
            scores = np.arange(4, dtype=np.float64)
            scores[int(action)] = -1
        else:
            scores = np.asarray(action, dtype=np.float64)
        if scores.shape != (4,) or not np.isfinite(scores).all():
            raise ValueError('expected four finite lower-is-better scores')
        requested = MPC_ACTION_NAMES[int(np.argmin(scores))]
        event = self.trace.events[self.index]
        state = self.engine.snapshot(step=self.index, time=event.time)
        start = perf_counter()
        decision = RtlolaReducerPolicy(_Scores(scores), self.catalog).choose(
            self.engine, state, event, self.budget)
        result = self.engine.live_step(event, decision.first_action, decision.first_action_budget,
                                       step=self.index + 1)
        execution_ms = (perf_counter() - start) * 1000
        reference = self._references[self.index]
        loss = self.engine.approx_loss_reference(reference.approximation, result.state)
        verdicts = {key: bool(result.verdict.get(key, False)) for key in self.scenario.trigger_keys}
        # Match benchmark semantics: positive means any trigger at this event.
        predicted, exact = any(verdicts.values()), any(reference.verdicts.values())
        tp, fp = int(predicted and exact), int(predicted and not exact)
        fn, tn = int(not predicted and exact), int(not predicted and not exact)
        record = {
            'step': self.index, 'requested_action': requested, 'executed_action': result.action_name,
            'fallback_used': decision.fallback_used, 'infeasible_candidate_count': decision.infeasible_candidate_count,
            'override_reason': ('under_budget' if result.action_name == 'none' else
                                'native_candidate_unavailable' if requested != result.action_name else None),
            'approx_loss': loss, 'execution_ms': execution_ms, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        }
        self.records.append(record)
        self.index += 1
        return dict(record)


def evaluate_monitor(policy: ReducerPolicy | str, traces: TraceSet, *, max_events: int | None = None,
                     reference_cache: str | Path | None = None) -> pd.DataFrame:
    """Evaluate validation traces, with explicit prefix diagnostics for quick runs.

    policy may be a trained selector or one of the four native action names.
    Prefix results are labelled and must not be reported as full-trajectory loss.
    """
    if max_events is not None and max_events < 1:
        raise ValueError('max_events must be positive')
    if isinstance(policy, str) and policy not in MPC_ACTION_NAMES:
        raise ValueError('unknown native selector')
    if not isinstance(policy, str) and (policy.candidate_names != MPC_ACTION_NAMES or
                                       policy.feature_schema != RTL_RANKING_FEATURE_SCHEMA):
        raise ValueError('policy schema differs')
    rows = []
    for original in traces.traces:
        trace = RtlolaTrace(original.scenario, original.trace_kind, original.seed,
                           original.events[:max_events] if max_events else original.events)
        cache = None
        if reference_cache is not None:
            cache = Path(reference_cache) / f'{trace.scenario}-{trace_sha256(trace.events)}.json'
            cache.parent.mkdir(parents=True, exist_ok=True)
        session = MonitorSession(trace, budget=traces.budget, reference_cache=cache)
        inference_ms = []
        start_run = perf_counter()
        while not session.done:
            obs = session.observe()
            start = perf_counter()
            action = (MPC_ACTION_NAMES.index(policy) if isinstance(policy, str) else
                      policy.predict_scores(obs.features) if obs.reduction_needed else 0)
            inference_ms.append((perf_counter() - start) * 1000)
            session.step(action)
        records = pd.DataFrame(session.records)
        tp, fp, fn, tn = (int(records[k].sum()) for k in ('tp', 'fp', 'fn', 'tn'))
        rows.append({
            'seed': trace.seed, 'trace_kind': trace.trace_kind, 'split': traces.split, 'budget': traces.budget,
            'events': len(trace.events), 'prefix_diagnostic': len(trace.events) != len(original.events),
            'final_approx_loss': records.approx_loss.iloc[-1], 'sum_approx_loss': records.approx_loss.sum(),
            'fallback_count': int(records.fallback_used.sum()),
            'infeasible_candidate_count': int(records.infeasible_candidate_count.sum()),
            'action_counts': dict(Counter(records.executed_action)),
            'fpr': fp / (fp + tn) if fp + tn else np.nan, 'fnr': fn / (fn + tp) if fn + tp else np.nan,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'mean_inference_ms': float(np.mean(inference_ms)),
            'event_loop_ms': (perf_counter() - start_run) * 1000,
        })
    return pd.DataFrame(rows)
