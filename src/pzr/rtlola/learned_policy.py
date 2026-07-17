"""Direct current-state inference for learned RTLola reducer rankings."""

from __future__ import annotations

import numpy as np

from pzr.learning.ranker import ReducerPolicy
from pzr.rtlola.actions import RtlolaActionCatalog
from pzr.rtlola.engine import (
    RtlolaBindingError,
    RtlolaEngine,
    RtlolaEvent,
    RtlolaStateRef,
)
from pzr.rtlola.features import RTL_RANKING_FEATURE_SCHEMA, extract_ranking_features
from pzr.rtlola.search import RtlolaNoFeasibleAction, RtlolaSearchResult


class RtlolaReducerPolicy:
    """Apply a fixed-catalog PyTorch ranking without inference-time rollouts."""

    def __init__(
        self,
        policy: ReducerPolicy,
        catalog: RtlolaActionCatalog,
    ) -> None:
        expected = tuple(catalog.mpc_candidate_names)
        if policy.candidate_names != expected:
            raise ValueError(
                "learned candidate catalog does not match RTLola candidates: "
                f"policy={policy.candidate_names}, expected={expected}"
            )
        if policy.feature_schema != RTL_RANKING_FEATURE_SCHEMA:
            raise ValueError("learned policy does not use the RTLola ranking feature schema")
        self.policy = policy
        self.catalog = catalog

    def choose(
        self,
        engine: RtlolaEngine,
        state: RtlolaStateRef,
        event: RtlolaEvent,
        budget: int,
    ) -> RtlolaSearchResult:
        metrics = engine.metrics(state)
        if metrics.dynamic_generator_count <= budget:
            step = engine.branch_step(state, event, self.catalog.no_op, budget)
            return RtlolaSearchResult(
                first_action=self.catalog.no_op,
                first_action_budget=budget,
                first_step=step,
                predicted_cost=0.0,
                predicted_sequence=(self.catalog.no_op.name,),
                evaluated_leaves=1,
                pruned_branches=0,
                mpc_variant="learned_direct",
                root_strategy="ranked_direct",
            )

        features = extract_ranking_features(engine, state, budget)
        scores = np.asarray(self.policy.predict_scores(features), dtype=np.float64)
        if scores.shape != (len(self.policy.candidate_names),):
            raise ValueError("learned policy returned an invalid score vector")
        order = np.argsort(scores, kind="stable")
        failures = 0
        for index in order:
            name = self.policy.candidate_names[int(index)]
            action = self.catalog.by_name[name]
            if action.explicit_budget and budget < metrics.dimension:
                failures += 1
                continue
            try:
                step = engine.branch_step(state, event, action, budget)
            except RtlolaBindingError:
                failures += 1
                continue
            return RtlolaSearchResult(
                first_action=action,
                first_action_budget=budget,
                first_step=step,
                predicted_cost=float(scores[index]),
                predicted_sequence=(action.name,),
                evaluated_leaves=1,
                pruned_branches=0,
                reducer_failure_count=failures,
                infeasible_candidate_count=failures,
                mpc_variant="learned_direct",
                root_strategy="ranked_direct",
            )

        try:
            step = engine.branch_step(state, event, self.catalog.fallback, budget)
        except RtlolaBindingError as exc:
            raise RtlolaNoFeasibleAction(
                "learned RTLola candidates and interval fallback were infeasible"
            ) from exc
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
            mpc_variant="learned_direct",
            root_strategy="ranked_direct",
        )
