"""Black-box monitor adapter boundary."""

from pzr.monitoring.base import (
    MonitorAdapter,
    MonitorResult,
    MonitorState,
    TriggerSpec,
    Verdict,
    evaluate_triggers,
    trigger_predicate_holds,
    trigger_satisfaction_fraction,
    trigger_straddles_threshold,
)

__all__ = [
    "MonitorAdapter",
    "MonitorResult",
    "MonitorState",
    "TriggerSpec",
    "Verdict",
    "evaluate_triggers",
    "trigger_predicate_holds",
    "trigger_satisfaction_fraction",
    "trigger_straddles_threshold",
]
