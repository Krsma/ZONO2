"""RTLola zonotope transform actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pzr.rtlola.binding import require_binding


ConfigFactory = Callable[[int], Any]


@dataclass(frozen=True)
class RtlolaAction:
    """Named RTLola zonotope transform action."""

    name: str
    _factory: ConfigFactory
    explicit_budget: bool = True

    def make_config(self, budget: int) -> Any:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        return self._factory(int(budget))


def default_actions() -> tuple[RtlolaAction, ...]:
    """Return the RTLola transforms exposed by the pinned binding."""
    _, _, ZonotopeConfig = require_binding()
    return (
        RtlolaAction("none", lambda _b: ZonotopeConfig.none(), explicit_budget=False),
        RtlolaAction("girard", lambda b: ZonotopeConfig.girard(b)),
        RtlolaAction("scott", lambda b: ZonotopeConfig.scott(b)),
        RtlolaAction("interval_hull", lambda b: ZonotopeConfig.interval_hull(b)),
        RtlolaAction("pca", lambda b: ZonotopeConfig.pca(b)),
        RtlolaAction("althoff_a", lambda b: ZonotopeConfig.althoff_a(b)),
        RtlolaAction("clustering", lambda b: ZonotopeConfig.clustering(b)),
        RtlolaAction("combastel", lambda b: ZonotopeConfig.combastel(b)),
        RtlolaAction("colinear_scale", lambda b: ZonotopeConfig.colinear_scale(b)),
        RtlolaAction("colinear", lambda _b: ZonotopeConfig.colinear(), explicit_budget=False),
        RtlolaAction("interval", lambda _b: ZonotopeConfig.interval(), explicit_budget=False),
    )


def action_by_name(actions: tuple[RtlolaAction, ...]) -> dict[str, RtlolaAction]:
    return {action.name: action for action in actions}


EXACT_BASELINE_ACTION_NAME = "none"
FALLBACK_ACTION_NAME = "interval"
COLINEAR_ACTION_NAME = "colinear"
MPC_ACTION_NAMES = (
    "girard",
    "scott",
    "interval_hull",
    "pca",
    "combastel",
    "clustering",
)
CORE_STATIC_ACTION_NAMES = (
    EXACT_BASELINE_ACTION_NAME,
    "girard",
    "scott",
    "interval_hull",
    "pca",
)
BOUNDED_STATIC_ACTION_NAMES = (
    "girard",
    "scott",
    "interval_hull",
    "pca",
    "althoff_a",
    "clustering",
    "combastel",
    "colinear_scale",
)
STATIC_ACTION_METHOD_NAMES = (
    EXACT_BASELINE_ACTION_NAME,
    *BOUNDED_STATIC_ACTION_NAMES,
)
EXPLICIT_ACTION_METHOD_NAMES = (
    *STATIC_ACTION_METHOD_NAMES,
    COLINEAR_ACTION_NAME,
    FALLBACK_ACTION_NAME,
)


@dataclass(frozen=True)
class RtlolaActionCatalog:
    """Experiment roles for binding-native zonotope transforms."""

    actions: tuple[RtlolaAction, ...]
    bounded_static_names: tuple[str, ...] = BOUNDED_STATIC_ACTION_NAMES
    mpc_candidate_names: tuple[str, ...] = MPC_ACTION_NAMES
    no_op_name: str = EXACT_BASELINE_ACTION_NAME
    fallback_name: str = FALLBACK_ACTION_NAME

    @property
    def by_name(self) -> dict[str, RtlolaAction]:
        return action_by_name(self.actions)

    @property
    def bounded_static(self) -> tuple[RtlolaAction, ...]:
        return tuple(self.by_name[name] for name in self.bounded_static_names)

    @property
    def mpc_candidates(self) -> tuple[RtlolaAction, ...]:
        return tuple(self.by_name[name] for name in self.mpc_candidate_names)

    @property
    def no_op(self) -> RtlolaAction:
        return self.by_name[self.no_op_name]

    @property
    def fallback(self) -> RtlolaAction:
        return self.by_name[self.fallback_name]


def default_action_catalog(
    mpc_candidate_names: tuple[str, ...] | None = None,
) -> RtlolaActionCatalog:
    """Return the authoritative action roles for benchmarks and learning."""
    candidate_names = (
        MPC_ACTION_NAMES
        if mpc_candidate_names is None
        else tuple(mpc_candidate_names)
    )
    if not candidate_names:
        raise ValueError("MPC candidate names must not be empty")
    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError("MPC candidate names must be unique")
    unsupported = set(candidate_names) - set(MPC_ACTION_NAMES)
    if unsupported:
        raise ValueError(
            "unsupported MPC candidates: "
            + ", ".join(sorted(unsupported))
        )
    return RtlolaActionCatalog(
        default_actions(),
        mpc_candidate_names=candidate_names,
    )
