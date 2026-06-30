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


MPC_ACTION_NAMES = ("girard", "scott", "interval_hull", "pca")
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


@dataclass(frozen=True)
class RtlolaActionCatalog:
    """Experiment roles for binding-native zonotope transforms."""

    actions: tuple[RtlolaAction, ...]
    bounded_static_names: tuple[str, ...] = BOUNDED_STATIC_ACTION_NAMES
    mpc_candidate_names: tuple[str, ...] = MPC_ACTION_NAMES
    no_op_name: str = "none"
    fallback_name: str = "interval"

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


def default_action_catalog() -> RtlolaActionCatalog:
    """Return the authoritative action roles for benchmarks and learning."""
    return RtlolaActionCatalog(default_actions())
