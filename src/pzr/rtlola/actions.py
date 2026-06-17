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
    """Return the v1 RTLola built-in action set."""
    _, _, ZonotopeConfig = require_binding()
    return (
        RtlolaAction("none", lambda _b: ZonotopeConfig.none(), explicit_budget=False),
        RtlolaAction("girard", lambda b: ZonotopeConfig.girard(b)),
        RtlolaAction("scott", lambda b: ZonotopeConfig.scott(b)),
        RtlolaAction("interval_hull", lambda b: ZonotopeConfig.interval_hull(b)),
        RtlolaAction("colinear_scale", lambda b: ZonotopeConfig.colinear_scale(b)),
        RtlolaAction("colinear", lambda _b: ZonotopeConfig.colinear(), explicit_budget=False),
        RtlolaAction("interval", lambda _b: ZonotopeConfig.interval(), explicit_budget=False),
    )


def action_by_name(actions: tuple[RtlolaAction, ...]) -> dict[str, RtlolaAction]:
    return {action.name: action for action in actions}
