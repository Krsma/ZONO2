"""Wall-clock timing utilities."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Timer:
    elapsed_ms: float = 0.0
    _start: float = field(default=0.0, repr=False)

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        return self.elapsed_ms


@contextmanager
def timed():
    """Context manager that yields a Timer and records elapsed time."""
    t = Timer()
    t.start()
    yield t
    t.stop()
