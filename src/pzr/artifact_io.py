"""Small atomic writers shared by versioned experiment artifacts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from io import TextIOWrapper
import json
from pathlib import Path

import pandas as pd


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV beside its destination and replace it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_text_atomic(value: str, path: Path) -> None:
    """Write text beside its destination and replace it atomically."""
    with atomic_text_writer(path) as handle:
        handle.write(value)


@contextmanager
def atomic_text_writer(path: Path, *, newline: str | None = None) -> Iterator[TextIOWrapper]:
    """Yield a temporary text handle and atomically replace ``path`` on success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline=newline) as handle:
        yield handle
    temporary.replace(path)


def write_json_atomic(payload: object, path: Path) -> None:
    """Write deterministic, human-readable JSON atomically."""
    write_text_atomic(json.dumps(payload, indent=2, sort_keys=True), path)
