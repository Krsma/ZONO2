"""JSON and CSV serialization utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def json_safe(obj: Any) -> Any:
    """Recursively convert numpy types and Paths for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_safe(data), f, indent=2)


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)
