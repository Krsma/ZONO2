"""Stable fingerprints for resumable learning experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


def model_sha256(directory: Path) -> str:
    return sha256_files((directory / "model.json", directory / "weights.pt"))


def pzr_source_sha256() -> str:
    root = Path(__file__).parents[1]
    return sha256_files(tuple(sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".lola"}
    )), relative_to=root)


def payload_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_files(
    paths: Sequence[Path],
    *,
    relative_to: Path | None = None,
) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise ValueError(f"fingerprinted artifact is missing: {path}")
        name = path.relative_to(relative_to) if relative_to is not None else path.name
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
