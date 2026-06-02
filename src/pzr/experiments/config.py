"""Experiment configuration with YAML support and preset profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pzr.mpc.objectives import CostWeights


@dataclass
class BenchmarkConfig:
    scenario: str = "omni_robot"
    length: int = 200
    budget: int = 10
    horizon: int = 4
    beam_width: int = 4
    seeds: int = 30
    method_set: str = "standard"
    cost_weights: CostWeights = field(default_factory=CostWeights)
    output_dir: str = "results"
    jobs: int = 1

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cost_weights"] = asdict(self.cost_weights)
        return d


PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {"length": 30, "seeds": 3, "horizon": 2},
    "standard": {"length": 200, "seeds": 10, "horizon": 4},
    "paper": {"length": 200, "seeds": 30, "horizon": 4},
}


def from_profile(profile: str, **overrides: Any) -> BenchmarkConfig:
    """Create a config from a named profile with optional overrides."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}. Options: {list(PROFILES)}")
    params = {**PROFILES[profile], **overrides}
    return BenchmarkConfig(**params)


def load_config(path: Path) -> BenchmarkConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    if "cost_weights" in data and isinstance(data["cost_weights"], dict):
        data["cost_weights"] = CostWeights(**data["cost_weights"])
    return BenchmarkConfig(**data)


def save_config(config: BenchmarkConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
