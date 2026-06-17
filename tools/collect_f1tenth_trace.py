#!/usr/bin/env python3
"""Collect an F1TENTH Gym trace as JSONL.

Run this inside an isolated F1TENTH sidecar environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import yaml


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _first_obs(reset_result: Any) -> Any:
    if isinstance(reset_result, tuple):
        return reset_result[0]
    return reset_result


def _obs_array(obs: dict[str, Any], key: str, default: float = 0.0) -> np.ndarray:
    value = obs.get(key)
    if value is None:
        return np.array([default], dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.array([default], dtype=float)
    return arr


def _reactive_action(obs: dict[str, Any]) -> np.ndarray:
    scans = np.asarray(obs["scans"], dtype=float)
    scan = scans[0] if scans.ndim == 2 else scans
    finite = np.isfinite(scan)
    if not np.any(finite):
        return np.array([[0.0, 0.5]], dtype=float)
    clean = np.where(finite, scan, np.nanmax(scan[finite]))
    n = clean.size
    center = n // 2
    window = max(n // 5, 1)
    sector = clean[center - window:center + window]
    best = int(np.argmax(sector)) - window
    steer = float(np.clip(best / max(window, 1) * 0.35, -0.35, 0.35))
    front = clean[center - max(n // 24, 1):center + max(n // 24, 1)]
    front_clearance = float(np.nanmin(front))
    speed = float(np.clip(0.45 + 0.22 * front_clearance, 0.35, 1.45))
    return np.array([[steer, speed]], dtype=float)


def _seeded_track_geometry(seed: int, *, stress_randomize: bool) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    stress = bool(stress_randomize)
    return {
        "amp1": float((0.46 if stress else 0.28) + rng.normal(0.0, 0.05)),
        "freq1": float((0.50 if stress else 0.45) + rng.normal(0.0, 0.03)),
        "phase1": float(rng.uniform(-0.45, 0.45)),
        "amp2": float((0.18 if stress else 0.06) + rng.normal(0.0, 0.025)),
        "freq2": float((1.05 if stress else 0.82) + rng.normal(0.0, 0.05)),
        "phase2": float(rng.uniform(-0.7, 0.7)),
        "half_width": float((0.70 if stress else 0.82) + rng.normal(0.0, 0.025)),
        "width_wave": float((0.17 if stress else 0.10) + rng.uniform(0.0, 0.025)),
        "width_freq": float((0.88 if stress else 0.90) + rng.normal(0.0, 0.04)),
        "width_phase": float(rng.uniform(-0.8, 0.8)),
        "bottleneck_x": float(rng.uniform(-1.6, 1.6) if stress else 0.0),
        "bottleneck_depth": float((0.24 + rng.uniform(0.0, 0.08)) if stress else 0.0),
        "bottleneck_sigma": float(0.85 + rng.uniform(0.0, 0.35)),
        "front_phase": float(rng.uniform(-1.2, 1.2)),
    }


def _center_y(x: float, geometry: dict[str, float]) -> float:
    return float(
        geometry["amp1"] * np.sin(geometry["freq1"] * x + geometry["phase1"])
        + geometry["amp2"] * np.sin(geometry["freq2"] * x + geometry["phase2"])
    )


def _track_width(x: float, geometry: dict[str, float]) -> float:
    bottleneck = geometry["bottleneck_depth"] * np.exp(
        -0.5 * ((x - geometry["bottleneck_x"]) / max(geometry["bottleneck_sigma"], 1e-9)) ** 2
    )
    wave = geometry["width_wave"] * np.sin(geometry["width_freq"] * x + geometry["width_phase"])
    return float(max(0.38, geometry["half_width"] + wave - bottleneck))


def _ensure_map(
    map_name: str,
    output_dir: Path,
    *,
    seed: int,
    stress_randomize: bool,
) -> tuple[str, np.ndarray, dict[str, float]]:
    path = Path(map_name)
    if path.with_suffix(".yaml").exists():
        return str(path.with_suffix("")), np.array([[0.0, 0.0, 0.0]], dtype=float), {}

    geometry = _seeded_track_geometry(seed, stress_randomize=stress_randomize)
    suffix = f"{map_name}_seed{seed}_probe_map" if stress_randomize else f"{map_name}_probe_map"
    stem = output_dir / suffix
    png_path = stem.with_suffix(".png")
    yaml_path = stem.with_suffix(".yaml")
    size = 420
    resolution = 0.05
    origin = [-10.5, -10.5, 0.0]
    img = np.zeros((size, size), dtype=np.uint8)

    def world_to_px(x: float, y: float) -> tuple[int, int]:
        px = int(round((x - origin[0]) / resolution))
        py = int(round((y - origin[1]) / resolution))
        return px, py

    # White corridor with black walls. The centerline drifts mildly so LiDAR
    # margins are near threshold without requiring a full planning stack.
    for x in np.arange(-9.0, 9.0, resolution):
        center_y = _center_y(float(x), geometry)
        half_width = _track_width(float(x), geometry)
        x0, y0 = world_to_px(x, center_y - half_width)
        x1, y1 = world_to_px(x + resolution, center_y + half_width)
        img[max(0, y0):min(size, y1), max(0, x0):min(size, x1 + 1)] = 255
    # Add two shallow side bays so the reactive controller has asymmetric scan
    # structure, while leaving the main corridor connected.
    for bx_min, bx_max, by_min, by_max in [(-2.5, -0.5, 0.7, 1.6), (2.5, 4.5, -1.6, -0.7)]:
        bx0, by0 = world_to_px(bx_min, by_min)
        bx1, by1 = world_to_px(bx_max, by_max)
        img[max(0, by0):min(size, by1), max(0, bx0):min(size, bx1)] = 255

    Image.fromarray(img).save(png_path)
    metadata = {
        "image": png_path.name,
        "resolution": resolution,
        "origin": origin,
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(metadata, f)
    rng = np.random.default_rng(seed + 1729)
    start_x = -7.5 + (rng.uniform(-0.18, 0.18) if stress_randomize else 0.0)
    start_y = _center_y(start_x, geometry) + (rng.uniform(-0.08, 0.08) if stress_randomize else 0.0)
    start_theta = float(np.arctan(
        geometry["amp1"] * geometry["freq1"] * np.cos(geometry["freq1"] * start_x + geometry["phase1"])
        + geometry["amp2"] * geometry["freq2"] * np.cos(geometry["freq2"] * start_x + geometry["phase2"])
    ))
    return str(stem), np.array([[start_x, start_y, start_theta]], dtype=float), geometry


def collect_trace(
    *,
    output: Path,
    length: int,
    seed: int,
    map_name: str,
    stress_randomize: bool,
) -> None:
    try:
        import gym
        import f110_gym  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(f"F1TENTH dependencies are unavailable: {exc}") from exc

    map_stem, reset_pose, map_geometry = _ensure_map(
        map_name, output.parent,
        seed=seed,
        stress_randomize=stress_randomize,
    )
    env = gym.make("f110_gym:f110-v0", map=map_stem, num_agents=1)
    rng = np.random.default_rng(seed)
    obs = _first_obs(env.reset(poses=reset_pose))
    records = []
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        for step in range(length):
            if not isinstance(obs, dict):
                raise RuntimeError(f"expected dict observation, got {type(obs)!r}")
            action = _reactive_action(obs)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = bool(terminated or truncated)
            else:
                next_obs, reward, done, info = step_result

            obs_record = {
                "step": step,
                "time": float(step),
                "obs": _json_safe(obs),
                "action": _json_safe(action),
                "reward": _json_safe(reward),
                "done": bool(done),
                "info": _json_safe(info),
                "map": map_name,
                "seed": int(seed),
                "map_stem": map_stem,
                "map_geometry": _json_safe(map_geometry),
                "reset_pose": _json_safe(reset_pose),
                "pzr_seeded_stress": bool(stress_randomize),
                "noise_probe": float(rng.normal(0.0, 1e-12)),
            }
            records.append(obs_record)
            obs = next_obs
            if done:
                break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    with open(output, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--map", dest="map_name", default="vegas")
    parser.add_argument("--stress-randomize", action="store_true")
    args = parser.parse_args()
    collect_trace(
        output=args.output,
        length=args.length,
        seed=args.seed,
        map_name=args.map_name,
        stress_randomize=args.stress_randomize,
    )


if __name__ == "__main__":
    main()
