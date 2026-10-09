"""Portable, hash-verified recorded tasks. Loading never collects teacher data."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pzr.learning.artifacts import load_reducer_cost_dataset
from pzr.learning.dataset import ReducerCostDataset
from pzr.learning.ranker import ReducerPolicy
from pzr.rtlola.actions import MPC_ACTION_NAMES
from pzr.rtlola.engine import RtlolaEvent
from pzr.rtlola.features import RTL_RANKING_FEATURE_SCHEMA
from pzr.rtlola.scenarios import RtlolaTrace, scenario_by_name


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TraceSet:
    traces: tuple[RtlolaTrace, ...]
    budget: int
    split: str


@dataclass(frozen=True)
class Task:
    dataset: ReducerCostDataset
    sample_metadata: pd.DataFrame
    budget: int
    bundle_sha256: str
    bundle_root: Path
    training_traces: TraceSet
    validation_traces: TraceSet
    metadata: dict[str, object]

    @property
    def training(self) -> ReducerCostDataset:
        return self.dataset.subset(self.dataset.indices_for_split("train"))

    @property
    def validation(self) -> ReducerCostDataset:
        return self.dataset.subset(self.dataset.indices_for_split("validation"))

    def frozen_baseline(self) -> ReducerPolicy:
        """Clean148 specialist, trained on more data than the starter task."""
        return ReducerPolicy.load(self.bundle_root / "baseline" / f"budget-{self.budget}")


def load_task(name: str = "robot_arm", *, bundle: str | Path = "starter", budget: int = 40) -> Task:
    if name != "robot_arm":
        raise ValueError("The recorded student bundles currently support robot_arm")
    root = Path(bundle)
    if str(bundle) == "starter":
        root = Path(__file__).parent / "bundles" / "starter"
    elif str(bundle) == "clean148":
        root = Path(os.environ.get("PZR_STUDENT_DATA", "data/student")) / "clean148"
    manifest_path = root / "bundle.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Student bundle missing: {root}. See docs/student/README.md")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "pzr.student-bundle.v1":
        raise ValueError("unsupported student bundle")
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != expected:
            raise ValueError(f"bundle file integrity failure: {relative}")
    scenario = scenario_by_name(name)
    if hashlib.sha256(scenario.spec.encode()).hexdigest() != manifest["spec_sha256"]:
        raise ValueError("bundle specification differs from installed specification")
    dataset, metadata, _ = load_reducer_cost_dataset(root / "dataset")
    if dataset.candidate_names != MPC_ACTION_NAMES or dataset.feature_names != RTL_RANKING_FEATURE_SCHEMA.feature_names:
        raise ValueError("bundle action or feature schema differs")
    rows = np.flatnonzero(metadata.budget.to_numpy() == budget)
    if rows.size == 0:
        raise ValueError(f"budget {budget} unavailable; choose from {manifest['budgets']}")
    dataset, metadata = dataset.subset(rows), metadata.iloc[rows].reset_index(drop=True)
    split_ids = {split: set(metadata.loc[metadata.split == split, 'trace_id']) for split in ('train', 'validation')}
    if not all(split_ids.values()) or split_ids['train'] & split_ids['validation']:
        raise ValueError("training/validation trajectories must be nonempty and disjoint")
    traces = {'train': [], 'validation': []}
    for record in manifest['traces']:
        if record['trace_id'] not in split_ids[record['split']]:
            continue
        payload = json.loads((root / record['path']).read_text())
        events = tuple(RtlolaEvent(float(t), tuple(values)) for t, values in payload)
        traces[record['split']].append(RtlolaTrace(name, record['trace_kind'], record['seed'], events))
    for split in traces:
        if len(traces[split]) != len(split_ids[split]):
            raise ValueError(f"missing recorded {split} trajectories")
    return Task(dataset, metadata, budget, file_hash(manifest_path), root,
                TraceSet(tuple(traces['train']), budget, 'train'),
                TraceSet(tuple(traces['validation']), budget, 'validation'), {
                    'spec': scenario.spec, 'spec_sha256': manifest['spec_sha256'],
                    'input_channels': scenario.input_channel_names,
                    'public_streams': scenario.public_stream_keys,
                    'trigger_names': scenario.trigger_keys, 'trigger_labels': scenario.trigger_labels,
                    'provenance': manifest['sources'],
                })
