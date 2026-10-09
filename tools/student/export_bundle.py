"""Export recorded teaching bundles; never collect, retrain, or modify results."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from pzr.artifact_io import write_json_atomic
from pzr.learning.artifacts import load_reducer_cost_dataset, write_reducer_cost_dataset
from pzr.learning.dataset import ReducerCostDataset
from pzr.learning.provenance import model_sha256
from pzr.learning.training import dataset_sha256
from pzr.rtlola.features import RTL_RANKING_FEATURE_SCHEMA
from pzr.rtlola.robot_arm import ROBOT_ARM_SPEC_SHA256
from pzr.rtlola.robot_arm_random import load_random_waypoint_trace
from pzr.student.data import file_hash

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'paper_clean20': 'paper-evaluation-v2/prepare/teacher/dataset',
    'dart_extra_clean16': 'dart-rescue-v1/prepare/extra-clean/dataset',
    'old_scaling_extra_clean48': 'clean-scaling-exploratory/prepare/new-clean/dataset',
    'new_clean64': 'prp-scale-robustness-exploratory/prepare/new-clean/dataset',
}


def export_bundle(output: Path, *, full: bool = False) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite bundle: {output}")
    import yaml
    contract = yaml.safe_load((ROOT / 'experiments/paper_evaluation_v3.yaml').read_text())['frozen_policy']
    freeze = json.loads((ROOT / contract['manifest']).read_text())
    if freeze['model_matrix_sha256'] != contract['model_matrix_sha256']:
        raise ValueError('frozen model matrix differs')
    train_seeds = set(range(20)) | set(range(26, 42)) | set(range(200, 312)) if full else set(range(4))
    val_seeds = set(range(20, 26)) if full else {20, 21}
    budgets = (40, 80, 120, 150, 200, 250, 500) if full else (40,)
    datasets, frames, sources, traces = [], [], {}, {}
    for label, relative in SOURCES.items():
        if not full and label != 'paper_clean20':
            continue
        path = ROOT / 'results' / relative
        digest = dataset_sha256(path)
        if digest != contract['source_dataset_sha256'][label]:
            raise ValueError(f'dataset hash differs: {label}')
        data, metadata, manifest = load_reducer_cost_dataset(path)
        selected = metadata.budget.isin(budgets) & (
            ((metadata.split == 'train') & metadata.seed.isin(train_seeds)) |
            ((metadata.split == 'validation') & metadata.seed.isin(val_seeds)))
        indices = np.flatnonzero(selected.to_numpy())
        datasets.append(data.subset(indices))
        frames.append(metadata.iloc[indices])
        sources[label] = digest
        trace_root = Path(manifest['trace_store'])
        if not trace_root.is_absolute():
            trace_root = ROOT / trace_root
        wanted = set(metadata.iloc[indices].trace_id)
        for record in manifest['traces']:
            if record['trace_id'] not in wanted:
                continue
            trace = load_random_waypoint_trace(trace_root / record['trace_store_relative_path'])
            if trace.metadata.trace_sha256 != record['trace_sha256']:
                raise ValueError('trace hash differs from teacher provenance')
            traces[record['trace_id']] = (record, trace)
    data = ReducerCostDataset.concatenate(datasets)
    metadata = pd.concat(frames, ignore_index=True)
    if metadata.sample_id.duplicated().any():
        raise ValueError('duplicate teacher samples')
    for split, seeds in [('train', train_seeds), ('validation', val_seeds)]:
        if set(metadata.loc[metadata.split == split, 'seed']) != seeds:
            raise ValueError(f'{split} trajectory coverage differs')
    output.mkdir(parents=True)
    write_reducer_cost_dataset(data, output / 'dataset', metadata, {
        'feature_schema': asdict(RTL_RANKING_FEATURE_SCHEMA), 'source_hashes': sources,
        'budgets': list(budgets), 'scenario': 'robot_arm',
    })
    (output / 'traces').mkdir()
    records = []
    for trace_id, (record, trace) in sorted(traces.items()):
        relative = f"traces/seed-{record['seed']}.json"
        write_json_atomic([[event.time, list(event.values)] for event in trace.events], output / relative)
        records.append({'trace_id': trace_id, 'seed': record['seed'], 'split': record['split'],
                        'trace_kind': record['condition'], 'path': relative,
                        'source_trace_sha256': record['trace_sha256']})
    for budget in budgets:
        model = freeze['models'][f'clean148:{budget}']
        source = ROOT / 'results/prp-scale-robustness-exploratory/train/clean148' / f'budget-{budget}'
        if model_sha256(source) != model['sha256']:
            raise ValueError('frozen model hash differs')
        destination = output / 'baseline' / f'budget-{budget}'
        destination.mkdir(parents=True)
        for filename in ('model.json', 'weights.pt'):
            shutil.copyfile(source / filename, destination / filename)
    write_json_atomic({
        'schema': 'pzr.student-bundle.v1', 'name': 'clean148' if full else 'starter',
        'sources': sources, 'spec_sha256': ROBOT_ARM_SPEC_SHA256, 'budgets': list(budgets),
        'traces': records, 'baseline': 'Clean148 optimizer seed 42; larger training set than starter',
        'files': {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()},
    }, output / 'bundle.json')
    print(f'Exported {data.num_samples} rows and {len(records)} traces to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    export_bundle(args.output, full=args.full)
