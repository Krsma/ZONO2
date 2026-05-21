"""Command-line training for distilled reducer-selection policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from pzr.learning.features import (
    DECISION_FEATURE_NAMES,
    DECISION_FEATURE_SCHEMA_VERSION,
)
from pzr.learning.dagger import (
    aggregate_dagger_rows,
    class_balanced_indices,
    load_dagger_iterations,
)
from pzr.learning.policy import _build_mlp

DEFAULT_CANDIDATE_REDUCER_NAMES = (
    "box",
    "girard",
    "girard_slack1",
    "keep_trigger",
    "keep_norm",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        train_policy(args)
        return 0
    if args.command == "dagger":
        train_dagger_policy(args)
        return 0
    raise ValueError(f"unsupported command: {args.command}")


def train_policy(args: argparse.Namespace) -> None:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for policy distillation. "
            "Install the learning extra with `python -m pip install -e .[learning]`."
        ) from exc

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = pd.read_csv(args.data)
    if "feature_schema_version" not in data.columns:
        raise ValueError("decision feature data is missing feature_schema_version")
    rows = data[
        (data["feature_schema_version"] == DECISION_FEATURE_SCHEMA_VERSION)
        & (data["method"] == args.expert_method)
        & (data["predictor_mode"] == args.predictor_mode)
        & data["chosen_reducer_label"].notna()
        & (data["chosen_reducer_label"].astype(str) != "")
    ].copy()
    if rows.empty:
        raise ValueError(
            f"no training rows found for method={args.expert_method!r}, "
            f"predictor_mode={args.predictor_mode!r}"
        )

    missing = [name for name in DECISION_FEATURE_NAMES if name not in rows.columns]
    if missing:
        raise ValueError(f"decision feature data is missing columns: {missing}")
    feature_names = tuple(DECISION_FEATURE_NAMES)
    features = rows.loc[:, feature_names].astype(float).to_numpy()
    if not np.isfinite(features).all():
        raise ValueError("decision feature data contains non-finite feature values")

    class_names = _ordered_classes(rows["chosen_reducer_label"].astype(str).tolist())
    class_to_index = {name: index for index, name in enumerate(class_names)}
    labels = rows["chosen_reducer_label"].astype(str).map(class_to_index).to_numpy(dtype=np.int64)
    if getattr(args, "class_balanced", False):
        selected = class_balanced_indices(rows["chosen_reducer_label"].astype(str), args.seed)
        rows = rows.iloc[selected].reset_index(drop=True)
        features = features[selected]
        labels = labels[selected]
    train_idx, val_idx = _train_validation_split(rows, args.validation_fraction, args.seed)

    mean = np.mean(features[train_idx], axis=0)
    std = np.std(features[train_idx], axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    x = (features - mean) / std
    x_train = torch.as_tensor(x[train_idx], dtype=torch.float32)
    y_train = torch.as_tensor(labels[train_idx], dtype=torch.long)
    x_val = torch.as_tensor(x[val_idx], dtype=torch.float32)
    y_val = torch.as_tensor(labels[val_idx], dtype=torch.long)

    hidden_sizes = tuple(int(value) for value in args.hidden_sizes)
    model = _build_mlp(len(feature_names), len(class_names), hidden_sizes, torch=torch)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(args.seed)

    for _epoch in range(args.epochs):
        order = torch.randperm(x_train.shape[0], generator=generator)
        for start in range(0, x_train.shape[0], args.batch_size):
            batch = order[start : start + args.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()

    train_metrics = _classification_metrics(model, x_train, y_train, class_names, torch=torch)
    val_metrics = _classification_metrics(model, x_val, y_val, class_names, torch=torch)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": DECISION_FEATURE_SCHEMA_VERSION,
        "model_state": model.state_dict(),
        "feature_names": list(feature_names),
        "class_names": list(class_names),
        "candidate_reducer_names": list(DEFAULT_CANDIDATE_REDUCER_NAMES),
        "normalizer_mean": mean.tolist(),
        "normalizer_std": std.tolist(),
        "hidden_sizes": list(hidden_sizes),
        "training_config": {
            "expert_method": args.expert_method,
            "predictor_mode": args.predictor_mode,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "validation_fraction": args.validation_fraction,
            "row_count": int(rows.shape[0]),
            "class_balanced": bool(getattr(args, "class_balanced", False)),
            "training_mode": getattr(args, "training_mode", "distillation"),
        },
    }
    dagger_metadata = getattr(args, "dagger_metadata", None)
    if dagger_metadata is not None:
        checkpoint["dagger"] = dagger_metadata
    torch.save(checkpoint, out)
    metrics = {
        "schema_version": DECISION_FEATURE_SCHEMA_VERSION,
        "data": str(args.data),
        "checkpoint": str(out),
        "expert_method": args.expert_method,
        "predictor_mode": args.predictor_mode,
        "row_count": int(rows.shape[0]),
        "class_balanced": bool(getattr(args, "class_balanced", False)),
        "training_mode": getattr(args, "training_mode", "distillation"),
        "train_row_count": int(len(train_idx)),
        "validation_row_count": int(len(val_idx)),
        "class_counts": {
            name: int(count)
            for name, count in rows["chosen_reducer_label"]
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        },
        "train": train_metrics,
        "validation": val_metrics,
    }
    out.with_suffix(".metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def train_dagger_policy(args: argparse.Namespace) -> None:
    """Aggregate DAgger iterations and train a reducer-ranking checkpoint."""

    iterations = load_dagger_iterations(
        args.data,
        expert_method=args.expert_method,
        predictor_mode=args.predictor_mode,
    )
    aggregate = aggregate_dagger_rows(iterations)
    if aggregate.empty:
        raise ValueError("DAgger aggregation produced no rows")
    out = Path(args.out)
    aggregate_path = args.aggregate_out or out.with_suffix(".dagger_dataset.csv")
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(aggregate_path, index=False)
    train_policy(
        argparse.Namespace(
            data=aggregate_path,
            expert_method=args.expert_method,
            predictor_mode=args.predictor_mode,
            out=out,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            validation_fraction=args.validation_fraction,
            hidden_sizes=args.hidden_sizes,
            class_balanced=True,
            training_mode="dagger",
            dagger_metadata={
                "iteration_count": len(iterations),
                "aggregate_dataset": str(aggregate_path),
                "source_rows": [
                    {
                        "iteration": iteration.iteration,
                        "row_count": int(iteration.rows.shape[0]),
                        "expert_policy": iteration.expert_policy,
                        "learner_policy": iteration.learner_policy,
                    }
                    for iteration in iterations
                ],
            },
        )
    )


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pzr-distill-policy",
        description="Train distilled reducer-selection policies from benchmark decisions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--expert-method", default="mpc_focused_sequence")
    train.add_argument("--predictor-mode", choices=("online", "oracle"), default="online")
    train.add_argument("--out", required=True, type=Path)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--validation-fraction", type=float, default=0.2)
    train.add_argument("--hidden-sizes", type=int, nargs="+", default=(64, 64))
    train.add_argument("--class-balanced", action="store_true")
    dagger = subparsers.add_parser("dagger")
    dagger.add_argument("--data", required=True, type=Path, nargs="+")
    dagger.add_argument("--expert-method", default="mpc_focused_sequence")
    dagger.add_argument("--predictor-mode", choices=("online", "oracle"), default="online")
    dagger.add_argument("--out", required=True, type=Path)
    dagger.add_argument("--aggregate-out", type=Path, default=None)
    dagger.add_argument("--seed", type=int, default=0)
    dagger.add_argument("--epochs", type=int, default=200)
    dagger.add_argument("--batch-size", type=int, default=64)
    dagger.add_argument("--lr", type=float, default=1e-3)
    dagger.add_argument("--weight-decay", type=float, default=0.0)
    dagger.add_argument("--validation-fraction", type=float, default=0.2)
    dagger.add_argument("--hidden-sizes", type=int, nargs="+", default=(64, 64))
    return parser


def _ordered_classes(labels: list[str]) -> tuple[str, ...]:
    present = set(labels)
    ordered = [name for name in DEFAULT_CANDIDATE_REDUCER_NAMES if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return tuple(ordered)


def _train_validation_split(
    rows: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if "seed" in rows.columns and rows["seed"].nunique() > 1:
        unique_seeds = np.asarray(sorted(rows["seed"].unique()))
        rng.shuffle(unique_seeds)
        val_count = max(1, int(round(unique_seeds.size * validation_fraction)))
        val_seeds = set(unique_seeds[:val_count])
        val_mask = rows["seed"].isin(val_seeds).to_numpy()
    else:
        val_mask = np.zeros(rows.shape[0], dtype=bool)
        val_count = max(1, int(round(rows.shape[0] * validation_fraction)))
        indices = np.arange(rows.shape[0])
        rng.shuffle(indices)
        val_mask[indices[:val_count]] = True
    train_mask = ~val_mask
    if not train_mask.any():
        train_mask[:] = True
        val_mask[:] = True
    elif not val_mask.any():
        val_mask = train_mask.copy()
    return np.flatnonzero(train_mask), np.flatnonzero(val_mask)


def _classification_metrics(
    model: Any,
    x: Any,
    y: Any,
    class_names: tuple[str, ...],
    *,
    torch: Any,
) -> dict[str, Any]:
    with torch.no_grad():
        logits = model(x)
        predictions = torch.argmax(logits, dim=1)
        correct = predictions.eq(y)
        top_k = min(3, len(class_names))
        topk = torch.topk(logits, k=top_k, dim=1).indices
        topk_correct = topk.eq(y.reshape(-1, 1)).any(dim=1)
    confusion: dict[str, dict[str, int]] = {name: {} for name in class_names}
    for true_index, pred_index in zip(y.cpu().numpy(), predictions.cpu().numpy()):
        true_name = class_names[int(true_index)]
        pred_name = class_names[int(pred_index)]
        confusion[true_name][pred_name] = confusion[true_name].get(pred_name, 0) + 1
    return {
        "accuracy": float(correct.float().mean().item()) if y.numel() else 0.0,
        f"top_{top_k}_accuracy": (
            float(topk_correct.float().mean().item()) if y.numel() else 0.0
        ),
        "confusion": confusion,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
