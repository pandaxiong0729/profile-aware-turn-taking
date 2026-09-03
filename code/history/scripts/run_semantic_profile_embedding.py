#!/usr/bin/env python
"""Prepare, cache, train, and evaluate the semantic-profile pilot."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from profile_turntaking.qwen25_omni_event_eval import prepare_event_eval
from profile_turntaking.semantic_profile_experiment import (
    SemanticTrainConfig,
    build_feature_cache,
    run_semantic_profile_experiment,
)
from profile_turntaking.utils import read_jsonl


def conversation_splits(manifest: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    seen: set[str] = set()
    for row in read_jsonl(manifest):
        conversation = str(row["conversation_id"])
        if conversation in seen:
            continue
        seen.add(conversation)
        split = str(row["split"])
        result[split].append(conversation)
    return {key: sorted(value) for key, value in result.items()}


def prepare(args: argparse.Namespace) -> None:
    source = Path(args.annotation_manifest)
    split_source = Path(args.split_manifest)
    splits = conversation_splits(split_source)
    destination = Path(args.data_dir)
    targets = {"train": args.train_per_class, "val": args.val_per_class, "test": args.test_per_class}
    for split, conversations in splits.items():
        per_class = targets[split]
        prepare_event_eval(
            source,
            destination / split,
            conversations=conversations,
            per_class=per_class,
            seed=args.seed + {"train": 0, "val": 1, "test": 2}[split],
            context_seconds=args.context_seconds,
            max_transcript_chars=6000,
            min_boundary_separation_s=args.min_boundary_separation_s,
            max_per_conversation_class=max(
                args.max_per_conversation_class,
                math.ceil(per_class / max(1, len(conversations))),
            ),
            forecast_lead_ms=100,
            forecast_horizon_ms=600,
            prompt_style="direct",
            phase=f"semantic-profile-{split}",
        )
    print(json.dumps({"prepared": str(destination.resolve()), "splits": splits}, indent=2))


def cache(args: argparse.Namespace) -> None:
    destination = Path(args.data_dir)
    reports = {}
    for split in ("train", "val", "test"):
        reports[split] = build_feature_cache(
            destination / split,
            destination / f"{split}.{args.cache_tag}.npz",
            sentence_model=args.sentence_model,
            batch_size=args.embedding_batch_size,
            profile_encoding=args.profile_encoding,
        )
    print(json.dumps(reports, indent=2))


def train(args: argparse.Namespace) -> None:
    root = Path(args.data_dir)
    config = SemanticTrainConfig(
        hidden_dimension=args.hidden_dimension,
        dropout=args.dropout,
        profile_dropout=args.profile_dropout,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seeds=tuple(args.seeds),
        device=args.device,
        profile_fusion=args.profile_fusion,
        profile_preprocessing=args.profile_preprocessing,
    )
    report = run_semantic_profile_experiment(
        root / f"train.{args.cache_tag}.npz",
        root / f"val.{args.cache_tag}.npz",
        root / f"test.{args.cache_tag}.npz",
        args.output_dir,
        config=config,
    )
    print(json.dumps(report["aggregate"], indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("command", choices=("prepare", "cache", "train", "all"))
    result.add_argument(
        "--annotation-manifest",
        default="data/processed/sbcsae_turn_events_v1/annotation_manifest.jsonl",
    )
    result.add_argument(
        "--split-manifest",
        default="data/processed/sbcsae_mvp_v2/event_onset_manifest.jsonl",
    )
    result.add_argument(
        "--data-dir",
        default="data/processed/sbcsae_semantic_profile_v1",
    )
    result.add_argument(
        "--output-dir",
        default="artifacts/semantic-profile-embedding/minilm-v1",
    )
    result.add_argument("--sentence-model", default="sentence-transformers/all-MiniLM-L6-v2")
    result.add_argument("--profile-encoding", choices=("whole", "linewise"), default="whole")
    result.add_argument("--cache-tag", default="semantic")
    result.add_argument("--train-per-class", type=int, default=300)
    result.add_argument("--val-per-class", type=int, default=50)
    result.add_argument("--test-per-class", type=int, default=50)
    result.add_argument("--context-seconds", type=float, default=5.9)
    result.add_argument("--min-boundary-separation-s", type=float, default=1.0)
    result.add_argument("--max-per-conversation-class", type=int, default=100)
    result.add_argument("--seed", type=int, default=113)
    result.add_argument("--embedding-batch-size", type=int, default=64)
    result.add_argument("--hidden-dimension", type=int, default=128)
    result.add_argument("--dropout", type=float, default=0.2)
    result.add_argument("--profile-dropout", type=float, default=0.5)
    result.add_argument("--epochs", type=int, default=30)
    result.add_argument("--patience", type=int, default=6)
    result.add_argument("--train-batch-size", type=int, default=64)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--seeds", type=int, nargs="+", default=[13, 37, 71])
    result.add_argument("--device", default="cpu")
    result.add_argument("--profile-fusion", choices=("additive", "interaction"), default="additive")
    result.add_argument(
        "--profile-preprocessing",
        choices=("raw", "center_l2"),
        default="raw",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command in {"prepare", "all"}:
        prepare(args)
    if args.command in {"cache", "all"}:
        cache(args)
    if args.command in {"train", "all"}:
        train(args)


if __name__ == "__main__":
    main()
