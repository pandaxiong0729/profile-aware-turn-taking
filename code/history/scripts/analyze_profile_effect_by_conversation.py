#!/usr/bin/env python
"""Decompose paired A/B profile effects by test conversation and task."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


MODES = ("hidden", "given", "shuffled")


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # Accuracy is computed only on the deterministic 50/50 task subsets used
    # by the paper-comparable aggregate.  No model is trained or selected here.
    groups: dict[tuple[str, str, str, int], list[int]] = defaultdict(list)
    with args.predictions.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not bool(row.get("paper_balanced_subset", False)):
                continue
            key = (
                str(row["conversation_id"]),
                str(row["task"]),
                str(row["profile_mode"]),
                int(row["seed"]),
            )
            groups[key].append(int(row["prediction_answer"] == row["reference_answer"]))

    rows: list[dict[str, Any]] = []
    conversations = sorted({key[0] for key in groups})
    tasks = sorted({key[1] for key in groups})
    seeds = sorted({key[3] for key in groups})
    for conversation in conversations:
        for task in tasks:
            by_mode: dict[str, list[float]] = {mode: [] for mode in MODES}
            counts: dict[str, int] = {}
            for mode in MODES:
                for seed in seeds:
                    values = groups.get((conversation, task, mode, seed), [])
                    if values:
                        by_mode[mode].append(sum(values) / len(values))
                        counts[mode] = len(values)
            if not all(by_mode.values()):
                continue
            scores = {mode: float(mean(by_mode[mode])) for mode in MODES}
            rows.append(
                {
                    "conversation_id": conversation,
                    "task": task,
                    "samples_per_seed": min(counts.values()),
                    **{f"{mode}_accuracy": scores[mode] for mode in MODES},
                    "given_minus_hidden": scores["given"] - scores["hidden"],
                    "given_minus_shuffled": scores["given"] - scores["shuffled"],
                }
            )

    conversation_rows: list[dict[str, Any]] = []
    for conversation in conversations:
        current = [row for row in rows if row["conversation_id"] == conversation]
        if not current:
            continue
        scores = {
            mode: float(mean([float(row[f"{mode}_accuracy"]) for row in current]))
            for mode in MODES
        }
        conversation_rows.append(
            {
                "conversation_id": conversation,
                "tasks_with_both_classes": len(current),
                **{f"{mode}_accuracy": scores[mode] for mode in MODES},
                "given_minus_hidden": scores["given"] - scores["hidden"],
                "given_minus_shuffled": scores["given"] - scores["shuffled"],
            }
        )

    macro = {
        mode: float(mean([float(row[f"{mode}_accuracy"]) for row in conversation_rows]))
        for mode in MODES
    }
    report = {
        "definition": "Five-seed mean accuracy on deterministic paper-balanced rows, decomposed by conversation.",
        "conversations": conversation_rows,
        "conversation_task_rows": rows,
        "conversation_macro": {
            **macro,
            "given_minus_hidden": macro["given"] - macro["hidden"],
            "given_minus_shuffled": macro["given"] - macro["shuffled"],
        },
        "warning": "Only three test conversations exist; conversation-level estimates have high variance.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "conversation_profile_effect.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for filename, payload in (
        ("conversation_profile_effect.csv", conversation_rows),
        ("conversation_task_profile_effect.csv", rows),
    ):
        if payload:
            with (args.output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(payload[0]))
                writer.writeheader()
                writer.writerows(payload)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
