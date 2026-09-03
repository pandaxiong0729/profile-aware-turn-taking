#!/usr/bin/env python
"""Average seed probabilities before making the final A/B decision."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


TASKS = ("turn_change", "backchannel", "interruption", "floor_taking")
MODES = ("hidden", "given", "shuffled")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    with args.predictions.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not bool(row.get("paper_balanced_subset", False)):
                continue
            key = (str(row["sample_id"]), str(row["task"]), str(row["profile_mode"]))
            entry = grouped.setdefault(
                key,
                {
                    "reference_answer": str(row["reference_answer"]),
                    "prob_B": [],
                    "seeds": [],
                },
            )
            if entry["reference_answer"] != str(row["reference_answer"]):
                raise ValueError(f"Reference answer changed across seeds for {key}")
            entry["prob_B"].append(float(row["prob_B"]))
            entry["seeds"].append(int(row["seed"]))

    accuracies: dict[str, dict[str, float]] = {task: {} for task in TASKS}
    counts: dict[str, dict[str, int]] = {task: {} for task in TASKS}
    ensemble_rows: list[dict[str, Any]] = []
    for task in TASKS:
        for mode in MODES:
            correct: list[int] = []
            for (sample_id, row_task, row_mode), entry in grouped.items():
                if row_task != task or row_mode != mode:
                    continue
                if len(set(entry["seeds"])) != len(entry["seeds"]):
                    raise ValueError(f"Duplicate seed prediction for {(sample_id, task, mode)}")
                mean_prob_b = sum(entry["prob_B"]) / len(entry["prob_B"])
                prediction = "B" if mean_prob_b >= 0.5 else "A"
                is_correct = int(prediction == entry["reference_answer"])
                correct.append(is_correct)
                ensemble_rows.append(
                    {
                        "sample_id": sample_id,
                        "task": task,
                        "profile_mode": mode,
                        "seed_count": len(entry["seeds"]),
                        "mean_prob_B": mean_prob_b,
                        "prediction_answer": prediction,
                        "reference_answer": entry["reference_answer"],
                        "correct": is_correct,
                    }
                )
            if not correct:
                raise ValueError(f"No balanced rows for {task}/{mode}")
            accuracies[task][mode] = sum(correct) / len(correct)
            counts[task][mode] = len(correct)

    overall = {
        mode: sum(accuracies[task][mode] for task in TASKS) / len(TASKS)
        for mode in MODES
    }
    report = {
        "method": "Unweighted arithmetic mean of five seed probabilities per identical sample/task/profile condition, followed by a 0.5 A/B threshold.",
        "task_accuracy": accuracies,
        "task_counts": counts,
        "overall_mean_four_task_accuracy": overall,
        "given_minus_hidden": overall["given"] - overall["hidden"],
        "given_minus_shuffled": overall["given"] - overall["shuffled"],
        "profile_order_pass": overall["given"] > overall["hidden"] and overall["given"] > overall["shuffled"],
        "accuracy_target_pass": overall["given"] >= 0.73,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions_path = args.output.with_name("ensemble_predictions.jsonl")
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in ensemble_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
