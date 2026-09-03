"""Dependency-light classification metrics and report formatting."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .constants import LABELS


def classification_metrics(
    targets: Sequence[int], predictions: Sequence[int]
) -> dict[str, Any]:
    size = len(LABELS)
    confusion = np.zeros((size, size), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        confusion[int(target), int(prediction)] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    recalls: list[float] = []
    for index, label in enumerate(LABELS):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum() - true_positive)
        false_negative = int(confusion[index, :].sum() - true_positive)
        support = int(confusion[index, :].sum())
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_values.append(f1)
        recalls.append(recall)
    accuracy = float(np.trace(confusion) / max(1, confusion.sum()))
    return {
        "macro_f1": float(np.mean(f1_values)),
        "balanced_accuracy": float(np.mean(recalls)),
        "accuracy": accuracy,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "labels": list(LABELS),
        "samples": int(confusion.sum()),
    }


def write_metrics_csv(path: str | Path, reports: dict[str, dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["profile_mode", "macro_f1", *[f"{label}_f1" for label in LABELS]])
        for mode, report in reports.items():
            writer.writerow(
                [
                    mode,
                    report["macro_f1"],
                    *[report["per_class"][label]["f1"] for label in LABELS],
                ]
            )
        if "given" in reports and "hidden" in reports:
            given = reports["given"]
            hidden = reports["hidden"]
            writer.writerow(
                [
                    "given_minus_hidden",
                    given["macro_f1"] - hidden["macro_f1"],
                    *[
                        given["per_class"][label]["f1"]
                        - hidden["per_class"][label]["f1"]
                        for label in LABELS
                    ],
                ]
            )
