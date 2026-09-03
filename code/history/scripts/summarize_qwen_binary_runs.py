"""Combine multiple calibrated Qwen runs without double-counting sample IDs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from profile_turntaking.constants import LABELS, LABEL_TO_ID
from profile_turntaking.metrics import classification_metrics
from profile_turntaking.prompt_baseline import PROFILE_MODES
from profile_turntaking.utils import write_json


def load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads((path / "predictions.json").read_text(encoding="utf-8"))


def summarize(run_paths: list[Path]) -> dict[str, Any]:
    seen: set[str] = set()
    combined: list[dict[str, Any]] = []
    run_counts: list[dict[str, Any]] = []
    for path in run_paths:
        rows = load_rows(path)
        unique_rows = [row for row in rows if str(row["sample_id"]) not in seen]
        duplicates = len(rows) - len(unique_rows)
        seen.update(str(row["sample_id"]) for row in unique_rows)
        combined.extend(unique_rows)
        run_counts.append(
            {
                "run": str(path.resolve()),
                "rows": len(rows),
                "new_unique_rows": len(unique_rows),
                "duplicates_excluded": duplicates,
            }
        )
    targets = [LABEL_TO_ID[str(row["reference_label"])] for row in combined]
    reports = {
        mode: classification_metrics(
            targets,
            [LABEL_TO_ID[str(row[f"{mode}_prediction"])] for row in combined],
        )
        for mode in PROFILE_MODES
    }
    distributions = {
        mode: {
            label: Counter(str(row[f"{mode}_prediction"]) for row in combined)[label]
            for label in LABELS
        }
        for mode in PROFILE_MODES
    }
    fixes = sum(
        row["hidden_prediction"] != row["reference_label"]
        and row["given_prediction"] == row["reference_label"]
        for row in combined
    )
    breaks = sum(
        row["hidden_prediction"] == row["reference_label"]
        and row["given_prediction"] != row["reference_label"]
        for row in combined
    )
    return {
        "runs": run_counts,
        "unique_samples": len(combined),
        "reference_distribution": dict(Counter(row["reference_label"] for row in combined)),
        "metrics": reports,
        "prediction_distribution": distributions,
        "profile_changes": {
            "hidden_vs_given": sum(
                row["hidden_prediction"] != row["given_prediction"] for row in combined
            ),
            "given_vs_shuffled": sum(
                row["given_prediction"] != row["shuffled_prediction"] for row in combined
            ),
            "given_fixes_hidden_error": fixes,
            "given_breaks_hidden_correct": breaks,
        },
        "macro_f1_deltas": {
            "given_minus_hidden": reports["given"]["macro_f1"]
            - reports["hidden"]["macro_f1"],
            "given_minus_shuffled": reports["given"]["macro_f1"]
            - reports["shuffled"]["macro_f1"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = summarize([Path(path) for path in args.run])
    write_json(Path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
