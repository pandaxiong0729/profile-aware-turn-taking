from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from profile_turntaking.constants import LABELS
from profile_turntaking.prompt_baseline import select_conversation_balanced_rows
from profile_turntaking.utils import read_jsonl, write_json, write_jsonl


def _parse_targets(value: str) -> dict[str, int]:
    targets: dict[str, int] = {}
    for part in value.split(","):
        label, count = part.split("=", 1)
        targets[label.strip().upper()] = int(count)
    if set(targets) != set(LABELS):
        raise argparse.ArgumentTypeError(
            f"class targets must specify exactly {', '.join(LABELS)}"
        )
    return targets


def _selection_report(
    rows: list[dict[str, Any]],
    *,
    class_targets: dict[str, int],
    max_per_conversation_class: int,
    min_boundary_separation_s: float,
    seed: int,
) -> dict[str, Any]:
    counts = Counter(str(row["label"]) for row in rows)
    by_label_conversation: dict[str, Counter[str]] = {
        label: Counter(
            str(row["conversation_id"])
            for row in rows
            if str(row["label"]) == label
        )
        for label in LABELS
    }
    times: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        times[str(row["conversation_id"])].append(float(row["prediction_time_s"]))
    observed_minimum = min(
        (
            right - left
            for values in times.values()
            for left, right in zip(sorted(values), sorted(values)[1:])
        ),
        default=None,
    )
    return {
        "selection_policy": "conversation_balanced_class_targets_v1",
        "seed": seed,
        "samples": len(rows),
        "conversations": len(times),
        "class_targets": class_targets,
        "class_counts": {label: counts[label] for label in LABELS},
        "max_per_conversation_class": max_per_conversation_class,
        "requested_min_boundary_separation_s": min_boundary_separation_s,
        "observed_min_boundary_separation_s": observed_minimum,
        "class_conversation_coverage": {
            label: len(by_label_conversation[label]) for label in LABELS
        },
        "class_max_conversation_share": {
            label: (
                max(by_label_conversation[label].values()) / counts[label]
                if counts[label]
                else None
            )
            for label in LABELS
        },
        "counts_by_conversation": {
            conversation_id: {
                label: by_label_conversation[label][conversation_id]
                for label in LABELS
            }
            for conversation_id in sorted(times)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a conversation-balanced prompt-review candidate manifest"
    )
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--split", default="all")
    parser.add_argument(
        "--class-targets",
        type=_parse_targets,
        default=_parse_targets("C=110,BC=110,T=110,I=110,NA=60"),
    )
    parser.add_argument("--max-per-conversation-class", type=int, default=10)
    parser.add_argument("--min-boundary-separation-s", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--review-items",
        help="optional review_items.json used to exclude automatically flagged rows",
    )
    parser.add_argument("--max-risk-score", type=int, default=None)
    args = parser.parse_args()

    source_rows = list(read_jsonl(args.input_manifest))
    if args.review_items is not None:
        review_items = json.loads(Path(args.review_items).read_text(encoding="utf-8"))
        maximum = 0 if args.max_risk_score is None else args.max_risk_score
        eligible_ids = {
            str(row["sample_id"])
            for row in review_items
            if int(row.get("risk_score", 0)) <= maximum
        }
        source_rows = [
            row for row in source_rows if str(row["sample_id"]) in eligible_ids
        ]
    selected = select_conversation_balanced_rows(
        source_rows,
        class_targets=args.class_targets,
        split=args.split,
        max_per_conversation_class=args.max_per_conversation_class,
        min_boundary_separation_s=args.min_boundary_separation_s,
        seed=args.seed,
    )
    output = Path(args.output_manifest)
    write_jsonl(output, selected)
    report = _selection_report(
        selected,
        class_targets=args.class_targets,
        max_per_conversation_class=args.max_per_conversation_class,
        min_boundary_separation_s=args.min_boundary_separation_s,
        seed=args.seed,
    )
    report["risk_filter"] = {
        "review_items": args.review_items,
        "max_risk_score": args.max_risk_score,
        "eligible_source_rows": len(source_rows),
    }
    write_json(output.with_suffix(".selection.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
