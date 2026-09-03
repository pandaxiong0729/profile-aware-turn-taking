"""Audit the five-class SBCSAE event labels used by the main experiment.

This audit checks the stored event evidence, the train/val/test references, and
the older dense 40 ms label package.  It does not relabel any sample.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


LABELS = ("C", "BC", "T", "I", "NA")
STRUCTURE_TO_LABEL = {
    "hold_after_pause": "C",
    "hold_after_unsuccessful_interruption": "C",
    "backchannel_candidate": "BC",
    "natural_turn_shift": "T",
    "shift_after_successful_interruption": "T",
    "interruption_candidate": "I",
    "mutual_silence_onset": "NA",
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else path.open
    with opener(path, "rt", encoding="utf-8") if path.suffix == ".gz" else opener(
        "r", encoding="utf-8"
    ) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def check_event(row: dict[str, Any], ipus: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    label = str(row.get("candidate_label"))
    structure = str(row.get("structure"))
    evidence = dict(row.get("evidence") or {})
    before = row.get("speaker_before")
    event_speaker = row.get("event_speaker")
    after = row.get("speaker_after")

    if label not in LABELS:
        errors.append("invalid_label")
    if STRUCTURE_TO_LABEL.get(structure) != label:
        errors.append("structure_label_mismatch")
    if not (
        abs(float(row["anchor_s"]) - float(row["event_start_s"])) <= 1e-6
        and abs(float(row["event_start_s"]) - float(row["event_end_s"])) <= 1e-6
    ):
        errors.append("event_not_point_anchored")

    if structure == "hold_after_pause":
        if not (before == event_speaker == after):
            errors.append("hold_speaker_mismatch")
        if float(evidence.get("silence_or_latch_ms", -1)) < 0:
            errors.append("negative_hold_gap")
    elif structure == "natural_turn_shift":
        if before is None or event_speaker is None or before == event_speaker or after != event_speaker:
            errors.append("turn_shift_speaker_mismatch")
        if float(evidence.get("silence_or_latch_ms", -1)) < 0:
            errors.append("negative_turn_gap")
    elif structure == "backchannel_candidate":
        duration = float(evidence.get("response_duration_ms", -1))
        if duration <= 0 or duration > 1500 + 1e-6:
            errors.append("backchannel_duration_out_of_rule")
        if not bool(evidence.get("inside_other_turn")):
            errors.append("backchannel_not_inside_other_turn")
        if not (bool(evidence.get("lexical_match")) or bool(evidence.get("structural_match"))):
            errors.append("backchannel_without_rule_match")
        if before is not None and event_speaker == before:
            errors.append("backchannel_same_as_floor_speaker")
    elif structure == "interruption_candidate":
        overlap_ms = float(evidence.get("overlap_ms", -1))
        if overlap_ms <= 0:
            errors.append("interruption_without_positive_overlap")
        if before is None or event_speaker is None or before == event_speaker:
            errors.append("interruption_speaker_mismatch")
        newcomer = ipus.get(str(evidence.get("newcomer_ipu_id")))
        incumbent = ipus.get(str(evidence.get("incumbent_ipu_id")))
        if newcomer is None or incumbent is None:
            errors.append("interruption_ipu_missing")
        else:
            actual = min(float(newcomer["end_s"]), float(incumbent["end_s"])) - max(
                float(newcomer["start_s"]), float(incumbent["start_s"])
            )
            if actual <= 0 or abs(actual * 1000 - overlap_ms) > 1.1:
                errors.append("interruption_overlap_evidence_mismatch")
    elif structure in {
        "shift_after_successful_interruption",
        "hold_after_unsuccessful_interruption",
    }:
        newcomer = ipus.get(str(evidence.get("interruption_ipu_id")))
        incumbent = ipus.get(str(evidence.get("incumbent_ipu_id")))
        if newcomer is None or incumbent is None:
            errors.append("floor_outcome_ipu_missing")
        elif structure == "shift_after_successful_interruption":
            if not float(newcomer["end_s"]) > float(incumbent["end_s"]) + 1e-6:
                errors.append("successful_floor_take_end_order_wrong")
            if not (before != event_speaker and after == event_speaker):
                errors.append("successful_floor_take_speaker_mismatch")
        else:
            if not float(incumbent["end_s"]) > float(newcomer["end_s"]) + 1e-6:
                errors.append("unsuccessful_floor_take_end_order_wrong")
            if not (before == event_speaker == after):
                errors.append("unsuccessful_floor_take_speaker_mismatch")
    elif structure == "mutual_silence_onset":
        if event_speaker is not None:
            errors.append("silence_has_event_speaker")
        if float(evidence.get("silence_duration_ms", -1)) < 200 - 1e-6:
            errors.append("silence_shorter_than_rule")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument(
        "--event-dir", default="data/processed/sbcsae_turn_events_v3"
    )
    parser.add_argument(
        "--experiment-data-dir",
        default="data/processed/sbcsae_qwen_shared_ab_30s_causal_v1",
    )
    parser.add_argument(
        "--dense-dir", default="data/processed/sbcsae_vad_fiveclass_v2"
    )
    parser.add_argument(
        "--output", default="collaboration/audits/fiveclass_label_audit.json"
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    event_dir = (root / args.event_dir).resolve()
    experiment_dir = (root / args.experiment_data_dir).resolve()
    dense_dir = (root / args.dense_dir).resolve()
    output = (root / args.output).resolve()

    ipus = {str(row["ipu_id"]): row for row in read_jsonl(event_dir / "ipus.jsonl")}
    events: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    structure_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    error_examples: dict[str, list[str]] = defaultdict(list)
    anchors: dict[str, set[int]] = defaultdict(set)
    duplicate_anchor_count = 0

    for row in read_jsonl(event_dir / "event_candidates.jsonl"):
        event_id = str(row["event_id"])
        if event_id in events:
            error_counts["duplicate_event_id"] += 1
        events[event_id] = row
        label = str(row["candidate_label"])
        counts[label] += 1
        structure_counts[str(row["structure"])] += 1
        anchor_ms = round(float(row["anchor_s"]) * 1000)
        conversation_id = str(row["conversation_id"])
        if anchor_ms in anchors[conversation_id]:
            duplicate_anchor_count += 1
        anchors[conversation_id].add(anchor_ms)
        for error in check_event(row, ipus):
            error_counts[error] += 1
            if len(error_examples[error]) < 5:
                error_examples[error].append(event_id)

    split_counts: dict[str, Any] = {}
    split_conversations: dict[str, set[str]] = {}
    selected_ids: set[str] = set()
    reference_mismatches = 0
    selected_duplicate_ids = 0
    for split in ("train", "val", "test"):
        labels = Counter()
        conversations: set[str] = set()
        rows = 0
        for row in read_jsonl(experiment_dir / split / "reference_labels.jsonl"):
            rows += 1
            event_id = str(row["sample_id"])
            if event_id in selected_ids:
                selected_duplicate_ids += 1
            selected_ids.add(event_id)
            source = events.get(event_id)
            if source is None or any(
                (
                    row.get("reference_label") != source.get("candidate_label"),
                    abs(float(row.get("event_time_in_conversation_s", -1)) - float(source.get("anchor_s", -2))) > 1e-6,
                    row.get("source_kind") != source.get("structure"),
                )
            ):
                reference_mismatches += 1
            labels[str(row["reference_label"])] += 1
            conversations.add(str(row["conversation_id"]))
        split_counts[split] = {
            "rows": rows,
            "labels": {label: labels[label] for label in LABELS},
            "conversations": sorted(conversations),
        }
        split_conversations[split] = conversations

    split_overlap = {
        "train_val": sorted(split_conversations["train"] & split_conversations["val"]),
        "train_test": sorted(split_conversations["train"] & split_conversations["test"]),
        "val_test": sorted(split_conversations["val"] & split_conversations["test"]),
    }

    dense_summary = json.loads((dense_dir / "summary.json").read_text(encoding="utf-8"))
    dense_verification = json.loads(
        (dense_dir / "verification.json").read_text(encoding="utf-8")
    )
    checks = {
        "event_ids_unique": error_counts["duplicate_event_id"] == 0,
        "one_event_per_millisecond_per_conversation": duplicate_anchor_count == 0,
        "event_evidence_matches_fiveclass_rules": not error_counts,
        "main_experiment_references_match_event_source": reference_mismatches == 0,
        "main_experiment_sample_ids_unique": selected_duplicate_ids == 0,
        "conversation_splits_disjoint": not any(split_overlap.values()),
        "all_five_labels_present_in_every_split": all(
            all(row["labels"][label] > 0 for label in LABELS)
            for row in split_counts.values()
        ),
        "dense_40ms_package_structurally_verified": bool(dense_verification.get("verified")),
        "dense_rule_version_matches_summary": dense_summary.get("rule_version")
        == "vad_trn_forced_fiveclass_v1",
    }
    result = {
        "verified": all(checks.values()),
        "scope": {
            "event_labels": str(event_dir.relative_to(root)),
            "main_experiment_labels": str(experiment_dir.relative_to(root)),
            "dense_40ms_labels": str(dense_dir.relative_to(root)),
        },
        "checks": checks,
        "event_dataset": {
            "events": len(events),
            "labels": {label: counts[label] for label in LABELS},
            "structures": dict(sorted(structure_counts.items())),
            "duplicate_anchor_count": duplicate_anchor_count,
            "rule_error_counts": dict(error_counts),
            "rule_error_examples": dict(error_examples),
        },
        "main_experiment": {
            "samples": len(selected_ids),
            "splits": split_counts,
            "reference_mismatches": reference_mismatches,
            "split_conversation_overlap": split_overlap,
        },
        "dense_40ms_dataset": {
            "frames": dense_summary["total_frames"],
            "labels": dense_summary["label_counts"],
            "verification_file_passed": dense_verification["verified"],
        },
        "interpretation": (
            "The stored labels are internally consistent with the declared VAD/IPU rules and "
            "the main experiment references the same event records. This proves deterministic "
            "rule consistency, not independent human semantic agreement."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
