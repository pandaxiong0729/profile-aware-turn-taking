from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from profile_turntaking.constants import LABELS, LABEL_TO_ID
from profile_turntaking.utils import read_jsonl, write_json, write_jsonl


SOURCE_LABEL = {
    "vad_majority_silence": "NA",
    "short_feedback_lexicon": "BC",
    "timed_speaker_overlap_ge_40ms": "I",
    "nonoverlap_speaker_change_onset": "T",
    "vad_majority_speech_default": "C",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify complete forced five-class labels")
    parser.add_argument(
        "--input-dir", default="data/processed/sbcsae_vad_fiveclass_v2"
    )
    args = parser.parse_args()
    root = Path(args.input_dir)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    conversation_counts: Counter[str] = Counter()
    invalid_labels = label_id_mismatches = source_mismatches = 0
    nonconsecutive_indices = time_order_errors = review_fields = 0
    interruption_without_two_speakers = 0
    last_index: dict[str, int] = {}
    last_end: dict[str, float] = {}
    examples: dict[str, list[dict]] = defaultdict(list)

    with gzip.open(root / "frame_labels.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            label = row.get("label")
            conversation_id = str(row["conversation_id"])
            counts[str(label)] += 1
            source_counts[str(row.get("label_source"))] += 1
            conversation_counts[conversation_id] += 1
            invalid_labels += int(label not in LABELS)
            label_id_mismatches += int(row.get("label_id") != LABEL_TO_ID.get(label))
            source_mismatches += int(SOURCE_LABEL.get(row.get("label_source")) != label)
            review_fields += int(
                any("review" in str(key).lower() for key in row)
            )
            index = int(row["frame_index"])
            if conversation_id in last_index:
                nonconsecutive_indices += int(index != last_index[conversation_id] + 1)
                time_order_errors += int(
                    abs(float(row["start_s"]) - last_end[conversation_id]) > 1e-6
                )
            else:
                nonconsecutive_indices += int(index != 0)
                time_order_errors += int(abs(float(row["start_s"])) > 1e-6)
            last_index[conversation_id] = index
            last_end[conversation_id] = float(row["end_s"])
            if label == "I" and len(row.get("active_speakers", [])) < 2:
                interruption_without_two_speakers += 1
            if label in LABELS and len(examples[label]) < 3:
                examples[label].append(row)

    span_counts: Counter[str] = Counter()
    span_frames = span_gaps = adjacent_same_label_spans = 0
    previous_span: dict[str, dict] = {}
    for row in read_jsonl(root / "label_spans.jsonl"):
        label = str(row["label"])
        frames = int(row["frame_count"])
        span_counts[label] += frames
        span_frames += frames
        conversation_id = str(row["conversation_id"])
        if conversation_id in previous_span:
            previous = previous_span[conversation_id]
            span_gaps += int(abs(float(row["start_s"]) - float(previous["end_s"])) > 1e-6)
            adjacent_same_label_spans += int(row["label"] == previous["label"])
        previous_span[conversation_id] = row

    total = sum(counts.values())
    checks = {
        "total_matches_summary": total == int(summary["total_frames"]),
        "all_frames_labeled": total == int(summary["labeled_frames"])
        and int(summary["unlabeled_frames"]) == 0,
        "coverage_is_one": float(summary["coverage"]) == 1.0,
        "only_five_labels": invalid_labels == 0 and set(counts) == set(LABELS),
        "label_counts_match_summary": all(
            counts[label] == int(summary["label_counts"][label]) for label in LABELS
        ),
        "label_ids_match": label_id_mismatches == 0,
        "sources_match_labels": source_mismatches == 0,
        "no_review_fields": review_fields == 0 and summary["review_states"] is False,
        "all_16_conversations_present": set(conversation_counts)
        == set(summary["conversation_ids"])
        and len(conversation_counts) == 16,
        "frame_indices_consecutive": nonconsecutive_indices == 0,
        "frame_times_contiguous": time_order_errors == 0,
        "interruption_has_two_timed_speakers": interruption_without_two_speakers == 0,
        "span_frames_match": span_frames == total
        and all(span_counts[label] == counts[label] for label in LABELS),
        "spans_contiguous_and_merged": span_gaps == 0
        and adjacent_same_label_spans == 0,
        "semantic_candidate_queue_unused": summary["semantic_candidate_queue_used"]
        is False,
    }
    result = {
        "verified": all(checks.values()),
        "checks": checks,
        "total_frames": total,
        "conversation_counts": dict(conversation_counts),
        "label_counts": {label: counts[label] for label in LABELS},
        "source_counts": dict(source_counts),
        "diagnostics": {
            "invalid_labels": invalid_labels,
            "label_id_mismatches": label_id_mismatches,
            "source_mismatches": source_mismatches,
            "review_fields": review_fields,
            "nonconsecutive_indices": nonconsecutive_indices,
            "time_order_errors": time_order_errors,
            "interruption_without_two_speakers": interruption_without_two_speakers,
            "span_gaps": span_gaps,
            "adjacent_same_label_spans": adjacent_same_label_spans,
        },
    }
    write_json(root / "verification.json", result)
    write_jsonl(
        root / "examples.jsonl",
        ({"label": label, "examples": examples[label]} for label in LABELS),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
