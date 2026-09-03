"""Force complete C/BC/T/I/NA weak labels from VAD and timed speakers.

This is intentionally a deterministic weak-label pipeline.  Every 40 ms frame
receives exactly one label; there are no review or uncertain states.  VAD gates
speech versus silence, while SBCSAE timed speaker/text rows provide the simple
BC, overlap, and speaker-change heuristics requested for the first experiment.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from .constants import LABELS, LABEL_TO_ID
from .data import is_backchannel
from .sbcsae_manifest import MonotonicWeakLabeler
from .schemas import Utterance
from .utils import read_jsonl, write_json, write_jsonl


RULE_VERSION = "vad_trn_forced_fiveclass_v1"


def _frame_slice(start_s: float, end_s: float, frame_s: float, frames: int) -> slice:
    start = max(0, min(frames, int(math.floor(start_s / frame_s))))
    end = max(start, min(frames, int(math.ceil(end_s / frame_s))))
    return slice(start, end)


def _merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _qualifying_overlaps(
    utterances: Sequence[Utterance], *, minimum_overlap_s: float
) -> list[tuple[float, float]]:
    """Return different-speaker overlap intervals lasting at least the minimum."""
    ordered = sorted(utterances, key=lambda row: (row.start_s, row.end_s))
    active: list[Utterance] = []
    intervals: list[tuple[float, float]] = []
    for current in ordered:
        active = [row for row in active if row.end_s > current.start_s]
        for previous in active:
            if previous.speaker == current.speaker:
                continue
            start = max(previous.start_s, current.start_s)
            end = min(previous.end_s, current.end_s)
            if end - start + 1e-9 >= minimum_overlap_s:
                intervals.append((start, end))
        active.append(current)
    return _merge_intervals(intervals)


def build_rule_masks(
    utterances: Sequence[Utterance],
    *,
    frames: int,
    frame_ms: int = 40,
    minimum_overlap_ms: int = 40,
) -> dict[str, np.ndarray]:
    """Build BC/I/T masks; final VAD speech gating happens separately."""
    frame_s = frame_ms / 1000.0
    ordered = sorted(utterances, key=lambda row: (row.start_s, row.end_s))
    helper = MonotonicWeakLabeler(ordered, horizon_ms=frame_ms)
    backchannel_rows = [
        row
        for index, row in enumerate(ordered)
        if helper.backchannels[index]
        and helper.previous_speaker[index]
        and helper.previous_speaker[index] != row.speaker
    ]
    bc = np.zeros(frames, dtype=bool)
    for row in backchannel_rows:
        bc[_frame_slice(row.start_s, row.end_s, frame_s, frames)] = True

    overlaps = _qualifying_overlaps(
        ordered, minimum_overlap_s=minimum_overlap_ms / 1000.0
    )
    interruption = np.zeros(frames, dtype=bool)
    for start, end in overlaps:
        interruption[_frame_slice(start, end, frame_s, frames)] = True

    turn_change = np.zeros(frames, dtype=bool)
    for index, row in enumerate(ordered):
        previous = helper.previous_speaker[index]
        if helper.backchannels[index] or previous is None or previous == row.speaker:
            continue
        onset = max(0, min(frames - 1, int(math.floor(row.start_s / frame_s))))
        if not interruption[onset]:
            turn_change[onset] = True
    return {"BC": bc, "I": interruption, "T": turn_change}


def force_frame_labels(
    vad_rows: Sequence[dict[str, Any]],
    utterances: Sequence[Utterance],
    *,
    frame_ms: int = 40,
    minimum_overlap_ms: int = 40,
) -> list[dict[str, Any]]:
    """Assign one and only one five-class label to every VAD frame."""
    masks = build_rule_masks(
        utterances,
        frames=len(vad_rows),
        frame_ms=frame_ms,
        minimum_overlap_ms=minimum_overlap_ms,
    )
    output: list[dict[str, Any]] = []
    for index, row in enumerate(vad_rows):
        if row["vad_state"] != "speech":
            label, source = "NA", "vad_majority_silence"
        elif masks["BC"][index]:
            label, source = "BC", "short_feedback_lexicon"
        elif masks["I"][index]:
            label, source = "I", "timed_speaker_overlap_ge_40ms"
        elif masks["T"][index]:
            label, source = "T", "nonoverlap_speaker_change_onset"
        else:
            label, source = "C", "vad_majority_speech_default"
        output.append(
            {
                "sample_id": f"{row['conversation_id']}-frame-{int(row['frame_index']):07d}",
                "conversation_id": row["conversation_id"],
                "frame_index": int(row["frame_index"]),
                "prediction_time_s": float(row["start_s"]),
                "start_s": float(row["start_s"]),
                "end_s": float(row["end_s"]),
                "horizon_ms": frame_ms,
                "label": label,
                "label_id": LABEL_TO_ID[label],
                "label_source": source,
                "vad_state": row["vad_state"],
                "vad_votes": int(row["vad_votes"]),
                "vad_source_count": int(row["vad_source_count"]),
                "active_speakers": list(row.get("trn_active_speakers", [])),
                "rule_version": RULE_VERSION,
            }
        )
    return output


def _read_gzip_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_gzip_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _label_spans(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    spans: list[dict[str, Any]] = []
    start = 0
    for index in range(1, len(rows) + 1):
        if index < len(rows) and rows[index]["label"] == rows[start]["label"]:
            continue
        spans.append(
            {
                "conversation_id": rows[start]["conversation_id"],
                "start_s": rows[start]["start_s"],
                "end_s": rows[index - 1]["end_s"],
                "label": rows[start]["label"],
                "frame_count": index - start,
                "rule_version": RULE_VERSION,
            }
        )
        start = index
    return spans


def build_forced_fiveclass_labels(
    *,
    vad_frames_path: str | Path,
    vad_summary_path: str | Path,
    utterances_path: str | Path,
    output_dir: str | Path,
    minimum_overlap_ms: int = 40,
) -> dict[str, Any]:
    vad_summary = json.loads(Path(vad_summary_path).read_text(encoding="utf-8"))
    conversation_ids = list(vad_summary["conversation_ids"])
    eligible = set(conversation_ids)
    utterances: dict[str, list[Utterance]] = defaultdict(list)
    for row in read_jsonl(utterances_path):
        if row.get("conversation_id") in eligible and row.get("is_person", True):
            utterances[str(row["conversation_id"])].append(
                Utterance(
                    float(row["start_s"]),
                    float(row["end_s"]),
                    str(row["speaker"]),
                    str(row.get("text") or row.get("clean_text") or ""),
                )
            )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_frames: list[dict[str, Any]] = []
    output_spans: list[dict[str, Any]] = []
    per_conversation: list[dict[str, Any]] = []
    current_id: str | None = None
    current_rows: list[dict[str, Any]] = []

    def finish(conversation_id: str, rows: list[dict[str, Any]]) -> None:
        labeled = force_frame_labels(
            rows,
            utterances[conversation_id],
            frame_ms=int(vad_summary["frame_ms"]),
            minimum_overlap_ms=minimum_overlap_ms,
        )
        output_frames.extend(labeled)
        output_spans.extend(_label_spans(labeled))
        counts = Counter(row["label"] for row in labeled)
        per_conversation.append(
            {
                "conversation_id": conversation_id,
                "frames": len(labeled),
                "label_counts": {label: counts[label] for label in LABELS},
            }
        )

    for row in _read_gzip_jsonl(vad_frames_path):
        conversation_id = str(row["conversation_id"])
        if current_id is None:
            current_id = conversation_id
        if conversation_id != current_id:
            finish(current_id, current_rows)
            current_id, current_rows = conversation_id, []
        current_rows.append(row)
    if current_id is not None:
        finish(current_id, current_rows)

    _write_gzip_jsonl(destination / "frame_labels.jsonl.gz", output_frames)
    write_jsonl(destination / "label_spans.jsonl", output_spans)
    counts = Counter(row["label"] for row in output_frames)
    source_counts = Counter(row["label_source"] for row in output_frames)
    frame_ms = int(vad_summary["frame_ms"])
    summary = {
        "schema_version": "1.0",
        "task": "forced_five_class_weak_annotation",
        "rule_version": RULE_VERSION,
        "scope": vad_summary["scope"],
        "conversations": len(per_conversation),
        "conversation_ids": conversation_ids,
        "duration_hours": vad_summary["duration_hours"],
        "frame_ms": frame_ms,
        "total_frames": len(output_frames),
        "labeled_frames": sum(counts.values()),
        "unlabeled_frames": len(output_frames) - sum(counts.values()),
        "coverage": sum(counts.values()) / len(output_frames) if output_frames else 0.0,
        "labels": list(LABELS),
        "label_counts": {label: counts[label] for label in LABELS},
        "label_duration_seconds": {
            label: round(counts[label] * frame_ms / 1000.0, 3) for label in LABELS
        },
        "label_source_counts": dict(source_counts),
        "minimum_interruption_overlap_ms": minimum_overlap_ms,
        "precedence": ["NA", "BC", "I", "T", "C"],
        "rules": {
            "NA": "VAD majority silence",
            "BC": "VAD speech and short feedback lexicon utterance from the other speaker",
            "I": "VAD speech and different-speaker timed overlap lasting at least 40 ms, unless BC",
            "T": "VAD speech at the first 40 ms frame of a non-overlapping speaker change",
            "C": "all remaining VAD speech frames",
        },
        "review_states": False,
        "semantic_candidate_queue_used": False,
        "per_conversation": per_conversation,
        "outputs": {
            "frame_labels": str((destination / "frame_labels.jsonl.gz").resolve()),
            "label_spans": str((destination / "label_spans.jsonl").resolve()),
        },
    }
    write_json(destination / "summary.json", summary)
    return summary


__all__ = [
    "RULE_VERSION",
    "build_forced_fiveclass_labels",
    "build_rule_masks",
    "force_frame_labels",
]
