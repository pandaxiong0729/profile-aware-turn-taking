"""Acoustic-first SBCSAE annotation with explicit uncertainty provenance.

VAD is allowed to assign only high-confidence speech/silence-derived labels.
BC, turn transfer, and interruption semantics are exported for later audio-MLLM
adjudication instead of being guessed from transcript words.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .audio import read_wav_channels
from .utils import read_jsonl, write_json, write_jsonl


FRAME_MS = 40
REASON_VAD_DISAGREEMENT = "vad_channel_disagreement"
REASON_VAD_BOUNDARY = "near_vad_boundary"
REASON_TRN_BOUNDARY = "near_transcript_boundary"
REASON_SHORT = "short_utterance_semantics"
REASON_OVERLAP = "two_speaker_overlap_semantics"
REASON_CHANGE = "speaker_change_semantics"
REASON_CONFLICT = "vad_transcript_activity_conflict"


@dataclass(frozen=True)
class VadConfig:
    sample_rate: int = 16_000
    frame_ms: int = FRAME_MS
    threshold: float = 0.5
    min_speech_duration_ms: int = 64
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30
    semantic_short_s: float = 1.5
    vad_boundary_margin_ms: int = 120
    transcript_boundary_margin_ms: int = 200
    review_join_gap_ms: int = 200
    context_before_s: float = 12.0
    context_after_s: float = 8.0


def _frame_slice(start_s: float, end_s: float, frame_s: float, size: int) -> slice:
    start = max(0, min(size, int(math.floor(start_s / frame_s))))
    end = max(start, min(size, int(math.ceil(end_s / frame_s))))
    return slice(start, end)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not np.any(mask):
        return mask.copy()
    kernel = np.ones(radius * 2 + 1, dtype=np.int16)
    return np.convolve(mask.astype(np.int16), kernel, mode="same") > 0


def _transition_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    edges = np.zeros(mask.shape, dtype=bool)
    if mask.size > 1:
        edges[1:] = mask[1:] != mask[:-1]
    return _dilate(edges, radius)


def _runs(mask: np.ndarray, *, join_gap: int = 0) -> list[tuple[int, int]]:
    if mask.size == 0 or not np.any(mask):
        return []
    work = mask.copy()
    if join_gap > 0:
        true_indices = np.flatnonzero(work)
        for left, right in zip(true_indices[:-1], true_indices[1:]):
            if 1 < right - left <= join_gap + 1:
                work[left : right + 1] = True
    padded = np.pad(work.astype(np.int8), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    return [(int(changes[i]), int(changes[i + 1])) for i in range(0, len(changes), 2)]


def _vad_sources(channels: np.ndarray) -> list[tuple[str, np.ndarray]]:
    sources = [(f"channel_{index}", channel) for index, channel in enumerate(channels)]
    if channels.shape[0] > 1:
        sources.append(("channel_mean", channels.mean(axis=0).astype(np.float32)))
    return sources


def run_silero_vad(
    audio_path: str | Path,
    *,
    config: VadConfig,
    model: Any | None = None,
) -> tuple[float, list[str], list[dict[str, Any]], np.ndarray]:
    """Return duration, source names, segment rows, and 40 ms vote counts."""
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError as error:  # pragma: no cover - dependency guidance
        raise RuntimeError(
            "Install the annotation dependencies with: pip install silero-vad onnxruntime"
        ) from error

    channels = read_wav_channels(audio_path, target_rate=config.sample_rate)
    duration_s = channels.shape[1] / config.sample_rate
    frames = int(math.ceil(duration_s * 1000 / config.frame_ms))
    votes = np.zeros(frames, dtype=np.uint8)
    segment_rows: list[dict[str, Any]] = []
    loaded_model = model or load_silero_vad(onnx=True)
    sources = _vad_sources(channels)
    for source_name, samples in sources:
        timestamps = get_speech_timestamps(
            torch.from_numpy(np.ascontiguousarray(samples)),
            loaded_model,
            sampling_rate=config.sample_rate,
            threshold=config.threshold,
            min_speech_duration_ms=config.min_speech_duration_ms,
            min_silence_duration_ms=config.min_silence_duration_ms,
            speech_pad_ms=config.speech_pad_ms,
            return_seconds=True,
        )
        for segment in timestamps:
            start_s = float(segment["start"])
            end_s = float(segment["end"])
            votes[_frame_slice(start_s, end_s, config.frame_ms / 1000.0, frames)] += 1
            segment_rows.append(
                {
                    "source": source_name,
                    "start_s": round(start_s, 6),
                    "end_s": round(end_s, 6),
                }
            )
    return duration_s, [name for name, _ in sources], segment_rows, votes


def derive_frame_annotations(
    *,
    conversation_id: str,
    duration_s: float,
    source_names: Sequence[str],
    vad_votes: np.ndarray,
    utterances: Sequence[dict[str, Any]],
    config: VadConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fuse VAD votes with speaker timing without inventing semantic labels."""
    frame_s = config.frame_ms / 1000.0
    frames = vad_votes.size
    person_rows = sorted(
        (row for row in utterances if row.get("is_person", True)),
        key=lambda row: (float(row["start_s"]), float(row["end_s"])),
    )
    speaker_ids = sorted({str(row["speaker"]) for row in person_rows})
    if len(speaker_ids) != 2:
        raise ValueError(
            f"{conversation_id} must have exactly two observed person speakers, got {speaker_ids}"
        )
    speaker_bit = {speaker: 1 << index for index, speaker in enumerate(speaker_ids)}
    active_bits = np.zeros(frames, dtype=np.uint8)
    short_mask = np.zeros(frames, dtype=bool)
    trn_edges = np.zeros(frames, dtype=bool)
    speaker_change = np.zeros(frames, dtype=bool)
    row_flags: list[tuple[dict[str, Any], bool, bool]] = []

    previous_speaker: str | None = None
    for row in person_rows:
        start_s, end_s = float(row["start_s"]), float(row["end_s"])
        is_short = end_s - start_s <= config.semantic_short_s
        is_change = previous_speaker is not None and previous_speaker != str(row["speaker"])
        span = _frame_slice(start_s, end_s, frame_s, frames)
        active_bits[span] |= speaker_bit[str(row["speaker"])]
        if is_short:
            short_mask[span] = True
        for point in (start_s, end_s):
            index = min(frames - 1, max(0, int(round(point / frame_s))))
            trn_edges[index] = True
        if is_change:
            index = min(frames - 1, max(0, int(math.floor(start_s / frame_s))))
            speaker_change[index] = True
        row_flags.append((row, is_short, is_change))
        previous_speaker = str(row["speaker"])

    trn_boundary = _dilate(
        trn_edges, int(math.ceil(config.transcript_boundary_margin_ms / config.frame_ms))
    )
    speaker_change = _dilate(speaker_change, 1)
    overlap = active_bits == sum(speaker_bit.values())
    source_count = len(source_names)
    consensus_threshold = 1 if source_count == 1 else int(math.ceil(source_count / 2))
    vad_speech = vad_votes >= consensus_threshold
    full_vad_speech = vad_votes == source_count
    full_vad_silence = vad_votes == 0
    vad_disagreement = (vad_votes > 0) & (vad_votes < source_count)
    vad_boundary = _transition_mask(
        vad_speech, int(math.ceil(config.vad_boundary_margin_ms / config.frame_ms))
    )
    trn_active = active_bits != 0
    conflict = vad_speech != trn_active
    exactly_one_speaker = np.isin(active_bits, list(speaker_bit.values()))

    high_na = full_vad_silence & ~vad_boundary
    high_c = (
        full_vad_speech
        & exactly_one_speaker
        & ~short_mask
        & ~overlap
        & ~speaker_change
        & ~trn_boundary
        & ~vad_boundary
    )
    semantic_review = ~high_na & ~high_c & (short_mask | overlap | speaker_change)
    acoustic_review = ~high_na & ~high_c & ~semantic_review

    rows: list[dict[str, Any]] = []
    reason_masks = [
        (REASON_VAD_DISAGREEMENT, vad_disagreement),
        (REASON_VAD_BOUNDARY, vad_boundary),
        (REASON_TRN_BOUNDARY, trn_boundary),
        (REASON_SHORT, short_mask),
        (REASON_OVERLAP, overlap),
        (REASON_CHANGE, speaker_change),
        (REASON_CONFLICT, conflict),
    ]
    for index in range(frames):
        if high_na[index]:
            status, label = "high_confidence", "NA"
        elif high_c[index]:
            status, label = "high_confidence", "C"
        elif semantic_review[index]:
            status, label = "needs_semantic_review", None
        else:
            status, label = "needs_acoustic_review", None
        active_speakers = [
            speaker for speaker, bit in speaker_bit.items() if active_bits[index] & bit
        ]
        reasons = [name for name, mask in reason_masks if mask[index]]
        rows.append(
            {
                "conversation_id": conversation_id,
                "frame_index": index,
                "start_s": round(index * frame_s, 6),
                "end_s": round(min(duration_s, (index + 1) * frame_s), 6),
                "vad_votes": int(vad_votes[index]),
                "vad_source_count": source_count,
                "vad_state": "speech" if vad_speech[index] else "silence",
                "trn_active_speakers": active_speakers,
                "automatic_label": label,
                "annotation_status": status,
                "review_reasons": reasons,
            }
        )

    high_spans: list[dict[str, Any]] = []
    for label, mask in (("C", high_c), ("NA", high_na)):
        for start, end in _runs(mask):
            high_spans.append(
                {
                    "conversation_id": conversation_id,
                    "start_s": round(start * frame_s, 6),
                    "end_s": round(min(duration_s, end * frame_s), 6),
                    "label": label,
                    "annotation_status": "high_confidence",
                    "source": "silero_vad_channel_consensus+speaker_timing_constraints",
                    "frame_count": end - start,
                }
            )

    # Event-level review targets are anchored to a specific short utterance,
    # speaker-change onset, or overlap onset.  Do not merge a chain of nearby
    # events into a long ambiguous target: the MLLM must know exactly which
    # local behavior it is judging.
    seeds: list[dict[str, Any]] = []

    def add_seed(start_s: float, end_s: float, reason: str) -> None:
        start_s = max(0.0, start_s)
        end_s = min(duration_s, max(start_s + frame_s, end_s))
        for seed in seeds:
            if start_s <= seed["end_s"] + 0.2 and end_s >= seed["start_s"] - 0.2:
                merged_start = min(seed["start_s"], start_s)
                merged_end = max(seed["end_s"], end_s)
                if merged_end - merged_start > 2.5:
                    continue
                if reason == REASON_CHANGE and not (
                    seed["start_s"] - 0.2 <= start_s <= seed["end_s"] + 0.2
                ):
                    continue
                seed["start_s"] = merged_start
                seed["end_s"] = merged_end
                seed["reasons"].add(reason)
                return
        seeds.append({"start_s": start_s, "end_s": end_s, "reasons": {reason}})

    for row, is_short, is_change in row_flags:
        start_s, end_s = float(row["start_s"]), float(row["end_s"])
        if is_short:
            add_seed(start_s, end_s, REASON_SHORT)
        if is_change:
            add_seed(start_s, min(end_s, start_s + 2.0), REASON_CHANGE)
    for start, end in _runs(overlap):
        overlap_start = start * frame_s
        overlap_end = min(duration_s, end * frame_s)
        add_seed(overlap_start, min(overlap_end, overlap_start + 2.0), REASON_OVERLAP)

    semantic_events: list[dict[str, Any]] = []
    for event_index, seed in enumerate(sorted(seeds, key=lambda item: item["start_s"])):
        interval_reasons = sorted(seed["reasons"])
        if REASON_OVERLAP in interval_reasons:
            candidate_labels = ["BC", "I"]
        elif REASON_CHANGE in interval_reasons:
            candidate_labels = ["BC", "T"]
        else:
            candidate_labels = ["BC", "C", "T"]
        start_s, end_s = float(seed["start_s"]), float(seed["end_s"])
        semantic_events.append(
            {
                "event_id": f"{conversation_id}-semantic-{event_index:06d}",
                "conversation_id": conversation_id,
                "target_start_s": round(start_s, 6),
                "target_end_s": round(end_s, 6),
                "context_start_s": round(max(0.0, start_s - config.context_before_s), 6),
                "context_end_s": round(min(duration_s, end_s + config.context_after_s), 6),
                "candidate_labels": candidate_labels,
                "review_reasons": interval_reasons,
                "annotation_status": "needs_semantic_review",
                "model_input_policy": "raw_audio_before+target+after; no profile; no weak label",
            }
        )
    return rows, high_spans, semantic_events


def _write_gzip_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_vad_annotations(
    *,
    catalog_dir: str | Path,
    scope_summary: str | Path,
    output_dir: str | Path,
    config: VadConfig | None = None,
    max_conversations: int | None = None,
) -> dict[str, Any]:
    """Run VAD on the eligible dyadic scope and write auditable artifacts."""
    cfg = config or VadConfig()
    catalog = Path(catalog_dir)
    scope = json.loads(Path(scope_summary).read_text(encoding="utf-8"))
    eligible = list(scope["eligible_conversation_ids"])
    if max_conversations is not None:
        eligible = eligible[:max_conversations]
    eligible_set = set(eligible)
    conversations = {
        str(row["conversation_id"]): row
        for row in read_jsonl(catalog / "conversations.jsonl")
        if row.get("conversation_id") in eligible_set
    }
    utterances: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(catalog / "utterances.jsonl"):
        if row.get("conversation_id") in eligible_set and row.get("is_person", True):
            utterances[str(row["conversation_id"])].append(row)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    all_segments: list[dict[str, Any]] = []
    all_frames: list[dict[str, Any]] = []
    all_high_spans: list[dict[str, Any]] = []
    all_semantic_events: list[dict[str, Any]] = []
    conversation_stats: list[dict[str, Any]] = []
    from silero_vad import load_silero_vad

    model = load_silero_vad(onnx=True)
    for conversation_id in eligible:
        conversation = conversations[conversation_id]
        duration_s, sources, segments, votes = run_silero_vad(
            conversation["audio_path"], config=cfg, model=model
        )
        for segment in segments:
            all_segments.append(
                {"conversation_id": conversation_id, **segment, "model": "silero_vad_6.2.1"}
            )
        frames, high_spans, semantic_events = derive_frame_annotations(
            conversation_id=conversation_id,
            duration_s=duration_s,
            source_names=sources,
            vad_votes=votes,
            utterances=utterances[conversation_id],
            config=cfg,
        )
        all_frames.extend(frames)
        all_high_spans.extend(high_spans)
        all_semantic_events.extend(semantic_events)
        counts = Counter(row["annotation_status"] for row in frames)
        labels = Counter(row["automatic_label"] for row in frames if row["automatic_label"])
        conversation_stats.append(
            {
                "conversation_id": conversation_id,
                "duration_s": round(duration_s, 6),
                "vad_sources": sources,
                "frame_count": len(frames),
                "status_counts": dict(counts),
                "automatic_label_counts": dict(labels),
                "semantic_events": len(semantic_events),
            }
        )

    write_jsonl(destination / "vad_segments.jsonl", all_segments)
    _write_gzip_jsonl(destination / "frame_annotations.jsonl.gz", all_frames)
    write_jsonl(destination / "high_confidence_spans.jsonl", all_high_spans)
    write_jsonl(destination / "semantic_review_queue.jsonl", all_semantic_events)
    total_status = Counter(row["annotation_status"] for row in all_frames)
    total_labels = Counter(row["automatic_label"] for row in all_frames if row["automatic_label"])
    summary = {
        "schema_version": "1.0",
        "task": "acoustic_first_five_class_annotation",
        "scope": "core_dyadic",
        "conversations": len(eligible),
        "conversation_ids": eligible,
        "duration_hours": sum(item["duration_s"] for item in conversation_stats) / 3600.0,
        "frame_ms": cfg.frame_ms,
        "vad_model": "silero-vad 6.2.1 ONNX",
        "status_counts": dict(total_status),
        "automatic_label_counts": dict(total_labels),
        "semantic_review_events": len(all_semantic_events),
        "label_policy": {
            "C": "unanimous channel VAD speech; exactly one timed speaker; not short, overlap, change, or boundary",
            "NA": "unanimous channel VAD silence away from a VAD boundary",
            "BC_T_I": "never assigned by VAD; exported to semantic_review_queue",
        },
        "important_limitations": [
            "VAD is not a semantic annotator and cannot distinguish BC, T, and I.",
            "Transcript speaker intervals are used only as timing constraints, not as BC labels.",
            "High-confidence labels remain automatic silver labels until a human audit estimates their error rate.",
        ],
        "config": cfg.__dict__,
        "conversation_stats": conversation_stats,
        "outputs": {
            "vad_segments": str((destination / "vad_segments.jsonl").resolve()),
            "frame_annotations": str((destination / "frame_annotations.jsonl.gz").resolve()),
            "high_confidence_spans": str((destination / "high_confidence_spans.jsonl").resolve()),
            "semantic_review_queue": str((destination / "semantic_review_queue.jsonl").resolve()),
        },
    }
    write_json(destination / "summary.json", summary)
    return summary


__all__ = [
    "VadConfig",
    "build_vad_annotations",
    "derive_frame_annotations",
    "run_silero_vad",
]
