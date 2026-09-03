"""Build leakage-safe five-class SBCSAE manifests from the normalized catalog."""

from __future__ import annotations

import copy
import heapq
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from .constants import LABELS
from .data import clean_transcript_text, is_backchannel
from .schemas import Utterance
from .utils import read_jsonl, write_json, write_jsonl


@dataclass(frozen=True)
class Candidate:
    conversation_id: str
    split: str
    prediction_time_s: float
    label: str


class MonotonicWeakLabeler:
    """Efficiently reproduce the five-class TRN heuristic on increasing time points."""

    def __init__(self, utterances: Sequence[Utterance], horizon_ms: int) -> None:
        self.utterances = sorted(utterances, key=lambda row: (row.start_s, row.end_s))
        self.horizon_s = horizon_ms / 1000.0
        self.backchannels = [is_backchannel(row) for row in self.utterances]
        self.previous_speaker = self._previous_floor_speakers()
        self.next_start = 0
        self.active: list[int] = []
        self.last_time = float("-inf")

    def _previous_floor_speakers(self) -> list[str | None]:
        """Match data._previous_floor at every utterance onset in O(n log n)."""

        result: list[str | None] = [None] * len(self.utterances)
        active_by_index: list[int] = []
        latest_by_end: list[tuple[float, int]] = []
        group_start = 0
        while group_start < len(self.utterances):
            time_s = self.utterances[group_start].start_s
            group_end = group_start + 1
            while (
                group_end < len(self.utterances)
                and self.utterances[group_end].start_s == time_s
            ):
                group_end += 1
            while (
                active_by_index
                and self.utterances[active_by_index[0]].end_s < time_s
            ):
                heapq.heappop(active_by_index)
            previous_index = (
                active_by_index[0]
                if active_by_index
                else latest_by_end[0][1]
                if latest_by_end
                else None
            )
            previous = (
                self.utterances[previous_index].speaker
                if previous_index is not None
                else None
            )
            for index in range(group_start, group_end):
                result[index] = previous
            for index in range(group_start, group_end):
                if not self.backchannels[index]:
                    heapq.heappush(active_by_index, index)
                    heapq.heappush(
                        latest_by_end, (-self.utterances[index].end_s, index)
                    )
            group_start = group_end
        return result

    def label(self, prediction_time_s: float) -> str:
        if prediction_time_s < self.last_time:
            raise ValueError("MonotonicWeakLabeler requires non-decreasing time points")
        self.last_time = prediction_time_s
        horizon_end = prediction_time_s + self.horizon_s
        while (
            self.next_start < len(self.utterances)
            and self.utterances[self.next_start].start_s < horizon_end
        ):
            self.active.append(self.next_start)
            self.next_start += 1
        self.active = [
            index for index in self.active if self.utterances[index].end_s > prediction_time_s
        ]
        for index in self.active:
            row = self.utterances[index]
            previous = self.previous_speaker[index]
            if self.backchannels[index] and previous and previous != row.speaker:
                return "BC"
        # Two utterances may both intersect the 40 ms chunk without
        # intersecting each other (A ends, then B starts).  That is a turn
        # transition, not overlapping speech.  Require true pairwise overlap.
        for position, first_index in enumerate(self.active):
            first = self.utterances[first_index]
            for second_index in self.active[position + 1 :]:
                second = self.utterances[second_index]
                if first.speaker == second.speaker:
                    continue
                overlap_start = max(
                    prediction_time_s, first.start_s, second.start_s
                )
                overlap_end = min(horizon_end, first.end_s, second.end_s)
                if overlap_start < overlap_end:
                    return "I"
        for index in self.active:
            row = self.utterances[index]
            previous = self.previous_speaker[index]
            if (
                not self.backchannels[index]
                and prediction_time_s - 1e-9 <= row.start_s < horizon_end
                and previous
                and previous != row.speaker
            ):
                return "T"
        return "C" if self.active else "NA"


def _group_splits(groups: Sequence[str], seed: int) -> dict[str, str]:
    shuffled = sorted(set(groups))
    if len(shuffled) < 3:
        raise ValueError("At least three speaker-connected groups are required")
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    minimum_holdout = 3 if count >= 10 else 1
    val_count = max(minimum_holdout, round(count * 0.15))
    test_count = max(minimum_holdout, round(count * 0.15))
    while val_count + test_count >= count:
        if val_count >= test_count and val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
    train_end = count - val_count - test_count
    val_end = train_end + val_count
    result = {group: "train" for group in shuffled[:train_end]}
    result.update({group: "val" for group in shuffled[train_end:val_end]})
    result.update({group: "test" for group in shuffled[val_end:]})
    return result


def _reservoir_add(
    reservoir: list[Candidate],
    candidate: Candidate,
    *,
    seen: int,
    limit: int,
    rng: random.Random,
) -> None:
    if len(reservoir) < limit:
        reservoir.append(candidate)
        return
    replacement = rng.randrange(seen)
    if replacement < limit:
        reservoir[replacement] = candidate


def _canonical_conversation(
    conversation: dict[str, Any], rows: Sequence[dict[str, Any]]
) -> tuple[list[Utterance], dict[str, Any], dict[str, str]]:
    person_rows = [row for row in rows if row["is_person"]]
    first_seen: list[str] = []
    for row in person_rows:
        if row["speaker"] not in first_seen:
            first_seen.append(row["speaker"])
    if len(first_seen) != 2:
        raise ValueError(
            f"{conversation['conversation_id']} is not an observed dyadic conversation"
        )
    speaker_mapping = {first_seen[0]: "speaker_A", first_seen[1]: "speaker_B"}
    participant_by_local = {
        row["local_speaker_id"]: row for row in conversation["participants"] if row["is_person"]
    }
    utterances = [
        Utterance(
            start_s=float(row["start_s"]),
            end_s=float(row["end_s"]),
            speaker=speaker_mapping[row["speaker"]],
            text=row["text"],
        )
        for row in person_rows
    ]
    profile = {
        "speaker_A": copy.deepcopy(participant_by_local[first_seen[0]]["profile"]),
        "speaker_B": copy.deepcopy(participant_by_local[first_seen[1]]["profile"]),
        "relationship": conversation["relationship"],
        "situation": conversation["situation"],
    }
    provenance = {
        "speaker_A": participant_by_local[first_seen[0]]["metadata_match_status"],
        "speaker_B": participant_by_local[first_seen[1]]["metadata_match_status"],
        "relationship_situation": conversation["context_mapping_confidence"],
    }
    return utterances, profile, provenance


def _transcript_prefix(
    utterances: Sequence[Utterance], start_s: float, end_s: float
) -> str:
    lines = [
        f"[{row.speaker} {row.start_s:.2f}-{row.end_s:.2f}] "
        f"{clean_transcript_text(row.text)}"
        for row in utterances
        if row.end_s <= end_s + 1e-9 and row.end_s >= start_s
    ]
    return " ".join(lines)


def _iter_grid(
    *, start_s: float, end_s: float, stride_ms: int
) -> Iterator[tuple[int, float]]:
    step = stride_ms / 1000.0
    index = 0
    time_s = start_s
    while time_s + 1e-9 < end_s:
        yield index, round(time_s, 6)
        index += 1
        time_s = start_s + index * step


def _sample_row(
    candidate: Candidate,
    *,
    conversation: dict[str, Any],
    utterances: Sequence[Utterance],
    profile: dict[str, Any],
    provenance: dict[str, str],
    context_seconds: float,
    horizon_ms: int,
) -> dict[str, Any]:
    prediction_time = candidate.prediction_time_s
    return {
        "sample_id": (
            f"{candidate.conversation_id}-{int(round(prediction_time * 1000)):09d}"
        ),
        "conversation_id": candidate.conversation_id,
        "split_group": conversation["split_group"],
        "split": candidate.split,
        "prediction_time_s": prediction_time,
        "horizon_ms": horizon_ms,
        "window_start_s": round(prediction_time - context_seconds, 6),
        "window_end_s": prediction_time,
        "audio_path": conversation["audio_path"],
        "audio_channel_policy": "mean_stereo_channels_at_load_time",
        "transcript_prefix": _transcript_prefix(
            utterances, prediction_time - context_seconds, prediction_time
        ),
        "text_source": "manual_trn_completed_units_not_streaming_asr",
        "profile": copy.deepcopy(profile),
        "profile_provenance": copy.deepcopy(provenance),
        "label": candidate.label,
        "label_source": "automatic_weak_chunk_state_from_trn_timestamps_v2",
        "gold_label": False,
    }


def _event_representative_time(
    event: dict[str, Any], *, frame_stride_ms: int, policy: str = "midpoint_grid"
) -> float:
    """Choose one observed grid frame near an event's midpoint."""

    if policy == "onset":
        return round(float(event["start_s"]), 6)
    if policy != "midpoint_grid":
        raise ValueError(f"Unknown event representative policy: {policy}")
    step = frame_stride_ms / 1000.0
    frame_count = max(1, int(round((event["end_s"] - event["start_s"]) / step)))
    return round(event["start_s"] + ((frame_count - 1) // 2) * step, 6)


def prepare_sbcsae_manifests(
    *,
    catalog_dir: str | Path,
    output_dir: str | Path,
    context_seconds: float = 30.0,
    horizon_ms: int = 40,
    frame_stride_ms: int = 40,
    evaluation_stride_ms: int = 200,
    max_train_per_class: int = 10000,
    max_evaluation_per_class: int = 5000,
    seed: int = 13,
    include_non_core_dyadic: bool = False,
) -> dict[str, Any]:
    source = Path(catalog_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    conversations = list(read_jsonl(source / "conversations.jsonl"))
    utterances_by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(source / "utterances.jsonl"):
        utterances_by_conversation[row["conversation_id"]].append(row)
    eligible = [
        row
        for row in conversations
        if row["observed_dyadic"]
        and (include_non_core_dyadic or row.get("core_dyadic", False))
        and row.get("audio_path")
    ]
    split_by_group = _group_splits([row["split_group"] for row in eligible], seed)
    rng = random.Random(seed)
    train_reservoirs: dict[str, list[Candidate]] = {label: [] for label in LABELS}
    train_seen: Counter[str] = Counter()
    eval_reservoirs: dict[tuple[str, str], list[Candidate]] = {
        (split, label): [] for split in ("val", "test") for label in LABELS
    }
    eval_seen: Counter[tuple[str, str]] = Counter()
    frame_counts: Counter[tuple[str, str]] = Counter()
    weak_events: list[dict[str, Any]] = []
    prepared: dict[str, tuple[list[Utterance], dict[str, Any], dict[str, str]]] = {}

    evaluation_multiple = max(1, round(evaluation_stride_ms / frame_stride_ms))
    for conversation in eligible:
        conversation_id = conversation["conversation_id"]
        split = split_by_group[conversation["split_group"]]
        canonical = _canonical_conversation(
            conversation, utterances_by_conversation[conversation_id]
        )
        prepared[conversation_id] = canonical
        rows, _, _ = canonical
        labeler = MonotonicWeakLabeler(rows, horizon_ms)
        end_s = min(
            float(conversation["duration_s"]),
            float(conversation["audio_info"]["duration_s"]),
        )
        event_label: str | None = None
        event_start = context_seconds
        last_time = context_seconds
        for frame_index, time_s in _iter_grid(
            start_s=context_seconds,
            end_s=end_s - horizon_ms / 1000.0,
            stride_ms=frame_stride_ms,
        ):
            label = labeler.label(time_s)
            frame_counts[(split, label)] += 1
            candidate = Candidate(conversation_id, split, time_s, label)
            if split == "train":
                train_seen[label] += 1
                _reservoir_add(
                    train_reservoirs[label],
                    candidate,
                    seen=train_seen[label],
                    limit=max_train_per_class,
                    rng=rng,
                )
            elif label != "C" or frame_index % evaluation_multiple == 0:
                eval_key = (split, label)
                eval_seen[eval_key] += 1
                _reservoir_add(
                    eval_reservoirs[eval_key],
                    candidate,
                    seen=eval_seen[eval_key],
                    limit=max_evaluation_per_class,
                    rng=rng,
                )
            if event_label is None:
                event_label = label
                event_start = time_s
            elif label != event_label:
                weak_events.append(
                    {
                        "conversation_id": conversation_id,
                        "split": split,
                        "start_s": event_start,
                        "end_s": time_s,
                        "label": event_label,
                        "source": "automatic_weak_chunk_state_from_trn_timestamps_v2",
                        "gold_label": False,
                    }
                )
                event_start = time_s
                event_label = label
            last_time = time_s
        if event_label is not None:
            weak_events.append(
                {
                    "conversation_id": conversation_id,
                    "split": split,
                    "start_s": event_start,
                    "end_s": last_time + frame_stride_ms / 1000.0,
                    "label": event_label,
                    "source": "automatic_weak_chunk_state_from_trn_timestamps_v2",
                    "gold_label": False,
                }
            )

    selected_candidates = [
        candidate for label in LABELS for candidate in train_reservoirs[label]
    ]
    selected_candidates.extend(
        candidate
        for split in ("val", "test")
        for label in LABELS
        for candidate in eval_reservoirs[(split, label)]
    )
    selected_candidates.sort(
        key=lambda row: (row.split, row.conversation_id, row.prediction_time_s)
    )
    conversation_by_id = {row["conversation_id"]: row for row in eligible}
    manifest_rows: list[dict[str, Any]] = []
    for candidate in selected_candidates:
        conversation = conversation_by_id[candidate.conversation_id]
        rows, profile, provenance = prepared[candidate.conversation_id]
        manifest_rows.append(
            _sample_row(
                candidate,
                conversation=conversation,
                utterances=rows,
                profile=profile,
                provenance=provenance,
                context_seconds=context_seconds,
                horizon_ms=horizon_ms,
            )
        )
    rng.shuffle(manifest_rows)
    write_jsonl(destination / "manifest.jsonl", manifest_rows)
    write_jsonl(destination / "weak_events.jsonl", weak_events)
    event_manifest_rows: list[dict[str, Any]] = []
    event_onset_manifest_rows: list[dict[str, Any]] = []
    for event_index, event in enumerate(weak_events):
        conversation = conversation_by_id[event["conversation_id"]]
        rows, profile, provenance = prepared[event["conversation_id"]]
        candidate = Candidate(
            event["conversation_id"],
            event["split"],
            _event_representative_time(event, frame_stride_ms=frame_stride_ms),
            event["label"],
        )
        row = _sample_row(
            candidate,
            conversation=conversation,
            utterances=rows,
            profile=profile,
            provenance=provenance,
            context_seconds=context_seconds,
            horizon_ms=horizon_ms,
        )
        row.update(
            {
                "weak_event_id": f"{event['conversation_id']}-event-{event_index:06d}",
                "weak_event_start_s": event["start_s"],
                "weak_event_end_s": event["end_s"],
                "event_representative": True,
                "event_representative_policy": "midpoint_grid",
            }
        )
        event_manifest_rows.append(row)
        onset_candidate = Candidate(
            event["conversation_id"],
            event["split"],
            _event_representative_time(
                event, frame_stride_ms=frame_stride_ms, policy="onset"
            ),
            event["label"],
        )
        onset_row = _sample_row(
            onset_candidate,
            conversation=conversation,
            utterances=rows,
            profile=profile,
            provenance=provenance,
            context_seconds=context_seconds,
            horizon_ms=horizon_ms,
        )
        onset_row.update(
            {
                "weak_event_id": f"{event['conversation_id']}-event-{event_index:06d}",
                "weak_event_start_s": event["start_s"],
                "weak_event_end_s": event["end_s"],
                "event_representative": True,
                "event_representative_policy": "onset",
            }
        )
        event_onset_manifest_rows.append(onset_row)
    write_jsonl(destination / "event_manifest.jsonl", event_manifest_rows)
    write_jsonl(destination / "event_onset_manifest.jsonl", event_onset_manifest_rows)
    write_json(
        destination / "split_map.json",
        {
            "seed": seed,
            "split_unit": "speaker_connected_component",
            "groups": split_by_group,
        },
    )
    selected_counts = Counter((row["split"], row["label"]) for row in manifest_rows)
    summary = {
        "schema_version": "1.0",
        "task": "five_class_profile_conditioned_next_turn_event_prediction",
        "labels": list(LABELS),
        "eligible_scope": "observed_dyadic" if include_non_core_dyadic else "core_dyadic",
        "eligible_conversations": len(eligible),
        "eligible_conversation_ids": [row["conversation_id"] for row in eligible],
        "speaker_connected_groups": len(split_by_group),
        "conversation_splits": dict(
            Counter(split_by_group[row["split_group"]] for row in eligible)
        ),
        "context_seconds": context_seconds,
        "horizon_ms": horizon_ms,
        "frame_stride_ms": frame_stride_ms,
        "evaluation_stride_ms": evaluation_stride_ms,
        "evaluation_sampling": (
            "all BC/T/I/NA candidates with a per-class cap; C sampled at evaluation stride"
        ),
        "full_weak_frame_counts": {
            split: {label: frame_counts[(split, label)] for label in LABELS}
            for split in ("train", "val", "test")
        },
        "selected_manifest_counts": {
            split: {label: selected_counts[(split, label)] for label in LABELS}
            for split in ("train", "val", "test")
        },
        "selected_samples": len(manifest_rows),
        "weak_events": len(weak_events),
        "event_manifest_counts": {
            split: {
                label: sum(
                    row["split"] == split and row["label"] == label
                    for row in event_manifest_rows
                )
                for label in LABELS
            }
            for split in ("train", "val", "test")
        },
        "profile_conditions_supported": ["given", "hidden", "shuffled"],
        "audio_channel_policy": "mean_stereo_channels_at_load_time",
        "known_limitations": [
            "TRN intervals are intonation units, not frame-accurate VAD labels.",
            "BC is lexical-duration weak supervision and needs manual audit.",
            "I is overlap weak supervision and is not split into cooperative/competitive.",
            "NA combines within-turn pause and between-turn gap until VAD refinement.",
            "Manual TRN text is a causal proxy; final runs should use streaming ASR prefixes.",
        ],
        "outputs": {
            "manifest": str((destination / "manifest.jsonl").resolve()),
            "weak_events": str((destination / "weak_events.jsonl").resolve()),
            "event_manifest": str((destination / "event_manifest.jsonl").resolve()),
            "event_onset_manifest": str(
                (destination / "event_onset_manifest.jsonl").resolve()
            ),
            "split_map": str((destination / "split_map.json").resolve()),
        },
    }
    write_json(destination / "summary.json", summary)
    return summary
