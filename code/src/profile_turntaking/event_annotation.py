"""Event-centred SBCSAE annotation preparation.

The dense 40 ms labels are deliberately not an input to this module.  It first
builds speaker IPUs by intersecting timed SBCSAE speaker rows with acoustic VAD
activity, then proposes one C/BC/T/I/NA label at each meaningful interaction
event.  These labels are candidates for human review, never human gold labels.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .audio import read_wav_window_robust_mix, write_wav_mono
from .constants import BACKCHANNEL_WORDS, LABELS
from .utils import read_jsonl, write_json, write_jsonl


RULE_VERSION = "sbcsae_event_candidates_v3_floor_safeguards"


def merge_intervals(
    intervals: Iterable[tuple[float, float]], *, max_gap_s: float = 0.0
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start - merged[-1][1] > max_gap_s + 1e-9:
            merged.append([float(start), float(end)])
        else:
            merged[-1][1] = max(merged[-1][1], float(end))
    return [(start, end) for start, end in merged]


def intersect_interval(
    start_s: float,
    end_s: float,
    intervals: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    output: list[tuple[float, float]] = []
    for other_start, other_end in intervals:
        if other_end <= start_s:
            continue
        if other_start >= end_s:
            break
        start = max(start_s, other_start)
        end = min(end_s, other_end)
        if end > start:
            output.append((start, end))
    return output


def complement_intervals(
    intervals: Sequence[tuple[float, float]], duration_s: float
) -> list[tuple[float, float]]:
    output: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in merge_intervals(intervals):
        start = max(0.0, min(duration_s, start))
        end = max(start, min(duration_s, end))
        if start > cursor:
            output.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration_s:
        output.append((cursor, duration_s))
    return output


def _clean_words(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def build_ipus(
    utterances: Sequence[dict[str, Any]],
    vad_intervals: Sequence[tuple[float, float]],
    *,
    ipu_silence_s: float = 0.2,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Intersect timed speaker rows with VAD and merge into speaker IPUs."""
    fragments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    transcript_only_rows = 0
    for row in utterances:
        if not row.get("is_person", True):
            continue
        start = float(row["start_s"])
        end = float(row["end_s"])
        pieces = intersect_interval(start, end, vad_intervals)
        if not pieces:
            # Do not invent acoustic activity when VAD found none.  Retain the
            # count so the extraction report exposes possible missed speech.
            transcript_only_rows += 1
            continue
        for piece_start, piece_end in pieces:
            fragments[str(row["speaker"])].append(
                {
                    "start_s": piece_start,
                    "end_s": piece_end,
                    "utterance_id": row.get("utterance_id", ""),
                    "clean_text": row.get("clean_text", ""),
                }
            )

    ipus: list[dict[str, Any]] = []
    for speaker, rows in sorted(fragments.items()):
        rows.sort(key=lambda item: (item["start_s"], item["end_s"]))
        current: dict[str, Any] | None = None
        for row in rows:
            if current is None or row["start_s"] - current["end_s"] > ipu_silence_s + 1e-9:
                if current is not None:
                    ipus.append(current)
                current = {
                    "speaker": speaker,
                    "start_s": float(row["start_s"]),
                    "end_s": float(row["end_s"]),
                    "utterance_ids": [row["utterance_id"]] if row["utterance_id"] else [],
                    "text_parts": [row["clean_text"]] if row["clean_text"] else [],
                }
            else:
                current["end_s"] = max(float(current["end_s"]), float(row["end_s"]))
                if row["utterance_id"] and row["utterance_id"] not in current["utterance_ids"]:
                    current["utterance_ids"].append(row["utterance_id"])
                if row["clean_text"] and row["clean_text"] not in current["text_parts"]:
                    current["text_parts"].append(row["clean_text"])
        if current is not None:
            ipus.append(current)

    ipus.sort(key=lambda item: (item["start_s"], item["end_s"], item["speaker"]))
    for index, ipu in enumerate(ipus):
        ipu["ipu_id"] = f"ipu-{index:06d}"
        ipu["duration_s"] = round(float(ipu["end_s"]) - float(ipu["start_s"]), 6)
        ipu["text"] = " ".join(ipu.pop("text_parts"))
        ipu["source"] = "sbcsae_timed_speaker_intersect_vad"
    return ipus, {"transcript_rows_without_vad": transcript_only_rows}


def _overlap_s(left: dict[str, Any], right: dict[str, Any]) -> float:
    return max(
        0.0,
        min(float(left["end_s"]), float(right["end_s"]))
        - max(float(left["start_s"]), float(right["start_s"])),
    )


def _nearest_before(
    ipus: Sequence[dict[str, Any]], time_s: float, *, speaker: str | None = None
) -> dict[str, Any] | None:
    candidates = [
        item
        for item in ipus
        if float(item["end_s"]) <= time_s + 1e-9
        and (speaker is None or item["speaker"] == speaker)
    ]
    return max(candidates, key=lambda item: (item["end_s"], item["start_s"])) if candidates else None


def _nearest_after(
    ipus: Sequence[dict[str, Any]], time_s: float, *, speaker: str | None = None
) -> dict[str, Any] | None:
    candidates = [
        item
        for item in ipus
        if float(item["start_s"]) >= time_s - 1e-9
        and (speaker is None or item["speaker"] == speaker)
    ]
    return min(candidates, key=lambda item: (item["start_s"], item["end_s"])) if candidates else None


def identify_backchannel_candidates(
    ipus: Sequence[dict[str, Any]],
    *,
    structural_max_s: float = 1.0,
    lexical_max_s: float = 1.5,
    pre_isolation_s: float = 1.0,
    post_isolation_s: float = 2.0,
    floor_context_s: float = 2.0,
) -> dict[str, dict[str, Any]]:
    """Find short listener responses; this remains a candidate, not truth."""
    output: dict[str, dict[str, Any]] = {}
    for current in ipus:
        speaker = str(current["speaker"])
        duration = float(current["end_s"]) - float(current["start_s"])
        same_before = _nearest_before(ipus, float(current["start_s"]), speaker=speaker)
        same_after = _nearest_after(ipus, float(current["end_s"]), speaker=speaker)
        pre_gap = (
            float(current["start_s"]) - float(same_before["end_s"])
            if same_before is not None
            else math.inf
        )
        post_gap = (
            float(same_after["start_s"]) - float(current["end_s"])
            if same_after is not None
            else math.inf
        )
        others = [item for item in ipus if item["speaker"] != speaker]
        overlapping = [item for item in others if _overlap_s(current, item) > 0.0]
        before = _nearest_before(others, float(current["start_s"]))
        after = _nearest_after(others, float(current["end_s"]))
        enclosed_by_other_floor = bool(
            before
            and after
            and before["speaker"] == after["speaker"]
            and float(current["start_s"]) - float(before["end_s"]) <= floor_context_s
            and float(after["start_s"]) - float(current["end_s"]) <= floor_context_s
        )
        inside_other_turn = bool(overlapping or enclosed_by_other_floor)
        text = _clean_words(current.get("text", ""))
        lexical = bool(text and text in BACKCHANNEL_WORDS and duration <= lexical_max_s)
        simultaneous_empty_other = any(
            abs(float(item["start_s"]) - float(current["start_s"])) <= 1e-6
            and not _clean_words(item.get("text", ""))
            for item in others
        )
        structural = bool(
            duration <= structural_max_s
            and pre_gap >= pre_isolation_s
            and post_gap >= post_isolation_s
            and inside_other_turn
            and not (bool(text) and simultaneous_empty_other)
        )
        if not inside_other_turn or not (lexical or structural):
            continue
        floor_speaker = None
        if overlapping:
            floor_speaker = min(overlapping, key=lambda item: item["start_s"])["speaker"]
        elif enclosed_by_other_floor and before is not None:
            floor_speaker = before["speaker"]
        confidence = "high" if lexical and structural else "medium"
        output[str(current["ipu_id"])] = {
            "lexical_match": lexical,
            "structural_match": structural,
            "pre_same_speaker_silence_s": None if math.isinf(pre_gap) else round(pre_gap, 6),
            "post_same_speaker_silence_s": None if math.isinf(post_gap) else round(post_gap, 6),
            "overlaps_other_speaker": bool(overlapping),
            "inside_other_turn": inside_other_turn,
            "floor_speaker": floor_speaker,
            "candidate_confidence": confidence,
        }

    # Two IPUs with the same onset can otherwise classify each other as a
    # structural backchannel.  Never discard both from the floor sequence.  A
    # lexical acknowledgement wins; otherwise an empty/shorter IPU is the
    # listener response and the content-bearing/longer IPU remains the floor.
    by_id = {str(item["ipu_id"]): item for item in ipus}
    ids = sorted(output)
    drop: set[str] = set()
    for index, left_id in enumerate(ids):
        left = by_id[left_id]
        for right_id in ids[index + 1 :]:
            right = by_id[right_id]
            if left["speaker"] == right["speaker"]:
                continue
            if abs(float(left["start_s"]) - float(right["start_s"])) > 1e-6:
                continue
            left_lexical = bool(output[left_id]["lexical_match"])
            right_lexical = bool(output[right_id]["lexical_match"])
            if left_lexical != right_lexical:
                drop.add(right_id if left_lexical else left_id)
                continue
            left_text = _clean_words(left.get("text", ""))
            right_text = _clean_words(right.get("text", ""))
            if bool(left_text) != bool(right_text):
                # Keep the empty IPU as the BC; drop the content-bearing floor
                # IPU from the BC set.
                drop.add(left_id if left_text else right_id)
                continue
            left_duration = float(left["end_s"]) - float(left["start_s"])
            right_duration = float(right["end_s"]) - float(right["start_s"])
            if abs(left_duration - right_duration) > 1e-6:
                drop.add(left_id if left_duration > right_duration else right_id)
            else:
                # With no acoustic/lexical basis for choosing a listener, keep
                # neither as an automatic BC candidate.
                drop.update((left_id, right_id))
    for ipu_id in drop:
        output.pop(ipu_id, None)
    return output


def _event(
    *,
    label: str,
    structure: str,
    anchor_s: float,
    event_start_s: float,
    event_end_s: float,
    speaker_before: str | None,
    event_speaker: str | None,
    speaker_after: str | None,
    confidence: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_label": label,
        "structure": structure,
        "anchor_s": round(anchor_s, 6),
        "event_start_s": round(event_start_s, 6),
        "event_end_s": round(max(event_start_s, event_end_s), 6),
        "speaker_before": speaker_before,
        "event_speaker": event_speaker,
        "speaker_after": speaker_after,
        "candidate_confidence": confidence,
        "evidence": evidence,
        "human_label": None,
        "review_status": "unreviewed",
        "rule_version": RULE_VERSION,
    }


def build_event_candidates(
    ipus: Sequence[dict[str, Any]],
    vad_intervals: Sequence[tuple[float, float]],
    *,
    duration_s: float,
    minimum_silence_event_s: float = 0.2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Map IPU relations to event-centred five-class candidate labels."""
    backchannels = identify_backchannel_candidates(ipus)
    bc_ids = set(backchannels)
    events: list[dict[str, Any]] = []
    structures: list[dict[str, Any]] = []
    interruption_by_newcomer: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

    # Backchannels and overlap onsets.
    for newcomer in ipus:
        incumbents = [
            other
            for other in ipus
            if other["speaker"] != newcomer["speaker"]
            and float(other["start_s"]) < float(newcomer["start_s"]) - 1e-6
            and float(other["end_s"]) > float(newcomer["start_s"]) + 1e-6
        ]
        incumbent = min(incumbents, key=lambda item: (item["start_s"], -item["end_s"])) if incumbents else None
        bc = backchannels.get(str(newcomer["ipu_id"]))
        if bc is not None:
            floor_speaker = bc.get("floor_speaker")
            structures.append(
                {
                    "structure": "short_response_candidate",
                    "start_s": round(float(newcomer["start_s"]), 6),
                    "end_s": round(float(newcomer["end_s"]), 6),
                    "floor_speaker": floor_speaker,
                    "response_speaker": newcomer["speaker"],
                    "duration_s": round(
                        float(newcomer["end_s"]) - float(newcomer["start_s"]), 6
                    ),
                }
            )
            events.append(
                _event(
                    label="BC",
                    structure="backchannel_candidate",
                    anchor_s=float(newcomer["start_s"]),
                    event_start_s=float(newcomer["start_s"]),
                    event_end_s=float(newcomer["start_s"]),
                    speaker_before=incumbent["speaker"] if incumbent else floor_speaker,
                    event_speaker=str(newcomer["speaker"]),
                    speaker_after=incumbent["speaker"] if incumbent else floor_speaker,
                    confidence=str(bc["candidate_confidence"]),
                    evidence={
                        "ipu_id": newcomer["ipu_id"],
                        "response_duration_ms": round(
                            (float(newcomer["end_s"]) - float(newcomer["start_s"]))
                            * 1000.0,
                            3,
                        ),
                        **bc,
                    },
                )
            )
            continue
        if incumbent is None:
            continue
        overlap_end = min(float(newcomer["end_s"]), float(incumbent["end_s"]))
        overlap = overlap_end - float(newcomer["start_s"])
        if overlap <= 0.0:
            continue
        if float(newcomer["end_s"]) > float(incumbent["end_s"]) + 1e-6:
            outcome = "successful_floor_take"
            after_speaker = str(newcomer["speaker"])
        elif float(newcomer["end_s"]) < float(incumbent["end_s"]) - 1e-6:
            outcome = "unsuccessful_butting_in_candidate"
            after_speaker = str(incumbent["speaker"])
        else:
            outcome = "unclear_simultaneous_end"
            after_speaker = None
        interruption_by_newcomer[str(newcomer["ipu_id"])] = (newcomer, incumbent)
        structures.append(
            {
                "structure": "overlap",
                "start_s": round(float(newcomer["start_s"]), 6),
                "end_s": round(overlap_end, 6),
                "speaker_1": incumbent["speaker"],
                "speaker_2": newcomer["speaker"],
                "duration_s": round(overlap, 6),
            }
        )
        events.append(
            _event(
                label="I",
                structure="interruption_candidate",
                anchor_s=float(newcomer["start_s"]),
                event_start_s=float(newcomer["start_s"]),
                event_end_s=float(newcomer["start_s"]),
                speaker_before=str(incumbent["speaker"]),
                event_speaker=str(newcomer["speaker"]),
                speaker_after=after_speaker,
                confidence="medium" if overlap >= 0.1 else "low",
                evidence={
                    "newcomer_ipu_id": newcomer["ipu_id"],
                    "incumbent_ipu_id": incumbent["ipu_id"],
                    "overlap_ms": round(overlap * 1000.0, 3),
                    "floor_outcome": outcome,
                    "requires_semantic_confirmation": True,
                },
            )
        )

    # Pause/HOLD and clean speaker shifts, excluding short BC IPUs as floor holders.
    floor_ipus = [
        item
        for item in ipus
        if str(item["ipu_id"]) not in bc_ids
        and not (
            not _clean_words(item.get("text", ""))
            and (
                float(item["end_s"]) - float(item["start_s"]) < 0.2
                or any(
                    other["speaker"] != item["speaker"]
                    and abs(float(other["start_s"]) - float(item["start_s"])) <= 1e-6
                    and bool(_clean_words(other.get("text", "")))
                    for other in ipus
                )
            )
        )
    ]
    for current in floor_ipus:
        if str(current["ipu_id"]) in interruption_by_newcomer:
            continue
        active_other = [
            item
            for item in floor_ipus
            if item["speaker"] != current["speaker"]
            and float(item["start_s"]) < float(current["start_s"]) + 1e-6
            and float(item["end_s"]) > float(current["start_s"]) + 1e-6
        ]
        if active_other:
            continue
        previous = _nearest_before(floor_ipus, float(current["start_s"]))
        if previous is None:
            continue
        gap = max(0.0, float(current["start_s"]) - float(previous["end_s"]))
        label = "C" if previous["speaker"] == current["speaker"] else "T"
        structure = "hold_after_pause" if label == "C" else "natural_turn_shift"
        structures.append(
            {
                "structure": "pause" if label == "C" else "gap",
                "start_s": round(float(previous["end_s"]), 6),
                "end_s": round(float(current["start_s"]), 6),
                "speaker_before": previous["speaker"],
                "speaker_after": current["speaker"],
                "duration_s": round(gap, 6),
            }
        )
        events.append(
            _event(
                label=label,
                structure=structure,
                anchor_s=float(current["start_s"]),
                event_start_s=float(current["start_s"]),
                event_end_s=float(current["start_s"]),
                speaker_before=str(previous["speaker"]),
                event_speaker=str(current["speaker"]),
                speaker_after=str(current["speaker"]),
                confidence="high" if gap >= minimum_silence_event_s else "medium",
                evidence={
                    "previous_ipu_id": previous["ipu_id"],
                    "current_ipu_id": current["ipu_id"],
                    "silence_or_latch_ms": round(gap * 1000.0, 3),
                },
            )
        )

    # A successful interruption is followed by a turn shift; a failed attempt
    # is followed by continuation by the incumbent.
    for newcomer, incumbent in interruption_by_newcomer.values():
        if float(newcomer["end_s"]) > float(incumbent["end_s"]) + 1e-6:
            anchor = float(incumbent["end_s"])
            events.append(
                _event(
                    label="T",
                    structure="shift_after_successful_interruption",
                    anchor_s=anchor,
                    event_start_s=anchor,
                    event_end_s=anchor,
                    speaker_before=str(incumbent["speaker"]),
                    event_speaker=str(newcomer["speaker"]),
                    speaker_after=str(newcomer["speaker"]),
                    confidence="medium",
                    evidence={
                        "interruption_ipu_id": newcomer["ipu_id"],
                        "incumbent_ipu_id": incumbent["ipu_id"],
                    },
                )
            )
        elif float(incumbent["end_s"]) > float(newcomer["end_s"]) + 1e-6:
            anchor = float(newcomer["end_s"])
            events.append(
                _event(
                    label="C",
                    structure="hold_after_unsuccessful_interruption",
                    anchor_s=anchor,
                    event_start_s=anchor,
                    event_end_s=anchor,
                    speaker_before=str(incumbent["speaker"]),
                    event_speaker=str(incumbent["speaker"]),
                    speaker_after=str(incumbent["speaker"]),
                    confidence="medium",
                    evidence={
                        "interruption_ipu_id": newcomer["ipu_id"],
                        "incumbent_ipu_id": incumbent["ipu_id"],
                    },
                )
            )

    # One NA event per meaningful mutual-silence interval, never one per frame.
    for start, end in complement_intervals(vad_intervals, duration_s):
        if end - start + 1e-9 < minimum_silence_event_s:
            continue
        structures.append(
            {
                "structure": "mutual_silence",
                "start_s": round(start, 6),
                "end_s": round(end, 6),
                "duration_s": round(end - start, 6),
            }
        )
        previous = _nearest_before(ipus, start)
        after = _nearest_after(ipus, end)
        # Leading/trailing recording padding is not a conversational event.
        if previous is None or after is None:
            continue
        events.append(
            _event(
                label="NA",
                structure="mutual_silence_onset",
                anchor_s=start,
                event_start_s=start,
                event_end_s=start,
                speaker_before=str(previous["speaker"]) if previous else None,
                event_speaker=None,
                speaker_after=str(after["speaker"]) if after else None,
                confidence="high",
                evidence={"silence_duration_ms": round((end - start) * 1000.0, 3)},
            )
        )

    # One annotation unit per millisecond.  Rules may propose multiple labels
    # at the same boundary; keep one deterministic primary proposal and retain
    # the other proposals as evidence instead of creating overlapping records.
    priority = {"I": 0, "BC": 1, "T": 2, "C": 3, "NA": 4}
    by_anchor: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in events:
        by_anchor[round(float(item["anchor_s"]) * 1000)].append(item)
    deduplicated: list[dict[str, Any]] = []
    for anchor_ms, proposals in sorted(by_anchor.items()):
        ordered = sorted(
            proposals,
            key=lambda row: (
                priority[str(row["candidate_label"])],
                str(row["structure"]),
                str(row["event_speaker"]),
            ),
        )
        primary = dict(ordered[0])
        if len(ordered) > 1:
            primary["evidence"] = dict(primary["evidence"])
            primary["evidence"]["same_time_alternative_proposals"] = [
                {
                    "candidate_label": row["candidate_label"],
                    "structure": row["structure"],
                    "event_speaker": row["event_speaker"],
                }
                for row in ordered[1:]
            ]
        primary["anchor_s"] = round(anchor_ms / 1000.0, 6)
        primary["event_start_s"] = primary["anchor_s"]
        primary["event_end_s"] = primary["anchor_s"]
        deduplicated.append(primary)
    return deduplicated, sorted(structures, key=lambda row: (row["start_s"], row["structure"]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _context_transcript(
    utterances: Sequence[dict[str, Any]], start_s: float, end_s: float
) -> list[dict[str, Any]]:
    return [
        {
            "speaker": row["speaker"],
            "start_in_clip_s": round(max(float(row["start_s"]), start_s) - start_s, 3),
            "end_in_clip_s": round(min(float(row["end_s"]), end_s) - start_s, 3),
            "text": row.get("clean_text", ""),
        }
        for row in utterances
        if float(row["start_s"]) < end_s and float(row["end_s"]) > start_s
    ]


def add_review_audio(
    events: Sequence[dict[str, Any]],
    *,
    conversation: dict[str, Any],
    utterances: Sequence[dict[str, Any]],
    destination: Path,
    context_before_s: float = 6.0,
    context_after_s: float = 4.0,
    sample_rate: int = 16_000,
    overwrite_audio: bool = False,
    previous_review_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Create one portable mono WAV for every event review item."""
    output: list[dict[str, Any]] = []
    audio_source = Path(str(conversation["audio_path"]))
    duration_s = float(conversation["audio_info"]["duration_s"])
    audio_dir = destination / "audio_clips"
    for item in events:
        clip_start = max(0.0, float(item["anchor_s"]) - context_before_s)
        clip_end = min(duration_s, float(item["anchor_s"]) + context_after_s)
        if clip_end <= clip_start:
            continue
        clip_path = audio_dir / f"{item['event_id']}.wav"
        previous = (previous_review_by_id or {}).get(str(item["event_id"]))
        reusable = (
            not overwrite_audio
            and clip_path.is_file()
            and previous is not None
            and abs(float(previous.get("anchor_s", -1.0)) - float(item["anchor_s"])) < 1e-9
            and abs(
                float(previous.get("clip_start_in_conversation_s", -1.0)) - clip_start
            )
            < 1e-6
            and abs(float(previous.get("clip_end_in_conversation_s", -1.0)) - clip_end)
            < 1e-6
            and previous.get("audio_sha256") == _sha256(clip_path)
        )
        if not reusable:
            samples = read_wav_window_robust_mix(
                audio_source, clip_start, clip_end, target_rate=sample_rate
            )
            write_wav_mono(clip_path, samples, sample_rate=sample_rate)
        row = dict(item)
        row.update(
            {
                "audio_path": str(Path("audio_clips") / clip_path.name).replace("\\", "/"),
                "audio_sha256": _sha256(clip_path),
                "audio_sample_rate_hz": sample_rate,
                "clip_start_in_conversation_s": round(clip_start, 6),
                "clip_end_in_conversation_s": round(clip_end, 6),
                "target_start_in_clip_s": round(float(item["event_start_s"]) - clip_start, 6),
                "target_end_in_clip_s": round(
                    min(float(item["event_end_s"]), clip_end) - clip_start, 6
                ),
                "context_transcript": _context_transcript(utterances, clip_start, clip_end),
            }
        )
        output.append(row)
    return output


def _load_jsonl_grouped(path: str | Path, key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        grouped[str(row[key])].append(row)
    return grouped


def build_sbcsae_event_annotations(
    *,
    catalog_dir: str | Path,
    vad_dir: str | Path,
    output_dir: str | Path,
    context_before_s: float = 6.0,
    context_after_s: float = 4.0,
    ipu_silence_ms: int = 200,
    minimum_silence_event_ms: int = 200,
    generate_audio: bool = True,
    overwrite_audio: bool = False,
) -> dict[str, Any]:
    catalog = Path(catalog_dir)
    vad = Path(vad_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    previous_review_by_id = (
        {
            str(row["event_id"]): row
            for row in read_jsonl(destination / "annotation_manifest.jsonl")
        }
        if (destination / "annotation_manifest.jsonl").is_file()
        else {}
    )
    conversations = {
        str(row["conversation_id"]): row
        for row in read_jsonl(catalog / "conversations.jsonl")
        if row.get("core_dyadic")
    }
    utterances_by_conversation = _load_jsonl_grouped(
        catalog / "utterances.jsonl", "conversation_id"
    )
    vad_by_conversation = _load_jsonl_grouped(vad / "vad_segments.jsonl", "conversation_id")

    all_ipus: list[dict[str, Any]] = []
    all_structures: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    all_review: list[dict[str, Any]] = []
    per_conversation: list[dict[str, Any]] = []
    event_index = 0
    for conversation_id, conversation in sorted(conversations.items()):
        utterances = [
            row
            for row in utterances_by_conversation.get(conversation_id, [])
            if row.get("is_person", True)
        ]
        # The union keeps quiet speech detected in only one physical channel;
        # timed speaker rows provide speaker identity.
        vad_intervals = merge_intervals(
            (
                (float(row["start_s"]), float(row["end_s"]))
                for row in vad_by_conversation.get(conversation_id, [])
            ),
            max_gap_s=0.05,
        )
        duration_s = float(conversation["audio_info"]["duration_s"])
        ipus, diagnostics = build_ipus(
            utterances,
            vad_intervals,
            ipu_silence_s=ipu_silence_ms / 1000.0,
        )
        for ipu in ipus:
            ipu["conversation_id"] = conversation_id
            ipu["ipu_id"] = f"{conversation_id}-{ipu['ipu_id']}"
        events, structures = build_event_candidates(
            ipus,
            vad_intervals,
            duration_s=duration_s,
            minimum_silence_event_s=minimum_silence_event_ms / 1000.0,
        )
        for structure_index, structure in enumerate(structures):
            structure["conversation_id"] = conversation_id
            structure["structure_id"] = f"{conversation_id}-structure-{structure_index:06d}"
        participants = {
            item["local_speaker_id"]: {
                "speaker_uid": item["speaker_uid"],
                "display_name": item["display_name"],
                "profile": item["profile"],
            }
            for item in conversation["participants"]
            if item.get("is_person")
        }
        for event in events:
            event["event_id"] = f"{conversation_id}-event-{event_index:07d}"
            event_index += 1
            event["conversation_id"] = conversation_id
            event["allowed_human_labels"] = list(LABELS)
            event["conversation_context"] = {
                "relationship": conversation["relationship"],
                "situation": conversation["situation"],
            }
            event["participants"] = participants
        review_rows = (
            add_review_audio(
                events,
                conversation=conversation,
                utterances=utterances,
                destination=destination,
                context_before_s=context_before_s,
                context_after_s=context_after_s,
                overwrite_audio=overwrite_audio,
                previous_review_by_id=previous_review_by_id,
            )
            if generate_audio
            else []
        )
        counts = Counter(item["candidate_label"] for item in events)
        per_conversation.append(
            {
                "conversation_id": conversation_id,
                "duration_s": duration_s,
                "utterances": len(utterances),
                "vad_intervals": len(vad_intervals),
                "ipus": len(ipus),
                "structures": len(structures),
                "events": len(events),
                "candidate_label_counts": dict(counts),
                **diagnostics,
            }
        )
        all_ipus.extend(ipus)
        all_structures.extend(structures)
        all_candidates.extend(events)
        all_review.extend(review_rows)
        print(
            f"{conversation_id}: ipus={len(ipus)} events={len(events)} "
            f"labels={dict(counts)}",
            flush=True,
        )

    write_jsonl(destination / "ipus.jsonl", all_ipus)
    write_jsonl(destination / "interaction_structures.jsonl", all_structures)
    write_jsonl(destination / "event_candidates.jsonl", all_candidates)
    if generate_audio:
        write_jsonl(destination / "annotation_manifest.jsonl", all_review)
    summary = {
        "schema_version": "1.0",
        "task": "event_centred_five_class_human_annotation_preparation",
        "rule_version": RULE_VERSION,
        "source_dense_40ms_labels_used": False,
        "conversations": len(conversations),
        "duration_hours": sum(item["duration_s"] for item in per_conversation) / 3600.0,
        "ipu_boundary_silence_ms": ipu_silence_ms,
        "minimum_silence_event_ms": minimum_silence_event_ms,
        "review_context_before_s": context_before_s,
        "review_context_after_s": context_after_s,
        "ipus": len(all_ipus),
        "structures": len(all_structures),
        "events": len(all_candidates),
        "review_audio_files": len(all_review),
        "candidate_label_counts": dict(Counter(item["candidate_label"] for item in all_candidates)),
        "structure_counts": dict(Counter(item["structure"] for item in all_candidates)),
        "confidence_counts": dict(Counter(item["candidate_confidence"] for item in all_candidates)),
        "per_conversation": per_conversation,
        "important_note": (
            "candidate_label is an automatic proposal for human review; human_label is intentionally empty"
        ),
    }
    write_json(destination / "summary.json", summary)
    return summary


def verify_event_annotation_package(output_dir: str | Path) -> dict[str, Any]:
    destination = Path(output_dir)
    summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    candidates = list(read_jsonl(destination / "event_candidates.jsonl"))
    review = list(read_jsonl(destination / "annotation_manifest.jsonl"))
    ids = [str(item["event_id"]) for item in candidates]
    review_by_id = {str(item["event_id"]): item for item in review}
    anchor_keys = [
        (str(item["conversation_id"]), round(float(item["anchor_s"]) * 1000))
        for item in candidates
    ]
    diagnostics = Counter()
    for item in candidates:
        if item.get("candidate_label") not in LABELS:
            diagnostics["invalid_label"] += 1
        if item.get("human_label") is not None:
            diagnostics["human_label_pre_filled"] += 1
        if float(item["event_end_s"]) < float(item["event_start_s"]):
            diagnostics["invalid_event_interval"] += 1
        if abs(float(item["event_end_s"]) - float(item["event_start_s"])) > 1e-9:
            diagnostics["non_point_event_target"] += 1
        row = review_by_id.get(str(item["event_id"]))
        if row is None:
            diagnostics["missing_review_row"] += 1
            continue
        audio = destination / str(row["audio_path"])
        if not audio.is_file():
            diagnostics["missing_audio"] += 1
        elif _sha256(audio) != row.get("audio_sha256"):
            diagnostics["audio_hash_mismatch"] += 1
        clip_duration = float(row["clip_end_in_conversation_s"]) - float(
            row["clip_start_in_conversation_s"]
        )
        if audio.is_file():
            try:
                with wave.open(str(audio), "rb") as wav_file:
                    if wav_file.getnchannels() != 1:
                        diagnostics["invalid_audio_channels"] += 1
                    if wav_file.getframerate() != 16000:
                        diagnostics["invalid_audio_sample_rate"] += 1
                    if wav_file.getsampwidth() != 2:
                        diagnostics["invalid_audio_sample_width"] += 1
                    actual_duration = wav_file.getnframes() / wav_file.getframerate()
                    if abs(actual_duration - clip_duration) > 0.02:
                        diagnostics["audio_duration_mismatch"] += 1
            except (EOFError, wave.Error, ZeroDivisionError):
                diagnostics["invalid_wav_file"] += 1
        if not (0.0 <= float(row["target_start_in_clip_s"]) <= clip_duration + 1e-6):
            diagnostics["target_outside_audio"] += 1
        if float(row["target_end_in_clip_s"]) < float(row["target_start_in_clip_s"]):
            diagnostics["invalid_target_interval"] += 1
    checks = {
        "sixteen_conversations": summary.get("conversations") == 16,
        "events_match_summary": len(candidates) == summary.get("events"),
        "unique_event_ids": len(ids) == len(set(ids)),
        "unique_event_times": len(anchor_keys) == len(set(anchor_keys)),
        "all_targets_are_points": diagnostics["non_point_event_target"] == 0,
        "all_five_candidate_labels_present": set(item["candidate_label"] for item in candidates)
        == set(LABELS),
        "one_review_audio_per_event": len(review) == len(candidates),
        "dense_40ms_labels_unused": summary.get("source_dense_40ms_labels_used") is False,
        "no_diagnostics": not diagnostics,
    }
    result = {
        "verified": all(checks.values()),
        "checks": checks,
        "diagnostics": dict(diagnostics),
        "events": len(candidates),
        "candidate_label_counts": dict(Counter(item["candidate_label"] for item in candidates)),
    }
    write_json(destination / "verification.json", result)
    return result


def build_static_review_site(output_dir: str | Path) -> dict[str, Any]:
    """Build the review page and its per-conversation data files."""
    destination = Path(output_dir)
    rows = list(read_jsonl(destination / "annotation_manifest.jsonl"))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["conversation_id"])].append(row)
    data_dir = destination / "review_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    for conversation_id, items in sorted(grouped.items()):
        public_items = [
            {
                "event_id": item["event_id"],
                "conversation_id": item["conversation_id"],
                "candidate_label": item["candidate_label"],
                "structure": item["structure"],
                "anchor_s": item["anchor_s"],
                "speaker_before": item.get("speaker_before"),
                "event_speaker": item.get("event_speaker"),
                "speaker_after": item.get("speaker_after"),
                "audio_path": item["audio_path"],
                "clip_start_in_conversation_s": item["clip_start_in_conversation_s"],
                "clip_end_in_conversation_s": item["clip_end_in_conversation_s"],
                "target_start_in_clip_s": item["target_start_in_clip_s"],
                "target_end_in_clip_s": item["target_end_in_clip_s"],
                "context_transcript": item["context_transcript"],
            }
            for item in items
        ]
        payload = json.dumps(public_items, ensure_ascii=False, separators=(",", ":"))
        (data_dir / f"{conversation_id}.js").write_text(
            f"window.REVIEW_EVENTS={payload};\n", encoding="utf-8"
        )
        index.append({"conversation_id": conversation_id, "events": len(items)})
    (data_dir / "index.js").write_text(
        "window.REVIEW_CONVERSATIONS="
        + json.dumps(index, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    title = html.escape("SBCSAE 对话事件人工标注")
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;background:#f5f7fa;color:#172033}}
.wrap{{max-width:980px;margin:auto;padding:20px}} .card{{background:white;border:1px solid #dce2ea;border-radius:12px;padding:18px;margin:12px 0}}
h1{{font-size:24px;margin:0 0 8px}} h2{{font-size:18px;margin:0 0 12px}} .muted{{color:#5f6b7a}}
.top{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} label{{display:block;font-size:13px;color:#4d5968;margin-bottom:4px}}
select,input,textarea,button{{font:inherit}} select,input,textarea{{box-sizing:border-box;width:100%;padding:9px;border:1px solid #bbc5d2;border-radius:7px}}
audio{{width:100%;margin:10px 0}} .timeline{{height:22px;background:#e8edf3;border-radius:6px;position:relative;overflow:hidden}}
.target{{position:absolute;top:0;bottom:0;background:#e53935;min-width:3px}} .targetText{{font-weight:700;color:#b71c1c;margin-top:7px}}
.autoEvent{{margin-top:10px;padding:10px 12px;background:#fff8e1;border:1px solid #efc96b;border-radius:8px;font-weight:700;color:#6d4c00}}
.labels{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:14px 0}} .labels button{{padding:12px 6px;border:2px solid #9aa7b5;background:#fff;border-radius:9px;cursor:pointer}}
.labels button.on{{border-color:#1565c0;background:#e3f2fd;color:#0d47a1;font-weight:700}} .issues{{display:flex;flex-wrap:wrap;gap:7px}}
.issues button,.nav button,.action{{padding:8px 12px;border:1px solid #9aa7b5;background:#fff;border-radius:7px;cursor:pointer}}
.nav .action{{display:inline-block;margin:0;color:#172033;box-sizing:border-box}} .navActions{{display:flex;gap:8px;align-items:center}}
.issues button.on{{background:#fff3cd;border-color:#d39e00}} .nav{{display:flex;gap:8px;align-items:center;justify-content:space-between}}
.transcript{{max-height:250px;overflow:auto;background:#f7f9fc;padding:10px;border-radius:8px;line-height:1.55}} .line{{padding:5px 7px;border-radius:6px}}
.time{{color:#667384;font-variant-numeric:tabular-nums}} .guide{{font-size:14px;line-height:1.6}} .guide b{{display:inline-block;min-width:34px}}
@media(max-width:650px){{.top{{grid-template-columns:1fr}}.labels{{grid-template-columns:1fr 1fr}}}}
</style>
</head>
<body><div class="wrap">
<h1>{title}</h1><p class="muted">听目标位置前后的对话，选择 C / BC / T / I / NA。页面同时显示自动标出的事件，方便核对。</p>
<div class="card top"><div><label for="reviewer">标注员姓名或编号</label><input id="reviewer" placeholder="例如：R01"></div><div><label for="conversation">选择会话</label><select id="conversation"></select></div></div>
<div class="card guide"><b>C</b>原说话人继续　 <b>BC</b>简短反馈、不接管对话　 <b>T</b>换人　 <b>I</b>未说完时插入并表达内容　 <b>NA</b>双方静音</div>
<div class="card" id="work" hidden>
<div class="nav"><button id="prev">上一条</button><strong id="progress"></strong><button id="next">下一条</button></div>
<h2 id="eventId"></h2><audio id="audio" controls preload="metadata"></audio>
<div class="timeline"><div class="target" id="target"></div></div><div class="targetText" id="targetText"></div><div class="autoEvent" id="autoEvent"></div>
<p><button class="action" id="playTarget">从目标前2秒播放</button></p>
<div class="transcript" id="transcript"></div>
<div class="labels" id="labels"></div>
<div class="issues" id="issues"></div>
<p><label for="notes">备注（可不填）</label><textarea id="notes" rows="2" placeholder="例如：像笑声、目标偏早"></textarea></p>
<p class="muted" id="saveStatus"></p>
<div class="nav"><button id="unlabeled">跳到下一条未标注</button><div class="navActions"><label class="action" for="importFile">读取已有结果</label><input id="importFile" type="file" accept="application/json,.json" hidden><button class="action" id="export">导出结果</button></div></div>
</div>
</div>
<script src="review_data/index.js"></script>
<script>
const LABELS=[['C','C 继续'],['BC','BC 简短反馈'],['T','T 换人'],['I','I 插话/打断'],['NA','NA 无人说话']];
const LABEL_NAMES={{C:'继续',BC:'简短反馈',T:'换人',I:'插话/打断',NA:'无人说话'}};
const ISSUES=[['UNCERTAIN','无法判断'],['BAD_TARGET','目标位置有误'],['AUDIO_ISSUE','音频问题']];
const SERVER_MODE=location.protocol==='http:'||location.protocol==='https:';
const LAST_REVIEWER_KEY='sbcsae-review-v1:last-reviewer',LAST_CONVERSATION_KEY='sbcsae-review-v1:last-conversation';
let events=[],pos=0,cid='',memoryStore={{}},persistentStorage=true,pendingImport=null,serverSaveQueue=Promise.resolve(),reviewerLoadTimer=null; const $=id=>document.getElementById(id);
function reviewer(){{return $('reviewer').value.trim()}}
function key(){{return `sbcsae-review-v1:${{reviewer()}}:${{cid}}`}}
function state(){{if(SERVER_MODE)return memoryStore[key()]||{{}};try{{return JSON.parse(localStorage.getItem(key())||JSON.stringify(memoryStore[key()]||{{}}))}}catch(e){{persistentStorage=false;return memoryStore[key()]||{{}}}}}}
function postServerReviews(rows){{if(!SERVER_MODE||!reviewer())return serverSaveQueue;const body={{conversation_id:cid,reviewer:reviewer(),last_position:pos,reviews:rows}};$('saveStatus').textContent='正在保存…';serverSaveQueue=serverSaveQueue.then(async()=>{{try{{const response=await fetch('/api/reviews',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}});const result=await response.json();if(!response.ok)throw new Error(result.error||'保存失败');if(body.conversation_id===cid&&body.reviewer===reviewer())$('saveStatus').textContent=`已保存到结果目录，共 ${{result.total_reviews}} 条。`}}catch(error){{$('saveStatus').textContent=`保存失败：${{error.message}}`}}}});return serverSaveQueue}}
function save(s,changedEventId=null){{memoryStore[key()]=s;if(SERVER_MODE){{const ids=changedEventId?[changedEventId]:Object.keys(s),rows=ids.filter(id=>s[id]).map(event_id=>({{event_id,...s[event_id]}}));postServerReviews(rows);return}}try{{localStorage.setItem(key(),JSON.stringify(s));persistentStorage=true}}catch(e){{persistentStorage=false}}updateSaveStatus(s)}}
function positionKey(){{return `${{key()}}:position`}}
function rememberPosition(){{if(SERVER_MODE){{postServerReviews([]);return}}try{{localStorage.setItem(positionKey(),String(pos))}}catch(e){{persistentStorage=false}}}}
function readPosition(){{try{{return Number(localStorage.getItem(positionKey())||0)}}catch(e){{persistentStorage=false;return 0}}}}
function updateSaveStatus(s=state()){{const n=Object.keys(s).length;if(SERVER_MODE){{$('saveStatus').textContent=reviewer()?`结果目录已保存 ${{n}} 条。`:'请先填写标注员编号。';return}}$('saveStatus').textContent=persistentStorage?`已在当前浏览器暂存 ${{n}} 条；完成后请点击“导出结果”。`:`当前浏览器不能暂存；请立即点击“导出结果”。`}}
async function restoreServerState(){{if(!reviewer()){{memoryStore[key()]={{}};pos=0;return}}const response=await fetch(`/api/reviews?conversation_id=${{encodeURIComponent(cid)}}&reviewer=${{encodeURIComponent(reviewer())}}`,{{cache:'no-store'}}),payload=await response.json();if(!response.ok)throw new Error(payload.error||'读取进度失败');const restored={{}};for(const row of payload.reviews||[]){{const eventId=String(row.event_id||'');if(eventId)restored[eventId]={{label:row.label||null,status:row.status||'',notes:row.notes||'',reviewer:payload.reviewer,updated_at:row.updated_at}}}}memoryStore[key()]=restored;pos=Math.max(0,Math.min(events.length-1,Number(payload.last_position||0)))}}
function loadConversation(id){{cid=id;try{{localStorage.setItem(LAST_CONVERSATION_KEY,id)}}catch(e){{persistentStorage=false}}window.REVIEW_EVENTS=null;const old=$('dataScript');if(old)old.remove();const s=document.createElement('script');s.id='dataScript';s.src=`review_data/${{id}}.js`;s.onload=async()=>{{events=window.REVIEW_EVENTS||[];try{{if(SERVER_MODE)await restoreServerState();else pos=Math.max(0,Math.min(events.length-1,readPosition()))}}catch(error){{$('saveStatus').textContent=`读取进度失败：${{error.message}}`}}$('work').hidden=false;render();if(pendingImport){{const payload=pendingImport;pendingImport=null;applyImportedReviews(payload)}}}};document.body.appendChild(s)}}
function applyImportedReviews(payload){{const rows=Array.isArray(payload)?payload:payload.reviews;if(!Array.isArray(rows))throw new Error('结果文件中缺少 reviews');const ids=new Set(events.map(e=>e.event_id)),labels=new Set(LABELS.map(x=>x[0])),issues=new Set(ISSUES.map(x=>x[0])),s=state();let loaded=0;for(const row of rows){{const eventId=String(row.event_id||row.sample_id||''),label=String(row.label||row.human_label||'').toUpperCase(),status=String(row.status||(label?'OK':'')).toUpperCase();if(!ids.has(eventId))continue;if(label&&!labels.has(label))continue;if(!label&&!issues.has(status))continue;s[eventId]={{...(s[eventId]||{{}}),label:label||null,status:label?'OK':status,notes:String(row.notes??row.note??''),reviewer:reviewer(),updated_at:String(row.updated_at||new Date().toISOString())}};loaded++}}save(s);const next=events.findIndex(e=>!s[e.event_id]);pos=next>=0?next:Math.max(0,events.length-1);rememberPosition();render();$('saveStatus').textContent=`已读取 ${{loaded}} 条结果并自动暂存；可从当前未标注位置继续。`}}
function importResultFile(file){{if(!file)return;const reader=new FileReader();reader.onload=()=>{{try{{const payload=JSON.parse(reader.result),target=String(payload.conversation_id||cid);if(payload.reviewer){{$('reviewer').value=String(payload.reviewer);try{{localStorage.setItem(LAST_REVIEWER_KEY,$('reviewer').value.trim())}}catch(e){{persistentStorage=false}}}}if(!reviewer())throw new Error('请先填写标注员编号');if(![...sel.options].some(o=>o.value===target))throw new Error(`找不到会话 ${{target}}`);if(SERVER_MODE||target!==cid){{pendingImport=payload;sel.value=target;loadConversation(target)}}else applyImportedReviews(payload)}}catch(error){{alert('读取失败：'+error.message)}}finally{{$('importFile').value=''}}}};reader.readAsText(file)}}
function render(){{if(!events.length)return;const e=events[pos],s=state(),r=s[e.event_id]||{{}};$('progress').textContent=`${{pos+1}} / ${{events.length}}`;$('eventId').textContent=e.event_id;$('audio').src=e.audio_path;
const duration=e.clip_end_in_conversation_s-e.clip_start_in_conversation_s,start=e.target_start_in_clip_s;$('target').style.left=`${{100*start/duration}}%`;$('target').style.width='3px';$('targetText').textContent=`事件发生时间：本段音频第 ${{start.toFixed(2)}} 秒`;$('autoEvent').textContent=`系统自动标记：${{e.candidate_label}}（${{LABEL_NAMES[e.candidate_label]||''}}）`;
$('transcript').innerHTML=(e.context_transcript||[]).map(x=>`<div class="line"><span class="time">[${{x.start_in_clip_s.toFixed(2)}}–${{x.end_in_clip_s.toFixed(2)}}]</span> <b>${{escapeHtml(x.speaker||'')}}</b>：${{escapeHtml(x.text||'（无文字）')}}</div>`).join('');
$('labels').innerHTML=LABELS.map(([v,t])=>`<button data-value="${{v}}" class="${{r.label===v?'on':''}}">${{t}}</button>`).join('');$('issues').innerHTML=ISSUES.map(([v,t])=>`<button data-value="${{v}}" class="${{r.status===v?'on':''}}">${{t}}</button>`).join('');$('notes').value=r.notes||'';
updateSaveStatus(s);document.querySelectorAll('#labels button').forEach(b=>b.onclick=()=>setReview({{label:b.dataset.value,status:'OK'}}));document.querySelectorAll('#issues button').forEach(b=>b.onclick=()=>setReview({{label:null,status:b.dataset.value}}));}}
function escapeHtml(x){{return String(x).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}}[c]))}}
function setReview(change){{if(!reviewer()){{alert('请先填写标注员编号');$('reviewer').focus();return}}const e=events[pos],s=state();s[e.event_id]={{...(s[e.event_id]||{{}}),...change,notes:$('notes').value,reviewer:reviewer(),updated_at:new Date().toISOString()}};save(s,e.event_id);render()}}
function move(delta){{const e=events[pos],s=state();if(s[e.event_id]){{s[e.event_id].notes=$('notes').value;save(s,e.event_id)}}pos=Math.max(0,Math.min(events.length-1,pos+delta));rememberPosition();render()}}
$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('playTarget').onclick=()=>{{const e=events[pos],a=$('audio');a.currentTime=Math.max(0,e.target_start_in_clip_s-2);a.play()}};$('notes').onchange=()=>{{const e=events[pos],s=state();if(s[e.event_id]){{s[e.event_id].notes=$('notes').value;save(s,e.event_id)}}}};
$('unlabeled').onclick=()=>{{const s=state(),i=events.findIndex((e,j)=>j>pos&&!s[e.event_id]);const any=i>=0?i:events.findIndex(e=>!s[e.event_id]);if(any>=0){{pos=any;render()}}}};
$('export').onclick=()=>{{if(!reviewer()){{alert('请先填写标注员编号');return}}if(SERVER_MODE){{const a=document.createElement('a');a.href=`/api/export?conversation_id=${{encodeURIComponent(cid)}}&reviewer=${{encodeURIComponent(reviewer())}}`;a.click();return}}const s=state(),payload={{schema_version:'1.0',conversation_id:cid,reviewer:reviewer(),exported_at:new Date().toISOString(),reviews:Object.entries(s).map(([event_id,v])=>({{event_id,...v}}))}};const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`${{cid}}_${{reviewer()}}_reviews.json`;a.click();URL.revokeObjectURL(a.href);$('saveStatus').textContent=`已生成结果文件：${{a.download}}`}};
$('importFile').onchange=()=>importResultFile($('importFile').files[0]);
const sel=$('conversation');(window.REVIEW_CONVERSATIONS||[]).forEach(x=>{{const o=document.createElement('option');o.value=x.conversation_id;o.textContent=`${{x.conversation_id}}（${{x.events}}条）`;sel.appendChild(o)}});try{{$('reviewer').value=localStorage.getItem(LAST_REVIEWER_KEY)||'';const last=localStorage.getItem(LAST_CONVERSATION_KEY);if(last&&[...sel.options].some(o=>o.value===last))sel.value=last}}catch(e){{persistentStorage=false}}sel.onchange=()=>loadConversation(sel.value);$('reviewer').oninput=()=>{{try{{localStorage.setItem(LAST_REVIEWER_KEY,$('reviewer').value.trim())}}catch(e){{persistentStorage=false}}clearTimeout(reviewerLoadTimer);reviewerLoadTimer=setTimeout(()=>loadConversation(cid||sel.value),300)}};if(sel.value)loadConversation(sel.value);
document.addEventListener('keydown',e=>{{if(e.target.matches('input,textarea,select'))return;const map={{'1':'C','2':'BC','3':'T','4':'I','5':'NA'}};if(map[e.key])setReview({{label:map[e.key],status:'OK'}});else if(e.key==='ArrowRight')move(1);else if(e.key==='ArrowLeft')move(-1)}});
</script></body></html>"""
    (destination / "review.html").write_text(page, encoding="utf-8")
    result = {
        "review_page": "review.html",
        "conversations": len(index),
        "events": len(rows),
        "portable": True,
        "requirements": "run the annotation server for automatic file saving; direct browser opening supports manual import/export only",
    }
    write_json(destination / "review_site.json", result)
    return result


__all__ = [
    "build_event_candidates",
    "build_ipus",
    "build_sbcsae_event_annotations",
    "build_static_review_site",
    "identify_backchannel_candidates",
    "verify_event_annotation_package",
]
