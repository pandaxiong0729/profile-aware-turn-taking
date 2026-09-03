from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from profile_turntaking.audio import read_wav_window_robust_mix, write_wav_mono
from profile_turntaking.utils import read_jsonl, write_json, write_jsonl


TASK_SPEC = {
    "turn_change": {
        "question": (
            "The current speaker has taken a pause at the exact end of the audio. "
            "Predict what happens immediately after the pause."
        ),
        "labels": ("CURRENT_SPEAKER_CONTINUES", "OTHER_SPEAKER_TAKES_TURN"),
    },
    "backchannel": {
        "question": (
            "Predict whether either listener produces a brief acknowledgement "
            "immediately after the exact end of the audio."
        ),
        "labels": ("BACKCHANNEL", "NO_BACKCHANNEL"),
    },
    "interruption": {
        "question": (
            "The clip ends while the current speaker is still speaking. Predict "
            "whether the other speaker begins a full contribution before the "
            "current speaker finishes."
        ),
        "labels": ("OTHER_SPEAKER_INTERRUPTS", "CURRENT_SPEAKER_CONTINUES"),
    },
    "floor_taking": {
        "question": (
            "At the end of the clip, the second speaker has just begun overlapping "
            "the first speaker. Predict whether the second speaker subsequently "
            "takes the floor."
        ),
        "labels": (
            "SECOND_SPEAKER_TAKES_FLOOR",
            "FIRST_SPEAKER_KEEPS_FLOOR",
        ),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def balanced_take(
    rows: list[dict[str, Any]],
    count: int,
    *,
    rng: random.Random,
    key: Callable[[dict[str, Any]], str] = lambda row: str(row["conversation_id"]),
) -> list[dict[str, Any]]:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in shuffled:
        by_conversation[key(row)].append(row)
    selected: list[dict[str, Any]] = []
    conversations = sorted(by_conversation)
    rng.shuffle(conversations)
    while len(selected) < count:
        progressed = False
        for conversation_id in conversations:
            bucket = by_conversation[conversation_id]
            if bucket:
                selected.append(bucket.pop())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(f"Requested {count} rows but found only {len(selected)}")
    return selected


def causal_transcript(
    utterances: list[dict[str, Any]], boundary_s: float, *, window_s: float = 30.0
) -> str:
    start_s = max(0.0, boundary_s - window_s)
    lines: list[str] = []
    for row in utterances:
        end_s = float(row["end_s"])
        if end_s > boundary_s + 1e-6 or end_s <= start_s:
            continue
        relative_start = max(float(row["start_s"]), start_s) - start_s
        relative_end = end_s - start_s
        text = str(row.get("clean_text", "")).strip() or "(no text)"
        lines.append(
            f"[{row['speaker']} {relative_start:.3f}-{relative_end:.3f}] {text}"
        )
    rendered = "\n".join(lines)
    return rendered[-7000:] if len(rendered) > 7000 else rendered


def build_prompt(task: str, transcript: str) -> str:
    spec = TASK_SPEC[task]
    first, second = spec["labels"]
    return "\n".join(
        [
            "Listen to the attached two-speaker conversation.",
            "The audio contains only the past and ends exactly at prediction time t.",
            "Predict only what happens after t. Never claim to hear audio after t.",
            "",
            str(spec["question"]),
            f"The two possible conclusions are {first} and {second}.",
            "",
            "Causal speaker-timed partial transcript; every listed unit ends no later than t:",
            transcript or "No completed transcript unit is available.",
            "",
            "Profile condition:",
            "Speaker profiles, relationship, and situation are unknown.",
            "",
            "Use the causal words, ending prosody, pause and listener activity.",
            f"Return only one conclusion: {first} or {second}.",
        ]
    )


def build_active_continuation_points(
    ipus_by_conversation: dict[str, list[dict[str, Any]]],
    events_by_conversation: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for conversation_id, ipus in ipus_by_conversation.items():
        events = events_by_conversation[conversation_id]
        for ipu in ipus:
            start = float(ipu["start_s"])
            end = float(ipu["end_s"])
            if end - start < 1.4 or start < 30.0:
                continue
            boundary = start + min(0.8, (end - start) / 2.0)
            if end < boundary + 0.5:
                continue
            other_active = any(
                other["speaker"] != ipu["speaker"]
                and float(other["start_s"]) <= boundary + 1e-6
                and float(other["end_s"]) >= boundary + 0.5 - 1e-6
                for other in ipus
            )
            future_event = any(
                boundary - 1e-6 <= float(event["anchor_s"]) <= boundary + 0.5 + 1e-6
                and event["candidate_label"] in {"BC", "I", "T"}
                for event in events
            )
            if other_active or future_event:
                continue
            points.append(
                {
                    "conversation_id": conversation_id,
                    "boundary_s": round(boundary, 6),
                    "source_kind": "active_continuation_control",
                    "source_id": str(ipu["ipu_id"]),
                    "speaker": ipu["speaker"],
                }
            )
    return points


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a strict-causal, balanced 50-sample hidden prompt benchmark."
    )
    parser.add_argument(
        "--events-dir", default="data/processed/sbcsae_turn_events_v3"
    )
    parser.add_argument(
        "--catalog-dir", default="data/processed/sbcsae_catalog_v2"
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/talking-turns-paper-aligned/qwen2.5-omni-3b-q8/"
            "hidden50-v3-reasoned"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()

    events_root = Path(args.events_dir)
    catalog_root = Path(args.catalog_dir)
    output_root = Path(args.output_dir)
    audio_root = output_root / "causal_audio"
    rng = random.Random(args.seed)

    conversations = {
        str(row["conversation_id"]): row
        for row in read_jsonl(catalog_root / "conversations.jsonl")
        if row.get("core_dyadic")
    }
    utterances_by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(catalog_root / "utterances.jsonl"):
        if row.get("is_person", True) and row["conversation_id"] in conversations:
            utterances_by_conversation[str(row["conversation_id"])].append(row)
    ipus_by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(events_root / "ipus.jsonl"):
        ipus_by_conversation[str(row["conversation_id"])].append(row)
    events_by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_events = list(read_jsonl(events_root / "event_candidates.jsonl"))
    for row in all_events:
        events_by_conversation[str(row["conversation_id"])].append(row)

    pause_events = [
        row
        for row in all_events
        if row["structure"] in {"hold_after_pause", "natural_turn_shift"}
        and 0.2 <= float(row["evidence"]["silence_or_latch_ms"]) / 1000.0 <= 2.5
        and float(row["anchor_s"]) >= 30.0
    ]
    bc_events = [
        row
        for row in all_events
        if row["structure"] == "backchannel_candidate"
        and float(row["anchor_s"]) >= 30.0
    ]
    interruption_events = [
        row
        for row in all_events
        if row["structure"] == "interruption_candidate"
        and float(row["evidence"].get("overlap_ms", 0.0)) >= 300.0
        and float(row["anchor_s"]) >= 30.0
    ]
    active_controls = build_active_continuation_points(
        ipus_by_conversation, events_by_conversation
    )

    selected: list[dict[str, Any]] = []

    def add_event(task: str, event: dict[str, Any], target: str, offset_s: float = 0.0) -> None:
        selected.append(
            {
                "task": task,
                "conversation_id": event["conversation_id"],
                "boundary_s": round(float(event["anchor_s"]) + offset_s, 6),
                "target": target,
                "source_kind": event["structure"],
                "source_id": event["event_id"],
            }
        )

    for event in balanced_take(
        [row for row in pause_events if row["candidate_label"] == "C"], 6, rng=rng
    ):
        add_event("turn_change", event, "CURRENT_SPEAKER_CONTINUES")
    for event in balanced_take(
        [row for row in pause_events if row["candidate_label"] == "T"], 6, rng=rng
    ):
        add_event("turn_change", event, "OTHER_SPEAKER_TAKES_TURN")
    for event in balanced_take(bc_events, 6, rng=rng):
        add_event("backchannel", event, "BACKCHANNEL")
    for control in balanced_take(active_controls, 12, rng=rng):
        selected.append(
            {
                "task": "backchannel" if len([r for r in selected if r["task"] == "backchannel"]) < 12 else "interruption",
                "conversation_id": control["conversation_id"],
                "boundary_s": control["boundary_s"],
                "target": (
                    "NO_BACKCHANNEL"
                    if len([r for r in selected if r["task"] == "backchannel"]) < 12
                    else "CURRENT_SPEAKER_CONTINUES"
                ),
                "source_kind": control["source_kind"],
                "source_id": control["source_id"],
            }
        )
    for event in balanced_take(interruption_events, 6, rng=rng):
        add_event("interruption", event, "OTHER_SPEAKER_INTERRUPTS")
    for outcome, target in (
        ("successful_floor_take", "SECOND_SPEAKER_TAKES_FLOOR"),
        ("unsuccessful_butting_in_candidate", "FIRST_SPEAKER_KEEPS_FLOOR"),
    ):
        candidates = [
            row
            for row in interruption_events
            if row["evidence"].get("floor_outcome") == outcome
        ]
        for event in balanced_take(candidates, 7, rng=rng):
            add_event("floor_taking", event, target, offset_s=0.2)

    if len(selected) != 50:
        raise AssertionError(f"Expected 50 selected samples, got {len(selected)}")
    rng.shuffle(selected)
    requests: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        sample_id = f"tt50-v3-{index:04d}"
        conversation_id = str(row["conversation_id"])
        boundary_s = float(row["boundary_s"])
        window_start = max(0.0, boundary_s - 30.0)
        audio_path = audio_root / f"{sample_id}.wav"
        samples = read_wav_window_robust_mix(
            conversations[conversation_id]["audio_path"], window_start, boundary_s
        )
        write_wav_mono(audio_path, samples, sample_rate=16_000)
        transcript = causal_transcript(
            utterances_by_conversation[conversation_id], boundary_s
        )
        prompt = build_prompt(str(row["task"]), transcript)
        relative_audio = audio_path.relative_to(output_root)
        request = {
            "request_id": f"{sample_id}::hidden",
            "sample_id": sample_id,
            "conversation_id": conversation_id,
            "task": row["task"],
            "audio_path": str(relative_audio),
            "audio_sha256": sha256_file(audio_path),
            "audio_duration_s": round(len(samples) / 16_000.0, 6),
            "prediction_boundary_in_conversation_s": boundary_s,
            "profile_mode": "hidden",
            "profile_text": "Speaker profiles, relationship, and situation are unknown.",
            "causal_partial_transcript": transcript,
            "transcript_sha256": sha256_text(transcript),
            "prompt": prompt,
            "prompt_sha256": sha256_text(prompt),
            "allowed_predictions": list(TASK_SPEC[str(row["task"])]["labels"]),
        }
        requests.append(request)
        references.append(
            {
                "sample_id": sample_id,
                "conversation_id": conversation_id,
                "task": row["task"],
                "target": row["target"],
                "reference_source": "automatic_structural_candidate_v3",
                "source_kind": row["source_kind"],
                "source_id": row["source_id"],
                "prediction_boundary_in_conversation_s": boundary_s,
            }
        )

    errors: list[str] = []
    for request in requests:
        audio_path = output_root / request["audio_path"]
        if sha256_file(audio_path) != request["audio_sha256"]:
            errors.append(f"{request['sample_id']}: audio hash mismatch")
        if sha256_text(request["causal_partial_transcript"]) != request["transcript_sha256"]:
            errors.append(f"{request['sample_id']}: transcript hash mismatch")
        end_times = [
            float(value)
            for value in re.findall(
                r"\[[^\]\n]+\s\d+\.\d+-(\d+\.\d+)\]",
                request["causal_partial_transcript"],
            )
        ]
        if end_times and max(end_times) > request["audio_duration_s"] + 1e-6:
            errors.append(f"{request['sample_id']}: transcript crosses t")
        if abs(request["audio_duration_s"] - 30.0) > 1e-6:
            errors.append(f"{request['sample_id']}: audio is not 30 seconds")
    audit = {
        "passed": not errors,
        "requests": len(requests),
        "task_counts": dict(Counter(row["task"] for row in requests)),
        "target_counts": dict(Counter(row["target"] for row in references)),
        "audio_sha256_verified": not any("audio hash" in error for error in errors),
        "transcript_sha256_verified": not any(
            "transcript hash" in error for error in errors
        ),
        "causal_transcript_timestamps_verified": not any(
            "crosses t" in error for error in errors
        ),
        "target_fields_absent_from_requests": all(
            key not in request for request in requests for key in ("target", "reference_label")
        ),
        "message_content_order": ["text", "input_audio"],
        "errors": errors,
    }
    write_jsonl(output_root / "requests.jsonl", requests)
    write_jsonl(output_root / "reference_labels.jsonl", references)
    write_json(output_root / "input_audit.json", audit)
    write_json(
        output_root / "run_config.json",
        {
            "diagnostic_only": True,
            "seed": args.seed,
            "event_rule_version": "sbcsae_event_candidates_v3_floor_safeguards",
            "input_contract": "causal mono audio + matching causal partial transcript + hidden profile",
            "reference_warning": "Automatic structural weak labels; not human gold labels.",
        },
    )
    if errors:
        raise ValueError("Input audit failed: " + "; ".join(errors[:10]))
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
