from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from profile_turntaking.audio import read_wav_window_robust_mix
from profile_turntaking.constants import LABELS
from profile_turntaking.prompt_baseline import PROFILE_MODES
from profile_turntaking.qwen25_omni_event_eval import (
    _profile_derangement,
    _sha256_file,
    _sha256_text,
    _unknown_profile,
    event_profile,
    render_profile,
)
from profile_turntaking.utils import read_jsonl, write_json, write_jsonl


PAPER_TASK_ORDER = ("turn_change", "backchannel", "interruption", "floor_taking")
DEFAULT_SPLIT_ROOT = Path("data/processed/sbcsae_semantic_profile_v1")
DEFAULT_EVENTS = Path("data/processed/sbcsae_turn_events_v3/event_candidates.jsonl")
DEFAULT_CATALOG = Path("data/processed/sbcsae_catalog_v2")


def _sha256_audio_array(samples: np.ndarray) -> str:
    audio = np.asarray(samples, dtype="<f4")
    return hashlib.sha256(audio.tobytes()).hexdigest()


def _prompt_template(prompt: str, profile_text: str) -> str:
    if prompt.count(profile_text) != 1:
        raise ValueError("Rendered profile must occur exactly once")
    return prompt.replace(profile_text, "<PROFILE_CONDITION>", 1)


def build_profile_prompt(
    *,
    audio_duration_s: float,
    transcript: str,
    profile_text: str,
    forecast_offset_ms: int,
    horizon_ms: int,
    transcript_mode: str,
) -> str:
    if transcript_mode not in {"catalog", "none"}:
        raise ValueError(f"Unknown transcript_mode: {transcript_mode}")
    lines = [
            "Listen to the attached two-person conversation and predict what happens next.",
            f"The mono audio is {audio_duration_s:.3f} seconds long and ends exactly at prediction time t.",
            "No future audio, future transcript, target label, or annotation evidence is provided.",
            f"Predict the turn-taking event beginning at t+{forecast_offset_ms} ms.",
            f"The evaluation interval is [t+{forecast_offset_ms} ms, t+{horizon_ms} ms).",
            "",
            "Possible labels are C, BC, T, I, and NA:",
            "C = current floor holder continues; no new listener response takes priority.",
            "BC = listener gives a short acknowledgement while the current speaker keeps the floor.",
            "T = current speaker yields and the other participant takes the floor.",
            "I = other participant starts a substantive contribution before the current speaker yields.",
            "NA = nobody is speaking at the decision point; silence begins or continues.",
            "",
    ]
    if transcript_mode == "catalog":
        lines.extend(
            [
                "Matching causal partial transcript (only units completed by t):",
                transcript or "No completed transcript unit is available in this 30-second window.",
                "",
            ]
        )
    else:
        lines.extend(["No transcript or ASR text is provided in this audio-only ablation.", ""])
    lines.extend(
        [
            "Speaker profile and conversation context:",
            profile_text,
            "",
            (
                "Use the causal audio, the matching causal partial transcript, and the profile condition. "
                "Return no explanation."
                if transcript_mode == "catalog"
                else "Use only the causal audio and the profile condition. Return no explanation."
            ),
        ]
    )
    return "\n".join(lines)


def build_audio_only_profile_prompt(
    *,
    audio_duration_s: float,
    profile_text: str,
    forecast_offset_ms: int,
    horizon_ms: int,
) -> str:
    """Backward-compatible wrapper for existing callers and artifacts."""

    return build_profile_prompt(
        audio_duration_s=audio_duration_s,
        transcript="",
        profile_text=profile_text,
        forecast_offset_ms=forecast_offset_ms,
        horizon_ms=horizon_ms,
        transcript_mode="none",
    )


def causal_catalog_transcript(
    utterances: list[dict[str, Any]],
    *,
    window_start_s: float,
    boundary_s: float,
    max_chars: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Render only utterance units fully completed by the prediction boundary."""

    units: list[dict[str, Any]] = []
    for row in utterances:
        start_s = float(row["start_s"])
        end_s = float(row["end_s"])
        if (
            end_s > boundary_s + 1e-6
            or end_s <= window_start_s + 1e-6
            or start_s < window_start_s - 1e-6
        ):
            continue
        units.append(
            {
                "speaker": str(row.get("speaker", "unknown")),
                "start_s": round(max(start_s, window_start_s) - window_start_s, 3),
                "end_s": round(end_s - window_start_s, 3),
                "text": str(row.get("clean_text") or row.get("text") or "").strip() or "(no text)",
            }
        )
    rendered = [
        f"[{unit['speaker']} {unit['start_s']:.3f}-{unit['end_s']:.3f}] {unit['text']}"
        for unit in units
    ]
    if max_chars > 0:
        kept_lines: list[str] = []
        kept_units: list[dict[str, Any]] = []
        used = 0
        for line, unit in reversed(list(zip(rendered, units))):
            added = len(line) + (1 if kept_lines else 0)
            if kept_lines and used + added > max_chars:
                break
            if not kept_lines and len(line) > max_chars:
                continue
            kept_lines.append(line)
            kept_units.append(unit)
            used += added
        rendered = list(reversed(kept_lines))
        units = list(reversed(kept_units))
    return "\n".join(rendered), units


def paper_targets(row: dict[str, Any]) -> dict[str, int | None]:
    label = str(row.get("candidate_label", ""))
    evidence = row.get("evidence", {}) if isinstance(row.get("evidence", {}), dict) else {}
    floor_outcome = str(evidence.get("floor_outcome", ""))
    return {
        "turn_change": 0 if label == "C" else (1 if label == "T" else None),
        "backchannel": 0 if label == "C" else (1 if label == "BC" else None),
        "interruption": 0 if label == "C" else (1 if label == "I" else None),
        "floor_taking": (
            1
            if floor_outcome == "successful_floor_take"
            else (0 if floor_outcome == "unsuccessful_butting_in_candidate" else None)
        ),
    }


def _load_split_conversations(split_root: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        config_path = split_root / split / "run_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing split run config: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        conversations = [str(item) for item in config.get("conversations", [])]
        if not conversations:
            raise ValueError(f"No conversations listed in {config_path}")
        result[split] = conversations
    seen: dict[str, str] = {}
    for split, conversations in result.items():
        for conversation in conversations:
            previous = seen.setdefault(conversation, split)
            if previous != split:
                raise ValueError(f"Conversation {conversation} appears in both {previous} and {split}")
    return result


def _first_event_eligible(
    rows: list[dict[str, Any]],
    *,
    forecast_lead_s: float,
    evaluation_window_s: float,
) -> set[str]:
    ordered = sorted(rows, key=lambda item: float(item["anchor_s"]))
    eligible: set[str] = set()
    for index, row in enumerate(ordered):
        event_time_s = float(row["anchor_s"])
        prediction_time_s = event_time_s - forecast_lead_s
        previous_anchor = float(ordered[index - 1]["anchor_s"]) if index > 0 else None
        next_anchor = float(ordered[index + 1]["anchor_s"]) if index + 1 < len(ordered) else None
        previous_clear = previous_anchor is None or previous_anchor < prediction_time_s - 1e-6
        following_clear = next_anchor is None or next_anchor >= event_time_s + evaluation_window_s - 1e-6
        if previous_clear and following_clear:
            eligible.add(str(row["event_id"]))
    return eligible


def _write_split(
    *,
    split: str,
    rows: list[dict[str, Any]],
    conversations: dict[str, dict[str, Any]],
    output_dir: Path,
    context_seconds: float,
    forecast_lead_ms: int,
    horizon_ms: int,
    sample_rate: int,
    limit: int | None,
    transcript_mode: str,
    transcript_max_chars: int,
    utterances_by_conversation: dict[str, list[dict[str, Any]]],
    hash_windows: str,
    hash_sample_per_split: int,
) -> dict[str, Any]:
    destination = output_dir / split
    forecast_lead_s = forecast_lead_ms / 1000.0
    evaluation_window_s = (horizon_ms - forecast_lead_ms) / 1000.0
    rows_by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_conversation[str(row["conversation_id"])].append(row)

    first_event_ids: set[str] = set()
    for conv_rows in rows_by_conversation.values():
        first_event_ids.update(
            _first_event_eligible(
                conv_rows,
                forecast_lead_s=forecast_lead_s,
                evaluation_window_s=evaluation_window_s,
            )
        )

    profiles = {conversation_id: event_profile(conv_rows[0]) for conversation_id, conv_rows in rows_by_conversation.items()}
    shuffled = _profile_derangement(profiles)

    selected: list[dict[str, Any]] = []
    skipped = Counter()
    for row in sorted(rows, key=lambda item: (str(item["conversation_id"]), float(item["anchor_s"]), str(item["event_id"]))):
        conversation_id = str(row["conversation_id"])
        label = str(row.get("candidate_label", ""))
        if label not in LABELS:
            skipped["invalid_label"] += 1
            continue
        if str(row["event_id"]) not in first_event_ids:
            skipped["not_isolated_first_event"] += 1
            continue
        event_time_s = float(row["anchor_s"])
        boundary_s = event_time_s - forecast_lead_s
        if boundary_s < context_seconds:
            skipped["insufficient_30s_history"] += 1
            continue
        conversation = conversations.get(conversation_id)
        if conversation is None:
            skipped["missing_conversation"] += 1
            continue
        duration_s = float(conversation["audio_info"]["duration_s"])
        if event_time_s + evaluation_window_s > duration_s + 1e-6:
            skipped["event_window_crosses_recording_end"] += 1
            continue
        selected.append(row)
    if limit is not None:
        selected = selected[: max(0, int(limit))]

    requests: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    window_hash_by_key: dict[tuple[str, float, float], str] = {}
    source_hash_by_path: dict[Path, str] = {}
    hashed_window_count = 0
    for row_index, row in enumerate(selected):
        sample_id = str(row["event_id"])
        conversation_id = str(row["conversation_id"])
        conversation = conversations[conversation_id]
        source_audio_path = Path(str(conversation["audio_path"])).resolve()
        source_hash = source_hash_by_path.get(source_audio_path)
        if source_hash is None:
            source_hash = _sha256_file(source_audio_path)
            source_hash_by_path[source_audio_path] = source_hash
        event_time_s = float(row["anchor_s"])
        boundary_s = round(event_time_s - forecast_lead_s, 6)
        window_start_s = round(boundary_s - context_seconds, 6)
        window_end_s = boundary_s
        key = (str(source_audio_path), round(window_start_s, 6), round(window_end_s, 6))
        logical_audio_hash = _sha256_text(
            f"{source_hash}\n{window_start_s:.6f}\n{window_end_s:.6f}\n{sample_rate}"
        )
        should_hash_window = hash_windows == "all" or (
            hash_windows == "sample" and row_index < max(0, hash_sample_per_split)
        )
        window_hash = window_hash_by_key.get(key, "")
        if should_hash_window and not window_hash:
            audio = read_wav_window_robust_mix(
                source_audio_path,
                window_start_s,
                window_end_s,
                target_rate=sample_rate,
            )
            window_hash = _sha256_audio_array(audio)
            window_hash_by_key[key] = window_hash
            hashed_window_count += 1
        transcript, transcript_units = causal_catalog_transcript(
            utterances_by_conversation.get(conversation_id, []),
            window_start_s=window_start_s,
            boundary_s=boundary_s,
            max_chars=transcript_max_chars,
        )
        if transcript_mode == "none":
            transcript = ""
            transcript_units = []
        transcript_hash = _sha256_text(transcript)
        targets = paper_targets(row)
        conditions = {
            "hidden": _unknown_profile(),
            "given": profiles[conversation_id],
            "shuffled": shuffled[conversation_id],
        }
        for profile_mode in PROFILE_MODES:
            profile_text = render_profile(conditions[profile_mode])
            prompt = build_profile_prompt(
                audio_duration_s=context_seconds,
                transcript=transcript,
                profile_text=profile_text,
                forecast_offset_ms=forecast_lead_ms,
                horizon_ms=horizon_ms,
                transcript_mode=transcript_mode,
            )
            prompt_template = _prompt_template(prompt, profile_text)
            request = {
                "request_id": f"{sample_id}::{profile_mode}",
                "sample_id": sample_id,
                "conversation_id": conversation_id,
                "profile_mode": profile_mode,
                "audio_path": str(source_audio_path),
                "source_audio_path": str(source_audio_path),
                "source_audio_sha256": source_hash,
                "audio_window_start_s": round(window_start_s, 6),
                "audio_window_end_s": round(window_end_s, 6),
                "audio_window_sha256": window_hash,
                "audio_logical_sha256": logical_audio_hash,
                "audio_sha256": window_hash or logical_audio_hash,
                "audio_duration_s": round(context_seconds, 6),
                "audio_sample_rate": sample_rate,
                "decision_time_in_conversation_s": round(boundary_s, 6),
                "source_clip_window_start_s": round(window_start_s, 6),
                "source_clip_boundary_s": round(window_end_s, 6),
                "forecast_offset_ms": forecast_lead_ms,
                "evaluation_window_ms": horizon_ms - forecast_lead_ms,
                "horizon_ms": horizon_ms,
                "transcript_prefix": transcript,
                "transcript_units": transcript_units,
                "transcript_sha256": transcript_hash,
                "boundary_state": {"active_speakers_at_t": [], "active_speaker_count": 0},
                "boundary_state_text": "",
                "boundary_state_sha256": _sha256_text(""),
                "causal_asr_transcript": "",
                "causal_asr_sha256": _sha256_text(""),
                "profile_text": profile_text,
                "profile_sha256": _sha256_text(profile_text),
                "prompt": prompt,
                "prompt_template_sha256": _sha256_text(prompt_template),
                "request_sha256": _sha256_text(
                    (window_hash or logical_audio_hash)
                    + "\n"
                    + transcript_hash
                    + "\n"
                    + _sha256_text("")
                    + "\n"
                    + _sha256_text("")
                    + "\n"
                    + prompt
                ),
            }
            requests.append(request)
        references.append(
            {
                "sample_id": sample_id,
                "conversation_id": conversation_id,
                "reference_label": str(row["candidate_label"]),
                "reference_source": "automatic_event_candidate_v3",
                "prediction_boundary_in_conversation_s": round(boundary_s, 6),
                "event_time_in_conversation_s": round(event_time_s, 6),
                "event_offset_ms": forecast_lead_ms,
                "source_kind": str(row.get("structure", "")),
                "candidate_confidence": str(row.get("candidate_confidence", "")),
                "paper_binary_targets": targets,
            }
        )
        selected_rows.append(
            {
                "sample_id": sample_id,
                "conversation_id": conversation_id,
                "reference_label": str(row["candidate_label"]),
                "prediction_boundary_in_conversation_s": round(boundary_s, 6),
                "event_time_in_conversation_s": round(event_time_s, 6),
                "source_audio_path": str(source_audio_path),
                "source_audio_sha256": source_hash,
                "audio_window_start_s": round(window_start_s, 6),
                "audio_window_end_s": round(window_end_s, 6),
                "audio_window_sha256": window_hash,
                "audio_logical_sha256": logical_audio_hash,
                "transcript_sha256": transcript_hash,
                "completed_transcript_units": len(transcript_units),
                "paper_binary_targets": targets,
            }
        )

    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "requests.jsonl", requests)
    write_jsonl(destination / "reference_labels.jsonl", references)
    write_jsonl(destination / "selected_inputs.jsonl", selected_rows)
    class_counts = Counter(row["reference_label"] for row in references)
    task_counts = {
        task: {
            "A": sum(ref["paper_binary_targets"][task] == 0 for ref in references),
            "B": sum(ref["paper_binary_targets"][task] == 1 for ref in references),
            "ignored": sum(ref["paper_binary_targets"][task] is None for ref in references),
        }
        for task in PAPER_TASK_ORDER
    }
    audit = {
        "passed": True,
        "split": split,
        "selected_samples": len(references),
        "requests": len(requests),
        "class_counts": {label: class_counts[label] for label in LABELS},
        "paper_binary_task_counts": task_counts,
        "skipped": dict(skipped),
        "context_seconds": context_seconds,
        "audio_window_duration_all_30s": all(
            abs(row["audio_window_end_s"] - row["audio_window_start_s"] - context_seconds) < 1e-6
            for row in selected_rows
        ),
        "transcript_mode": transcript_mode,
        "transcript_included": transcript_mode == "catalog",
        "causal_transcript_timestamps_checked": all(
            all(float(unit["end_s"]) <= context_seconds + 1e-6 for unit in request["transcript_units"])
            for request in requests
        ),
        "profile_modes": list(PROFILE_MODES),
        "profile_only_change_fields": [
            "request_id",
            "profile_mode",
            "profile_text",
            "profile_sha256",
            "prompt",
            "prompt_template_sha256",
            "request_sha256",
        ],
        "audio_window_hash_policy": hash_windows,
        "audio_window_hash_samples": hashed_window_count,
        "audio_window_sha256_checked": hashed_window_count > 0,
        "reference_labels_kept_outside_requests": True,
    }
    write_json(destination / "input_audit.json", audit)
    return audit


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    split_conversations = _load_split_conversations(Path(args.split_root))
    conversations = {
        str(row["conversation_id"]): row
        for row in read_jsonl(Path(args.catalog_dir) / "conversations.jsonl")
        if row.get("core_dyadic")
    }
    events = [
        row
        for row in read_jsonl(args.events)
        if str(row.get("conversation_id", "")) in {
            conv for values in split_conversations.values() for conv in values
        }
    ]
    allowed_conversations = {conv for values in split_conversations.values() for conv in values}
    utterances_by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if args.transcript_mode == "catalog":
        for row in read_jsonl(Path(args.catalog_dir) / "utterances.jsonl"):
            conversation_id = str(row.get("conversation_id", ""))
            if conversation_id in allowed_conversations and bool(row.get("is_person", True)):
                utterances_by_conversation[conversation_id].append(row)
        for conv_rows in utterances_by_conversation.values():
            conv_rows.sort(key=lambda item: (float(item["start_s"]), float(item["end_s"])))
    by_split = {
        split: [
            row
            for row in events
            if str(row.get("conversation_id", "")) in set(conversation_ids)
        ]
        for split, conversation_ids in split_conversations.items()
    }
    split_reports = {
        split: _write_split(
            split=split,
            rows=rows,
            conversations=conversations,
            output_dir=output_dir,
            context_seconds=float(args.context_seconds),
            forecast_lead_ms=int(args.forecast_lead_ms),
            horizon_ms=int(args.horizon_ms),
            sample_rate=int(args.sample_rate),
            limit=args.limit_per_split,
            transcript_mode=str(args.transcript_mode),
            transcript_max_chars=int(args.transcript_max_chars),
            utterances_by_conversation=utterances_by_conversation,
            hash_windows=str(args.hash_windows),
            hash_sample_per_split=int(args.hash_sample_per_split),
        )
        for split, rows in by_split.items()
    }
    summary = {
        "dataset": (
            "sbcsae_qwen_shared_ab_30s_causal_v1"
            if args.transcript_mode == "catalog"
            else "sbcsae_qwen_shared_ab_30s_audio_only_v1"
        ),
        "purpose": (
            "Primary Qwen shared A/B adapter data with 30s causal audio, matching causal transcript, and profile."
            if args.transcript_mode == "catalog"
            else "Paper-aligned audio-only ablation with 30s causal audio and profile."
        ),
        "events": str(Path(args.events).resolve()),
        "catalog_dir": str(Path(args.catalog_dir).resolve()),
        "split_root": str(Path(args.split_root).resolve()),
        "output_dir": str(output_dir.resolve()),
        "context_seconds": float(args.context_seconds),
        "forecast_lead_ms": int(args.forecast_lead_ms),
        "horizon_ms": int(args.horizon_ms),
        "sample_rate": int(args.sample_rate),
        "transcript_mode": str(args.transcript_mode),
        "transcript_included": args.transcript_mode == "catalog",
        "audio_window_hash_policy": str(args.hash_windows),
        "profile_modes": list(PROFILE_MODES),
        "paper_binary_tasks": list(PAPER_TASK_ORDER),
        "splits": split_reports,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 30s causal data for Qwen shared A/B adapter.")
    parser.add_argument("--events", default=str(DEFAULT_EVENTS))
    parser.add_argument("--catalog-dir", default=str(DEFAULT_CATALOG))
    parser.add_argument("--split-root", default=str(DEFAULT_SPLIT_ROOT))
    parser.add_argument("--output-dir", default="data/processed/sbcsae_qwen_shared_ab_30s_causal_v1")
    parser.add_argument("--context-seconds", type=float, default=30.0)
    parser.add_argument("--forecast-lead-ms", type=int, default=100)
    parser.add_argument("--horizon-ms", type=int, default=600)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--transcript-mode", choices=("catalog", "none"), default="catalog")
    parser.add_argument("--transcript-max-chars", type=int, default=6000)
    parser.add_argument("--hash-windows", choices=("none", "sample", "all"), default="sample")
    parser.add_argument("--hash-sample-per-split", type=int, default=50)
    parser.add_argument("--limit-per-split", type=int)
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
