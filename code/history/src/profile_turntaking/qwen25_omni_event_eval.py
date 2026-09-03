"""Strict-causal Qwen2.5-Omni-3B evaluation on SBCSAE event points.

The review package contains audio after each event so a person can verify the
automatic proposal.  That future context must never enter a prediction request.
This module therefore rebuilds a causal clip ending before the event point and
keeps event times and reference labels in a separate file.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import shutil
import time
import urllib.error
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .audio import read_wav_window, write_wav_mono
from .constants import LABELS, LABEL_TO_ID
from .metrics import classification_metrics, write_metrics_csv
from .mllm_prompt_baseline import (
    OUTPUT_SCHEMA,
    _server_chat_completion,
    prepare_silenced_audio_control,
    run_mllm_server_requests,
    score_silenced_audio_control,
)
from .prompt_baseline import PROFILE_MODES, select_conversation_balanced_rows
from .utils import read_jsonl, write_json, write_jsonl

DEFAULT_CONFIG_PATH = Path("code/configs/qwen25_omni_event_eval_v1.json")
PROMPT_VERSION = "event_first_audio_transcript_profile_v11_causal_asr"
DEFAULT_FORECAST_LEAD_MS = 100
DEFAULT_FORECAST_HORIZON_MS = 140
_PROFILE_PLACEHOLDER = "<PROFILE_CONDITION>"
_SAFE_FILE_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
SCORE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 100}},
    "required": ["score"],
    "additionalProperties": False,
}
_FORBIDDEN_REQUEST_KEYS = {
    "candidate_label",
    "candidate_confidence",
    "evidence",
    "human_label",
    "label_source",
    "reference_label",
    "review_status",
    "structure",
    "target",
    "event_offset_ms",
    "event_time_in_conversation_s",
    "training_target",
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(_nested_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(_nested_keys(nested))
    return keys


def _spoken(value: Any) -> str:
    rendered = str(value or "unknown").strip()
    return rendered.replace("_", " ") if rendered else "unknown"


def _unknown_profile() -> dict[str, Any]:
    speaker = {
        "age_group": "unknown",
        "gender": "unknown",
        "social_role": "unknown",
        "background": "unknown",
    }
    return {
        "speaker_00": dict(speaker),
        "speaker_01": dict(speaker),
        "relationship": "unknown",
        "situation": "unknown",
    }


def event_profile(row: dict[str, Any]) -> dict[str, Any]:
    """Extract only the intended profile fields, never names or annotation evidence."""

    participants = row.get("participants", {})
    context = row.get("conversation_context", {})
    result: dict[str, Any] = {}
    for speaker in ("speaker_00", "speaker_01"):
        source = participants.get(speaker, {}).get("profile", {})
        result[speaker] = {
            "age_group": source.get("age_group", "unknown"),
            "gender": source.get("gender", "unknown"),
            "social_role": source.get("social_role", "unknown"),
            "background": source.get("background", "unknown"),
        }
    result["relationship"] = context.get("relationship", "unknown")
    result["situation"] = context.get("situation", "unknown")
    return result


def render_profile(profile: dict[str, Any]) -> str:
    """Render all conditions with one fixed natural-language template."""

    lines: list[str] = []
    for speaker in ("speaker_00", "speaker_01"):
        fields = profile.get(speaker, {})
        lines.append(
            f"{speaker} has age group {_spoken(fields.get('age_group'))}, "
            f"gender {_spoken(fields.get('gender'))}, social role "
            f"{_spoken(fields.get('social_role'))}, and background "
            f"{_spoken(fields.get('background'))}."
        )
    lines.append(
        f"Their relationship is {_spoken(profile.get('relationship'))}."
    )
    lines.append(
        f"The conversation situation is {_spoken(profile.get('situation'))}."
    )
    return "\n".join(lines)


def _profile_derangement(
    profiles: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    conversation_ids = sorted(profiles)
    if len(conversation_ids) < 2:
        raise ValueError("At least two conversations are required for shuffled profiles")
    return {
        conversation_id: profiles[
            conversation_ids[(index + 1) % len(conversation_ids)]
        ]
        for index, conversation_id in enumerate(conversation_ids)
    }


def causal_transcript_units(
    row: dict[str, Any],
    *,
    window_start_in_clip_s: float,
    boundary_in_clip_s: float,
    max_chars: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Keep only utterance units that finish no later than the boundary.

    A unit crossing the boundary is omitted in full because its text may contain
    words spoken after the boundary.
    """

    units: list[dict[str, Any]] = []
    for source in row.get("context_transcript", []):
        start_s = float(source.get("start_in_clip_s", 0.0))
        end_s = float(source.get("end_in_clip_s", 0.0))
        if end_s > boundary_in_clip_s + 1e-6:
            continue
        if end_s <= window_start_in_clip_s + 1e-6:
            continue
        units.append(
            {
                "speaker": str(source.get("speaker", "unknown")),
                "start_s": round(max(start_s, window_start_in_clip_s) - window_start_in_clip_s, 3),
                "end_s": round(end_s - window_start_in_clip_s, 3),
                "text": str(source.get("text", "")).strip() or "(no text)",
            }
        )
    rendered = [
        f"[{unit['speaker']} {unit['start_s']:.3f}-{unit['end_s']:.3f}] {unit['text']}"
        for unit in units
    ]
    if max_chars > 0:
        kept: list[str] = []
        used = 0
        kept_units: list[dict[str, Any]] = []
        for line, unit in reversed(list(zip(rendered, units))):
            added = len(line) + (1 if kept else 0)
            if kept and used + added > max_chars:
                break
            if not kept and len(line) > max_chars:
                continue
            kept.append(line)
            kept_units.append(unit)
            used += added
        rendered = list(reversed(kept))
        units = list(reversed(kept_units))
    transcript = "\n".join(rendered)
    return transcript, units


def causal_boundary_state(
    row: dict[str, Any],
    *,
    window_start_in_clip_s: float,
    boundary_in_clip_s: float,
) -> dict[str, Any]:
    """Describe only speaker activity observable at the causal audio boundary.

    The text and the annotated end time of a unit that crosses the boundary are
    deliberately excluded.  Only its speaker identity and already-observed start
    time are retained.  This restores the missing "who is speaking now" state
    without exposing words or timing after the end of the model audio.
    """

    active: list[dict[str, Any]] = []
    for source in row.get("context_transcript", []):
        start_s = float(source.get("start_in_clip_s", 0.0))
        end_s = float(source.get("end_in_clip_s", 0.0))
        if start_s <= boundary_in_clip_s + 1e-6 and end_s > boundary_in_clip_s + 1e-6:
            active.append(
                {
                    "speaker": str(source.get("speaker", "unknown")),
                    "observed_start_s": round(
                        max(start_s, window_start_in_clip_s) - window_start_in_clip_s,
                        3,
                    ),
                }
            )
    active.sort(key=lambda item: (item["observed_start_s"], item["speaker"]))
    return {
        "active_speakers_at_t": active,
        "active_speaker_count": len(active),
    }


def render_boundary_state(state: dict[str, Any], *, audio_duration_s: float) -> str:
    active = list(state.get("active_speakers_at_t", []))
    if not active:
        return "At t (the end of the audio), no speaker is currently detected as speaking."
    rendered = ", ".join(
        f"{item['speaker']} (audible since {float(item['observed_start_s']):.3f}s)"
        for item in active
    )
    return (
        f"At t={audio_duration_s:.3f}s (the end of the audio), the currently audible "
        f"speaker(s) are: {rendered}."
    )


def build_event_prompt(
    *,
    audio_duration_s: float,
    transcript: str,
    profile_text: str,
    boundary_state_text: str = "",
    causal_asr_transcript: str = "",
    forecast_offset_ms: int = DEFAULT_FORECAST_LEAD_MS,
    horizon_ms: int = DEFAULT_FORECAST_HORIZON_MS,
    prompt_style: str = "direct",
) -> str:
    """Build a five-class prompt for the first event after the causal boundary."""

    if horizon_ms <= forecast_offset_ms:
        raise ValueError("horizon_ms must be greater than forecast_offset_ms")
    if prompt_style not in {"direct", "decision_tree", "reasoned", "hierarchical"}:
        raise ValueError(f"Unknown prompt_style: {prompt_style}")
    transcript_text = transcript or "No completed transcript unit is available."
    boundary_text = boundary_state_text or (
        "No separate speaker-activity summary is available; infer it from the audio."
    )
    live_asr_text = causal_asr_transcript or (
        "No separate causal ASR transcript is available; use the attached audio directly."
    )
    shared = [
            "Listen to the attached two-person conversation and predict what happens next.",
            f"The mono audio is {audio_duration_s:.3f} seconds long and contains only the conversation up to now.",
            f"<PREDICTION_BOUNDARY t=END_OF_AUDIO={audio_duration_s:.3f}s>",
            f"The exact decision point is t+{forecast_offset_ms} ms. Predict the interaction state beginning at that point.",
            f"Use [t+{forecast_offset_ms} ms, t+{horizon_ms} ms) only as the duration over which that event would be classified.",
            "No future audio or future transcript is provided. Predict from the conversation before t.",
            "Use the audio, the matching partial transcript, and the speaker profile together.",
            "",
            "Classify that next event:",
            "C = the floor holder is the speaker active at the decision point, with no new listener response taking priority.",
            "BC = the listener gives only a short acknowledgement such as mm-hm, yeah, or right; the current speaker keeps the floor.",
            "T = the current speaker yields, then the other participant takes the floor.",
            "I = the other participant starts a substantive contribution before the current speaker has yielded.",
            "NA = neither participant is speaking at the decision point; a silent interval begins or continues.",
            "",
            "Causal transcript available before t:",
            transcript_text,
            "",
            "Speaker activity exactly at t (derived only from audio up to t):",
            boundary_text,
            "",
            "Causal ASR of this exact audio, including the unfinished words nearest t:",
            live_asr_text,
            "",
            "Speaker profile and conversation context:",
            profile_text,
            "",
    ]
    if prompt_style == "hierarchical":
        ending = [
            "Predict the next event in three stages, but output only its final class:",
            f"1. At exactly t+{forecast_offset_ms} ms, will nobody be speaking? If yes, choose NA.",
            "2. Otherwise, decide whether a new listener onset occurs at that point. If not and the floor holder remains active, choose C.",
            "3. If the other person starts, choose BC only for a brief listener acknowledgement; choose T for a substantive turn after a yield; choose I for a substantive start before a yield.",
            "Important: an active speaker at t does not by itself imply I. Use completion, prosody, overlap risk, the unfinished causal words, and profile together.",
            "Important: BC, T, I, and NA are genuine possibilities; do not default to C or I merely because they are common.",
            'Return exactly one JSON object such as {"label":"BC"}. Return no explanation.',
        ]
    elif prompt_style == "decision_tree":
        ending = [
            "Use this decision process:",
            "1. Decide whether the current speaker is likely to keep the floor.",
            "2. If the listener responds, decide whether it is only a short acknowledgement or a substantive turn.",
            "3. If it is substantive, decide whether the current speaker has yielded before it begins.",
            "4. If nobody begins, choose silence.",
            "Map the decision to C, BC, T, I, or NA using the definitions above.",
            'Return exactly one JSON object such as {"label":"C"}. Return no explanation.',
        ]
    elif prompt_style == "reasoned":
        ending = [
            "Briefly explain who holds the floor, whether the last utterance sounds complete, and how the profile affects the likely next response.",
            'Finish with one JSON object such as {"label":"C"}.',
            "The final label must be exactly C, BC, T, I, or NA.",
        ]
    else:
        ending = [
            "Silently consider who currently holds the floor, whether the utterance sounds complete, and how the profile changes the likely response style.",
            'Return exactly one JSON object such as {"label":"C"}.',
            "The label must be exactly C, BC, T, I, or NA. Return no explanation.",
        ]
    return "\n".join(shared + ending)


_HYPOTHESIS_DESCRIPTIONS = {
    "C": "the floor holder remains the active speaker at the decision point, with no new listener response taking priority",
    "BC": "the listener begins only a brief acknowledgement at the decision point while the floor holder keeps the floor",
    "T": "the other participant begins a substantive new turn at the decision point after the floor holder yields",
    "I": "the other participant begins a substantive contribution at the decision point before the floor holder yields",
    "NA": "neither participant is speaking at the decision point, so a silent interval begins or continues",
}


def build_candidate_score_prompt(
    *,
    audio_duration_s: float,
    transcript: str,
    profile_text: str,
    boundary_state_text: str,
    causal_asr_transcript: str,
    hypothesis_label: str,
    forecast_offset_ms: int,
    horizon_ms: int,
) -> str:
    """Ask for one calibrated one-vs-rest score instead of an enum choice."""

    if hypothesis_label not in LABELS:
        raise ValueError(f"Unknown hypothesis label: {hypothesis_label}")
    transcript_text = transcript or "No completed transcript unit is available."
    asr_text = causal_asr_transcript or (
        "No separate causal ASR transcript is available; use the attached audio directly."
    )
    description = _HYPOTHESIS_DESCRIPTIONS[hypothesis_label]
    return "\n".join(
        [
            "Listen to the attached two-person conversation and estimate one turn-taking hypothesis.",
            f"The mono audio is {audio_duration_s:.3f} seconds long and contains only the conversation up to now.",
            f"<PREDICTION_BOUNDARY t=END_OF_AUDIO={audio_duration_s:.3f}s>",
            f"The exact decision point is t+{forecast_offset_ms} ms.",
            f"The interval [t+{forecast_offset_ms} ms, t+{horizon_ms} ms) is used only to classify the event beginning at that point.",
            "No future audio, future transcript, event answer, or annotation evidence is provided.",
            "Use the causal audio, matching causal transcript, and profile together.",
            "",
            "The five mutually exclusive possibilities are:",
            "C: floor holder remains active, with no new listener response taking priority.",
            "BC: listener starts only a brief acknowledgement while the floor holder keeps the floor.",
            "T: other participant starts a substantive turn after the floor holder yields.",
            "I: other participant starts a substantive contribution before the floor holder yields.",
            "NA: nobody is speaking at the decision point; silence begins or continues.",
            "",
            "Completed speaker-timed transcript before t:",
            transcript_text,
            "",
            "Speaker activity exactly at t:",
            boundary_state_text,
            "",
            "Causal ASR from this exact audio, including unfinished words nearest t:",
            asr_text,
            "",
            "Speaker profile and conversation context:",
            profile_text,
            "",
            f"Score this hypothesis only: {hypothesis_label} = {description}.",
            "Give an absolute likelihood from 0 to 100, where 0 means incompatible with the causal evidence, 50 means genuinely uncertain, and 100 means strongly expected.",
            "Do not choose a class and do not make the five scores sum to 100; this request independently scores one hypothesis.",
            'Return exactly one JSON object such as {"score":63}. Return no explanation.',
        ]
    )

def _prompt_template(prompt: str, profile_text: str) -> str:
    if prompt.count(profile_text) != 1:
        raise ValueError("Rendered profile must occur exactly once in the prompt")
    return prompt.replace(profile_text, _PROFILE_PLACEHOLDER, 1)


def _load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def phase_parameters(
    phase: str, config_path: str | Path = DEFAULT_CONFIG_PATH
) -> dict[str, Any]:
    config = _load_config(config_path)
    phases = config.get("phases", {})
    if phase not in phases:
        raise ValueError(f"Unknown phase {phase!r}; choose from {sorted(phases)}")
    result = dict(phases[phase])
    result["phase"] = phase
    result["config_path"] = str(Path(config_path))
    return result


def verify_formal_readiness(
    gate_run_dir: str | Path,
    audio_control_dir: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Require a non-collapsed, complete, audio-sensitive development gate."""

    gate_root = Path(gate_run_dir)
    control_root = Path(audio_control_dir)
    config = _load_config(config_path)
    errors: list[str] = []
    required = {
        "run_config": gate_root / "run_config.json",
        "input_audit": gate_root / "input_audit.json",
        "diagnostics": gate_root / "diagnostics.json",
        "audio_sensitivity": control_root / "audio_sensitivity.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        errors.append(f"missing readiness files: {missing}")
        payloads: dict[str, dict[str, Any]] = {}
    else:
        payloads = {
            name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in required.items()
        }
    run_config = payloads.get("run_config", {})
    input_audit = payloads.get("input_audit", {})
    diagnostics = payloads.get("diagnostics", {})
    sensitivity = payloads.get("audio_sensitivity", {})
    thresholds = config.get("collapse_gate", {})
    if run_config and run_config.get("phase") != "gate50":
        errors.append("development gate phase is not gate50")
    if run_config and run_config.get("prompt_version") != config.get("prompt_version"):
        errors.append("development gate prompt version differs from the locked config")
    if run_config and run_config.get("message_content_order") != ["text", "input_audio"]:
        errors.append("development gate did not use the verified text-before-audio order")
    if input_audit and not input_audit.get("passed"):
        errors.append("development input audit did not pass")
    expected_counts = {label: 10 for label in LABELS}
    if input_audit and input_audit.get("class_counts") != expected_counts:
        errors.append("development gate is not exactly balanced at 10 samples per class")
    collapse = diagnostics.get("collapse_gate", {})
    if diagnostics and not collapse.get("hidden_noncollapsed"):
        errors.append("hidden development baseline is label-collapsed")
    if diagnostics and not collapse.get("complete_response_set"):
        errors.append("development response set is incomplete")
    minimum_controls = int(thresholds.get("minimum_audio_control_samples", 50))
    comparable = int(sensitivity.get("comparable_samples", 0) or 0)
    if sensitivity and comparable < minimum_controls:
        errors.append(
            f"audio control has {comparable} comparable samples; need {minimum_controls}"
        )
    minimum_change = float(thresholds.get("minimum_audio_change_fraction", 0.2))
    changed_fraction = sensitivity.get("changed_fraction")
    if sensitivity and (
        changed_fraction is None or float(changed_fraction) < minimum_change
    ):
        errors.append(
            f"audio change fraction is {changed_fraction}; need at least {minimum_change}"
        )
    report = {
        "passed": not errors,
        "gate_run_dir": str(gate_root.resolve()),
        "audio_control_dir": str(control_root.resolve()),
        "prompt_version": run_config.get("prompt_version"),
        "message_content_order": run_config.get("message_content_order"),
        "hidden_noncollapsed": collapse.get("hidden_noncollapsed"),
        "hidden_dominant_label_fraction": collapse.get(
            "hidden_dominant_label_fraction"
        ),
        "complete_response_set": collapse.get("complete_response_set"),
        "audio_comparable_samples": comparable,
        "audio_changed_fraction": changed_fraction,
        "thresholds": thresholds,
        "errors": errors,
    }
    write_json(gate_root / "formal_readiness.json", report)
    if errors:
        raise ValueError("Formal evaluation readiness failed: " + "; ".join(errors))
    return report


def prepare_event_eval(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    conversations: Sequence[str],
    per_class: int,
    seed: int = 13,
    context_seconds: float = 12.0,
    max_transcript_chars: int = 6000,
    min_boundary_separation_s: float = 3.0,
    max_per_conversation_class: int = 10,
    sample_rate: int = 16_000,
    forecast_lead_ms: int = DEFAULT_FORECAST_LEAD_MS,
    forecast_horizon_ms: int = DEFAULT_FORECAST_HORIZON_MS,
    prompt_style: str = "direct",
    phase: str = "custom",
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Prepare paired requests ending before a future event reference point."""

    if per_class <= 0:
        raise ValueError("per_class must be positive")
    if context_seconds <= 0:
        raise ValueError("context_seconds must be positive")
    if forecast_lead_ms <= 0:
        raise ValueError("forecast_lead_ms must be positive")
    if forecast_horizon_ms <= forecast_lead_ms:
        raise ValueError("forecast_horizon_ms must be greater than forecast_lead_ms")
    forecast_lead_s = forecast_lead_ms / 1000.0
    evaluation_window_s = (forecast_horizon_ms - forecast_lead_ms) / 1000.0
    source_manifest = Path(manifest_path).resolve()
    manifest_root = source_manifest.parent
    allowed_conversations = set(conversations)
    all_rows = list(read_jsonl(source_manifest))
    rows_by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        conversation_id = str(row.get("conversation_id", ""))
        if conversation_id in allowed_conversations and str(row.get("candidate_label", "")) in LABELS:
            rows_by_conversation[conversation_id].append(row)

    first_event_ids: set[str] = set()
    for rows in rows_by_conversation.values():
        ordered = sorted(rows, key=lambda item: float(item["anchor_s"]))
        for index, row in enumerate(ordered):
            event_time_s = float(row["anchor_s"])
            prediction_time_s = event_time_s - forecast_lead_s
            previous_anchor = (
                float(ordered[index - 1]["anchor_s"]) if index > 0 else None
            )
            next_anchor = (
                float(ordered[index + 1]["anchor_s"])
                if index + 1 < len(ordered)
                else None
            )
            previous_is_clear = (
                previous_anchor is None
                or previous_anchor < prediction_time_s - 1e-6
            )
            following_window_is_clear = (
                next_anchor is None
                or next_anchor >= event_time_s + evaluation_window_s - 1e-6
            )
            if previous_is_clear and following_window_is_clear:
                first_event_ids.add(str(row["event_id"]))

    eligible: list[dict[str, Any]] = []
    for row in all_rows:
        label = str(row.get("candidate_label", ""))
        conversation_id = str(row.get("conversation_id", ""))
        target_in_clip_s = float(row.get("target_start_in_clip_s", 0.0))
        event_id = str(row.get("event_id", ""))
        if (
            conversation_id in allowed_conversations
            and label in LABELS
            and event_id in first_event_ids
            and target_in_clip_s > forecast_lead_s + 0.04
        ):
            event_time_s = float(row["anchor_s"])
            adapted = dict(row)
            adapted["sample_id"] = event_id
            adapted["label"] = label
            adapted["prediction_time_s"] = event_time_s - forecast_lead_s
            adapted["split"] = "all"
            eligible.append(adapted)
    found_conversations = {str(row["conversation_id"]) for row in eligible}
    missing_conversations = sorted(allowed_conversations - found_conversations)
    if missing_conversations:
        raise ValueError(f"No eligible rows for conversations: {missing_conversations}")
    selected = select_conversation_balanced_rows(
        eligible,
        class_targets={label: per_class for label in LABELS},
        split="all",
        max_per_conversation_class=max_per_conversation_class,
        min_boundary_separation_s=min_boundary_separation_s,
        seed=seed,
    )
    profiles: dict[str, dict[str, Any]] = {}
    for row in eligible:
        profiles.setdefault(str(row["conversation_id"]), event_profile(row))
    shuffled = _profile_derangement(profiles)

    destination = Path(output_dir)
    clips_dir = destination / "causal_audio"
    requests: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for row in selected:
        sample_id = str(row["sample_id"])
        conversation_id = str(row["conversation_id"])
        source_audio = Path(str(row["audio_path"]))
        if not source_audio.is_absolute():
            source_audio = manifest_root / source_audio
        event_in_clip_s = float(row["target_start_in_clip_s"])
        boundary_in_clip_s = event_in_clip_s - forecast_lead_s
        window_start_in_clip_s = max(0.0, boundary_in_clip_s - context_seconds)
        audio = read_wav_window(
            source_audio,
            window_start_in_clip_s,
            boundary_in_clip_s,
            target_rate=sample_rate,
        )
        safe_id = _SAFE_FILE_PATTERN.sub("_", sample_id)
        clip_path = clips_dir / f"{safe_id}.wav"
        write_wav_mono(clip_path, audio, sample_rate=sample_rate)
        audio_hash = _sha256_file(clip_path)
        audio_duration_s = len(audio) / sample_rate
        transcript, transcript_units = causal_transcript_units(
            row,
            window_start_in_clip_s=window_start_in_clip_s,
            boundary_in_clip_s=boundary_in_clip_s,
            max_chars=max_transcript_chars,
        )
        transcript_hash = _sha256_text(transcript)
        boundary_state = causal_boundary_state(
            row,
            window_start_in_clip_s=window_start_in_clip_s,
            boundary_in_clip_s=boundary_in_clip_s,
        )
        boundary_state_text = render_boundary_state(
            boundary_state,
            audio_duration_s=audio_duration_s,
        )
        boundary_state_hash = _sha256_text(boundary_state_text)
        conditions = {
            "hidden": _unknown_profile(),
            "given": profiles[conversation_id],
            "shuffled": shuffled[conversation_id],
        }
        for profile_mode in PROFILE_MODES:
            profile_text = render_profile(conditions[profile_mode])
            prompt = build_event_prompt(
                audio_duration_s=audio_duration_s,
                transcript=transcript,
                profile_text=profile_text,
                boundary_state_text=boundary_state_text,
                causal_asr_transcript="",
                forecast_offset_ms=forecast_lead_ms,
                horizon_ms=forecast_horizon_ms,
                prompt_style=prompt_style,
            )
            prompt_template = _prompt_template(prompt, profile_text)
            requests.append(
                {
                    "request_id": f"{sample_id}::{profile_mode}",
                    "sample_id": sample_id,
                    "conversation_id": conversation_id,
                    "profile_mode": profile_mode,
                    "audio_path": str(Path("causal_audio") / clip_path.name),
                    "audio_sha256": audio_hash,
                    "audio_duration_s": round(audio_duration_s, 6),
                    "audio_sample_rate": sample_rate,
                    "decision_time_in_conversation_s": round(float(row["prediction_time_s"]), 6),
                    "source_clip_window_start_s": round(window_start_in_clip_s, 6),
                    "source_clip_boundary_s": round(boundary_in_clip_s, 6),
                    "forecast_offset_ms": forecast_lead_ms,
                    "evaluation_window_ms": forecast_horizon_ms - forecast_lead_ms,
                    "horizon_ms": forecast_horizon_ms,
                    "transcript_prefix": transcript,
                    "transcript_units": transcript_units,
                    "transcript_sha256": transcript_hash,
                    "boundary_state": boundary_state,
                    "boundary_state_text": boundary_state_text,
                    "boundary_state_sha256": boundary_state_hash,
                    "causal_asr_transcript": "",
                    "causal_asr_sha256": _sha256_text(""),
                    "profile_text": profile_text,
                    "profile_sha256": _sha256_text(profile_text),
                    "prompt": prompt,
                    "prompt_template_sha256": _sha256_text(prompt_template),
                    "request_sha256": _sha256_text(
                        audio_hash
                        + "\n"
                        + transcript_hash
                        + "\n"
                        + boundary_state_hash
                        + "\n"
                        + _sha256_text("")
                        + "\n"
                        + prompt
                    ),
                }
            )
        references.append(
            {
                "sample_id": sample_id,
                "conversation_id": conversation_id,
                "reference_label": str(row["candidate_label"]),
                "reference_source": "accepted_five_class_reference",
                "candidate_confidence": row.get("candidate_confidence"),
                "prediction_boundary_in_conversation_s": round(float(row["prediction_time_s"]), 6),
                "event_time_in_conversation_s": round(float(row["anchor_s"]), 6),
                "event_offset_ms": forecast_lead_ms,
            }
        )
        selected_rows.append(
            {
                "sample_id": sample_id,
                "conversation_id": conversation_id,
                "prediction_boundary_in_conversation_s": round(float(row["prediction_time_s"]), 6),
                "event_time_in_conversation_s": round(float(row["anchor_s"]), 6),
                "event_offset_ms": forecast_lead_ms,
                "source_audio_sha256": row.get("audio_sha256"),
                "causal_audio_sha256": audio_hash,
                "causal_audio_duration_s": round(audio_duration_s, 6),
                "completed_transcript_units": len(transcript_units),
                "active_speakers_at_boundary": [
                    item["speaker"]
                    for item in boundary_state["active_speakers_at_t"]
                ],
            }
        )

    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "requests.jsonl", requests)
    write_jsonl(destination / "reference_labels.jsonl", references)
    write_jsonl(destination / "selected_inputs.jsonl", selected_rows)
    summary = {
        "experiment": "qwen2.5-omni-3b-strict-causal-event-eval",
        "phase": phase,
        "training_performed": False,
        "model": "Qwen2.5-Omni-3B-Q8_0",
        "backend": "llama.cpp persistent server",
        "message_content_order": ["text", "input_audio"],
        "prompt_version": PROMPT_VERSION,
        "prompt_style": prompt_style,
        "manifest": str(source_manifest),
        "config_path": str(config_path) if config_path is not None else None,
        "conversations": sorted(allowed_conversations),
        "seed": seed,
        "per_class": per_class,
        "selected_samples": len(selected),
        "requests": len(requests),
        "profile_modes": list(PROFILE_MODES),
        "context_seconds": context_seconds,
        "sample_rate": sample_rate,
        "forecast_lead_ms": forecast_lead_ms,
        "horizon_ms": forecast_horizon_ms,
        "first_event_filter": "no other candidate boundary in [t, reference_event) or within the following evaluation window",
        "evaluation_window_ms": forecast_horizon_ms - forecast_lead_ms,
        "max_transcript_chars": max_transcript_chars,
        "min_boundary_separation_s": min_boundary_separation_s,
        "max_per_conversation_class": max_per_conversation_class,
        "request_file_contains_reference_labels": False,
        "reference_label_status": "accepted five-class reference labels",
        "model_input_contract": [
            "causal mono audio ending exactly at prediction boundary t",
            "matching transcript units completed no later than t",
            "speaker activity state observable exactly at t",
            "optional ASR generated once from the exact causal audio and shared by all profile modes",
            "fixed-template profile condition",
            "forecast one exact future evaluation window [t+offset, t+horizon)",
        ],
        "paired_invariant": (
            "Within each sample only profile_text/profile_sha256/profile_mode/request_id/"
            "request_sha256 may change across hidden, given, and shuffled requests."
        ),
        "causal_asr_applied": False,
        "output_schema": OUTPUT_SCHEMA,
        "output_schema_sha256": _sha256_text(
            json.dumps(OUTPUT_SCHEMA, sort_keys=True, separators=(",", ":"))
        ),
    }
    write_json(destination / "run_config.json", summary)
    audit_event_eval(
        destination,
        expected_samples=len(selected),
        expected_per_class=per_class,
    )
    return summary


def prepare_phase(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    phase: str,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    prompt_style: str | None = None,
    selection_seed: int | None = None,
) -> dict[str, Any]:
    parameters = phase_parameters(phase, config_path)
    return prepare_event_eval(
        manifest_path,
        output_dir,
        conversations=parameters["conversations"],
        per_class=int(parameters["per_class"]),
        seed=(
            int(selection_seed)
            if selection_seed is not None
            else int(parameters.get("seed", 13))
        ),
        context_seconds=float(parameters.get("context_seconds", 12.0)),
        max_transcript_chars=int(parameters.get("max_transcript_chars", 6000)),
        min_boundary_separation_s=float(
            parameters.get("min_boundary_separation_s", 3.0)
        ),
        max_per_conversation_class=int(
            parameters.get("max_per_conversation_class", 10)
        ),
        sample_rate=int(parameters.get("sample_rate", 16_000)),
        forecast_lead_ms=int(parameters.get("forecast_lead_ms", DEFAULT_FORECAST_LEAD_MS)),
        forecast_horizon_ms=int(parameters.get("forecast_horizon_ms", DEFAULT_FORECAST_HORIZON_MS)),
        prompt_style=prompt_style or str(parameters.get("prompt_style", "direct")),
        phase=phase,
        config_path=config_path,
    )


def apply_causal_asr(
    run_dir: str | Path,
    *,
    asr_path: str | Path | None = None,
) -> dict[str, Any]:
    """Attach one causal ASR result per audio sample to all profile conditions.

    This operation intentionally has no access to ``reference_labels.jsonl``.
    The same ASR text and hash are reused for hidden/given/shuffled so profile
    remains the only experimental variable.
    """

    root = Path(run_dir)
    requests_path = root / "requests.jsonl"
    asr_file = Path(asr_path) if asr_path is not None else root / "asr.jsonl"
    requests = list(read_jsonl(requests_path))
    asr_rows = list(read_jsonl(asr_file))
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    errors: list[str] = []
    for row in asr_rows:
        key = (str(row.get("sample_id", "")), str(row.get("audio_sha256", "")))
        if not all(key):
            errors.append("ASR row is missing sample_id or audio_sha256")
            continue
        if row.get("error"):
            errors.append(f"ASR failed for {key[0]}: {row['error']}")
            continue
        transcript = str(row.get("transcript", "")).strip()
        if not transcript:
            errors.append(f"ASR transcript is empty for {key[0]}")
            continue
        if key in by_key and str(by_key[key].get("transcript", "")).strip() != transcript:
            errors.append(f"ASR has conflicting transcripts for {key[0]}")
            continue
        by_key[key] = row
    request_keys = {
        (str(row.get("sample_id", "")), str(row.get("audio_sha256", "")))
        for row in requests
    }
    missing = sorted(request_keys - set(by_key))
    if missing:
        errors.append(f"missing ASR for {len(missing)} samples: {[key[0] for key in missing[:5]]}")
    if errors:
        raise ValueError("Cannot apply causal ASR: " + "; ".join(errors[:10]))

    config_path = root / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prompt_style = str(config.get("prompt_style", "hierarchical"))
    updated: list[dict[str, Any]] = []
    for request in requests:
        key = (str(request["sample_id"]), str(request["audio_sha256"]))
        causal_asr = str(by_key[key]["transcript"]).strip()
        causal_asr_hash = _sha256_text(causal_asr)
        prompt = build_event_prompt(
            audio_duration_s=float(request["audio_duration_s"]),
            transcript=str(request.get("transcript_prefix", "")),
            profile_text=str(request["profile_text"]),
            boundary_state_text=str(request.get("boundary_state_text", "")),
            causal_asr_transcript=causal_asr,
            forecast_offset_ms=int(request["forecast_offset_ms"]),
            horizon_ms=int(request["horizon_ms"]),
            prompt_style=prompt_style,
        )
        rewritten = dict(request)
        rewritten["causal_asr_transcript"] = causal_asr
        rewritten["causal_asr_sha256"] = causal_asr_hash
        rewritten["prompt"] = prompt
        rewritten["prompt_template_sha256"] = _sha256_text(
            _prompt_template(prompt, str(request["profile_text"]))
        )
        rewritten["request_sha256"] = _sha256_text(
            str(request["audio_sha256"])
            + "\n"
            + str(request["transcript_sha256"])
            + "\n"
            + str(request["boundary_state_sha256"])
            + "\n"
            + causal_asr_hash
            + "\n"
            + prompt
        )
        updated.append(rewritten)
    write_jsonl(requests_path, updated)
    config["prompt_version"] = PROMPT_VERSION
    config["causal_asr_applied"] = True
    config["causal_asr_file"] = str(asr_file.resolve())
    config["causal_asr_samples"] = len(request_keys)
    config["model_input_contract"] = [
        "causal mono audio ending exactly at prediction boundary t",
        "matching transcript units completed no later than t",
        "speaker activity state observable exactly at t",
        "ASR generated once from the exact causal audio and shared by all profile modes",
        "fixed-template profile condition",
        "forecast one exact future evaluation window [t+offset, t+horizon)",
    ]
    write_json(config_path, config)
    audit = audit_event_eval(root)
    return {
        "run_dir": str(root.resolve()),
        "samples": len(request_keys),
        "requests": len(updated),
        "causal_asr_applied": True,
        "audit_passed": audit["passed"],
    }


def prepare_candidate_score_eval(
    source_run_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create five one-vs-rest score requests for every paired model input."""

    source = Path(source_run_dir)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Candidate score output directory is not empty: {destination}")
    base_audit = audit_event_eval(source)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / "causal_audio", destination / "causal_audio")
    for name in (
        "requests.jsonl",
        "reference_labels.jsonl",
        "selected_inputs.jsonl",
        "run_config.json",
    ):
        shutil.copy2(source / name, destination / name)
    if (source / "asr.jsonl").exists():
        shutil.copy2(source / "asr.jsonl", destination / "asr.jsonl")
    audit_event_eval(destination)

    base_requests = list(read_jsonl(destination / "requests.jsonl"))
    candidate_requests: list[dict[str, Any]] = []
    for base in base_requests:
        for hypothesis_label in LABELS:
            profile_text = str(base["profile_text"])
            prompt = build_candidate_score_prompt(
                audio_duration_s=float(base["audio_duration_s"]),
                transcript=str(base.get("transcript_prefix", "")),
                profile_text=profile_text,
                boundary_state_text=str(base["boundary_state_text"]),
                causal_asr_transcript=str(base["causal_asr_transcript"]),
                hypothesis_label=hypothesis_label,
                forecast_offset_ms=int(base["forecast_offset_ms"]),
                horizon_ms=int(base["horizon_ms"]),
            )
            candidate_requests.append(
                {
                    "request_id": f"{base['request_id']}::score_{hypothesis_label}",
                    "base_request_id": base["request_id"],
                    "sample_id": base["sample_id"],
                    "conversation_id": base["conversation_id"],
                    "profile_mode": base["profile_mode"],
                    "hypothesis_label": hypothesis_label,
                    "audio_path": base["audio_path"],
                    "audio_sha256": base["audio_sha256"],
                    "audio_duration_s": base["audio_duration_s"],
                    "decision_time_in_conversation_s": base[
                        "decision_time_in_conversation_s"
                    ],
                    "forecast_offset_ms": base["forecast_offset_ms"],
                    "horizon_ms": base["horizon_ms"],
                    "transcript_prefix": base["transcript_prefix"],
                    "transcript_sha256": base["transcript_sha256"],
                    "boundary_state_text": base["boundary_state_text"],
                    "boundary_state_sha256": base["boundary_state_sha256"],
                    "causal_asr_transcript": base["causal_asr_transcript"],
                    "causal_asr_sha256": base["causal_asr_sha256"],
                    "profile_text": profile_text,
                    "profile_sha256": base["profile_sha256"],
                    "prompt": prompt,
                    "prompt_template_sha256": _sha256_text(
                        _prompt_template(prompt, profile_text)
                    ),
                    "request_sha256": _sha256_text(
                        str(base["audio_sha256"])
                        + "\n"
                        + str(base["transcript_sha256"])
                        + "\n"
                        + str(base["boundary_state_sha256"])
                        + "\n"
                        + str(base["causal_asr_sha256"])
                        + "\n"
                        + hypothesis_label
                        + "\n"
                        + prompt
                    ),
                }
            )
    write_jsonl(destination / "candidate_requests.jsonl", candidate_requests)
    config_path = destination / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["inference_method"] = "one_vs_rest_candidate_scoring_v1"
    config["candidate_hypotheses"] = list(LABELS)
    config["candidate_requests"] = len(candidate_requests)
    config["candidate_output_schema"] = SCORE_OUTPUT_SCHEMA
    write_json(config_path, config)
    candidate_audit = audit_candidate_score_eval(destination)
    return {
        "source_run_dir": str(source.resolve()),
        "output_dir": str(destination.resolve()),
        "samples": base_audit["selected_samples"],
        "base_requests": base_audit["requests"],
        "candidate_requests": len(candidate_requests),
        "audit_passed": candidate_audit["passed"],
    }


def audit_candidate_score_eval(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    base = audit_event_eval(root)
    base_requests = {
        str(row["request_id"]): row for row in read_jsonl(root / "requests.jsonl")
    }
    candidates = list(read_jsonl(root / "candidate_requests.jsonl"))
    errors: list[str] = []
    expected = base["requests"] * len(LABELS)
    if len(candidates) != expected:
        errors.append(f"expected {expected} candidate requests, found {len(candidates)}")
    request_ids = [str(row.get("request_id", "")) for row in candidates]
    if len(request_ids) != len(set(request_ids)):
        errors.append("candidate requests contain duplicate request_id values")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        request_id = str(row.get("request_id", ""))
        base_request = base_requests.get(str(row.get("base_request_id", "")))
        if base_request is None:
            errors.append(f"candidate {request_id} has no base request")
            continue
        hypothesis = str(row.get("hypothesis_label", ""))
        if hypothesis not in LABELS:
            errors.append(f"candidate {request_id} has invalid hypothesis {hypothesis}")
        forbidden = _nested_keys(row) & _FORBIDDEN_REQUEST_KEYS
        if forbidden:
            errors.append(f"candidate {request_id} contains forbidden keys {sorted(forbidden)}")
        for field in (
            "sample_id",
            "conversation_id",
            "profile_mode",
            "audio_path",
            "audio_sha256",
            "transcript_prefix",
            "transcript_sha256",
            "boundary_state_text",
            "boundary_state_sha256",
            "causal_asr_transcript",
            "causal_asr_sha256",
            "profile_text",
            "profile_sha256",
        ):
            if row.get(field) != base_request.get(field):
                errors.append(f"candidate {request_id} changes base field {field}")
        profile_text = str(row.get("profile_text", ""))
        prompt = str(row.get("prompt", ""))
        if not profile_text or prompt.count(profile_text) != 1:
            errors.append(f"candidate {request_id} profile rendering mismatch")
        elif _sha256_text(_prompt_template(prompt, profile_text)) != row.get(
            "prompt_template_sha256"
        ):
            errors.append(f"candidate {request_id} prompt template mismatch")
        expected_hash = _sha256_text(
            str(row.get("audio_sha256", ""))
            + "\n"
            + str(row.get("transcript_sha256", ""))
            + "\n"
            + str(row.get("boundary_state_sha256", ""))
            + "\n"
            + str(row.get("causal_asr_sha256", ""))
            + "\n"
            + hypothesis
            + "\n"
            + prompt
        )
        if expected_hash != row.get("request_sha256"):
            errors.append(f"candidate {request_id} request hash mismatch")
        grouped[(str(row.get("sample_id", "")), hypothesis)].append(row)
    expected_groups = base["selected_samples"] * len(LABELS)
    if len(grouped) != expected_groups:
        errors.append(f"expected {expected_groups} sample/hypothesis groups, found {len(grouped)}")
    for key, rows in grouped.items():
        modes = {str(row.get("profile_mode", "")) for row in rows}
        if modes != set(PROFILE_MODES):
            errors.append(f"candidate group {key} has modes {sorted(modes)}")
        for field in (
            "audio_sha256",
            "transcript_sha256",
            "boundary_state_sha256",
            "causal_asr_sha256",
            "prompt_template_sha256",
        ):
            if len({str(row.get(field, "")) for row in rows}) != 1:
                errors.append(f"candidate group {key} changes non-profile field {field}")
    report = {
        "passed": not errors,
        "samples": base["selected_samples"],
        "base_requests": base["requests"],
        "candidate_requests": len(candidates),
        "hypotheses": list(LABELS),
        "profile_modes": list(PROFILE_MODES),
        "output_schema": SCORE_OUTPUT_SCHEMA,
        "errors": errors,
    }
    write_json(root / "candidate_input_audit.json", report)
    if errors:
        raise ValueError("Candidate score input audit failed: " + "; ".join(errors[:10]))
    return report


def _parse_candidate_score(raw: str) -> int | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'"score"\s*:\s*(\d{1,3})', raw)
        if not match:
            return None
        value = int(match.group(1))
    else:
        value = payload.get("score") if isinstance(payload, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if int(value) != float(value):
            return None
        value = int(value)
    return value if 0 <= value <= 100 else None


def run_candidate_score_server(
    run_dir: str | Path,
    *,
    endpoint: str = "http://127.0.0.1:8091/v1/chat/completions",
    model: str = "Qwen2.5-Omni-3B-Q8_0",
    timeout_s: float = 180.0,
    retries: int = 2,
    seed: int = 13,
    limit: int | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    audit_candidate_score_eval(root)
    rows = list(read_jsonl(root / "candidate_requests.jsonl"))
    if limit is not None:
        rows = rows[:limit]
    destination = root / "candidate_responses.jsonl"
    existing = list(read_jsonl(destination)) if destination.exists() else []
    completed = {str(row["request_id"]) for row in existing}
    written = 0
    invalid = 0
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            request_id = str(row["request_id"])
            if request_id in completed:
                continue
            request_row = dict(row)
            audio_path = Path(str(request_row["audio_path"]))
            if not audio_path.is_absolute():
                audio_path = root / audio_path
            request_row["audio_path"] = str(audio_path.resolve())
            raw = ""
            error = ""
            started = time.perf_counter()
            for attempt in range(retries + 1):
                try:
                    raw = _server_chat_completion(
                        endpoint,
                        request_row,
                        model=model,
                        timeout_s=timeout_s,
                        seed=seed,
                        structured_output=True,
                        max_tokens=16,
                        output_schema=SCORE_OUTPUT_SCHEMA,
                        schema_name="turn_taking_hypothesis_score",
                    )
                    error = ""
                    break
                except (KeyError, ValueError, OSError, urllib.error.URLError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt < retries:
                        time.sleep(min(2**attempt, 8))
            score = _parse_candidate_score(raw) if not error else None
            valid = score is not None
            if not valid:
                invalid += 1
            response = {
                "request_id": request_id,
                "base_request_id": row["base_request_id"],
                "sample_id": row["sample_id"],
                "conversation_id": row["conversation_id"],
                "profile_mode": row["profile_mode"],
                "hypothesis_label": row["hypothesis_label"],
                "request_sha256": row["request_sha256"],
                "audio_sha256": row["audio_sha256"],
                "score": score,
                "valid": valid,
                "raw_response": raw,
                "error": error,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
            handle.write(json.dumps(response, ensure_ascii=False) + "\n")
            handle.flush()
            completed.add(request_id)
            written += 1
    return {
        "requested": len(rows),
        "already_completed": len(rows) - written,
        "newly_written": written,
        "new_invalid": invalid,
        "responses": str(destination.resolve()),
    }


def aggregate_candidate_scores(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    audit = audit_candidate_score_eval(root)
    candidate_responses = list(read_jsonl(root / "candidate_responses.jsonl"))
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    invalid = 0
    for row in candidate_responses:
        if not row.get("valid") or row.get("score") is None:
            invalid += 1
            continue
        key = (str(row["sample_id"]), str(row["profile_mode"]))
        grouped[key][str(row["hypothesis_label"])] = row
    predictions: list[dict[str, Any]] = []
    standard_responses: list[dict[str, Any]] = []
    tie_count = 0
    incomplete: list[str] = []
    for base in read_jsonl(root / "requests.jsonl"):
        key = (str(base["sample_id"]), str(base["profile_mode"]))
        scores_by_label = grouped.get(key, {})
        if set(scores_by_label) != set(LABELS):
            incomplete.append(f"{key[0]}::{key[1]}")
            continue
        scores = {label: int(scores_by_label[label]["score"]) for label in LABELS}
        maximum = max(scores.values())
        tied = [label for label in LABELS if scores[label] == maximum]
        if len(tied) > 1:
            tie_count += 1
        prediction = tied[0]
        predictions.append(
            {
                "sample_id": key[0],
                "conversation_id": base["conversation_id"],
                "profile_mode": key[1],
                "scores": scores,
                "prediction": prediction,
                "maximum_score": maximum,
                "tie_labels": tied,
            }
        )
        standard_responses.append(
            {
                "request_id": base["request_id"],
                "sample_id": key[0],
                "conversation_id": base["conversation_id"],
                "profile_mode": key[1],
                "model": "Qwen2.5-Omni-3B-Q8_0 candidate scoring",
                "request_sha256": base["request_sha256"],
                "audio_sha256": base["audio_sha256"],
                "transcript_sha256": base["transcript_sha256"],
                "prediction": prediction,
                "valid": True,
                "raw_response": json.dumps(scores, ensure_ascii=False),
                "error": "",
                "latency_ms": sum(
                    float(scores_by_label[label].get("latency_ms", 0.0)) for label in LABELS
                ),
            }
        )
    if incomplete:
        raise ValueError(f"Incomplete candidate scores for {len(incomplete)} inputs: {incomplete[:5]}")
    write_jsonl(root / "candidate_score_predictions.jsonl", predictions)
    write_jsonl(root / "responses.jsonl", standard_responses)
    summary = {
        "samples": audit["samples"],
        "paired_inputs": len(predictions),
        "candidate_responses": len(candidate_responses),
        "invalid_candidate_responses": invalid,
        "ties": tie_count,
        "prediction_distribution": {
            mode: {
                label: sum(
                    row["profile_mode"] == mode and row["prediction"] == label
                    for row in predictions
                )
                for label in LABELS
            }
            for mode in PROFILE_MODES
        },
    }
    write_json(root / "candidate_aggregation.json", summary)
    return summary


def audit_event_eval(
    run_dir: str | Path,
    *,
    expected_samples: int | None = None,
    expected_per_class: int | None = None,
) -> dict[str, Any]:
    """Fail closed on leakage, causality, pairing, audio, and class balance."""

    root = Path(run_dir)
    requests = list(read_jsonl(root / "requests.jsonl"))
    references = list(read_jsonl(root / "reference_labels.jsonl"))
    config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    expected_samples = expected_samples or int(config.get("selected_samples", 0))
    expected_per_class = expected_per_class or int(config.get("per_class", 0))
    forecast_lead_ms = int(config.get("forecast_lead_ms", 0))
    forecast_horizon_ms = int(config.get("horizon_ms", 0))
    errors: list[str] = []
    warnings: list[str] = []
    if forecast_lead_ms <= 0:
        errors.append("run_config forecast_lead_ms must be positive")
    if forecast_horizon_ms <= forecast_lead_ms:
        errors.append("run_config horizon must be greater than forecast lead")

    request_ids = [str(row.get("request_id", "")) for row in requests]
    if len(request_ids) != len(set(request_ids)):
        errors.append("requests.jsonl contains duplicate request_id values")
    reference_ids = [str(row.get("sample_id", "")) for row in references]
    if len(reference_ids) != len(set(reference_ids)):
        errors.append("reference_labels.jsonl contains duplicate sample_id values")
    if len(references) != expected_samples:
        errors.append(f"expected {expected_samples} references, found {len(references)}")
    reference_by_sample = {str(row.get("sample_id", "")): row for row in references}
    for reference in references:
        sample_id = str(reference.get("sample_id", ""))
        boundary_s = float(reference.get("prediction_boundary_in_conversation_s", math.nan))
        event_s = float(reference.get("event_time_in_conversation_s", math.nan))
        offset_ms = int(reference.get("event_offset_ms", -1))
        if not math.isfinite(boundary_s) or not math.isfinite(event_s):
            errors.append(f"reference {sample_id} has invalid boundary/event time")
        elif abs((event_s - boundary_s) * 1000.0 - forecast_lead_ms) > 1e-3:
            errors.append(f"reference {sample_id} event offset does not match forecast lead")
        if offset_ms != forecast_lead_ms:
            errors.append(f"reference {sample_id} event_offset_ms mismatch")
    class_counts = Counter(str(row.get("reference_label", "")) for row in references)
    for label in LABELS:
        if class_counts[label] != expected_per_class:
            errors.append(
                f"expected {expected_per_class} reference rows for {label}, "
                f"found {class_counts[label]}"
            )

    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in requests:
        by_sample[str(request.get("sample_id", ""))].append(request)
    if set(by_sample) != set(reference_ids):
        errors.append("sample_id sets differ between requests and references")
    if len(requests) != expected_samples * len(PROFILE_MODES):
        errors.append("request count is not exactly three per selected sample")

    manifest_path = Path(str(config.get("manifest", "")))
    if not manifest_path.is_file():
        errors.append("source manifest needed for first-event audit is missing")
    else:
        anchors_by_conversation: dict[str, list[float]] = defaultdict(list)
        for source_row in read_jsonl(manifest_path):
            if str(source_row.get("candidate_label", "")) in LABELS:
                anchors_by_conversation[str(source_row.get("conversation_id", ""))].append(
                    float(source_row["anchor_s"])
                )
        for anchors in anchors_by_conversation.values():
            anchors.sort()
        for reference in references:
            sample_id = str(reference.get("sample_id", ""))
            conversation_id = str(reference.get("conversation_id", ""))
            boundary_s = float(reference.get("prediction_boundary_in_conversation_s", math.nan))
            event_s = float(reference.get("event_time_in_conversation_s", math.nan))
            intervening = [
                anchor
                for anchor in anchors_by_conversation.get(conversation_id, [])
                if boundary_s - 1e-6 <= anchor < event_s - 1e-6
            ]
            if intervening:
                errors.append(
                    f"reference {sample_id} is not the first candidate event after boundary"
                )
            following = [
                anchor
                for anchor in anchors_by_conversation.get(conversation_id, [])
                if event_s + 1e-6 < anchor < event_s + (forecast_horizon_ms - forecast_lead_ms) / 1000.0 - 1e-6
            ]
            if following:
                errors.append(
                    f"reference {sample_id} has another candidate inside its evaluation window"
                )

    invariant_fields = (
        "sample_id",
        "conversation_id",
        "audio_path",
        "audio_sha256",
        "audio_duration_s",
        "audio_sample_rate",
        "decision_time_in_conversation_s",
        "source_clip_window_start_s",
        "source_clip_boundary_s",
        "forecast_offset_ms",
        "evaluation_window_ms",
        "horizon_ms",
        "transcript_prefix",
        "transcript_units",
        "transcript_sha256",
        "boundary_state",
        "boundary_state_text",
        "boundary_state_sha256",
        "causal_asr_transcript",
        "causal_asr_sha256",
        "prompt_template_sha256",
    )
    allowed_changing = {
        "profile_mode",
        "profile_text",
        "profile_sha256",
        "prompt",
        "request_id",
        "request_sha256",
    }
    audio_cache: dict[Path, tuple[str, int, int, int]] = {}
    for sample_id, rows in by_sample.items():
        modes = Counter(str(row.get("profile_mode", "")) for row in rows)
        if modes != Counter({mode: 1 for mode in PROFILE_MODES}):
            errors.append(f"sample {sample_id} profile modes are {dict(modes)}")
            continue
        all_keys = set().union(*(row.keys() for row in rows))
        unexpected_change = []
        for field in sorted(all_keys - allowed_changing):
            values = {json.dumps(row.get(field), sort_keys=True) for row in rows}
            if len(values) != 1:
                unexpected_change.append(field)
        for field in invariant_fields:
            values = {json.dumps(row.get(field), sort_keys=True) for row in rows}
            if len(values) != 1:
                unexpected_change.append(field)
        if unexpected_change:
            errors.append(
                f"sample {sample_id} changes non-profile fields "
                f"{sorted(set(unexpected_change))}"
            )
        profile_texts = {str(row["profile_mode"]): str(row["profile_text"]) for row in rows}
        if profile_texts.get("given") == profile_texts.get("shuffled"):
            errors.append(f"sample {sample_id} given and shuffled profiles are identical")
        for request in rows:
            request_id = str(request.get("request_id", ""))
            forbidden = _nested_keys(request) & _FORBIDDEN_REQUEST_KEYS
            if forbidden:
                errors.append(
                    f"request {request_id} contains forbidden keys {sorted(forbidden)}"
                )
            transcript = str(request.get("transcript_prefix", ""))
            if _sha256_text(transcript) != request.get("transcript_sha256"):
                errors.append(f"request {request_id} transcript hash mismatch")
            boundary_state_text = str(request.get("boundary_state_text", ""))
            if _sha256_text(boundary_state_text) != request.get("boundary_state_sha256"):
                errors.append(f"request {request_id} boundary state hash mismatch")
            boundary_state = request.get("boundary_state", {})
            if set(boundary_state) != {"active_speakers_at_t", "active_speaker_count"}:
                errors.append(f"request {request_id} boundary state schema mismatch")
            for active in boundary_state.get("active_speakers_at_t", []):
                if set(active) != {"speaker", "observed_start_s"}:
                    errors.append(f"request {request_id} leaks extra boundary state fields")
                elif float(active.get("observed_start_s", math.inf)) > float(
                    request.get("audio_duration_s", -1)
                ) + 1e-6:
                    errors.append(f"request {request_id} boundary speaker starts after t")
            causal_asr = str(request.get("causal_asr_transcript", ""))
            if _sha256_text(causal_asr) != request.get("causal_asr_sha256"):
                errors.append(f"request {request_id} causal ASR hash mismatch")
            duration = float(request.get("audio_duration_s", -1.0))
            offset_ms = int(request.get("forecast_offset_ms", -1))
            window_ms = int(request.get("evaluation_window_ms", -1))
            horizon_ms = int(request.get("horizon_ms", -1))
            if offset_ms != forecast_lead_ms:
                errors.append(f"request {request_id} forecast offset mismatch")
            if window_ms != forecast_horizon_ms - forecast_lead_ms:
                errors.append(f"request {request_id} evaluation window mismatch")
            if horizon_ms != forecast_horizon_ms:
                errors.append(f"request {request_id} forecast horizon mismatch")
            units = request.get("transcript_units", [])
            if any(float(unit.get("end_s", math.inf)) > duration + 1e-6 for unit in units):
                errors.append(f"request {request_id} contains future transcript")
            profile_text = str(request.get("profile_text", ""))
            prompt = str(request.get("prompt", ""))
            boundary_marker = f"<PREDICTION_BOUNDARY t=END_OF_AUDIO={duration:.3f}s>"
            if boundary_marker not in prompt:
                errors.append(f"request {request_id} lacks the exact audio-end boundary marker")
            expected_interval = f"[t+{forecast_lead_ms} ms, t+{forecast_horizon_ms} ms)"
            if expected_interval not in prompt:
                errors.append(f"request {request_id} lacks the locked forecast interval")
            reference = reference_by_sample.get(str(request.get("sample_id", "")), {})
            if abs(
                float(request.get("decision_time_in_conversation_s", math.nan))
                - float(reference.get("prediction_boundary_in_conversation_s", math.nan))
            ) > 1e-6:
                errors.append(f"request {request_id} prediction boundary mismatch")
            if not profile_text or prompt.count(profile_text) != 1:
                errors.append(f"request {request_id} profile rendering mismatch")
            elif _sha256_text(_prompt_template(prompt, profile_text)) != request.get(
                "prompt_template_sha256"
            ):
                errors.append(f"request {request_id} prompt template mismatch")
            if transcript:
                if transcript not in prompt:
                    errors.append(f"request {request_id} transcript is absent from prompt")
            elif "No completed transcript unit is available." not in prompt:
                errors.append(f"request {request_id} empty transcript marker is absent")
            if boundary_state_text not in prompt:
                errors.append(f"request {request_id} boundary state is absent from prompt")
            if causal_asr:
                if causal_asr not in prompt:
                    errors.append(f"request {request_id} causal ASR is absent from prompt")
            elif "No separate causal ASR transcript is available" not in prompt:
                errors.append(f"request {request_id} empty causal ASR marker is absent")
            expected_request_hash = _sha256_text(
                str(request.get("audio_sha256", ""))
                + "\n"
                + str(request.get("transcript_sha256", ""))
                + "\n"
                + str(request.get("boundary_state_sha256", ""))
                + "\n"
                + str(request.get("causal_asr_sha256", ""))
                + "\n"
                + prompt
            )
            if expected_request_hash != request.get("request_sha256"):
                errors.append(f"request {request_id} request hash mismatch")
            audio_path = Path(str(request.get("audio_path", "")))
            if not audio_path.is_absolute():
                audio_path = root / audio_path
            audio_path = audio_path.resolve()
            if not audio_path.is_file():
                errors.append(f"request {request_id} audio file is missing")
                continue
            if audio_path not in audio_cache:
                with wave.open(str(audio_path), "rb") as wav:
                    audio_cache[audio_path] = (
                        _sha256_file(audio_path),
                        wav.getnchannels(),
                        wav.getframerate(),
                        wav.getnframes(),
                    )
            observed_hash, channels, rate, frames = audio_cache[audio_path]
            if observed_hash != request.get("audio_sha256"):
                errors.append(f"request {request_id} audio hash mismatch")
            if channels != 1:
                errors.append(f"request {request_id} audio is not mono")
            if rate != int(request.get("audio_sample_rate", -1)):
                errors.append(f"request {request_id} audio sample rate mismatch")
            if abs(frames / rate - duration) > 1e-4:
                errors.append(f"request {request_id} audio duration mismatch")
            window_duration = float(request.get("source_clip_boundary_s", -1)) - float(
                request.get("source_clip_window_start_s", -1)
            )
            if abs(window_duration - duration) > 1e-4:
                errors.append(f"request {request_id} causal window mismatch")

    schema_hash = _sha256_text(
        json.dumps(OUTPUT_SCHEMA, sort_keys=True, separators=(",", ":"))
    )
    if config.get("message_content_order") != ["text", "input_audio"]:
        errors.append("run_config does not lock the verified text-before-audio ordering")
    if schema_hash != config.get("output_schema_sha256"):
        errors.append("run_config output schema does not match the runner schema")
    report = {
        "passed": not errors,
        "selected_samples": len(references),
        "requests": len(requests),
        "class_counts": {label: class_counts[label] for label in LABELS},
        "conversations": sorted({str(row["conversation_id"]) for row in references}),
        "causal_audio_files_checked": len(audio_cache),
        "paired_profile_modes": list(PROFILE_MODES),
        "causal_asr_applied": bool(config.get("causal_asr_applied", False)),
        "causal_asr_nonempty_samples": len(
            {
                str(row.get("sample_id", ""))
                for row in requests
                if str(row.get("causal_asr_transcript", "")).strip()
            }
        ),
        "forecast_lead_ms": forecast_lead_ms,
        "forecast_horizon_ms": forecast_horizon_ms,
        "evaluation_window_ms": forecast_horizon_ms - forecast_lead_ms,
        "first_event_after_boundary_checked": True,
        "isolated_evaluation_window_checked": True,
        "request_file_contains_reference_labels": False,
        "output_schema": OUTPUT_SCHEMA,
        "errors": errors,
        "warnings": warnings,
    }
    write_json(root / "input_audit.json", report)
    if errors:
        raise ValueError("Event evaluation input audit failed: " + "; ".join(errors[:10]))
    return report


def _paired_change_summary(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    result = {
        "given_fixes_hidden_error": 0,
        "given_breaks_hidden_correct": 0,
        "both_correct": 0,
        "both_wrong": 0,
    }
    for row in rows:
        hidden_correct = row["hidden_prediction"] == row["reference_label"]
        given_correct = row["given_prediction"] == row["reference_label"]
        if given_correct and not hidden_correct:
            result["given_fixes_hidden_error"] += 1
        elif hidden_correct and not given_correct:
            result["given_breaks_hidden_correct"] += 1
        elif hidden_correct and given_correct:
            result["both_correct"] += 1
        else:
            result["both_wrong"] += 1
    return result


def _exact_mcnemar_pvalue(fixes: int, breaks: int) -> float:
    discordant = fixes + breaks
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(fixes, breaks) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _bootstrap(
    rows: Sequence[dict[str, Any]],
    reports: dict[str, dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_conversation[str(row["conversation_id"])].append(row)
    conversation_ids = sorted(by_conversation)
    rng = random.Random(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(resamples):
        sampled: list[dict[str, Any]] = []
        for _ in conversation_ids:
            sampled.extend(by_conversation[rng.choice(conversation_ids)])
        targets = [LABEL_TO_ID[row["reference_label"]] for row in sampled]
        sampled_reports: dict[str, dict[str, Any]] = {}
        for mode in PROFILE_MODES:
            predictions = [LABEL_TO_ID[row[f"{mode}_prediction"]] for row in sampled]
            sampled_reports[mode] = classification_metrics(targets, predictions)
            draws[f"{mode}_macro_f1"].append(sampled_reports[mode]["macro_f1"])
        draws["given_minus_hidden_macro_f1"].append(
            sampled_reports["given"]["macro_f1"]
            - sampled_reports["hidden"]["macro_f1"]
        )
        draws["given_minus_shuffled_macro_f1"].append(
            sampled_reports["given"]["macro_f1"]
            - sampled_reports["shuffled"]["macro_f1"]
        )
    points = {
        **{f"{mode}_macro_f1": reports[mode]["macro_f1"] for mode in PROFILE_MODES},
        "given_minus_hidden_macro_f1": (
            reports["given"]["macro_f1"] - reports["hidden"]["macro_f1"]
        ),
        "given_minus_shuffled_macro_f1": (
            reports["given"]["macro_f1"] - reports["shuffled"]["macro_f1"]
        ),
    }
    intervals = {}
    for metric, values in draws.items():
        array = np.asarray(values, dtype=np.float64)
        intervals[metric] = {
            "point": float(points[metric]),
            "lower_2_5": float(np.percentile(array, 2.5)),
            "upper_97_5": float(np.percentile(array, 97.5)),
        }
    return {
        "cluster_unit": "conversation_id",
        "clusters": len(conversation_ids),
        "resamples": resamples,
        "seed": seed,
        "intervals": intervals,
    }


def _write_predictions_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "sample_id",
        "conversation_id",
        "reference_label",
        "hidden_prediction",
        "given_prediction",
        "shuffled_prediction",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def score_event_eval(
    run_dir: str | Path,
    *,
    bootstrap_resamples: int = 2000,
    seed: int = 13,
) -> dict[str, Any]:
    """Score only samples with valid outputs in all three profile conditions."""

    root = Path(run_dir)
    audit = audit_event_eval(root)
    references = {
        str(row["sample_id"]): row for row in read_jsonl(root / "reference_labels.jsonl")
    }
    responses = list(read_jsonl(root / "responses.jsonl"))
    by_sample: dict[str, dict[str, Any]] = {}
    invalid_by_mode = {mode: 0 for mode in PROFILE_MODES}
    distribution = {mode: Counter() for mode in PROFILE_MODES}
    latencies: list[float] = []
    for response in responses:
        sample_id = str(response.get("sample_id", ""))
        mode = str(response.get("profile_mode", ""))
        if sample_id not in references or mode not in PROFILE_MODES:
            continue
        prediction = str(response.get("prediction", ""))
        if not response.get("valid") or prediction not in LABELS:
            invalid_by_mode[mode] += 1
            continue
        reference = references[sample_id]
        row = by_sample.setdefault(
            sample_id,
            {
                "sample_id": sample_id,
                "conversation_id": reference["conversation_id"],
                "reference_label": reference["reference_label"],
            },
        )
        row[f"{mode}_prediction"] = prediction
        distribution[mode][prediction] += 1
        if isinstance(response.get("latency_ms"), (int, float)):
            latencies.append(float(response["latency_ms"]))
    paired = [
        row
        for row in by_sample.values()
        if all(f"{mode}_prediction" in row for mode in PROFILE_MODES)
    ]
    paired.sort(key=lambda row: str(row["sample_id"]))
    if not paired:
        raise ValueError("No samples have valid hidden/given/shuffled responses")
    targets = [LABEL_TO_ID[row["reference_label"]] for row in paired]
    reports: dict[str, dict[str, Any]] = {}
    for mode in PROFILE_MODES:
        predictions = [LABEL_TO_ID[row[f"{mode}_prediction"]] for row in paired]
        reports[mode] = classification_metrics(targets, predictions)
    write_json(root / "metrics.json", reports)
    write_metrics_csv(root / "profile_comparison.csv", reports)
    write_json(root / "predictions.json", paired)
    _write_predictions_csv(root / "predictions.csv", paired)
    paired_changes = _paired_change_summary(paired)
    write_json(root / "paired_changes.json", paired_changes)
    bootstrap = _bootstrap(
        paired, reports, resamples=bootstrap_resamples, seed=seed
    )
    write_json(root / "bootstrap_95ci.json", bootstrap)

    hidden_total = sum(distribution["hidden"].values())
    hidden_dominant = max(distribution["hidden"].values(), default=0)
    hidden_distinct = len(distribution["hidden"])
    hidden_dominant_fraction = hidden_dominant / hidden_total if hidden_total else None
    hidden_noncollapsed = bool(
        hidden_total
        and hidden_distinct >= 3
        and hidden_dominant_fraction is not None
        and hidden_dominant_fraction <= 0.80
    )
    complete = len(paired) == len(references) and not any(invalid_by_mode.values())
    latency_array = np.asarray(latencies, dtype=np.float64)
    diagnostics = {
        "reference_status": "accepted five-class reference labels",
        "prediction_distribution": {
            mode: {label: distribution[mode][label] for label in LABELS}
            for mode in PROFILE_MODES
        },
        "response_validity": {
            "expected_requests": audit["requests"],
            "received_responses": len(responses),
            "paired_valid_samples": len(paired),
            "invalid_by_mode": invalid_by_mode,
            "complete": complete,
        },
        "profile_pair_changes": {
            f"{left}_vs_{right}": {
                "changed": sum(
                    row[f"{left}_prediction"] != row[f"{right}_prediction"]
                    for row in paired
                ),
                "samples": len(paired),
            }
            for left, right in (
                ("hidden", "given"),
                ("hidden", "shuffled"),
                ("given", "shuffled"),
            )
        },
        "given_vs_hidden_exact_mcnemar_p": _exact_mcnemar_pvalue(
            paired_changes["given_fixes_hidden_error"],
            paired_changes["given_breaks_hidden_correct"],
        ),
        "latency_ms": {
            "count": int(latency_array.size),
            "mean": float(latency_array.mean()) if latency_array.size else None,
            "median": float(np.median(latency_array)) if latency_array.size else None,
            "p95": float(np.percentile(latency_array, 95)) if latency_array.size else None,
        },
        "collapse_gate": {
            "hidden_distinct_predicted_labels": hidden_distinct,
            "hidden_dominant_label_fraction": hidden_dominant_fraction,
            "hidden_noncollapsed": hidden_noncollapsed,
            "complete_response_set": complete,
            "passed": hidden_noncollapsed and complete,
            "profile_effect_claim_allowed": False,
            "reason": (
                "The hidden baseline is non-collapsed; audio sensitivity must still pass "
                "before interpreting profile differences."
                if hidden_noncollapsed and complete
                else "The hidden baseline is collapsed or incomplete; diagnose it on the development set."
            ),
        },
    }
    write_json(root / "diagnostics.json", diagnostics)
    result = {
        "metrics": reports,
        "paired_changes": paired_changes,
        "bootstrap_95ci": bootstrap,
        "diagnostics": diagnostics,
    }
    return result


def run_event_server(
    run_dir: str | Path,
    *,
    endpoint: str = "http://127.0.0.1:8091/v1/chat/completions",
    model: str = "qwen2.5-omni-3b-q4_k_m",
    timeout_s: float = 180.0,
    retries: int = 2,
    seed: int = 13,
    limit: int | None = None,
    structured_output: bool = True,
    max_tokens: int = 16,
) -> dict[str, Any]:
    root = Path(run_dir)
    audit_event_eval(root)
    return run_mllm_server_requests(
        root / "requests.jsonl",
        root / "responses.jsonl",
        endpoint=endpoint,
        model=model,
        timeout_s=timeout_s,
        retries=retries,
        seed=seed,
        limit=limit,
        structured_output=structured_output,
        max_tokens=max_tokens,
    )


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "apply_causal_asr",
    "audit_event_eval",
    "build_event_prompt",
    "causal_boundary_state",
    "causal_transcript_units",
    "event_profile",
    "phase_parameters",
    "prepare_event_eval",
    "prepare_phase",
    "prepare_silenced_audio_control",
    "render_boundary_state",
    "render_profile",
    "run_event_server",
    "score_event_eval",
    "score_silenced_audio_control",
    "verify_formal_readiness",
]
