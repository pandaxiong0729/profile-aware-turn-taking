"""Audio-MLLM adjudication for semantic turn-taking candidates.

Unlike the causal prompt baseline, this annotator receives audio before and
after a known target interval.  Its job is retrospective semantic annotation,
not prediction.  Model agreement creates silver suggestions only; it never
silently turns model output into gold labels.
"""

from __future__ import annotations

import base64
import hashlib
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from .audio import read_wav_window_robust_mix, write_wav_mono
from .utils import read_jsonl, write_json, write_jsonl


ANNOTATION_LABELS = ("C", "BC", "T", "I", "NA", "UNCERTAIN")
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(ANNOTATION_LABELS)},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "needs_review": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": [
        "label",
        "confidence",
        "needs_review",
        "reason",
    ],
    "additionalProperties": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _balanced_select(
    events: Sequence[dict[str, Any]], *, per_group: int, seed: int
) -> list[dict[str, Any]]:
    pools: dict[tuple[str, ...], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for event in events:
        pools[tuple(event["candidate_labels"])][str(event["conversation_id"])].append(event)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for group in sorted(pools):
        by_conversation = pools[group]
        for rows in by_conversation.values():
            rng.shuffle(rows)
        conversation_ids = sorted(by_conversation)
        rng.shuffle(conversation_ids)
        while len([row for row in selected if tuple(row["candidate_labels"]) == group]) < per_group:
            made_progress = False
            for conversation_id in conversation_ids:
                if by_conversation[conversation_id]:
                    selected.append(by_conversation[conversation_id].pop())
                    made_progress = True
                    if len(
                        [row for row in selected if tuple(row["candidate_labels"]) == group]
                    ) >= per_group:
                        break
            if not made_progress:
                break
    return sorted(selected, key=lambda row: str(row["event_id"]))


def build_annotation_prompt(
    event: dict[str, Any], *, view: str, transcript_context: str = ""
) -> str:
    common = [
        "You are retrospectively annotating one event in a natural two-person conversation.",
        "You receive three consecutive recordings explicitly named BEFORE, TARGET, and AFTER.",
        "Judge only TARGET, using BEFORE and AFTER to understand voice identity and floor control.",
        "Do not predict a later event. No profile, candidate label, or weak label is provided.",
        "Choose exactly one label. Return only that label, confidence, review flag, and one short reason.",
        "Labels: C=same floor holder continues; BC=brief listener feedback without taking the floor; T=other speaker cleanly takes the floor; I=non-feedback overlapping/interruptive attempt; NA=no intelligible target speech; UNCERTAIN=not reliably decidable.",
        "Reason internally about voice identity, overlap, brief feedback, and who controls the floor after TARGET.",
        "If the distinction cannot be heard reliably, choose UNCERTAIN, set needs_review=true, and lower confidence.",
    ]
    if transcript_context:
        common.extend(
            [
                "The following time-aligned partial transcript is observation context, not a label. Rows overlapping TARGET are marked [TARGET].",
                transcript_context,
                "Use the audio for prosody and overlap; do not classify from word identity alone.",
            ]
        )
    if view == "floor_questions":
        common.extend(
            [
                "Determine who holds the floor in BEFORE, whether TARGET contains speech and overlap, whether its main voice matches that floor holder, and who controls AFTER.",
                "A short word is not automatically feedback: prosody and subsequent floor behavior decide brief_feedback_only.",
            ]
        )
    elif view == "counterfactual_check":
        common.extend(
            [
                "Check the counterfactual: if TARGET were removed, would the BEFORE speaker's turn continue essentially unchanged in AFTER? Use that only to fill the observable fields.",
                "Use prosody, timing, overlap, and post-target continuation, not transcript word identity alone.",
            ]
        )
    else:
        raise ValueError(f"Unknown annotation view: {view}")
    common.append("Return only the required JSON object.")
    return "\n".join(common)


def prepare_annotation_pilot(
    *,
    events_path: str | Path,
    conversations_path: str | Path,
    utterances_path: str | Path | None = None,
    output_dir: str | Path,
    per_group: int = 20,
    seed: int = 41,
    sample_rate: int = 16_000,
) -> dict[str, Any]:
    events = list(read_jsonl(events_path))
    conversations = {
        str(row["conversation_id"]): row for row in read_jsonl(conversations_path)
    }
    utterances: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if utterances_path is not None:
        for row in read_jsonl(utterances_path):
            if row.get("is_person", True):
                utterances[str(row["conversation_id"])].append(row)
        for rows in utterances.values():
            rows.sort(key=lambda row: (float(row["start_s"]), float(row["end_s"])))
    selected = _balanced_select(events, per_group=per_group, seed=seed)
    destination = Path(output_dir)
    clips = destination / "audio_clips"
    requests: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for event in selected:
        conversation = conversations[str(event["conversation_id"])]
        audio = read_wav_window_robust_mix(
            conversation["audio_path"],
            float(event["context_start_s"]),
            float(event["context_end_s"]),
            target_rate=sample_rate,
        )
        clip_path = write_wav_mono(
            clips / f"{event['event_id']}.wav", audio, sample_rate=sample_rate
        )
        audio_hash = _sha256(clip_path)
        context_rows = [
            row
            for row in utterances.get(str(event["conversation_id"]), [])
            if float(row["end_s"]) > float(event["context_start_s"])
            and float(row["start_s"]) < float(event["context_end_s"])
        ]
        transcript_lines: list[str] = []
        for row in context_rows:
            relative_start = float(row["start_s"]) - float(event["context_start_s"])
            relative_end = float(row["end_s"]) - float(event["context_start_s"])
            overlaps_target = (
                float(row["end_s"]) > float(event["target_start_s"])
                and float(row["start_s"]) < float(event["target_end_s"])
            )
            marker = " [TARGET]" if overlaps_target else ""
            text = str(row.get("clean_text") or row.get("text") or "").strip()
            transcript_lines.append(
                f"{max(0.0, relative_start):06.2f}-{max(0.0, relative_end):06.2f} "
                f"{row['speaker']}{marker}: {text}"
            )
        transcript_context = "\n".join(transcript_lines)
        target_start_sample = max(
            0,
            round(
                (float(event["target_start_s"]) - float(event["context_start_s"]))
                * sample_rate
            ),
        )
        target_end_sample = min(
            len(audio),
            round(
                (float(event["target_end_s"]) - float(event["context_start_s"]))
                * sample_rate
            ),
        )
        parts = {
            "BEFORE": audio[:target_start_sample],
            "TARGET": audio[target_start_sample:target_end_sample],
            "AFTER": audio[target_end_sample:],
        }
        audio_parts: list[dict[str, str]] = []
        for part_name, part_audio in parts.items():
            if len(part_audio) == 0:
                part_audio = [0.0] * round(0.08 * sample_rate)
            part_path = write_wav_mono(
                clips / f"{event['event_id']}.{part_name.lower()}.wav",
                part_audio,
                sample_rate=sample_rate,
            )
            audio_parts.append(
                {
                    "name": part_name,
                    "path": str(part_path.resolve()),
                    "sha256": _sha256(part_path),
                }
            )
        selected_row = {
            **event,
            "audio_path": str(clip_path.resolve()),
            "audio_sha256": audio_hash,
            "audio_parts": audio_parts,
            "context_transcript": transcript_context,
            "annotation_policy": "retrospective_audio_semantics_no_profile_no_weak_label",
        }
        selected_rows.append(selected_row)
        for view in ("floor_questions", "counterfactual_check"):
            prompt = build_annotation_prompt(
                event, view=view, transcript_context=transcript_context
            )
            requests.append(
                {
                    "request_id": f"{event['event_id']}::{view}",
                    "event_id": event["event_id"],
                    "conversation_id": event["conversation_id"],
                    "view": view,
                    "audio_path": str(clip_path.resolve()),
                    "audio_sha256": audio_hash,
                    "audio_parts": audio_parts,
                    "prompt": prompt,
                    "request_sha256": hashlib.sha256(
                        (audio_hash + "\n" + prompt).encode("utf-8")
                    ).hexdigest(),
                }
            )
    write_jsonl(destination / "selected_events.jsonl", selected_rows)
    write_jsonl(destination / "requests.jsonl", requests)
    audit = {
        "events": len(selected_rows),
        "requests": len(requests),
        "views_per_event": 2,
        "candidate_group_counts": dict(
            Counter("/".join(row["candidate_labels"]) for row in selected_rows)
        ),
        "conversation_counts": dict(
            Counter(str(row["conversation_id"]) for row in selected_rows)
        ),
        "label_leakage": False,
        "profile_in_prompt": False,
        "transcript_in_prompt": utterances_path is not None,
        "future_audio_included_for_retrospective_annotation": True,
        "gold_status": "none_model_outputs_are_silver_suggestions",
    }
    write_json(destination / "audit.json", audit)
    return audit


def _derive_label(payload: dict[str, Any]) -> str:
    if not payload["speech_in_target"]:
        contradictory = any(
            payload[field]
            for field in (
                "target_matches_pre_target_floor_voice",
                "brief_feedback_only",
                "audible_overlap",
                "target_speaker_controls_floor_after",
            )
        )
        return "UNCERTAIN" if contradictory else "NA"
    same_voice = payload["target_matches_pre_target_floor_voice"]
    feedback = payload["brief_feedback_only"]
    overlap = payload["audible_overlap"]
    target_controls = payload["target_speaker_controls_floor_after"]
    original_continues = payload["original_floor_holder_continues_after"]
    if feedback and (same_voice or target_controls or not original_continues):
        return "UNCERTAIN"
    if target_controls and original_continues:
        return "UNCERTAIN"
    if same_voice:
        return "UNCERTAIN" if overlap else "C"
    if feedback and original_continues:
        return "BC"
    if overlap:
        return "I"
    if target_controls and not original_continues:
        return "T"
    return "UNCERTAIN"


def _parse_annotation(raw: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
    try:
        payload["confidence"] = float(payload["confidence"])
    except (KeyError, TypeError, ValueError):
        return None
    if not 0.0 <= payload["confidence"] <= 1.0:
        return None
    if payload.get("label") not in ANNOTATION_LABELS:
        return None
    if type(payload.get("needs_review")) is not bool:
        return None
    observable_fields = (
        "speech_in_target",
        "target_matches_pre_target_floor_voice",
        "brief_feedback_only",
        "audible_overlap",
        "target_speaker_controls_floor_after",
        "original_floor_holder_continues_after",
    )
    has_observations = any(field in payload for field in observable_fields)
    if has_observations:
        if any(type(payload.get(field)) is not bool for field in observable_fields):
            return None
        payload["derived_label"] = _derive_label(payload)
        payload["label_consistent"] = payload["label"] == payload["derived_label"]
        if payload["derived_label"] == "UNCERTAIN" or not payload["label_consistent"]:
            payload["needs_review"] = True
    else:
        payload["derived_label"] = payload["label"]
        payload["label_consistent"] = True
        if payload["label"] == "UNCERTAIN":
            payload["needs_review"] = True
    return payload


def _chat_completion(
    endpoint: str,
    request_row: dict[str, Any],
    *,
    model: str,
    timeout_s: float,
    seed: int,
) -> str:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "The following are consecutive BEFORE, TARGET, and AFTER audio segments.",
        }
    ]
    for part in request_row.get("audio_parts", []):
        audio_data = base64.b64encode(Path(part["path"]).read_bytes()).decode("ascii")
        content.extend(
            [
                {"type": "text", "text": f"{part['name']} audio:"},
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_data, "format": "wav"},
                },
            ]
        )
    if not request_row.get("audio_parts"):
        audio_data = base64.b64encode(Path(request_row["audio_path"]).read_bytes()).decode(
            "ascii"
        )
        content.append(
            {
                "type": "input_audio",
                "input_audio": {"data": audio_data, "format": "wav"},
            }
        )
    content.append({"type": "text", "text": request_row["prompt"]})
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "temperature": 0,
            "max_tokens": 160,
            "seed": seed,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "turn_taking_annotation",
                    "strict": True,
                    "schema": OUTPUT_SCHEMA,
                },
            },
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    return str(content)


def run_annotation_requests(
    *,
    requests_path: str | Path,
    responses_path: str | Path,
    endpoint: str = "http://127.0.0.1:8091/v1/chat/completions",
    model: str = "qwen2.5-omni-3b-q4_k_m",
    timeout_s: float = 180.0,
    retries: int = 2,
    seed: int = 41,
    limit: int | None = None,
) -> dict[str, Any]:
    requests = list(read_jsonl(requests_path))
    if limit is not None:
        requests = requests[:limit]
    destination = Path(responses_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = list(read_jsonl(destination)) if destination.exists() else []
    completed = {str(row["request_id"]) for row in existing}
    written = invalid = 0
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        for request in requests:
            if str(request["request_id"]) in completed:
                continue
            started = time.perf_counter()
            raw = error = ""
            for attempt in range(retries + 1):
                try:
                    raw = _chat_completion(
                        endpoint, request, model=model, timeout_s=timeout_s, seed=seed
                    )
                    error = ""
                    break
                except (KeyError, ValueError, OSError, urllib.error.URLError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt < retries:
                        time.sleep(min(2**attempt, 8))
            annotation = _parse_annotation(raw) if not error else None
            valid = annotation is not None
            invalid += int(not valid)
            row = {
                "request_id": request["request_id"],
                "event_id": request["event_id"],
                "conversation_id": request["conversation_id"],
                "view": request["view"],
                "model": model,
                "annotation": annotation,
                "valid": valid,
                "raw_response": raw,
                "error": error,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
    return {"requested": len(requests), "newly_written": written, "invalid": invalid}


def analyze_annotation_pilot(
    run_dir: str | Path, *, confidence_threshold: float = 0.75
) -> dict[str, Any]:
    root = Path(run_dir)
    selected = {row["event_id"]: row for row in read_jsonl(root / "selected_events.jsonl")}
    responses: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for response in read_jsonl(root / "responses.jsonl"):
        responses[str(response["event_id"])].append(response)
    decisions: list[dict[str, Any]] = []
    for event_id, event in selected.items():
        valid = [row for row in responses.get(event_id, []) if row.get("valid")]
        labels = [row["annotation"]["label"] for row in valid]
        confident = [
            row
            for row in valid
            if float(row["annotation"]["confidence"]) >= confidence_threshold
            and not row["annotation"]["needs_review"]
            and row["annotation"]["label_consistent"]
            and row["annotation"]["label"] != "UNCERTAIN"
        ]
        consensus = (
            confident[0]["annotation"]["label"]
            if len(confident) == 2
            and confident[0]["annotation"]["label"]
            == confident[1]["annotation"]["label"]
            else None
        )
        decisions.append(
            {
                "event_id": event_id,
                "conversation_id": event["conversation_id"],
                "candidate_labels": event["candidate_labels"],
                "model_labels": labels,
                "suggested_label": consensus,
                "annotation_status": "mllm_consensus_silver" if consensus else "needs_human_review",
            }
        )
    write_jsonl(root / "decisions.jsonl", decisions)
    total = len(decisions)
    consensus_rows = [row for row in decisions if row["suggested_label"]]
    report = {
        "events": total,
        "events_with_two_responses": sum(len(responses.get(event_id, [])) == 2 for event_id in selected),
        "consensus_silver": len(consensus_rows),
        "needs_human_review": total - len(consensus_rows),
        "consensus_rate": len(consensus_rows) / total if total else 0.0,
        "suggested_label_counts": dict(
            Counter(row["suggested_label"] for row in consensus_rows)
        ),
        "all_model_label_counts": dict(
            Counter(
                label
                for rows in responses.values()
                for response in rows
                if response.get("valid")
                for label in [response["annotation"]["label"]]
            )
        ),
        "all_derived_label_counts": dict(
            Counter(
                response["annotation"]["derived_label"]
                for rows in responses.values()
                for response in rows
                if response.get("valid")
            )
        ),
        "important": "Consensus labels are silver suggestions, not human-verified gold.",
    }
    write_json(root / "analysis.json", report)
    return report


__all__ = [
    "analyze_annotation_pilot",
    "build_annotation_prompt",
    "prepare_annotation_pilot",
    "run_annotation_requests",
]
