"""Paper-aligned causal prompt diagnostics for turn-taking prediction.

This module deliberately keeps references out of model requests.  It consumes
already prepared causal audio clips plus their matching partial transcripts and
runs each binary task twice with the semantic answer order reversed.  Stable
semantic predictions across the two orders are required before accuracy is
interpreted.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .utils import read_jsonl, write_json, write_jsonl


TASK_LABELS: dict[str, tuple[str, str]] = {
    "turn_change": ("CURRENT_SPEAKER_CONTINUES", "OTHER_SPEAKER_TAKES_TURN"),
    "backchannel": ("BACKCHANNEL", "NO_BACKCHANNEL"),
    "interruption": ("OTHER_SPEAKER_INTERRUPTS", "CURRENT_SPEAKER_CONTINUES"),
    "floor_taking": ("SECOND_SPEAKER_TAKES_FLOOR", "FIRST_SPEAKER_KEEPS_FLOOR"),
}

# Meaning of the legacy A/B weak reference used by the prepared diagnostic set.
REFERENCE_TO_SEMANTIC: dict[str, dict[str, str]] = {
    "turn_change": {
        "A": "CURRENT_SPEAKER_CONTINUES",
        "B": "OTHER_SPEAKER_TAKES_TURN",
    },
    "backchannel": {"A": "BACKCHANNEL", "B": "NO_BACKCHANNEL"},
    "interruption": {
        "A": "OTHER_SPEAKER_INTERRUPTS",
        "B": "CURRENT_SPEAKER_CONTINUES",
    },
    "floor_taking": {
        "A": "SECOND_SPEAKER_TAKES_FLOOR",
        "B": "FIRST_SPEAKER_KEEPS_FLOOR",
    },
}

_TASK_QUESTION = {
    "turn_change": (
        "The current speaker has just paused at the end of the audio. Predict "
        "what happens immediately after that pause."
    ),
    "backchannel": (
        "Predict whether either listener produces a brief acknowledgement "
        "immediately after the end of the audio."
    ),
    "interruption": (
        "The clip ends while the current speaker is still speaking. Predict "
        "whether the other speaker starts a full contribution before the current "
        "speaker finishes."
    ),
    "floor_taking": (
        "At the end of the clip, the second speaker has just begun overlapping "
        "the first speaker. Predict which speaker holds the floor after this onset."
    ),
}

_POSITIVE_OUTCOME = {
    "turn_change": "the other speaker takes the next turn after the pause",
    "backchannel": "a listener produces a brief backchannel immediately after t",
    "interruption": "the other speaker begins a full contribution before the current speaker finishes",
    "floor_taking": "the second speaker successfully takes the floor after the overlap onset",
}

_POSITIVE_SEMANTIC = {
    "turn_change": "OTHER_SPEAKER_TAKES_TURN",
    "backchannel": "BACKCHANNEL",
    "interruption": "OTHER_SPEAKER_INTERRUPTS",
    "floor_taking": "SECOND_SPEAKER_TAKES_FLOOR",
}

_FORBIDDEN_REQUEST_KEYS = {
    "target",
    "target_option",
    "reference_label",
    "candidate_label",
    "human_label",
    "training_target",
    "label_evidence",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _all_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(_all_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(_all_keys(nested))
    return keys


def build_semantic_prompt(
    *,
    task: str,
    transcript: str,
    profile_text: str,
    reverse_order: bool,
) -> str:
    """Build one causal prediction prompt without arbitrary A/B answer tokens."""

    if task not in TASK_LABELS:
        raise ValueError(f"Unsupported task: {task}")
    labels = list(TASK_LABELS[task])
    if reverse_order:
        labels.reverse()
    transcript = transcript.strip() or "No completed transcript unit is available."
    profile_text = profile_text.strip() or "Speaker profiles are unknown."
    return "\n".join(
        [
            "Listen to the attached two-speaker conversation and predict the immediate future.",
            "The attached audio contains only the conversation up to prediction time t and ends exactly at t.",
            "Use no sound or words after t.",
            "",
            _TASK_QUESTION[task],
            "",
            "Allowed predictions (the order is not a ranking):",
            *[f"- {label}" for label in labels],
            "",
            "Causal speaker-timed partial transcript (all listed units end no later than t):",
            transcript,
            "",
            "Profile condition:",
            profile_text,
            "",
            'Return exactly one JSON object: {"prediction":"ALLOWED_VALUE"}.',
            "Use exactly one allowed prediction value and no explanation.",
        ]
    )


def build_probability_prompt(
    *, task: str, transcript: str, profile_text: str
) -> str:
    """Ask for a continuous future-event probability instead of a forced option."""

    if task not in _POSITIVE_OUTCOME:
        raise ValueError(f"Unsupported task: {task}")
    transcript = transcript.strip() or "No completed transcript unit is available."
    profile_text = profile_text.strip() or "Speaker profiles are unknown."
    return "\n".join(
        [
            "Listen to the attached two-speaker conversation and forecast the immediate future.",
            "The attached audio contains only the conversation up to prediction time t and ends exactly at t.",
            "Use no sound or words after t.",
            "",
            f"Estimate the probability that {_POSITIVE_OUTCOME[task]}.",
            "Use an integer from 0 to 100, where 0 means impossible, 50 means genuinely uncertain, and 100 means certain.",
            "Use the ending prosody, whether the current utterance sounds complete, speaker activity, and the causal words to distinguish this sample from others.",
            "Do not output 50 merely because future behavior cannot be known with certainty.",
            "",
            "Causal speaker-timed partial transcript (all listed units end no later than t):",
            transcript,
            "",
            "Profile condition:",
            profile_text,
            "",
            'Return exactly one JSON object: {"probability":INTEGER}.',
            "Return no explanation.",
        ]
    )


def prepare_semantic_requests(run_dir: str | Path) -> dict[str, Any]:
    """Create forward/reversed semantic requests and run the input audit."""

    root = Path(run_dir).resolve()
    source_path = root / "requests.jsonl"
    rows = list(read_jsonl(source_path))
    prepared: list[dict[str, Any]] = []
    for row in rows:
        task = str(row["task"])
        transcript = str(row.get("completed_causal_transcript", ""))
        for order_name, reverse_order in (("forward", False), ("reversed", True)):
            prompt = build_semantic_prompt(
                task=task,
                transcript=transcript,
                profile_text=str(row["profile_text"]),
                reverse_order=reverse_order,
            )
            prepared.append(
                {
                    "request_id": f"{row['request_id']}::semantic_{order_name}",
                    "sample_id": row["sample_id"],
                    "conversation_id": row["conversation_id"],
                    "task": task,
                    "order": order_name,
                    "audio_path": row["audio_path"],
                    "audio_sha256": row["audio_sha256"],
                    "audio_duration_s": row["audio_duration_s"],
                    "prediction_boundary_in_conversation_s": row[
                        "prediction_boundary_in_conversation_s"
                    ],
                    "profile_mode": row["profile_mode"],
                    "profile_text": row["profile_text"],
                    "causal_partial_transcript": transcript,
                    "transcript_sha256": _sha256_text(transcript),
                    "allowed_predictions": list(TASK_LABELS[task]),
                    "prompt": prompt,
                    "prompt_sha256": _sha256_text(prompt),
                }
            )
    output_path = root / "requests.semantic.jsonl"
    write_jsonl(output_path, prepared)
    audit = audit_semantic_requests(root)
    return {"requests": len(prepared), "path": str(output_path), "audit": audit}


def audit_semantic_requests(run_dir: str | Path) -> dict[str, Any]:
    """Verify strict causal request structure before contacting the model."""

    root = Path(run_dir).resolve()
    rows = list(read_jsonl(root / "requests.semantic.jsonl"))
    errors: list[str] = []
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        forbidden = _FORBIDDEN_REQUEST_KEYS.intersection(_all_keys(row))
        if forbidden:
            errors.append(f"{row['request_id']}: forbidden keys {sorted(forbidden)}")
        audio_path = Path(str(row["audio_path"]))
        if not audio_path.is_absolute():
            audio_path = root / audio_path
        if not audio_path.is_file():
            errors.append(f"{row['request_id']}: missing audio")
        elif _sha256_file(audio_path) != row["audio_sha256"]:
            errors.append(f"{row['request_id']}: audio SHA-256 mismatch")
        transcript = str(row["causal_partial_transcript"])
        if _sha256_text(transcript) != row["transcript_sha256"]:
            errors.append(f"{row['request_id']}: transcript SHA-256 mismatch")
        end_times = [float(value) for value in re.findall(r"\[[^\]\n]+\s\d+\.\d+-(\d+\.\d+)\]", transcript)]
        if end_times and max(end_times) > float(row["audio_duration_s"]) + 1e-6:
            errors.append(f"{row['request_id']}: transcript extends beyond audio boundary")
        by_sample[str(row["sample_id"])].append(row)
    for sample_id, pair in by_sample.items():
        if sorted(row["order"] for row in pair) != ["forward", "reversed"]:
            errors.append(f"{sample_id}: missing forward/reversed pair")
            continue
        if len({row["audio_sha256"] for row in pair}) != 1:
            errors.append(f"{sample_id}: paired audio differs")
        if len({row["transcript_sha256"] for row in pair}) != 1:
            errors.append(f"{sample_id}: paired transcript differs")
        if len({row["profile_text"] for row in pair}) != 1:
            errors.append(f"{sample_id}: paired profile differs")
    report = {
        "passed": not errors,
        "requests": len(rows),
        "samples": len(by_sample),
        "tasks": dict(Counter(str(row["task"]) for row in rows)),
        "message_content_order": ["text", "input_audio"],
        "target_fields_absent": not any(
            _FORBIDDEN_REQUEST_KEYS.intersection(_all_keys(row)) for row in rows
        ),
        "errors": errors,
    }
    write_json(root / "input_audit.semantic.json", report)
    if errors:
        raise ValueError("Semantic request audit failed: " + "; ".join(errors[:5]))
    return report


def _chat_completion(
    endpoint: str,
    row: dict[str, Any],
    *,
    model: str,
    timeout_s: float,
    seed: int,
) -> str:
    root = Path(str(row["_root"]))
    audio_path = Path(str(row["audio_path"]))
    if not audio_path.is_absolute():
        audio_path = root / audio_path
    audio_data = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    allowed = list(row["allowed_predictions"])
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": row["prompt"]},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_data, "format": "wav"},
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 32,
            "seed": seed,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "turn_taking_prediction",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "prediction": {"type": "string", "enum": allowed}
                        },
                        "required": ["prediction"],
                        "additionalProperties": False,
                    },
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


def _parse_prediction(raw: str, allowed: list[str]) -> str | None:
    try:
        parsed = json.loads(raw)
        value = str(parsed.get("prediction", "")).strip().upper()
        if value in allowed:
            return value
    except (json.JSONDecodeError, AttributeError):
        pass
    upper = raw.upper()
    hits = [label for label in allowed if re.search(rf"\b{re.escape(label)}\b", upper)]
    return hits[0] if len(hits) == 1 else None


def run_semantic_requests(
    run_dir: str | Path,
    *,
    endpoint: str = "http://127.0.0.1:8091/v1/chat/completions",
    model: str = "Qwen2.5-Omni-3B-Q8_0",
    timeout_s: float = 180.0,
    seed: int = 20260719,
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    audit_semantic_requests(root)
    rows = list(read_jsonl(root / "requests.semantic.jsonl"))
    destination = root / "responses.semantic.jsonl"
    output: list[dict[str, Any]] = []
    for index, source in enumerate(rows, start=1):
        row = dict(source)
        row["_root"] = str(root)
        started = time.perf_counter()
        raw = ""
        error = ""
        try:
            raw = _chat_completion(
                endpoint, row, model=model, timeout_s=timeout_s, seed=seed
            )
        except (KeyError, ValueError, OSError, urllib.error.URLError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        prediction = _parse_prediction(raw, list(row["allowed_predictions"])) if not error else None
        result = {
            "request_id": row["request_id"],
            "sample_id": row["sample_id"],
            "conversation_id": row["conversation_id"],
            "task": row["task"],
            "order": row["order"],
            "model": model,
            "audio_sha256": row["audio_sha256"],
            "transcript_sha256": row["transcript_sha256"],
            "prompt_sha256": row["prompt_sha256"],
            "prediction": prediction,
            "valid": prediction in row["allowed_predictions"],
            "raw_response": raw,
            "error": error,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        output.append(result)
        write_jsonl(destination, output)
        print(
            json.dumps(
                {
                    "n": index,
                    "task": row["task"],
                    "order": row["order"],
                    "prediction": prediction,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return {"responses": len(output), "path": str(destination)}


def score_semantic_requests(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    responses = list(read_jsonl(root / "responses.semantic.jsonl"))
    references = {
        str(row["sample_id"]): row
        for row in read_jsonl(root / "reference_labels.jsonl")
    }
    paired: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in responses:
        paired[str(row["sample_id"])][str(row["order"])] = row
    task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_rows: list[dict[str, Any]] = []
    for sample_id, orders in paired.items():
        if set(orders) != {"forward", "reversed"}:
            continue
        reference = references[sample_id]
        task = str(reference["task"])
        target = REFERENCE_TO_SEMANTIC[task][str(reference["target_option"])]
        forward = orders["forward"].get("prediction")
        reversed_value = orders["reversed"].get("prediction")
        row = {
            "sample_id": sample_id,
            "task": task,
            "weak_reference": target,
            "forward_prediction": forward,
            "reversed_prediction": reversed_value,
            "order_stable": forward == reversed_value,
            "forward_correct": forward == target,
            "reversed_correct": reversed_value == target,
        }
        task_rows[task].append(row)
        all_rows.append(row)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        return {
            "samples": n,
            "forward_accuracy_against_weak_reference": (
                sum(row["forward_correct"] for row in rows) / n if n else None
            ),
            "reversed_accuracy_against_weak_reference": (
                sum(row["reversed_correct"] for row in rows) / n if n else None
            ),
            "semantic_order_stability": (
                sum(row["order_stable"] for row in rows) / n if n else None
            ),
            "forward_distribution": dict(
                Counter(str(row["forward_prediction"]) for row in rows)
            ),
            "reversed_distribution": dict(
                Counter(str(row["reversed_prediction"]) for row in rows)
            ),
        }

    report = {
        "diagnostic_only": True,
        "reference_warning": "References are automatic structural weak labels, not human gold labels.",
        "overall": summarize(all_rows),
        "by_task": {task: summarize(rows) for task, rows in sorted(task_rows.items())},
        "rows": all_rows,
    }
    write_json(root / "metrics.semantic.json", report)
    return report


def prepare_probability_requests(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    source_rows = list(read_jsonl(root / "requests.jsonl"))
    output: list[dict[str, Any]] = []
    for row in source_rows:
        transcript = str(row.get("completed_causal_transcript", ""))
        prompt = build_probability_prompt(
            task=str(row["task"]),
            transcript=transcript,
            profile_text=str(row["profile_text"]),
        )
        output.append(
            {
                "request_id": f"{row['request_id']}::probability",
                "sample_id": row["sample_id"],
                "conversation_id": row["conversation_id"],
                "task": row["task"],
                "audio_path": row["audio_path"],
                "audio_sha256": row["audio_sha256"],
                "audio_duration_s": row["audio_duration_s"],
                "prediction_boundary_in_conversation_s": row[
                    "prediction_boundary_in_conversation_s"
                ],
                "profile_mode": row["profile_mode"],
                "profile_text": row["profile_text"],
                "causal_partial_transcript": transcript,
                "transcript_sha256": _sha256_text(transcript),
                "prompt": prompt,
                "prompt_sha256": _sha256_text(prompt),
            }
        )
    path = root / "requests.probability.jsonl"
    write_jsonl(path, output)
    errors: list[str] = []
    for row in output:
        if _FORBIDDEN_REQUEST_KEYS.intersection(_all_keys(row)):
            errors.append(f"{row['request_id']}: target-like field present")
        audio_path = Path(str(row["audio_path"]))
        if not audio_path.is_absolute():
            audio_path = root / audio_path
        if not audio_path.is_file() or _sha256_file(audio_path) != row["audio_sha256"]:
            errors.append(f"{row['request_id']}: audio missing or hash mismatch")
        end_times = [
            float(value)
            for value in re.findall(
                r"\[[^\]\n]+\s\d+\.\d+-(\d+\.\d+)\]",
                str(row["causal_partial_transcript"]),
            )
        ]
        if end_times and max(end_times) > float(row["audio_duration_s"]) + 1e-6:
            errors.append(f"{row['request_id']}: transcript extends past t")
    audit = {
        "passed": not errors,
        "requests": len(output),
        "message_content_order": ["text", "input_audio"],
        "target_fields_absent": not errors,
        "errors": errors,
    }
    write_json(root / "input_audit.probability.json", audit)
    if errors:
        raise ValueError("Probability request audit failed: " + "; ".join(errors[:5]))
    return {"requests": len(output), "path": str(path), "audit": audit}


def run_probability_requests(
    run_dir: str | Path,
    *,
    endpoint: str = "http://127.0.0.1:8091/v1/chat/completions",
    model: str = "Qwen2.5-Omni-3B-Q8_0",
    timeout_s: float = 180.0,
    seed: int = 20260719,
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    rows = list(read_jsonl(root / "requests.probability.jsonl"))
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        audio_path = Path(str(row["audio_path"]))
        if not audio_path.is_absolute():
            audio_path = root / audio_path
        audio_data = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        body = json.dumps(
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": row["prompt"]},
                            {
                                "type": "input_audio",
                                "input_audio": {"data": audio_data, "format": "wav"},
                            },
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 16,
                "seed": seed,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "turn_taking_probability",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "probability": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 100,
                                }
                            },
                            "required": ["probability"],
                            "additionalProperties": False,
                        },
                    },
                },
            },
            ensure_ascii=False,
        ).encode("utf-8")
        started = time.perf_counter()
        raw = ""
        error = ""
        probability: int | None = None
        try:
            request = urllib.request.Request(
                endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
            raw_content = payload["choices"][0]["message"]["content"]
            if isinstance(raw_content, list):
                raw_content = "".join(
                    str(part.get("text", ""))
                    for part in raw_content
                    if isinstance(part, dict)
                )
            raw = str(raw_content)
            parsed = json.loads(raw)
            value = int(parsed["probability"])
            if 0 <= value <= 100:
                probability = value
        except (KeyError, ValueError, OSError, json.JSONDecodeError, urllib.error.URLError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        result = {
            "request_id": row["request_id"],
            "sample_id": row["sample_id"],
            "conversation_id": row["conversation_id"],
            "task": row["task"],
            "model": model,
            "audio_sha256": row["audio_sha256"],
            "transcript_sha256": row["transcript_sha256"],
            "prompt_sha256": row["prompt_sha256"],
            "probability": probability,
            "valid": probability is not None,
            "raw_response": raw,
            "error": error,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        output.append(result)
        write_jsonl(root / "responses.probability.jsonl", output)
        print(
            json.dumps(
                {"n": index, "task": row["task"], "probability": probability},
                ensure_ascii=False,
            ),
            flush=True,
        )
    return {"responses": len(output), "path": str(root / "responses.probability.jsonl")}


def _pairwise_auc(targets: list[int], scores: list[float]) -> float | None:
    positive = [score for target, score in zip(targets, scores) if target == 1]
    negative = [score for target, score in zip(targets, scores) if target == 0]
    if not positive or not negative:
        return None
    wins = sum(p > n for p in positive for n in negative)
    ties = sum(p == n for p in positive for n in negative)
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def score_probability_requests(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    responses = list(read_jsonl(root / "responses.probability.jsonl"))
    references = {
        str(row["sample_id"]): row
        for row in read_jsonl(root / "reference_labels.jsonl")
    }
    rows: list[dict[str, Any]] = []
    for response in responses:
        sample_id = str(response["sample_id"])
        reference = references[sample_id]
        task = str(reference["task"])
        semantic = REFERENCE_TO_SEMANTIC[task][str(reference["target_option"])]
        positive = int(semantic == _POSITIVE_SEMANTIC[task])
        rows.append(
            {
                "sample_id": sample_id,
                "task": task,
                "weak_reference_positive": positive,
                "probability": response.get("probability"),
            }
        )
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task"])].append(row)

    def summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
        valid = [row for row in values if row["probability"] is not None]
        targets = [int(row["weak_reference_positive"]) for row in valid]
        scores = [float(row["probability"]) for row in valid]
        predictions = [int(score >= 50) for score in scores]
        return {
            "samples": len(values),
            "valid": len(valid),
            "unique_probabilities": len(set(scores)),
            "probability_min": min(scores) if scores else None,
            "probability_max": max(scores) if scores else None,
            "accuracy_at_50_against_weak_reference": (
                sum(p == y for p, y in zip(predictions, targets)) / len(valid)
                if valid
                else None
            ),
            "roc_auc_against_weak_reference": _pairwise_auc(targets, scores),
        }

    report = {
        "diagnostic_only": True,
        "reference_warning": "References are automatic structural weak labels, not human gold labels.",
        "overall": summarize(rows),
        "by_task": {task: summarize(values) for task, values in sorted(by_task.items())},
        "rows": rows,
    }
    write_json(root / "metrics.probability.json", report)
    return report


__all__ = [
    "REFERENCE_TO_SEMANTIC",
    "TASK_LABELS",
    "audit_semantic_requests",
    "build_probability_prompt",
    "build_semantic_prompt",
    "prepare_probability_requests",
    "prepare_semantic_requests",
    "run_probability_requests",
    "run_semantic_requests",
    "score_probability_requests",
    "score_semantic_requests",
]
