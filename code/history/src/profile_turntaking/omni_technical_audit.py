"""Audited Qwen2.5-Omni turn-taking input and prompt diagnostics.

The primary future-prediction requests preserve the project contract:
causal audio + matching causal partial transcript + hidden profile. Auxiliary
ASR and occurred-event controls isolate whether the local audio path works.
References are stored separately and never sent in model requests.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import re
import shutil
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .audio import read_wav_window_robust_mix, write_wav_mono
from .utils import read_jsonl, write_json, write_jsonl


TASK_ORDER = ("turn_change", "backchannel", "interruption", "floor_taking")

TASK_SPEC: dict[str, dict[str, Any]] = {
    "turn_change": {
        "question": (
            "The current speaker has just paused at the exact end of the audio. "
            "Predict what happens immediately after this pause."
        ),
        "observed_question": (
            "The prediction boundary is exactly 5.0 seconds into this clip. "
            "Immediately after that boundary, does the other speaker take the "
            "turn, or does the same speaker continue?"
        ),
        "labels": ("CURRENT_SPEAKER_CONTINUES", "OTHER_SPEAKER_TAKES_TURN"),
    },
    "backchannel": {
        "question": (
            "Predict whether a listener produces a brief acknowledgement "
            "immediately after the exact end of the audio without taking the floor."
        ),
        "observed_question": (
            "The prediction boundary is exactly 5.0 seconds into this clip. "
            "Immediately after that boundary, does a listener produce a brief "
            "backchannel without taking the floor?"
        ),
        "labels": ("BACKCHANNEL", "NO_BACKCHANNEL"),
    },
    "interruption": {
        "question": (
            "The clip ends while the current speaker is still speaking. Predict "
            "whether the other speaker begins a full contribution before the "
            "current speaker finishes."
        ),
        "observed_question": (
            "The prediction boundary is exactly 5.0 seconds into this clip. "
            "After that boundary, does the other speaker begin a full contribution "
            "before the current speaker finishes?"
        ),
        "labels": ("OTHER_SPEAKER_INTERRUPTS", "CURRENT_SPEAKER_CONTINUES"),
    },
    "floor_taking": {
        "question": (
            "At the exact end of the audio, the second speaker has just begun "
            "overlapping the first speaker. Predict whether the second speaker "
            "subsequently takes the floor."
        ),
        "observed_question": (
            "The prediction boundary is exactly 5.0 seconds into this clip, where "
            "the second speaker begins overlapping the first. In the following "
            "audio, does the second speaker take the floor or does the first "
            "speaker keep it?"
        ),
        "labels": (
            "SECOND_SPEAKER_TAKES_FLOOR",
            "FIRST_SPEAKER_KEEPS_FLOOR",
        ),
    },
}

PREDICTION_CONDITIONS = {
    "future_current",
    "future_current_reversed_labels",
    "future_current_audio_first",
    "future_paper_style",
    "future_silence",
    "future_wrong_audio",
    "occurred_recognition",
    "occurred_recognition_reversed_labels",
}

FORBIDDEN_KEYS = {
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
    result: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            result.add(str(key))
            result.update(_all_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            result.update(_all_keys(nested))
    return result


def _normalise_words(text: str) -> list[str]:
    without_timestamps = re.sub(r"\[[^\]]+\]", " ", text.lower())
    return re.findall(r"[a-z']+", without_timestamps)


def word_overlap_f1(reference: str, hypothesis: str) -> float:
    """Bag-of-words overlap F1 used only as a coarse ASR path diagnostic."""
    reference_counts = Counter(_normalise_words(reference))
    hypothesis_counts = Counter(_normalise_words(hypothesis))
    overlap = sum((reference_counts & hypothesis_counts).values())
    reference_total = sum(reference_counts.values())
    hypothesis_total = sum(hypothesis_counts.values())
    if not reference_total or not hypothesis_total or not overlap:
        return 0.0
    precision = overlap / hypothesis_total
    recall = overlap / reference_total
    return 2.0 * precision * recall / (precision + recall)


def build_future_prompt(
    *,
    task: str,
    transcript: str,
    profile_text: str,
    style: str,
    reverse_labels: bool = False,
) -> str:
    """Build a causal future prompt while preserving all three primary inputs."""
    spec = TASK_SPEC[task]
    first, second = spec["labels"]
    if reverse_labels:
        first, second = second, first
    transcript = transcript.strip() or "No completed transcript unit is available."
    profile_text = profile_text.strip() or "Speaker profiles are unknown."
    if style == "current":
        opening = [
            "Listen to the attached two-speaker conversation.",
            "The audio contains only the past and ends exactly at prediction time t.",
            "Predict only what happens after t. Never claim to hear audio after t.",
            "",
            str(spec["question"]),
        ]
        ending = [
            "Use the causal words, ending prosody, pause and listener activity.",
            f"Return only one conclusion: {first} or {second}.",
        ]
    elif style == "paper":
        opening = [
            "You are provided the audio of a two-speaker conversation.",
            "The provided audio ends exactly at the prediction point.",
            str(spec["question"]),
            "Choose the more likely of the two conclusions based only on the "
            "conversation available up to this point.",
        ]
        ending = [
            "Briefly reason internally, but output only the final conclusion.",
            f"Your answer must be exactly {first} or {second}.",
        ]
    else:
        raise ValueError(f"Unsupported prompt style: {style}")
    return "\n".join(
        [
            *opening,
            f"The two possible conclusions are {first} and {second}.",
            "",
            "Causal speaker-timed partial transcript; every listed unit ends no later than t:",
            transcript,
            "",
            "Profile condition:",
            profile_text,
            "",
            *ending,
        ]
    )


def build_occurred_prompt(task: str, *, reverse_labels: bool = False) -> str:
    """Ask about an event already audible in an auxiliary diagnostic clip."""
    spec = TASK_SPEC[task]
    first, second = spec["labels"]
    if reverse_labels:
        first, second = second, first
    return "\n".join(
        [
            "Listen to this complete two-speaker diagnostic clip.",
            str(spec["observed_question"]),
            f"The two possible conclusions are {first} and {second}.",
            "This is event recognition, not future prediction: the relevant audio "
            "after the 5.0-second boundary is included in the clip.",
            f"Return only one conclusion: {first} or {second}.",
        ]
    )


def build_asr_prompt() -> str:
    return (
        "Transcribe the spoken English in the attached audio. Return only the "
        "transcript. Do not describe the task, the speakers, or background noise."
    )


def _select_balanced_references(
    references: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for task in TASK_ORDER:
        labels = TASK_SPEC[task]["labels"]
        task_rows = [row for row in references if row["task"] == task]
        for label in labels:
            matches = sorted(
                (row for row in task_rows if row["target"] == label),
                key=lambda row: str(row["sample_id"]),
            )
            if not matches:
                raise ValueError(f"No reference for {task}/{label}")
            selected.append(matches[0])
    return selected


def prepare_audit(
    *,
    source_run_dir: str | Path,
    catalog_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source_root = Path(source_run_dir).resolve()
    catalog_root = Path(catalog_dir).resolve()
    output_root = Path(output_dir).resolve()
    audio_root = output_root / "audio"
    causal_root = audio_root / "causal"
    observed_root = audio_root / "observed"
    silence_root = audio_root / "silence"
    wrong_root = audio_root / "wrong"
    output_root.mkdir(parents=True, exist_ok=True)

    source_requests = {
        str(row["sample_id"]): row
        for row in read_jsonl(source_root / "requests.jsonl")
    }
    all_references = list(read_jsonl(source_root / "reference_labels.jsonl"))
    selected = _select_balanced_references(all_references)
    selected_by_task_target = {
        (str(row["task"]), str(row["target"])): row for row in selected
    }
    conversations = {
        str(row["conversation_id"]): row
        for row in read_jsonl(catalog_root / "core_dyadic_conversations.jsonl")
    }

    references: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for reference in selected:
        sample_id = str(reference["sample_id"])
        task = str(reference["task"])
        target = str(reference["target"])
        source = source_requests[sample_id]
        source_audio = source_root / str(source["audio_path"])
        causal_path = causal_root / f"{sample_id}.wav"
        causal_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_audio, causal_path)

        duration_s = float(source["audio_duration_s"])
        silence_path = silence_root / f"{sample_id}.wav"
        write_wav_mono(
            silence_path,
            np.zeros(int(round(duration_s * 16_000)), dtype=np.float32),
        )

        boundary_s = float(reference["prediction_boundary_in_conversation_s"])
        conversation = conversations[str(reference["conversation_id"])]
        observed_path = observed_root / f"{sample_id}.wav"
        observed_audio = read_wav_window_robust_mix(
            conversation["audio_path"],
            max(0.0, boundary_s - 5.0),
            boundary_s + 3.0,
        )
        write_wav_mono(observed_path, observed_audio)

        opposite_label = next(
            label for label in TASK_SPEC[task]["labels"] if label != target
        )
        wrong_reference = selected_by_task_target[(task, opposite_label)]
        wrong_sample_id = str(wrong_reference["sample_id"])
        wrong_source = source_requests[wrong_sample_id]
        wrong_source_audio = source_root / str(wrong_source["audio_path"])
        wrong_audio_path = wrong_root / f"{sample_id}__from__{wrong_sample_id}.wav"
        wrong_audio_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(wrong_source_audio, wrong_audio_path)

        transcript = str(source["causal_partial_transcript"])
        profile_text = str(source["profile_text"])
        references.append(
            {
                "sample_id": sample_id,
                "conversation_id": reference["conversation_id"],
                "task": task,
                "target": target,
                "prediction_boundary_in_conversation_s": boundary_s,
                "wrong_audio_sample_id": wrong_sample_id,
            }
        )

        prompt_current = build_future_prompt(
            task=task,
            transcript=transcript,
            profile_text=profile_text,
            style="current",
        )
        prompt_paper = build_future_prompt(
            task=task,
            transcript=transcript,
            profile_text=profile_text,
            style="paper",
        )
        prompt_reversed = build_future_prompt(
            task=task,
            transcript=transcript,
            profile_text=profile_text,
            style="current",
            reverse_labels=True,
        )
        common = {
            "sample_id": sample_id,
            "conversation_id": reference["conversation_id"],
            "task": task,
            "profile_mode": "hidden",
            "profile_text": profile_text,
            "causal_partial_transcript": transcript,
            "transcript_sha256": _sha256_text(transcript),
            "allowed_predictions": list(TASK_SPEC[task]["labels"]),
            "diagnostic_control": False,
        }

        def add_prediction(
            condition: str,
            audio_path: Path,
            prompt: str,
            *,
            content_order: str = "text_audio",
            diagnostic_control: bool = False,
            allowed_predictions: list[str] | None = None,
        ) -> None:
            allowed = allowed_predictions or list(TASK_SPEC[task]["labels"])
            requests.append(
                {
                    **common,
                    "request_id": f"{sample_id}::{condition}",
                    "condition": condition,
                    "audio_path": str(audio_path.relative_to(output_root)),
                    "audio_sha256": _sha256_file(audio_path),
                    "audio_duration_s": (
                        8.0 if condition == "occurred_recognition" else duration_s
                    ),
                    "prompt": prompt,
                    "prompt_sha256": _sha256_text(prompt),
                    "content_order": content_order,
                    "output_schema": "semantic_prediction",
                    "diagnostic_control": diagnostic_control,
                    "allowed_predictions": allowed,
                }
            )

        add_prediction("future_current", causal_path, prompt_current)
        add_prediction(
            "future_current_reversed_labels",
            causal_path,
            prompt_reversed,
            diagnostic_control=True,
            allowed_predictions=list(reversed(TASK_SPEC[task]["labels"])),
        )
        add_prediction(
            "future_current_audio_first",
            causal_path,
            prompt_current,
            content_order="audio_text",
        )
        add_prediction("future_paper_style", causal_path, prompt_paper)
        add_prediction(
            "future_silence",
            silence_path,
            prompt_current,
            diagnostic_control=True,
        )
        add_prediction(
            "future_wrong_audio",
            wrong_audio_path,
            prompt_current,
            diagnostic_control=True,
        )
        add_prediction(
            "occurred_recognition",
            observed_path,
            build_occurred_prompt(task),
            diagnostic_control=True,
        )
        add_prediction(
            "occurred_recognition_reversed_labels",
            observed_path,
            build_occurred_prompt(task, reverse_labels=True),
            diagnostic_control=True,
            allowed_predictions=list(reversed(TASK_SPEC[task]["labels"])),
        )

        for condition, order in (
            ("asr_text_first", "text_audio"),
            ("asr_audio_first", "audio_text"),
        ):
            asr_prompt = build_asr_prompt()
            requests.append(
                {
                    "request_id": f"{sample_id}::{condition}",
                    "sample_id": sample_id,
                    "conversation_id": reference["conversation_id"],
                    "task": task,
                    "condition": condition,
                    "audio_path": str(causal_path.relative_to(output_root)),
                    "audio_sha256": _sha256_file(causal_path),
                    "audio_duration_s": duration_s,
                    "causal_partial_transcript": transcript,
                    "transcript_sha256": _sha256_text(transcript),
                    "profile_mode": "hidden",
                    "profile_text": profile_text,
                    "prompt": asr_prompt,
                    "prompt_sha256": _sha256_text(asr_prompt),
                    "content_order": order,
                    "output_schema": "transcript_text",
                    "diagnostic_control": True,
                }
            )

    write_jsonl(output_root / "reference_labels.jsonl", references)
    write_jsonl(output_root / "requests.jsonl", requests)
    config = {
        "model": "Qwen2.5-Omni-3B-Q8_0",
        "samples": len(references),
        "requests": len(requests),
        "primary_input_contract": (
            "causal audio + matching causal partial transcript + hidden profile"
        ),
        "primary_conditions": [
            "future_current",
            "future_current_audio_first",
            "future_paper_style",
        ],
        "diagnostic_controls": [
            "future_current_reversed_labels",
            "future_silence",
            "future_wrong_audio",
            "occurred_recognition",
            "occurred_recognition_reversed_labels",
            "asr_text_first",
            "asr_audio_first",
        ],
    }
    write_json(output_root / "run_config.json", config)
    audit = audit_requests(output_root)
    return {**config, "output_dir": str(output_root), "input_audit": audit}


def audit_requests(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    requests = list(read_jsonl(root / "requests.jsonl"))
    references = list(read_jsonl(root / "reference_labels.jsonl"))
    errors: list[str] = []
    class_counts = Counter(
        f"{row['task']}::{row['target']}" for row in references
    )
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in requests:
        request_id = str(row["request_id"])
        forbidden = FORBIDDEN_KEYS.intersection(_all_keys(row))
        if forbidden:
            errors.append(f"{request_id}: forbidden target keys {sorted(forbidden)}")
        audio_path = Path(str(row["audio_path"]))
        if not audio_path.is_absolute():
            audio_path = root / audio_path
        if not audio_path.is_file():
            errors.append(f"{request_id}: missing audio")
        elif _sha256_file(audio_path) != row["audio_sha256"]:
            errors.append(f"{request_id}: audio SHA-256 mismatch")
        transcript = str(row.get("causal_partial_transcript", ""))
        if _sha256_text(transcript) != row.get("transcript_sha256"):
            errors.append(f"{request_id}: transcript SHA-256 mismatch")
        if row["condition"].startswith("future_") and row["condition"] not in {
            "future_silence",
            "future_wrong_audio",
        }:
            end_times = [
                float(value)
                for value in re.findall(
                    r"\[[^\]\n]+\s\d+\.\d+-(\d+\.\d+)\]", transcript
                )
            ]
            if end_times and max(end_times) > float(row["audio_duration_s"]) + 1e-6:
                errors.append(f"{request_id}: transcript extends beyond audio")
            if str(row.get("profile_mode")) != "hidden":
                errors.append(f"{request_id}: primary audit must use hidden profile")
        if row["content_order"] not in {"text_audio", "audio_text"}:
            errors.append(f"{request_id}: invalid content order")
        if row["output_schema"] == "semantic_prediction":
            if sorted(row.get("allowed_predictions", [])) != sorted(
                TASK_SPEC[str(row["task"])]["labels"]
            ):
                errors.append(f"{request_id}: output label schema mismatch")
        by_sample[str(row["sample_id"])].append(row)

    expected_conditions = {
        "future_current",
        "future_current_reversed_labels",
        "future_current_audio_first",
        "future_paper_style",
        "future_silence",
        "future_wrong_audio",
        "occurred_recognition",
        "occurred_recognition_reversed_labels",
        "asr_text_first",
        "asr_audio_first",
    }
    for sample_id, rows in by_sample.items():
        conditions = {str(row["condition"]) for row in rows}
        if conditions != expected_conditions:
            errors.append(
                f"{sample_id}: condition mismatch {sorted(conditions)}"
            )
        primary = [
            row
            for row in rows
            if row["condition"]
            in {
                "future_current",
                "future_current_audio_first",
                "future_paper_style",
            }
        ]
        if len({row["audio_sha256"] for row in primary}) != 1:
            errors.append(f"{sample_id}: primary-condition audio differs")
        if len({row["transcript_sha256"] for row in primary}) != 1:
            errors.append(f"{sample_id}: primary-condition transcript differs")
        if len({row["profile_text"] for row in primary}) != 1:
            errors.append(f"{sample_id}: primary-condition profile differs")

    report = {
        "passed": not errors,
        "samples": len(by_sample),
        "requests": len(requests),
        "class_counts": dict(sorted(class_counts.items())),
        "condition_counts": dict(
            sorted(Counter(str(row["condition"]) for row in requests).items())
        ),
        "audio_sha256_verified": not any("audio" in error for error in errors),
        "transcript_sha256_verified": not any(
            "transcript SHA" in error for error in errors
        ),
        "causal_transcript_timestamps_verified": not any(
            "extends beyond" in error for error in errors
        ),
        "target_fields_absent": not any(
            FORBIDDEN_KEYS.intersection(_all_keys(row)) for row in requests
        ),
        "output_schema_verified": not any(
            "output label schema" in error for error in errors
        ),
        "errors": errors,
    }
    write_json(root / "input_audit.json", report)
    if errors:
        raise ValueError("Omni technical-audit preflight failed: " + "; ".join(errors[:8]))
    return report


def _completion(
    *,
    endpoint: str,
    model: str,
    root: Path,
    row: dict[str, Any],
    timeout_s: float,
    seed: int,
) -> str:
    audio_path = Path(str(row["audio_path"]))
    if not audio_path.is_absolute():
        audio_path = root / audio_path
    audio_data = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    text_item = {"type": "text", "text": row["prompt"]}
    audio_item = {
        "type": "input_audio",
        "input_audio": {"data": audio_data, "format": "wav"},
    }
    content = (
        [text_item, audio_item]
        if row["content_order"] == "text_audio"
        else [audio_item, text_item]
    )
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": (
            160 if row["output_schema"] == "transcript_text" else 48
        ),
        "seed": seed,
    }
    if row["output_schema"] == "semantic_prediction":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "turn_taking_prediction",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "prediction": {
                            "type": "string",
                            "enum": row["allowed_predictions"],
                        }
                    },
                    "required": ["prediction"],
                    "additionalProperties": False,
                },
            },
        }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    result = payload["choices"][0]["message"]["content"]
    if isinstance(result, list):
        result = "".join(
            str(part.get("text", "")) for part in result if isinstance(part, dict)
        )
    return str(result)


def _parse_prediction(raw: str, allowed: list[str]) -> str | None:
    try:
        parsed = json.loads(raw)
        prediction = str(parsed.get("prediction", "")).strip().upper()
        if prediction in allowed:
            return prediction
    except (json.JSONDecodeError, AttributeError):
        pass
    upper = raw.upper()
    hits = [
        label
        for label in sorted(allowed, key=len, reverse=True)
        if re.search(rf"\b{re.escape(label)}\b", upper)
    ]
    return hits[0] if len(hits) == 1 else None


def run_audit(
    run_dir: str | Path,
    *,
    endpoint: str = "http://127.0.0.1:8091/v1/chat/completions",
    model: str = "Qwen2.5-Omni-3B-Q8_0",
    timeout_s: float = 180.0,
    seed: int = 20260731,
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    audit_requests(root)
    requests = list(read_jsonl(root / "requests.jsonl"))
    destination = root / "responses.jsonl"
    existing = list(read_jsonl(destination)) if destination.is_file() else []
    completed = {str(row["request_id"]) for row in existing}
    output = list(existing)
    for index, row in enumerate(requests, start=1):
        if str(row["request_id"]) in completed:
            continue
        started = time.perf_counter()
        raw = ""
        error = ""
        try:
            raw = _completion(
                endpoint=endpoint,
                model=model,
                root=root,
                row=row,
                timeout_s=timeout_s,
                seed=seed,
            )
        except (KeyError, ValueError, OSError, urllib.error.URLError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        prediction = None
        transcript = None
        if row["output_schema"] == "semantic_prediction" and not error:
            prediction = _parse_prediction(raw, list(row["allowed_predictions"]))
        elif row["output_schema"] == "transcript_text" and not error:
            transcript = raw.strip()
        result = {
            "request_id": row["request_id"],
            "sample_id": row["sample_id"],
            "conversation_id": row["conversation_id"],
            "task": row["task"],
            "condition": row["condition"],
            "content_order": row["content_order"],
            "model": model,
            "audio_sha256": row["audio_sha256"],
            "transcript_sha256": row["transcript_sha256"],
            "prompt_sha256": row["prompt_sha256"],
            "prediction": prediction,
            "transcript": transcript,
            "valid": (
                prediction in row.get("allowed_predictions", [])
                if row["output_schema"] == "semantic_prediction"
                else bool(transcript)
            ),
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
                    "request": row["request_id"],
                    "prediction": prediction,
                    "transcript_chars": len(transcript or ""),
                    "error": error,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return {
        "responses": len(output),
        "expected": len(requests),
        "path": str(destination),
    }


def _summarise_predictions(
    rows: list[dict[str, Any]], references: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    scored = [
        {
            **row,
            "target": references[str(row["sample_id"])]["target"],
            "correct": row.get("prediction")
            == references[str(row["sample_id"])]["target"],
        }
        for row in rows
    ]
    by_task: dict[str, Any] = {}
    for task in TASK_ORDER:
        values = [row for row in scored if row["task"] == task]
        distribution = Counter(str(row.get("prediction")) for row in values)
        by_task[task] = {
            "samples": len(values),
            "correct": sum(bool(row["correct"]) for row in values),
            "accuracy": (
                sum(bool(row["correct"]) for row in values) / len(values)
                if values
                else None
            ),
            "prediction_distribution": dict(distribution),
            "noncollapsed": len(distribution) >= 2,
        }
    distribution = Counter(str(row.get("prediction")) for row in scored)
    return {
        "samples": len(scored),
        "valid": sum(bool(row.get("valid")) for row in scored),
        "correct": sum(bool(row["correct"]) for row in scored),
        "accuracy": (
            sum(bool(row["correct"]) for row in scored) / len(scored)
            if scored
            else None
        ),
        "prediction_distribution": dict(distribution),
        "all_tasks_noncollapsed": all(
            summary["noncollapsed"] for summary in by_task.values()
        ),
        "by_task": by_task,
    }


def score_audit(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    references = {
        str(row["sample_id"]): row
        for row in read_jsonl(root / "reference_labels.jsonl")
    }
    requests = {
        str(row["request_id"]): row for row in read_jsonl(root / "requests.jsonl")
    }
    responses = list(read_jsonl(root / "responses.jsonl"))
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in responses:
        by_condition[str(row["condition"])].append(row)

    prediction_summaries = {
        condition: _summarise_predictions(values, references)
        for condition, values in by_condition.items()
        if condition in PREDICTION_CONDITIONS
    }

    response_index = {
        (str(row["sample_id"]), str(row["condition"])): row for row in responses
    }
    audio_sensitivity_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    label_order_rows: list[dict[str, Any]] = []
    asr_rows: list[dict[str, Any]] = []
    for sample_id, reference in references.items():
        current = response_index[(sample_id, "future_current")]
        silence = response_index[(sample_id, "future_silence")]
        wrong = response_index[(sample_id, "future_wrong_audio")]
        audio_first = response_index[(sample_id, "future_current_audio_first")]
        future_reversed = response_index[
            (sample_id, "future_current_reversed_labels")
        ]
        occurred = response_index[(sample_id, "occurred_recognition")]
        occurred_reversed = response_index[
            (sample_id, "occurred_recognition_reversed_labels")
        ]
        audio_sensitivity_rows.append(
            {
                "sample_id": sample_id,
                "task": reference["task"],
                "correct_audio_prediction": current.get("prediction"),
                "silence_prediction": silence.get("prediction"),
                "wrong_audio_prediction": wrong.get("prediction"),
                "changes_on_silence": current.get("prediction")
                != silence.get("prediction"),
                "changes_on_wrong_audio": current.get("prediction")
                != wrong.get("prediction"),
            }
        )
        order_rows.append(
            {
                "sample_id": sample_id,
                "task": reference["task"],
                "text_audio_prediction": current.get("prediction"),
                "audio_text_prediction": audio_first.get("prediction"),
                "same_prediction": current.get("prediction")
                == audio_first.get("prediction"),
            }
        )
        label_order_rows.append(
            {
                "sample_id": sample_id,
                "task": reference["task"],
                "future_normal": current.get("prediction"),
                "future_reversed": future_reversed.get("prediction"),
                "future_same_semantic_prediction": current.get("prediction")
                == future_reversed.get("prediction"),
                "occurred_normal": occurred.get("prediction"),
                "occurred_reversed": occurred_reversed.get("prediction"),
                "occurred_same_semantic_prediction": occurred.get("prediction")
                == occurred_reversed.get("prediction"),
            }
        )
        request = requests[f"{sample_id}::asr_text_first"]
        for condition in ("asr_text_first", "asr_audio_first"):
            response = response_index[(sample_id, condition)]
            asr_rows.append(
                {
                    "sample_id": sample_id,
                    "task": reference["task"],
                    "condition": condition,
                    "word_overlap_f1": round(
                        word_overlap_f1(
                            str(request["causal_partial_transcript"]),
                            str(response.get("transcript") or ""),
                        ),
                        4,
                    ),
                    "transcript": response.get("transcript"),
                }
            )

    asr_by_condition: dict[str, dict[str, Any]] = {}
    for condition in ("asr_text_first", "asr_audio_first"):
        values = [row for row in asr_rows if row["condition"] == condition]
        scores = [float(row["word_overlap_f1"]) for row in values]
        asr_by_condition[condition] = {
            "samples": len(values),
            "nonempty": sum(bool(row.get("transcript")) for row in values),
            "mean_word_overlap_f1": (
                round(sum(scores) / len(scores), 4) if scores else None
            ),
            "min_word_overlap_f1": min(scores) if scores else None,
            "max_word_overlap_f1": max(scores) if scores else None,
        }

    report = {
        "model": next(
            (str(row["model"]) for row in responses if row.get("model")), None
        ),
        "samples": len(references),
        "responses": len(responses),
        "prediction_conditions": prediction_summaries,
        "audio_sensitivity": {
            "samples": len(audio_sensitivity_rows),
            "changed_on_silence": sum(
                bool(row["changes_on_silence"]) for row in audio_sensitivity_rows
            ),
            "changed_on_wrong_audio": sum(
                bool(row["changes_on_wrong_audio"])
                for row in audio_sensitivity_rows
            ),
            "rows": audio_sensitivity_rows,
        },
        "content_order": {
            "samples": len(order_rows),
            "same_prediction": sum(
                bool(row["same_prediction"]) for row in order_rows
            ),
            "rows": order_rows,
        },
        "label_order": {
            "samples": len(label_order_rows),
            "future_same_semantic_prediction": sum(
                bool(row["future_same_semantic_prediction"])
                for row in label_order_rows
            ),
            "occurred_same_semantic_prediction": sum(
                bool(row["occurred_same_semantic_prediction"])
                for row in label_order_rows
            ),
            "rows": label_order_rows,
        },
        "asr": {"by_condition": asr_by_condition, "rows": asr_rows},
        "interpretation_gate": {
            "audio_received": all(
                summary["nonempty"] == summary["samples"]
                for summary in asr_by_condition.values()
            ),
            "occurred_event_recognition_noncollapsed": prediction_summaries.get(
                "occurred_recognition", {}
            ).get("all_tasks_noncollapsed", False),
            "future_hidden_noncollapsed": prediction_summaries.get(
                "future_current", {}
            ).get("all_tasks_noncollapsed", False),
            "profile_comparison_allowed": False,
        },
    }
    report["interpretation_gate"]["profile_comparison_allowed"] = bool(
        report["interpretation_gate"]["audio_received"]
        and report["interpretation_gate"]["future_hidden_noncollapsed"]
    )
    write_json(root / "metrics.json", report)
    return report


def _metric_card(title: str, value: str, note: str) -> str:
    return (
        '<div class="metric"><div class="metric-title">'
        + html.escape(title)
        + '</div><div class="metric-value">'
        + html.escape(value)
        + '</div><div class="metric-note">'
        + html.escape(note)
        + "</div></div>"
    )


def render_frontend(run_dir: str | Path) -> Path:
    root = Path(run_dir).resolve()
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    requests = list(read_jsonl(root / "requests.jsonl"))
    responses = list(read_jsonl(root / "responses.jsonl"))
    references = {
        str(row["sample_id"]): row
        for row in read_jsonl(root / "reference_labels.jsonl")
    }
    request_index = {
        (str(row["sample_id"]), str(row["condition"])): row for row in requests
    }
    response_index = {
        (str(row["sample_id"]), str(row["condition"])): row for row in responses
    }

    current = metrics["prediction_conditions"]["future_current"]
    occurred = metrics["prediction_conditions"]["occurred_recognition"]
    paper = metrics["prediction_conditions"]["future_paper_style"]
    cards = "".join(
        [
            _metric_card(
                "未来预测",
                f"{current['correct']}/{current['samples']}",
                "当前因果提示",
            ),
            _metric_card(
                "已发生事件识别",
                f"{occurred['correct']}/{occurred['samples']}",
                "事件后音频已包含",
            ),
            _metric_card(
                "论文风格提示",
                f"{paper['correct']}/{paper['samples']}",
                "保留音频、转写与 hidden profile",
            ),
            _metric_card(
                "静音改变答案",
                f"{metrics['audio_sensitivity']['changed_on_silence']}/"
                f"{metrics['audio_sensitivity']['samples']}",
                "其余文字不变",
            ),
            _metric_card(
                "错配音频改变答案",
                f"{metrics['audio_sensitivity']['changed_on_wrong_audio']}/"
                f"{metrics['audio_sensitivity']['samples']}",
                "同任务相反参考音频",
            ),
            _metric_card(
                "交换答案后语义不变",
                f"{metrics['label_order']['future_same_semantic_prediction']}/"
                f"{metrics['label_order']['samples']}",
                "越低说明答案位置偏置越严重",
            ),
            _metric_card(
                "允许 Profile 对照",
                (
                    "是"
                    if metrics["interpretation_gate"][
                        "profile_comparison_allowed"
                    ]
                    else "否"
                ),
                "要求听音正常且未来预测不塌缩",
            ),
        ]
    )

    condition_labels = {
        "future_current": "当前因果提示",
        "future_current_reversed_labels": "当前提示：答案倒序",
        "future_current_audio_first": "音频在前",
        "future_paper_style": "论文风格提示",
        "future_silence": "静音对照",
        "future_wrong_audio": "错配音频",
        "occurred_recognition": "已发生事件识别",
        "occurred_recognition_reversed_labels": "事件识别：答案倒序",
        "asr_text_first": "ASR：文字在前",
        "asr_audio_first": "ASR：音频在前",
    }
    sample_sections: list[str] = []
    for sample_id in sorted(references):
        reference = references[sample_id]
        task = str(reference["task"])
        causal = request_index[(sample_id, "future_current")]
        observed = request_index[(sample_id, "occurred_recognition")]
        wrong = request_index[(sample_id, "future_wrong_audio")]
        result_rows: list[str] = []
        for condition, label in condition_labels.items():
            request = request_index[(sample_id, condition)]
            response = response_index.get((sample_id, condition), {})
            answer = response.get("prediction") or response.get("transcript") or "无输出"
            result_rows.append(
                "<tr>"
                f"<td>{html.escape(label)}</td>"
                f"<td>{html.escape(str(request['content_order']))}</td>"
                f"<td>{html.escape(str(answer))}</td>"
                f"<td>{html.escape(str(response.get('latency_ms', '')))}</td>"
                "<td><details><summary>查看</summary>"
                f"<h4>Prompt</h4><pre>{html.escape(str(request['prompt']))}</pre>"
                f"<h4>Raw output</h4><pre>{html.escape(str(response.get('raw_response', '')))}</pre>"
                "</details></td></tr>"
            )
        sample_sections.append(
            f"""
            <section class="sample" data-task="{html.escape(task)}">
              <div class="sample-head">
                <div>
                  <h2>{html.escape(sample_id)}</h2>
                  <div class="badges">
                    <span>{html.escape(task)}</span>
                    <span class="target">参考：{html.escape(str(reference['target']))}</span>
                  </div>
                </div>
              </div>
              <div class="audio-grid">
                <label>因果音频（结束处为预测点）
                  <audio controls preload="none" src="{html.escape(str(causal['audio_path']).replace(chr(92), '/'))}"></audio>
                </label>
                <label>已发生事件音频（第 5 秒为预测点）
                  <audio controls preload="none" src="{html.escape(str(observed['audio_path']).replace(chr(92), '/'))}"></audio>
                </label>
                <label>错配音频
                  <audio controls preload="none" src="{html.escape(str(wrong['audio_path']).replace(chr(92), '/'))}"></audio>
                </label>
              </div>
              <details>
                <summary>查看因果部分转写</summary>
                <pre>{html.escape(str(causal['causal_partial_transcript']))}</pre>
              </details>
              <div class="table-wrap">
                <table>
                  <thead><tr><th>检查</th><th>输入顺序</th><th>模型输出</th><th>延迟 ms</th><th>详情</th></tr></thead>
                  <tbody>{''.join(result_rows)}</tbody>
                </table>
              </div>
            </section>
            """
        )

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen2.5-Omni-3B 话轮技术排错</title>
<style>
:root {{ color-scheme: light; --ink:#182230; --muted:#64748b; --line:#dbe3ec; --blue:#155eef; --bg:#f5f7fb; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Inter,"Microsoft YaHei",system-ui,sans-serif; color:var(--ink); background:var(--bg); }}
main {{ width:min(1380px,96vw); margin:32px auto 80px; }}
h1 {{ margin:0 0 8px; font-size:32px; }}
.lead {{ color:var(--muted); margin:0 0 24px; }}
.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin-bottom:24px; }}
.metric,.sample {{ background:white; border:1px solid var(--line); border-radius:14px; box-shadow:0 6px 24px rgba(15,23,42,.05); }}
.metric {{ padding:18px; }}
.metric-title,.metric-note {{ color:var(--muted); font-size:13px; }}
.metric-value {{ font-size:28px; font-weight:750; margin:6px 0; }}
.filters {{ position:sticky; top:0; z-index:5; background:rgba(245,247,251,.94); backdrop-filter:blur(8px); padding:12px 0; display:flex; gap:8px; }}
button {{ border:1px solid var(--line); background:white; padding:9px 14px; border-radius:999px; cursor:pointer; }}
button.active {{ background:var(--blue); color:white; border-color:var(--blue); }}
.sample {{ margin:16px 0; padding:20px; }}
.sample h2 {{ margin:0; font-size:20px; }}
.badges {{ display:flex; gap:8px; margin-top:8px; }}
.badges span {{ background:#eef4ff; color:#1649a3; padding:5px 9px; border-radius:999px; font-size:12px; }}
.badges .target {{ background:#ecfdf3; color:#067647; }}
.audio-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:12px; margin:18px 0; }}
.audio-grid label {{ border:1px solid var(--line); border-radius:10px; padding:12px; font-size:13px; color:var(--muted); }}
audio {{ width:100%; margin-top:8px; }}
details {{ margin:10px 0; }}
summary {{ cursor:pointer; color:#344054; font-weight:650; }}
pre {{ white-space:pre-wrap; word-break:break-word; background:#f8fafc; padding:12px; border-radius:8px; max-height:340px; overflow:auto; }}
.table-wrap {{ overflow:auto; }}
table {{ width:100%; border-collapse:collapse; margin-top:12px; min-width:820px; }}
th,td {{ text-align:left; border-top:1px solid var(--line); padding:10px 8px; vertical-align:top; font-size:13px; }}
th {{ color:var(--muted); }}
.hidden {{ display:none; }}
</style>
</head>
<body>
<main>
  <h1>Qwen2.5-Omni-3B 话轮技术排错</h1>
  <p class="lead">8 条平衡样本；主预测输入为因果音频＋匹配部分转写＋hidden profile。页面可以直接播放音频并核对每次请求。</p>
  <div class="metrics">{cards}</div>
  <div class="filters">
    <button class="active" data-filter="all">全部</button>
    <button data-filter="turn_change">换人</button>
    <button data-filter="backchannel">Backchannel</button>
    <button data-filter="interruption">打断</button>
    <button data-filter="floor_taking">取得话轮</button>
  </div>
  {''.join(sample_sections)}
</main>
<script>
document.querySelectorAll('[data-filter]').forEach(button => {{
  button.addEventListener('click', () => {{
    document.querySelectorAll('[data-filter]').forEach(x => x.classList.remove('active'));
    button.classList.add('active');
    const value = button.dataset.filter;
    document.querySelectorAll('.sample').forEach(card => {{
      card.classList.toggle('hidden', value !== 'all' && card.dataset.task !== value);
    }});
  }});
}});
</script>
</body>
</html>
"""
    destination = root / "review.html"
    destination.write_text(document, encoding="utf-8")
    return destination


__all__ = [
    "audit_requests",
    "build_asr_prompt",
    "build_future_prompt",
    "build_occurred_prompt",
    "prepare_audit",
    "render_frontend",
    "run_audit",
    "score_audit",
    "word_overlap_f1",
]
