"""Inference-only foundation-model prompt baseline for profile ablations."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .constants import LABELS, LABEL_TO_ID, UNKNOWN_PROFILE
from .metrics import classification_metrics, write_metrics_csv
from .utils import read_jsonl, write_json, write_jsonl

PROFILE_MODES = ("hidden", "given", "shuffled")
_LABEL_PATTERN = re.compile(r"(?<![A-Z])(BC|NA|C|T|I)(?![A-Z])", re.IGNORECASE)

SYSTEM_PROMPT = """You are a strict five-class turn-taking event classifier.
You receive only information available before one prediction time. Predict the event in the next 40 milliseconds.

Labels:
- C: the current speaker continues speaking.
- BC: the listener gives a short backchannel without taking the floor.
- T: the floor transfers to the other speaker.
- I: non-backchannel overlapping speech or interruption occurs.
- NA: no one is speaking.

Use only the supplied completed dialogue history and profile condition. Do not invent future words.
Return exactly one JSON object such as {"label":"BC"}. The label must be one of C, BC, T, I, NA."""


def _spoken(value: Any) -> str:
    rendered = str(value or "unknown").strip()
    return rendered.replace("_", " ") if rendered else "unknown"


def profile_to_prompt(profile: dict[str, Any]) -> str:
    """Serialize structured profile fields with one fixed, target-free template."""

    if profile == UNKNOWN_PROFILE:
        return "Profile information is unavailable."
    speaker_a = profile.get("speaker_A", {})
    speaker_b = profile.get("speaker_B", {})
    return "\n".join(
        [
            (
                "Speaker A: age group="
                f"{_spoken(speaker_a.get('age_group'))}; gender="
                f"{_spoken(speaker_a.get('gender'))}; social role="
                f"{_spoken(speaker_a.get('social_role'))}; background="
                f"{_spoken(speaker_a.get('background'))}."
            ),
            (
                "Speaker B: age group="
                f"{_spoken(speaker_b.get('age_group'))}; gender="
                f"{_spoken(speaker_b.get('gender'))}; social role="
                f"{_spoken(speaker_b.get('social_role'))}; background="
                f"{_spoken(speaker_b.get('background'))}."
            ),
            f"Relationship: {_spoken(profile.get('relationship'))}.",
            f"Situation: {_spoken(profile.get('situation'))}.",
        ]
    )


def build_messages(
    row: dict[str, Any],
    profile: dict[str, Any],
    *,
    max_transcript_chars: int = 6000,
) -> list[dict[str, str]]:
    """Build a request from an explicit whitelist of causal input fields."""

    transcript = str(row.get("transcript_prefix", "")).strip()
    if max_transcript_chars > 0 and len(transcript) > max_transcript_chars:
        transcript = "[earlier history omitted] " + transcript[-max_transcript_chars:]
    if not transcript:
        transcript = "No completed dialogue history is available."
    user_prompt = "\n".join(
        [
            f"Prediction time: {float(row['prediction_time_s']):.3f} seconds.",
            "",
            "Completed dialogue history:",
            transcript,
            "",
            "Profile condition:",
            profile_to_prompt(profile),
            "",
            "Predict the next-40-ms label now.",
        ]
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def select_prompt_rows(
    rows: Iterable[dict[str, Any]],
    *,
    split: str = "test",
    max_per_class: int = 20,
    seed: int = 13,
) -> list[dict[str, Any]]:
    """Select a deterministic class-stratified pilot.

    ``split='all'`` is intended only for inference-only evaluation where no
    checkpoint was trained on the corpus (for example, a zero-shot MLLM).
    """

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (split == "all" or row.get("split") == split) and row.get("label") in LABELS:
            by_label[str(row["label"])].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for label in LABELS:
        candidates = sorted(by_label[label], key=lambda item: str(item["sample_id"]))
        rng.shuffle(candidates)
        selected.extend(candidates[:max_per_class])
    return sorted(selected, key=lambda item: str(item["sample_id"]))


def shuffled_profile_map(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Rotate whole-conversation profiles as the same negative control used elsewhere."""

    profiles: dict[str, dict[str, Any]] = {}
    for row in rows:
        profiles.setdefault(str(row["conversation_id"]), row["profile"])
    conversation_ids = sorted(profiles)
    if len(conversation_ids) < 2:
        return {conversation_id: UNKNOWN_PROFILE for conversation_id in conversation_ids}
    return {
        conversation_id: profiles[conversation_ids[(index + 1) % len(conversation_ids)]]
        for index, conversation_id in enumerate(conversation_ids)
    }


def _request_id(sample_id: str, profile_mode: str) -> str:
    return f"{sample_id}::{profile_mode}"


def prepare_prompt_run(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    max_per_class: int = 20,
    seed: int = 13,
    max_transcript_chars: int = 6000,
) -> dict[str, Any]:
    """Write target-free requests and a separate local-only gold file."""

    all_rows = list(read_jsonl(manifest_path))
    selected = select_prompt_rows(
        all_rows,
        split=split,
        max_per_class=max_per_class,
        seed=seed,
    )
    if not selected:
        raise ValueError(f"No eligible rows found for split={split!r}")
    split_rows = [row for row in all_rows if row.get("split") == split]
    shuffled = shuffled_profile_map(split_rows)
    requests: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    for row in selected:
        profiles = {
            "hidden": UNKNOWN_PROFILE,
            "given": row["profile"],
            "shuffled": shuffled[str(row["conversation_id"])],
        }
        for profile_mode in PROFILE_MODES:
            request_id = _request_id(str(row["sample_id"]), profile_mode)
            messages = build_messages(
                row,
                profiles[profile_mode],
                max_transcript_chars=max_transcript_chars,
            )
            request_hash = hashlib.sha256(
                json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            requests.append(
                {
                    "request_id": request_id,
                    "sample_id": row["sample_id"],
                    "conversation_id": row["conversation_id"],
                    "profile_mode": profile_mode,
                    "messages": messages,
                    "request_sha256": request_hash,
                }
            )
            gold.append(
                {
                    "request_id": request_id,
                    "sample_id": row["sample_id"],
                    "conversation_id": row["conversation_id"],
                    "profile_mode": profile_mode,
                    "target": row["label"],
                }
            )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "requests.jsonl", requests)
    write_jsonl(destination / "gold.jsonl", gold)
    summary = {
        "task": "inference_only_text_prompt_baseline",
        "training_performed": False,
        "manifest": str(Path(manifest_path).resolve()),
        "split": split,
        "seed": seed,
        "max_per_class": max_per_class,
        "max_transcript_chars": max_transcript_chars,
        "selected_samples": len(selected),
        "api_requests": len(requests),
        "profile_modes": list(PROFILE_MODES),
        "request_file_contains_targets": False,
        "limitations": [
            "Text-only prompts do not expose prosody, energy, or overlap acoustics.",
            "This pilot does not train or fine-tune any model.",
            "Weak SBCSAE labels are not frame-accurate human gold labels.",
        ],
    }
    write_json(destination / "run_config.json", summary)
    return summary


def parse_label(response_text: str) -> str | None:
    """Parse one unambiguous five-class label from JSON or short model text."""

    stripped = response_text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        label = str(payload.get("label", "")).strip().upper()
        return label if label in LABELS else None
    matches = {match.upper() for match in _LABEL_PATTERN.findall(stripped)}
    return next(iter(matches)) if len(matches) == 1 else None


def _chat_completion(
    endpoint: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    timeout_s: float,
) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 16,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload["choices"][0]["message"]["content"])


def run_prompt_requests(
    requests_path: str | Path,
    responses_path: str | Path,
    *,
    endpoint: str,
    model: str,
    api_key_env: str = "PROMPT_API_KEY",
    timeout_s: float = 60.0,
    retries: int = 2,
    delay_s: float = 0.0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Call an OpenAI-compatible endpoint with append-only resume semantics."""

    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Environment variable {api_key_env} is not set")
    requests = list(read_jsonl(requests_path))
    if limit is not None:
        requests = requests[:limit]
    destination = Path(responses_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = list(read_jsonl(destination)) if destination.exists() else []
    completed = {str(row["request_id"]) for row in existing}
    written = 0
    failed = 0
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        for row in requests:
            request_id = str(row["request_id"])
            if request_id in completed:
                continue
            started = time.perf_counter()
            raw_response = ""
            error = ""
            for attempt in range(retries + 1):
                try:
                    raw_response = _chat_completion(
                        endpoint,
                        api_key,
                        model,
                        row["messages"],
                        timeout_s=timeout_s,
                    )
                    error = ""
                    break
                except (KeyError, ValueError, urllib.error.URLError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt < retries:
                        time.sleep(min(2**attempt, 8))
            prediction = parse_label(raw_response) if not error else None
            response_row = {
                "request_id": request_id,
                "sample_id": row["sample_id"],
                "conversation_id": row["conversation_id"],
                "profile_mode": row["profile_mode"],
                "model": model,
                "request_sha256": row["request_sha256"],
                "prediction": prediction,
                "valid": prediction is not None,
                "raw_response": raw_response,
                "error": error,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
            handle.write(json.dumps(response_row, ensure_ascii=False) + "\n")
            handle.flush()
            completed.add(request_id)
            written += 1
            if error or prediction is None:
                failed += 1
            if delay_s > 0:
                time.sleep(delay_s)
    return {
        "requested": len(requests),
        "already_completed": len(requests) - written,
        "newly_written": written,
        "new_failures_or_invalid": failed,
        "responses": str(destination.resolve()),
    }


def _paired_change_summary(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "given_fixes_hidden_error": 0,
        "given_breaks_hidden_correct": 0,
        "both_correct": 0,
        "both_wrong": 0,
    }
    for row in rows:
        target = row["target"]
        hidden_correct = row["hidden_prediction"] == target
        given_correct = row["given_prediction"] == target
        if given_correct and not hidden_correct:
            summary["given_fixes_hidden_error"] += 1
        elif hidden_correct and not given_correct:
            summary["given_breaks_hidden_correct"] += 1
        elif hidden_correct and given_correct:
            summary["both_correct"] += 1
        else:
            summary["both_wrong"] += 1
    return summary


def score_prompt_run(run_dir: str | Path) -> dict[str, Any]:
    """Score the intersection of samples valid under all three profile conditions."""

    root = Path(run_dir)
    gold = {str(row["request_id"]): row for row in read_jsonl(root / "gold.jsonl")}
    responses = list(read_jsonl(root / "responses.jsonl"))
    predictions_by_sample: dict[str, dict[str, Any]] = defaultdict(dict)
    invalid_by_mode = {mode: 0 for mode in PROFILE_MODES}
    for response in responses:
        request_id = str(response["request_id"])
        if request_id not in gold:
            continue
        mode = str(response["profile_mode"])
        if mode not in PROFILE_MODES:
            continue
        if not response.get("valid") or response.get("prediction") not in LABELS:
            invalid_by_mode[mode] += 1
            continue
        gold_row = gold[request_id]
        sample = predictions_by_sample[str(response["sample_id"])]
        sample["sample_id"] = response["sample_id"]
        sample["conversation_id"] = response["conversation_id"]
        sample["target"] = gold_row["target"]
        sample[f"{mode}_prediction"] = response["prediction"]
    paired_rows = [
        row
        for row in predictions_by_sample.values()
        if all(f"{mode}_prediction" in row for mode in PROFILE_MODES)
    ]
    paired_rows.sort(key=lambda row: str(row["sample_id"]))
    if not paired_rows:
        raise ValueError("No samples have valid responses for hidden/given/shuffled")
    reports: dict[str, dict[str, Any]] = {}
    targets = [LABEL_TO_ID[row["target"]] for row in paired_rows]
    for mode in PROFILE_MODES:
        predictions = [LABEL_TO_ID[row[f"{mode}_prediction"]] for row in paired_rows]
        reports[mode] = classification_metrics(targets, predictions)
    write_json(root / "metrics.json", reports)
    write_metrics_csv(root / "profile_comparison.csv", reports)
    write_json(root / "predictions.json", paired_rows)
    write_predictions_csv(root / "predictions.csv", paired_rows)
    paired_changes = _paired_change_summary(paired_rows)
    write_json(root / "paired_changes.json", paired_changes)
    validity = {
        "gold_requests": len(gold),
        "received_responses": len(responses),
        "paired_valid_samples": len(paired_rows),
        "invalid_by_mode": invalid_by_mode,
    }
    write_json(root / "response_validity.json", validity)
    return {
        "metrics": reports,
        "paired_changes": paired_changes,
        "validity": validity,
    }


def write_predictions_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    """Optional compact export for manual review."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "conversation_id",
        "target",
        "hidden_prediction",
        "given_prediction",
        "shuffled_prediction",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
