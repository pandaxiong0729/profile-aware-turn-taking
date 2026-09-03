"""Paper-style binary turn-taking probes aggregated into five classes.

The ICLR 2025 audio-FM benchmark asks separate A/B questions for turn change,
backchannel, and interruption instead of asking one five-way question.  This
module follows that output design while preserving this project's strict input
contract: causal audio + matching causal transcript + profile.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import urllib.error
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .constants import LABELS
from .mllm_prompt_baseline import _server_chat_completion_payload
from .prompt_baseline import PROFILE_MODES
from .qwen25_omni_event_eval import audit_event_eval
from .utils import read_jsonl, write_json, write_jsonl


BINARY_STAGES = ("silence", "listener_onset", "brief_response", "yield")
BINARY_ORDERS = ("ab", "ba")
PROMPT_VERSION = "paper_style_binary_hierarchy_v2_order_calibrated"
_PROFILE_PLACEHOLDER = "<PROFILE_CONDITION>"
_FORBIDDEN_KEYS = {
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
_ANSWER_PATTERN = re.compile(r"(?<![A-Za-z])([ABab])(?![A-Za-z])")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _prompt_template(prompt: str, profile_text: str) -> str:
    if prompt.count(profile_text) != 1:
        raise ValueError("Rendered profile must occur exactly once")
    return prompt.replace(profile_text, _PROFILE_PLACEHOLDER, 1)


def _stage_question(
    stage: str, *, forecast_offset_ms: int, option_order: str
) -> list[str]:
    if option_order not in BINARY_ORDERS:
        raise ValueError(f"Unknown binary option order: {option_order}")
    if stage == "silence":
        original = [
            f"At exactly t+{forecast_offset_ms} ms, which is more likely?",
            "(A) Neither speaker is speaking; a silent interval begins or continues.",
            "(B) At least one speaker is speaking.",
        ]
    elif stage == "listener_onset":
        original = [
            f"At exactly t+{forecast_offset_ms} ms, which is more likely?",
            "(A) No new listener response begins; the current floor holder continues or resumes.",
            "(B) The other participant begins a new response, either a brief acknowledgement or a substantive contribution.",
        ]
    elif stage == "brief_response":
        original = [
            "Assume the other participant begins a response immediately after the audio. Which is more likely?",
            "(A) It is only a brief listener acknowledgement such as mm-hm, yeah, or right, and does not take the floor.",
            "(B) It is a substantive contribution that attempts to take or share the floor.",
        ]
    elif stage == "yield":
        original = [
            "Assume the other participant begins a substantive contribution immediately after the audio. Which is more likely?",
            "(A) The current floor holder yields before it begins, so this is a natural turn change.",
            "(B) It begins before the current floor holder yields, so this is an interruption.",
        ]
    else:
        raise ValueError(f"Unknown binary stage: {stage}")
    if option_order == "ab":
        return original
    return [
        original[0],
        "(A) " + original[2].removeprefix("(B) "),
        "(B) " + original[1].removeprefix("(A) "),
    ]


def build_binary_prompt(
    base: dict[str, Any], *, stage: str, option_order: str = "ab"
) -> str:
    """Build a short A/B question inspired by the paper's best Qwen prompts."""

    transcript = str(base.get("transcript_prefix", "")).strip()
    if not transcript:
        transcript = "No completed transcript unit is available."
    causal_asr = str(base.get("causal_asr_transcript", "")).strip()
    if not causal_asr:
        causal_asr = "No separate causal ASR is available; use the audio directly."
    lines = [
        "You are given the causal audio of a two-speaker conversation. The audio ends abruptly at time t.",
        f"The audio is {float(base['audio_duration_s']):.3f} seconds long and contains no future information.",
        "Use the audio, its matching causal transcript, and the profile. Predict only what happens immediately after t.",
        "",
        "Completed speaker-timed transcript before t:",
        transcript,
        "",
        "Speaker activity at the end of the audio:",
        str(base["boundary_state_text"]),
        "",
        "Causal ASR of the same audio, including unfinished final words:",
        causal_asr,
        "",
        "Speaker profile and conversation context:",
        str(base["profile_text"]),
        "",
        *_stage_question(
            stage,
            forecast_offset_ms=int(base["forecast_offset_ms"]),
            option_order=option_order,
        ),
        "Based only on the causal inputs, output only A or B, nothing else.",
        "Among A or B, the answer is",
    ]
    return "\n".join(lines)


def prepare_binary_hierarchy_eval(
    source_run_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source = Path(source_run_dir)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Binary hierarchy output directory is not empty: {destination}")
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

    probes: list[dict[str, Any]] = []
    for base in read_jsonl(destination / "requests.jsonl"):
        for stage in BINARY_STAGES:
            for option_order in BINARY_ORDERS:
                prompt = build_binary_prompt(
                    base, stage=stage, option_order=option_order
                )
                profile_text = str(base["profile_text"])
                probes.append(
                {
                    "request_id": f"{base['request_id']}::binary_{stage}_{option_order}",
                    "base_request_id": base["request_id"],
                    "sample_id": base["sample_id"],
                    "conversation_id": base["conversation_id"],
                    "profile_mode": base["profile_mode"],
                    "binary_stage": stage,
                    "binary_order": option_order,
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
                        + stage
                        + "\n"
                        + option_order
                        + "\n"
                        + prompt
                    ),
                }
            )
    write_jsonl(destination / "binary_requests.jsonl", probes)
    config_path = destination / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["inference_method"] = PROMPT_VERSION
    config["binary_stages"] = list(BINARY_STAGES)
    config["binary_orders"] = list(BINARY_ORDERS)
    config["binary_requests"] = len(probes)
    config["binary_output_schema"] = "one literal letter: A or B"
    write_json(config_path, config)
    audit = audit_binary_hierarchy_eval(destination)
    return {
        "source_run_dir": str(source.resolve()),
        "output_dir": str(destination.resolve()),
        "samples": base_audit["selected_samples"],
        "base_requests": base_audit["requests"],
        "binary_requests": len(probes),
        "audit_passed": audit["passed"],
    }


def audit_binary_hierarchy_eval(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    base = audit_event_eval(root)
    base_requests = {
        str(row["request_id"]): row for row in read_jsonl(root / "requests.jsonl")
    }
    probes = list(read_jsonl(root / "binary_requests.jsonl"))
    errors: list[str] = []
    expected = base["requests"] * len(BINARY_STAGES) * len(BINARY_ORDERS)
    if len(probes) != expected:
        errors.append(f"expected {expected} binary requests, found {len(probes)}")
    request_ids = [str(row.get("request_id", "")) for row in probes]
    if len(request_ids) != len(set(request_ids)):
        errors.append("binary requests contain duplicate request IDs")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in probes:
        request_id = str(row.get("request_id", ""))
        base_request = base_requests.get(str(row.get("base_request_id", "")))
        if base_request is None:
            errors.append(f"binary request {request_id} has no base request")
            continue
        stage = str(row.get("binary_stage", ""))
        if stage not in BINARY_STAGES:
            errors.append(f"binary request {request_id} has invalid stage {stage}")
        option_order = str(row.get("binary_order", ""))
        if option_order not in BINARY_ORDERS:
            errors.append(
                f"binary request {request_id} has invalid option order {option_order}"
            )
        forbidden = _nested_keys(row) & _FORBIDDEN_KEYS
        if forbidden:
            errors.append(f"binary request {request_id} contains forbidden keys {sorted(forbidden)}")
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
                errors.append(f"binary request {request_id} changes base field {field}")
        profile_text = str(row.get("profile_text", ""))
        prompt = str(row.get("prompt", ""))
        if not profile_text or prompt.count(profile_text) != 1:
            errors.append(f"binary request {request_id} profile rendering mismatch")
        elif _sha256_text(_prompt_template(prompt, profile_text)) != row.get(
            "prompt_template_sha256"
        ):
            errors.append(f"binary request {request_id} prompt template mismatch")
        expected_hash = _sha256_text(
            str(row.get("audio_sha256", ""))
            + "\n"
            + str(row.get("transcript_sha256", ""))
            + "\n"
            + str(row.get("boundary_state_sha256", ""))
            + "\n"
            + str(row.get("causal_asr_sha256", ""))
            + "\n"
            + stage
            + "\n"
            + option_order
            + "\n"
            + prompt
        )
        if expected_hash != row.get("request_sha256"):
            errors.append(f"binary request {request_id} request hash mismatch")
        groups[(str(row.get("sample_id", "")), f"{stage}:{option_order}")].append(row)
    expected_groups = (
        base["selected_samples"] * len(BINARY_STAGES) * len(BINARY_ORDERS)
    )
    if len(groups) != expected_groups:
        errors.append(f"expected {expected_groups} sample/stage groups, found {len(groups)}")
    for key, rows in groups.items():
        if {str(row.get("profile_mode", "")) for row in rows} != set(PROFILE_MODES):
            errors.append(f"binary group {key} does not contain all profile modes")
        for field in (
            "audio_sha256",
            "transcript_sha256",
            "boundary_state_sha256",
            "causal_asr_sha256",
            "prompt_template_sha256",
        ):
            if len({str(row.get(field, "")) for row in rows}) != 1:
                errors.append(f"binary group {key} changes non-profile field {field}")
    report = {
        "passed": not errors,
        "samples": base["selected_samples"],
        "base_requests": base["requests"],
        "binary_requests": len(probes),
        "stages": list(BINARY_STAGES),
        "orders": list(BINARY_ORDERS),
        "profile_modes": list(PROFILE_MODES),
        "errors": errors,
    }
    write_json(root / "binary_input_audit.json", report)
    if errors:
        raise ValueError("Binary hierarchy input audit failed: " + "; ".join(errors[:10]))
    return report


def parse_binary_answer(raw: str) -> str | None:
    stripped = raw.strip().upper()
    if stripped in {"A", "B"}:
        return stripped
    matches = [match.upper() for match in _ANSWER_PATTERN.findall(raw)]
    return matches[-1] if matches else None


def _binary_payload_result(payload: dict[str, Any]) -> tuple[str, str | None, float, float]:
    choice = payload["choices"][0]
    content = str(choice["message"]["content"])
    answer = parse_binary_answer(content)
    token_rows = choice.get("logprobs", {}).get("content", [])
    if not token_rows:
        raise ValueError("binary response has no token log probabilities")
    top = token_rows[0].get("top_logprobs", [])
    scores = {
        str(item.get("token", "")).strip().upper(): float(item["logprob"])
        for item in top
        if str(item.get("token", "")).strip().upper() in {"A", "B"}
    }
    if set(scores) != {"A", "B"}:
        raise ValueError("binary response top log probabilities do not contain A and B")
    return content, answer, scores["A"], scores["B"]


def run_binary_hierarchy_server(
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
    audit_binary_hierarchy_eval(root)
    rows = list(read_jsonl(root / "binary_requests.jsonl"))
    if limit is not None:
        rows = rows[:limit]
    destination = root / "binary_responses.jsonl"
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
                    payload = _server_chat_completion_payload(
                        endpoint,
                        request_row,
                        model=model,
                        timeout_s=timeout_s,
                        seed=seed,
                        structured_output=False,
                        max_tokens=1,
                        logprobs=True,
                        top_logprobs=20,
                    )
                    raw, answer, logprob_a, logprob_b = _binary_payload_result(
                        payload
                    )
                    error = ""
                    break
                except (KeyError, ValueError, OSError, urllib.error.URLError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt < retries:
                        time.sleep(min(2**attempt, 8))
            if error:
                answer = None
                logprob_a = None
                logprob_b = None
            valid = answer in {"A", "B"}
            if not valid:
                invalid += 1
            response = {
                "request_id": request_id,
                "base_request_id": row["base_request_id"],
                "sample_id": row["sample_id"],
                "conversation_id": row["conversation_id"],
                "profile_mode": row["profile_mode"],
                "binary_stage": row["binary_stage"],
                "binary_order": row["binary_order"],
                "request_sha256": row["request_sha256"],
                "audio_sha256": row["audio_sha256"],
                "answer": answer,
                "token_logprob_A": logprob_a,
                "token_logprob_B": logprob_b,
                "semantic_A_log_odds": (
                    (float(logprob_a) - float(logprob_b))
                    if row["binary_order"] == "ab" and logprob_a is not None
                    else (
                        float(logprob_b) - float(logprob_a)
                        if logprob_a is not None
                        else None
                    )
                ),
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


def _answers_to_label(answers: dict[str, str]) -> str:
    if answers["silence"] == "A":
        return "NA"
    if answers["listener_onset"] == "A":
        return "C"
    if answers["brief_response"] == "A":
        return "BC"
    return "T" if answers["yield"] == "A" else "I"


def aggregate_binary_hierarchy(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    audit = audit_binary_hierarchy_eval(root)
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    rows = list(read_jsonl(root / "binary_responses.jsonl"))
    invalid = 0
    for row in rows:
        if not row.get("valid") or row.get("answer") not in {"A", "B"}:
            invalid += 1
            continue
        grouped[(str(row["sample_id"]), str(row["profile_mode"]))][
            f"{row['binary_stage']}:{row['binary_order']}"
        ] = row
    predictions: list[dict[str, Any]] = []
    standard: list[dict[str, Any]] = []
    incomplete: list[str] = []
    contradictions = 0
    option_order_disagreements = 0
    expected_stage_orders = {
        f"{stage}:{option_order}"
        for stage in BINARY_STAGES
        for option_order in BINARY_ORDERS
    }
    for base in read_jsonl(root / "requests.jsonl"):
        key = (str(base["sample_id"]), str(base["profile_mode"]))
        stage_rows = grouped.get(key, {})
        if set(stage_rows) != expected_stage_orders:
            incomplete.append(f"{key[0]}::{key[1]}")
            continue
        answers: dict[str, str] = {}
        stage_log_odds: dict[str, float] = {}
        raw_answers: dict[str, dict[str, str]] = {}
        for stage in BINARY_STAGES:
            original = stage_rows[f"{stage}:ab"]
            swapped = stage_rows[f"{stage}:ba"]
            log_odds = (
                float(original["semantic_A_log_odds"])
                + float(swapped["semantic_A_log_odds"])
            ) / 2.0
            stage_log_odds[stage] = round(log_odds, 6)
            answers[stage] = "A" if log_odds >= 0.0 else "B"
            semantic_original = str(original["answer"])
            semantic_swapped = "B" if str(swapped["answer"]) == "A" else "A"
            if semantic_original != semantic_swapped:
                option_order_disagreements += 1
            raw_answers[stage] = {
                "ab": str(original["answer"]),
                "ba": str(swapped["answer"]),
            }
        if answers["silence"] == "A" and answers["listener_onset"] == "B":
            contradictions += 1
        prediction = _answers_to_label(answers)
        predictions.append(
            {
                "sample_id": key[0],
                "conversation_id": base["conversation_id"],
                "profile_mode": key[1],
                "answers": answers,
                "semantic_A_log_odds": stage_log_odds,
                "raw_token_answers": raw_answers,
                "prediction": prediction,
            }
        )
        standard.append(
            {
                "request_id": base["request_id"],
                "sample_id": key[0],
                "conversation_id": base["conversation_id"],
                "profile_mode": key[1],
                "model": "Qwen2.5-Omni-3B-Q8_0 paper-style binary hierarchy",
                "request_sha256": base["request_sha256"],
                "audio_sha256": base["audio_sha256"],
                "transcript_sha256": base["transcript_sha256"],
                "prediction": prediction,
                "valid": True,
                "raw_response": json.dumps(answers, ensure_ascii=False),
                "error": "",
                "latency_ms": sum(
                    float(stage_rows[f"{stage}:{option_order}"].get("latency_ms", 0.0))
                    for stage in BINARY_STAGES
                    for option_order in BINARY_ORDERS
                ),
            }
        )
    if incomplete:
        raise ValueError(f"Incomplete binary answers for {len(incomplete)} inputs: {incomplete[:5]}")
    write_jsonl(root / "binary_predictions.jsonl", predictions)
    write_jsonl(root / "responses.jsonl", standard)
    distribution = {
        mode: {
            label: sum(
                row["profile_mode"] == mode and row["prediction"] == label
                for row in predictions
            )
            for label in LABELS
        }
        for mode in PROFILE_MODES
    }
    stage_distribution = {
        mode: {
            stage: dict(
                Counter(
                    row["answers"][stage]
                    for row in predictions
                    if row["profile_mode"] == mode
                )
            )
            for stage in BINARY_STAGES
        }
        for mode in PROFILE_MODES
    }
    summary = {
        "samples": audit["samples"],
        "paired_inputs": len(predictions),
        "binary_responses": len(rows),
        "invalid_binary_responses": invalid,
        "cross_stage_contradictions": contradictions,
        "option_order_disagreements": option_order_disagreements,
        "prediction_distribution": distribution,
        "stage_answer_distribution": stage_distribution,
    }
    write_json(root / "binary_aggregation.json", summary)
    return summary


__all__ = [
    "BINARY_ORDERS",
    "BINARY_STAGES",
    "PROMPT_VERSION",
    "aggregate_binary_hierarchy",
    "audit_binary_hierarchy_eval",
    "build_binary_prompt",
    "parse_binary_answer",
    "prepare_binary_hierarchy_eval",
    "run_binary_hierarchy_server",
]
