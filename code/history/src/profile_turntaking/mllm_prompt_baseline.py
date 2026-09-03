"""Audio+causal-transcript+profile MLLM baseline with paired controls."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import random
import re
import subprocess
import time
import urllib.error
import urllib.request
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .audio import read_wav_window
from .constants import LABELS, UNKNOWN_PROFILE
from .prompt_baseline import (
    PROFILE_MODES,
    profile_to_prompt,
    score_prompt_run as _score_prompt_run,
    select_prompt_rows,
    shuffled_profile_map,
)
from .utils import read_jsonl, write_json, write_jsonl

_JSON_LABEL_PATTERN = re.compile(
    r'\{\s*"label"\s*:\s*"(BC|NA|C|T|I)"\s*\}', re.IGNORECASE
)
_SAFE_FILE_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
_PREFIX_TIME_RE = re.compile(
    r"\[[^\]]+\s+(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\]"
)
_PROFILE_PLACEHOLDER = "<PROFILE_CONDITION>"
_FORBIDDEN_REQUEST_KEYS = {
    "label",
    "target",
    "training_target",
    "annotation_only_not_model_input",
    "label_evidence",
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": list(LABELS)}},
    "required": ["label"],
    "additionalProperties": False,
}


def require_reviewed_labels(
    run_dir: str | Path, *, allow_weak_labels: bool = False
) -> dict[str, Any]:
    """Block formal inference/scoring until every selected target was reviewed."""

    config_path = Path(run_dir) / "run_config.json"
    if not config_path.is_file():
        raise ValueError(f"Missing run configuration: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quality = config.get("label_quality", {})
    if not quality.get("formal_claim_allowed", False) and not allow_weak_labels:
        raise ValueError(
            "This run contains unreviewed weak labels. Complete label review and "
            "regenerate the run, or pass --allow-weak-labels for a diagnostic-only run."
        )
    return quality


def _causal_transcript(row: dict[str, Any], max_transcript_chars: int) -> str:
    transcript = str(row.get("transcript_prefix", "")).strip()
    if max_transcript_chars > 0 and len(transcript) > max_transcript_chars:
        transcript = "[earlier causal history omitted] " + transcript[-max_transcript_chars:]
    return transcript


def build_audio_prompt(
    row: dict[str, Any],
    profile: dict[str, Any],
    *,
    max_transcript_chars: int = 6000,
) -> str:
    """Build a target-free prompt from all three required causal inputs."""

    duration_s = float(row["window_end_s"]) - float(row["window_start_s"])
    transcript = _causal_transcript(row, max_transcript_chars)
    if not transcript:
        transcript = "No completed dialogue unit is available before time t."
    return "\n".join(
        [
            "You are a strict audio turn-taking event classifier.",
            (
                f"The attached {duration_s:.3f}-second mono conversation audio contains only "
                "information available before the prediction boundary and ends exactly at time t."
            ),
            "Classify the conversational state during the next 40-millisecond chunk, [t, t+40 ms].",
            "",
            "Labels:",
            "- C: exactly one participant is speaking and keeps the current floor.",
            "- BC: a short listener backchannel is present while the other participant keeps the floor.",
            "- T: exactly one participant is speaking after the floor transfers to that participant in this chunk.",
            "- I: both participants speak in this chunk and the overlap is not a backchannel.",
            "- NA: neither participant speaks in this chunk.",
            "",
            "Use audible speech activity, pauses, overlap, turn-final prosody, and backchannel cues.",
            "The recording is mono and may contain both participants; do not invent a speaker identity.",
            "The transcript below contains completed causal transcript units; every listed unit ended no later than time t.",
            "Use it together with the audio; do not infer or invent future words.",
            "",
            "Completed causal transcript units available before time t:",
            transcript,
            "",
            "Profile condition:",
            profile_to_prompt(profile),
            "",
            'Return exactly one JSON object with the single key "label".',
            "Its value must be exactly one of C, BC, T, I, NA.",
        ]
    )


def _write_pcm16_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _request_id(sample_id: str, profile_mode: str) -> str:
    return f"{sample_id}::{profile_mode}"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _prompt_template(prompt: str, profile_text: str) -> str:
    if prompt.count(profile_text) != 1:
        raise ValueError("Rendered profile must occur exactly once in the prompt")
    return prompt.replace(profile_text, _PROFILE_PLACEHOLDER, 1)


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


def prepare_mllm_prompt_run(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    max_per_class: int = 1,
    seed: int = 13,
    context_seconds: float = 30.0,
    sample_rate: int = 16_000,
    max_transcript_chars: int = 6000,
) -> dict[str, Any]:
    """Create paired requests where only the rendered profile may change."""

    if context_seconds <= 0:
        raise ValueError("context_seconds must be positive")
    all_rows = list(read_jsonl(manifest_path))
    selected = select_prompt_rows(
        all_rows,
        split=split,
        max_per_class=max_per_class,
        seed=seed,
    )
    if not selected:
        raise ValueError(f"No eligible rows found for split={split!r}")
    split_rows = [
        row for row in all_rows if split == "all" or row.get("split") == split
    ]
    shuffled = shuffled_profile_map(split_rows)
    destination = Path(output_dir)
    clips_dir = destination / "audio_clips"
    requests: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []

    for row in selected:
        prediction_time_s = float(row["prediction_time_s"])
        start_s = max(0.0, prediction_time_s - context_seconds)
        prepared_row = {
            "window_start_s": start_s,
            "window_end_s": prediction_time_s,
            "prediction_time_s": prediction_time_s,
            "transcript_prefix": row.get("transcript_prefix", ""),
        }
        audio = read_wav_window(
            row["audio_path"],
            start_s,
            prediction_time_s,
            target_rate=sample_rate,
        )
        safe_sample_id = _SAFE_FILE_PATTERN.sub("_", str(row["sample_id"]))
        clip_path = clips_dir / f"{safe_sample_id}.wav"
        _write_pcm16_wav(clip_path, audio, sample_rate)
        audio_hash = _sha256_file(clip_path)
        transcript = _causal_transcript(prepared_row, max_transcript_chars)
        transcript_hash = _sha256_text(transcript)
        transcript_times = _PREFIX_TIME_RE.findall(transcript)
        transcript_max_end_s = max(
            (float(end_s) for _, end_s in transcript_times), default=None
        )
        profiles = {
            "hidden": UNKNOWN_PROFILE,
            "given": row["profile"],
            "shuffled": shuffled[str(row["conversation_id"])],
        }
        for profile_mode in PROFILE_MODES:
            profile_text = profile_to_prompt(profiles[profile_mode])
            prompt = build_audio_prompt(
                prepared_row,
                profiles[profile_mode],
                max_transcript_chars=max_transcript_chars,
            )
            prompt_template = _prompt_template(prompt, profile_text)
            request_id = _request_id(str(row["sample_id"]), profile_mode)
            request_fingerprint = hashlib.sha256(
                (audio_hash + "\n" + prompt).encode("utf-8")
            ).hexdigest()
            requests.append(
                {
                    "request_id": request_id,
                    "sample_id": row["sample_id"],
                    "conversation_id": row["conversation_id"],
                    "profile_mode": profile_mode,
                    "audio_path": str(Path("audio_clips") / clip_path.name),
                    "audio_sha256": audio_hash,
                    "audio_duration_s": round(prediction_time_s - start_s, 3),
                    "audio_sample_rate": sample_rate,
                    "window_start_s": start_s,
                    "window_end_s": prediction_time_s,
                    "prediction_time_s": prediction_time_s,
                    "horizon_ms": 40,
                    "transcript_prefix": transcript,
                    "transcript_sha256": transcript_hash,
                    "transcript_chars": len(transcript),
                    "transcript_interval_count": len(transcript_times),
                    "transcript_max_end_s": transcript_max_end_s,
                    "profile_text": profile_text,
                    "prompt": prompt,
                    "prompt_template_sha256": _sha256_text(prompt_template),
                    "request_sha256": request_fingerprint,
                }
            )
            gold.append(
                {
                    "request_id": request_id,
                    "sample_id": row["sample_id"],
                    "conversation_id": row["conversation_id"],
                    "profile_mode": profile_mode,
                    "target": row["label"],
                    "label_source": row.get("label_source", "unknown"),
                    "gold_label": bool(row.get("gold_label", False)),
                }
            )

    write_jsonl(destination / "requests.jsonl", requests)
    write_jsonl(destination / "gold.jsonl", gold)
    summary = {
        "task": "inference_only_audio_causal_transcript_profile_mllm_baseline",
        "training_performed": False,
        "manifest": str(Path(manifest_path).resolve()),
        "split": split,
        "seed": seed,
        "max_per_class": max_per_class,
        "selected_samples": len(selected),
        "mllm_requests": len(requests),
        "profile_modes": list(PROFILE_MODES),
        "context_seconds": context_seconds,
        "sample_rate": sample_rate,
        "horizon_ms": 40,
        "max_transcript_chars": max_transcript_chars,
        "request_file_contains_targets": False,
        "model_inputs": [
            "causal_mono_audio_[t-30s,t]",
            "completed_causal_transcript_units_available_by_t",
            "fixed_template_profile_text",
        ],
        "model_output": "one_of_C_BC_T_I_NA_for_[t,t+40ms]",
        "transcript_is_model_input": True,
        "label_quality": {
            "human_gold_samples": sum(bool(row.get("gold_label", False)) for row in selected),
            "weak_label_samples": sum(not bool(row.get("gold_label", False)) for row in selected),
            "formal_claim_allowed": all(bool(row.get("gold_label", False)) for row in selected),
        },
        "paired_invariant": (
            "Within a sample, audio, causal transcript, prediction boundary, task prompt, "
            "output schema, and decoding are identical; only profile_text changes."
        ),
        "limitations": [
            "This is zero-shot prompting; no model training or fine-tuning is performed.",
            "Automatic SBCSAE labels are candidate weak labels until the selected events are reviewed.",
            "A general MLLM was not specifically trained to forecast a 40-ms event horizon.",
        ],
    }
    write_json(destination / "run_config.json", summary)
    audit_mllm_prompt_run(
        destination,
        expected_samples=len(selected),
        expected_per_class=max_per_class if max_per_class > 0 else None,
    )
    return summary


def audit_mllm_prompt_run(
    run_dir: str | Path,
    *,
    expected_samples: int | None = None,
    expected_per_class: int | None = None,
) -> dict[str, Any]:
    """Fail closed unless the three-input paired experiment contract holds."""

    root = Path(run_dir)
    requests = list(read_jsonl(root / "requests.jsonl"))
    gold = list(read_jsonl(root / "gold.jsonl"))
    config_path = root / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    if expected_samples is None:
        expected_samples = int(config.get("selected_samples", 0)) or None
    if expected_per_class is None:
        expected_per_class = int(config.get("max_per_class", 0)) or None

    errors: list[str] = []
    warnings: list[str] = []
    request_ids = [str(row.get("request_id", "")) for row in requests]
    gold_ids = [str(row.get("request_id", "")) for row in gold]
    if len(request_ids) != len(set(request_ids)):
        errors.append("requests.jsonl contains duplicate request_id values")
    if len(gold_ids) != len(set(gold_ids)):
        errors.append("gold.jsonl contains duplicate request_id values")
    if set(request_ids) != set(gold_ids):
        errors.append("request_id sets differ between requests.jsonl and gold.jsonl")

    requests_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in requests:
        requests_by_sample[str(request.get("sample_id", ""))].append(request)
    unique_samples = len(requests_by_sample)
    if expected_samples is not None and unique_samples != expected_samples:
        errors.append(f"expected {expected_samples} samples, found {unique_samples}")
    if len(requests) != unique_samples * len(PROFILE_MODES):
        errors.append("request count is not exactly three per unique sample")

    gold_target_by_sample: dict[str, set[str]] = defaultdict(set)
    for row in gold:
        gold_target_by_sample[str(row.get("sample_id", ""))].add(str(row.get("target", "")))
    class_counts: Counter[str] = Counter()
    for sample_id, targets in gold_target_by_sample.items():
        if len(targets) != 1:
            errors.append(f"sample {sample_id} does not have exactly one gold target")
            continue
        target = next(iter(targets))
        if target not in LABELS:
            errors.append(f"sample {sample_id} has invalid target {target!r}")
        else:
            class_counts[target] += 1
    if expected_per_class is not None:
        bad_counts = {
            label: class_counts.get(label, 0)
            for label in LABELS
            if class_counts.get(label, 0) != expected_per_class
        }
        if bad_counts:
            errors.append(
                f"expected {expected_per_class} samples per class; mismatches={bad_counts}"
            )

    no_timestamp_samples = 0
    audio_observations: dict[Path, tuple[str, int, int, int]] = {}
    for sample_id, rows in requests_by_sample.items():
        modes = Counter(str(row.get("profile_mode", "")) for row in rows)
        if modes != Counter({mode: 1 for mode in PROFILE_MODES}):
            errors.append(f"sample {sample_id} profile modes are {dict(modes)}")
            continue
        invariant_fields = (
            "audio_sha256",
            "audio_path",
            "audio_duration_s",
            "audio_sample_rate",
            "window_start_s",
            "window_end_s",
            "transcript_sha256",
            "transcript_prefix",
            "prediction_time_s",
            "horizon_ms",
            "prompt_template_sha256",
        )
        for field in invariant_fields:
            if len({json.dumps(row.get(field), sort_keys=True) for row in rows}) != 1:
                errors.append(f"sample {sample_id} changes non-profile field {field}")
        for request in rows:
            forbidden = _nested_keys(request) & _FORBIDDEN_REQUEST_KEYS
            if forbidden:
                errors.append(
                    f"request {request.get('request_id')} contains forbidden keys {sorted(forbidden)}"
                )
            transcript = str(request.get("transcript_prefix", ""))
            if _sha256_text(transcript) != request.get("transcript_sha256"):
                errors.append(f"request {request.get('request_id')} transcript hash mismatch")
            profile_text = str(request.get("profile_text", ""))
            prompt = str(request.get("prompt", ""))
            if not transcript:
                transcript_in_prompt = "No completed dialogue unit is available before time t."
            else:
                transcript_in_prompt = transcript
            if transcript_in_prompt not in prompt:
                errors.append(f"request {request.get('request_id')} transcript is absent from prompt")
            if not profile_text or prompt.count(profile_text) != 1:
                errors.append(f"request {request.get('request_id')} profile rendering mismatch")
            elif _sha256_text(_prompt_template(prompt, profile_text)) != request.get(
                "prompt_template_sha256"
            ):
                errors.append(f"request {request.get('request_id')} prompt template hash mismatch")
            prediction_time_s = float(request.get("prediction_time_s", -1))
            intervals = _PREFIX_TIME_RE.findall(transcript)
            if transcript and not intervals:
                no_timestamp_samples += 1
            if any(float(end_s) > prediction_time_s + 1e-6 for _, end_s in intervals):
                errors.append(f"request {request.get('request_id')} contains future transcript")
            relative_audio = Path(str(request.get("audio_path", "")))
            audio_path = relative_audio if relative_audio.is_absolute() else root / relative_audio
            if not audio_path.is_file():
                errors.append(f"request {request.get('request_id')} audio file is missing")
            else:
                resolved_audio = audio_path.resolve()
                if resolved_audio not in audio_observations:
                    with wave.open(str(resolved_audio), "rb") as wav:
                        audio_observations[resolved_audio] = (
                            _sha256_file(resolved_audio),
                            wav.getnchannels(),
                            wav.getframerate(),
                            wav.getnframes(),
                        )
                observed_hash, channels, rate, frames = audio_observations[resolved_audio]
                if observed_hash != request.get("audio_sha256"):
                    errors.append(f"request {request.get('request_id')} audio hash mismatch")
                if channels != 1:
                    errors.append(f"request {request.get('request_id')} audio is not mono")
                if rate != int(request.get("audio_sample_rate", -1)):
                    errors.append(f"request {request.get('request_id')} audio rate mismatch")
                observed_duration = frames / rate
                if abs(observed_duration - float(request.get("audio_duration_s", -1))) > 1e-3:
                    errors.append(f"request {request.get('request_id')} audio duration mismatch")
            window_start_s = float(request.get("window_start_s", -1))
            window_end_s = float(request.get("window_end_s", -1))
            if abs(window_end_s - prediction_time_s) > 1e-6:
                errors.append(f"request {request.get('request_id')} audio does not end at t")
            if abs((window_end_s - window_start_s) - float(request.get("audio_duration_s", -1))) > 1e-3:
                errors.append(f"request {request.get('request_id')} audio window mismatch")
            if int(request.get("horizon_ms", -1)) != 40:
                errors.append(f"request {request.get('request_id')} horizon is not 40 ms")
            expected_request_hash = _sha256_text(
                str(request.get("audio_sha256", "")) + "\n" + prompt
            )
            if expected_request_hash != request.get("request_sha256"):
                errors.append(f"request {request.get('request_id')} request hash mismatch")
    if no_timestamp_samples:
        warnings.append(
            f"{no_timestamp_samples} requests have non-empty transcripts without parseable timestamps"
        )

    report = {
        "passed": not errors,
        "expected_samples": expected_samples,
        "unique_samples": unique_samples,
        "requests": len(requests),
        "expected_per_class": expected_per_class,
        "class_counts": {label: class_counts.get(label, 0) for label in LABELS},
        "profile_modes": list(PROFILE_MODES),
        "input_contract": {
            "audio": "mono [t-context,t]",
            "transcript": "completed causal transcript units available by t",
            "profile": "hidden/given/shuffled fixed-template natural language",
            "output": "one label in C/BC/T/I/NA for [t,t+40ms]",
        },
        "output_schema": OUTPUT_SCHEMA,
        "errors": errors,
        "warnings": warnings,
    }
    write_json(root / "input_audit.json", report)
    if errors:
        raise ValueError("MLLM input audit failed: " + "; ".join(errors[:10]))
    return report


def prepare_silenced_audio_control(
    main_run_dir: str | Path,
    output_dir: str | Path,
    *,
    samples: int = 50,
    seed: int = 29,
) -> dict[str, Any]:
    """Create a diagnostic set that changes only hidden-mode audio to silence."""

    main_root = Path(main_run_dir).resolve()
    hidden = [
        row
        for row in read_jsonl(main_root / "requests.jsonl")
        if row.get("profile_mode") == "hidden"
    ]
    rng = random.Random(seed)
    hidden.sort(key=lambda row: str(row["sample_id"]))
    rng.shuffle(hidden)
    selected = hidden[:samples]
    if len(selected) != samples:
        raise ValueError(f"Requested {samples} controls but only found {len(selected)}")
    destination = Path(output_dir)
    silence_dir = destination / "silence_clips"
    controls: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for original in selected:
        original_audio = Path(str(original["audio_path"]))
        if not original_audio.is_absolute():
            original_audio = main_root / original_audio
        silence_path = silence_dir / f"{original['sample_id']}.wav"
        with wave.open(str(original_audio), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            frames = source.getnframes()
        silence_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(silence_path), "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(sample_width)
            wav.setframerate(sample_rate)
            wav.writeframes(b"\x00" * frames * channels * sample_width)
        control = dict(original)
        control["request_id"] = f"{original['sample_id']}::hidden_silenced_audio"
        control["profile_mode"] = "hidden_silenced_audio"
        control["reference_request_id"] = original["request_id"]
        control["reference_audio_sha256"] = original["audio_sha256"]
        control["audio_path"] = str(Path("silence_clips") / silence_path.name)
        control["audio_sha256"] = _sha256_file(silence_path)
        control["request_sha256"] = _sha256_text(
            control["audio_sha256"] + "\n" + str(control["prompt"])
        )
        controls.append(control)
        mapping.append(
            {
                "sample_id": original["sample_id"],
                "reference_request_id": original["request_id"],
                "control_request_id": control["request_id"],
            }
        )
    write_jsonl(destination / "requests.jsonl", controls)
    write_jsonl(destination / "mapping.jsonl", mapping)
    summary = {
        "diagnostic_only": True,
        "control": "replace causal audio with duration-matched digital silence",
        "samples": len(controls),
        "seed": seed,
        "invariants": [
            "causal_partial_transcript",
            "hidden_profile_text",
            "prompt",
            "prediction_boundary",
            "output_schema",
        ],
        "changed_field": "audio only",
    }
    write_json(destination / "run_config.json", summary)
    return summary


def score_silenced_audio_control(
    main_run_dir: str | Path,
    control_run_dir: str | Path,
) -> dict[str, Any]:
    """Compare original hidden predictions with their silenced-audio controls."""

    main_responses = {
        str(row["request_id"]): row for row in read_jsonl(Path(main_run_dir) / "responses.jsonl")
    }
    controls = list(read_jsonl(Path(control_run_dir) / "responses.jsonl"))
    control_requests = {
        str(row["request_id"]): row
        for row in read_jsonl(Path(control_run_dir) / "requests.jsonl")
    }
    comparisons: list[dict[str, Any]] = []
    for control in controls:
        request = control_requests.get(str(control.get("request_id")))
        if request is None:
            continue
        original = main_responses.get(str(request.get("reference_request_id")))
        if original is None or not original.get("valid") or not control.get("valid"):
            continue
        comparisons.append(
            {
                "sample_id": control["sample_id"],
                "original_prediction": original["prediction"],
                "silenced_audio_prediction": control["prediction"],
                "changed": original["prediction"] != control["prediction"],
            }
        )
    changed = sum(row["changed"] for row in comparisons)
    report = {
        "diagnostic_only": True,
        "comparable_samples": len(comparisons),
        "changed_predictions": changed,
        "changed_fraction": changed / len(comparisons) if comparisons else None,
        "original_distribution": dict(
            Counter(row["original_prediction"] for row in comparisons)
        ),
        "silenced_audio_distribution": dict(
            Counter(row["silenced_audio_prediction"] for row in comparisons)
        ),
        "interpretation": (
            "A low change fraction suggests that this checkpoint's decision is not strongly "
            "sensitive to the supplied audio under this prompt; it is not proof that audio is ignored."
        ),
    }
    write_json(Path(control_run_dir) / "audio_sensitivity.json", report)
    write_json(Path(control_run_dir) / "comparisons.json", comparisons)
    return report


def parse_mllm_cli_label(raw_output: str) -> str | None:
    """Read the final schema-constrained JSON object from llama.cpp output."""

    matches = [match.upper() for match in _JSON_LABEL_PATTERN.findall(raw_output)]
    return matches[-1] if matches else None


def _exact_mcnemar_pvalue(fixes: int, breaks: int) -> float:
    discordant = fixes + breaks
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(fixes, breaks) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _prompt_run_diagnostics(
    responses: list[dict[str, Any]],
    paired_changes: dict[str, int],
) -> dict[str, Any]:
    distributions: dict[str, Counter[str]] = {
        mode: Counter() for mode in PROFILE_MODES
    }
    predictions_by_sample: dict[str, dict[str, str]] = defaultdict(dict)
    valid_latencies: list[float] = []
    for row in responses:
        mode = str(row.get("profile_mode", ""))
        prediction = str(row.get("prediction", ""))
        if mode in PROFILE_MODES and prediction in LABELS and row.get("valid"):
            distributions[mode][prediction] += 1
            predictions_by_sample[str(row.get("sample_id"))][mode] = prediction
        if isinstance(row.get("latency_ms"), (int, float)):
            valid_latencies.append(float(row["latency_ms"]))
    pair_changes = {}
    for left, right in (("hidden", "given"), ("hidden", "shuffled"), ("given", "shuffled")):
        comparable = [
            modes for modes in predictions_by_sample.values() if left in modes and right in modes
        ]
        changed = sum(modes[left] != modes[right] for modes in comparable)
        pair_changes[f"{left}_vs_{right}"] = {
            "comparable_samples": len(comparable),
            "changed_predictions": changed,
            "changed_fraction": changed / len(comparable) if comparable else 0.0,
        }
    hidden_total = sum(distributions["hidden"].values())
    hidden_dominant = max(distributions["hidden"].values(), default=0)
    hidden_distinct = len(distributions["hidden"])
    hidden_noncollapsed = (
        hidden_distinct >= 3
        and hidden_total > 0
        and hidden_dominant / hidden_total <= 0.80
    )
    latencies = np.asarray(valid_latencies, dtype=np.float64)
    return {
        "prediction_distribution": {
            mode: {label: distributions[mode].get(label, 0) for label in LABELS}
            for mode in PROFILE_MODES
        },
        "paired_prediction_changes": pair_changes,
        "given_vs_hidden_exact_mcnemar_p": _exact_mcnemar_pvalue(
            paired_changes["given_fixes_hidden_error"],
            paired_changes["given_breaks_hidden_correct"],
        ),
        "latency_ms": {
            "count": int(latencies.size),
            "mean": float(latencies.mean()) if latencies.size else None,
            "median": float(np.median(latencies)) if latencies.size else None,
            "p95": float(np.percentile(latencies, 95)) if latencies.size else None,
            "min": float(latencies.min()) if latencies.size else None,
            "max": float(latencies.max()) if latencies.size else None,
        },
        "scientific_validity_gate": {
            "hidden_distinct_predicted_labels": hidden_distinct,
            "hidden_dominant_label_fraction": (
                hidden_dominant / hidden_total if hidden_total else None
            ),
            "hidden_noncollapsed": hidden_noncollapsed,
            "profile_effect_claim_allowed": False,
            "reason": (
                "Hidden baseline is sufficiently non-collapsed. Profile significance still "
                "requires a predeclared paired test."
                if hidden_noncollapsed
                else "Hidden baseline is label-collapsed; do not claim profile efficacy."
            ),
        },
    }


def score_prompt_run(run_dir: str | Path) -> dict[str, Any]:
    """Score the paired run and add MLLM-specific collapse/latency diagnostics."""

    result = _score_prompt_run(run_dir)
    root = Path(run_dir)
    responses = list(read_jsonl(root / "responses.jsonl"))
    diagnostics = _prompt_run_diagnostics(responses, result["paired_changes"])
    write_json(root / "diagnostics.json", diagnostics)
    result["diagnostics"] = diagnostics
    return result


def _run_llama_request(
    request: dict[str, Any],
    *,
    executable: str | Path,
    model_path: str | Path,
    mmproj_path: str | Path,
    timeout_s: float,
    seed: int,
    context_size: int,
    gpu_layers: str,
) -> tuple[str, str, int]:
    command = [
        str(Path(executable).resolve()),
        "-m",
        str(Path(model_path).resolve()),
        "--mmproj",
        str(Path(mmproj_path).resolve()),
        "--audio",
        str(Path(request["audio_path"]).resolve()),
        "-p",
        str(request["prompt"]),
        "--json-schema",
        json.dumps(OUTPUT_SCHEMA, separators=(",", ":")),
        "--predict",
        "16",
        "--ctx-size",
        str(context_size),
        "--gpu-layers",
        gpu_layers,
        "--temperature",
        "0",
        "--top-k",
        "1",
        "--seed",
        str(seed),
        "--no-warmup",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
        check=False,
    )
    return completed.stdout.strip(), completed.stderr.strip(), completed.returncode


def run_mllm_prompt_requests(
    requests_path: str | Path,
    responses_path: str | Path,
    *,
    executable: str | Path,
    model_path: str | Path,
    mmproj_path: str | Path,
    timeout_s: float = 180.0,
    seed: int = 13,
    context_size: int = 4096,
    gpu_layers: str = "all",
    limit: int | None = None,
) -> dict[str, Any]:
    """Run local audio MLLM requests with append-only resume semantics."""

    required_paths = [Path(executable), Path(model_path), Path(mmproj_path)]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing MLLM runtime files: {missing}")
    requests_file = Path(requests_path).resolve()
    requests = list(read_jsonl(requests_file))
    for request in requests:
        audio_path = Path(str(request["audio_path"]))
        if not audio_path.is_absolute():
            request["audio_path"] = str((requests_file.parent / audio_path).resolve())
    if limit is not None:
        requests = requests[:limit]
    destination = Path(responses_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = list(read_jsonl(destination)) if destination.exists() else []
    completed_ids = {str(row["request_id"]) for row in existing}
    written = 0
    invalid = 0
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        for request in requests:
            request_id = str(request["request_id"])
            if request_id in completed_ids:
                continue
            started = time.perf_counter()
            raw_output = ""
            stderr = ""
            error = ""
            returncode = -1
            try:
                raw_output, stderr, returncode = _run_llama_request(
                    request,
                    executable=executable,
                    model_path=model_path,
                    mmproj_path=mmproj_path,
                    timeout_s=timeout_s,
                    seed=seed,
                    context_size=context_size,
                    gpu_layers=gpu_layers,
                )
                if returncode != 0:
                    error = f"llama-mtmd-cli exited with code {returncode}"
            except subprocess.TimeoutExpired:
                error = f"llama-mtmd-cli timed out after {timeout_s:.1f}s"
            prediction = parse_mllm_cli_label(raw_output) if not error else None
            valid = prediction in LABELS
            if not valid:
                invalid += 1
            response = {
                "request_id": request_id,
                "sample_id": request["sample_id"],
                "conversation_id": request["conversation_id"],
                "profile_mode": request["profile_mode"],
                "model": Path(model_path).name,
                "request_sha256": request["request_sha256"],
                "audio_sha256": request["audio_sha256"],
                "prediction": prediction,
                "valid": valid,
                "raw_response": raw_output,
                "error": error,
                "returncode": returncode,
                "stderr_tail": stderr[-2000:],
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
            handle.write(json.dumps(response, ensure_ascii=False) + "\n")
            handle.flush()
            completed_ids.add(request_id)
            written += 1
    return {
        "requested": len(requests),
        "already_completed": len(requests) - written,
        "newly_written": written,
        "new_invalid": invalid,
        "responses": str(destination.resolve()),
    }


def _server_chat_completion_payload(
    endpoint: str,
    request_row: dict[str, Any],
    *,
    model: str,
    timeout_s: float,
    seed: int,
    structured_output: bool = True,
    max_tokens: int = 16,
    output_schema: dict[str, Any] | None = None,
    schema_name: str = "turn_taking_label",
    logprobs: bool = False,
    top_logprobs: int | None = None,
) -> dict[str, Any]:
    audio_data = base64.b64encode(Path(request_row["audio_path"]).read_bytes()).decode("ascii")
    payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        # Qwen2.5-Omni must receive the instruction before the
                        # audio marker.  With the reverse order this llama.cpp
                        # build accepts the request but the model treats real
                        # speech like silence and the five-class output
                        # collapses.  This ordering matches llama.cpp's
                        # documented Qwen2.5-Omni example.
                        {"type": "text", "text": request_row["prompt"]},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_data, "format": "wav"},
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "seed": seed,
        }
    if structured_output:
        response_schema = output_schema or OUTPUT_SCHEMA
        payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": response_schema,
                },
            }
    if logprobs:
        payload["logprobs"] = True
        if top_logprobs is not None:
            payload["top_logprobs"] = top_logprobs
    body = json.dumps(
        payload,
        ensure_ascii=False,
    ).encode("utf-8")
    http_request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(http_request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _server_chat_completion(
    endpoint: str,
    request_row: dict[str, Any],
    *,
    model: str,
    timeout_s: float,
    seed: int,
    structured_output: bool = True,
    max_tokens: int = 16,
    output_schema: dict[str, Any] | None = None,
    schema_name: str = "turn_taking_label",
) -> str:
    payload = _server_chat_completion_payload(
        endpoint,
        request_row,
        model=model,
        timeout_s=timeout_s,
        seed=seed,
        structured_output=structured_output,
        max_tokens=max_tokens,
        output_schema=output_schema,
        schema_name=schema_name,
    )
    content = payload["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content)


def run_mllm_server_requests(
    requests_path: str | Path,
    responses_path: str | Path,
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
    """Run multimodal requests against one persistent llama.cpp server."""

    requests_file = Path(requests_path).resolve()
    requests = list(read_jsonl(requests_file))
    for request in requests:
        audio_path = Path(str(request["audio_path"]))
        if not audio_path.is_absolute():
            request["audio_path"] = str((requests_file.parent / audio_path).resolve())
    if limit is not None:
        requests = requests[:limit]
    destination = Path(responses_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = list(read_jsonl(destination)) if destination.exists() else []
    completed_ids = {str(row["request_id"]) for row in existing}
    written = 0
    invalid = 0
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        for request in requests:
            request_id = str(request["request_id"])
            if request_id in completed_ids:
                continue
            started = time.perf_counter()
            raw_output = ""
            error = ""
            for attempt in range(retries + 1):
                try:
                    raw_output = _server_chat_completion(
                        endpoint,
                        request,
                        model=model,
                        timeout_s=timeout_s,
                        seed=seed,
                        structured_output=structured_output,
                        max_tokens=max_tokens,
                    )
                    error = ""
                    break
                except (KeyError, ValueError, OSError, urllib.error.URLError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt < retries:
                        time.sleep(min(2**attempt, 8))
            prediction = parse_mllm_cli_label(raw_output) if not error else None
            valid = prediction in LABELS
            if not valid:
                invalid += 1
            response = {
                "request_id": request_id,
                "sample_id": request["sample_id"],
                "conversation_id": request["conversation_id"],
                "profile_mode": request["profile_mode"],
                "model": model,
                "backend": "llama.cpp-server-input_audio",
                "request_sha256": request["request_sha256"],
                "audio_sha256": request["audio_sha256"],
                "transcript_sha256": request["transcript_sha256"],
                "prediction": prediction,
                "valid": valid,
                "raw_response": raw_output,
                "error": error,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
            handle.write(json.dumps(response, ensure_ascii=False) + "\n")
            handle.flush()
            completed_ids.add(request_id)
            written += 1
    return {
        "requested": len(requests),
        "already_completed": len(requests) - written,
        "newly_written": written,
        "new_invalid": invalid,
        "responses": str(destination.resolve()),
    }


__all__ = [
    "OUTPUT_SCHEMA",
    "audit_mllm_prompt_run",
    "build_audio_prompt",
    "parse_mllm_cli_label",
    "prepare_silenced_audio_control",
    "prepare_mllm_prompt_run",
    "require_reviewed_labels",
    "run_mllm_prompt_requests",
    "run_mllm_server_requests",
    "score_silenced_audio_control",
    "score_prompt_run",
]
