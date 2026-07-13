"""Audio-plus-profile MLLM prompt baseline with paired profile controls."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .audio import read_wav_window
from .constants import LABELS, UNKNOWN_PROFILE
from .prompt_baseline import (
    PROFILE_MODES,
    profile_to_prompt,
    score_prompt_run,
    select_prompt_rows,
    shuffled_profile_map,
)
from .utils import read_jsonl, write_json, write_jsonl

_JSON_LABEL_PATTERN = re.compile(
    r'\{\s*"label"\s*:\s*"(BC|NA|C|T|I)"\s*\}', re.IGNORECASE
)
_SAFE_FILE_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": list(LABELS)}},
    "required": ["label"],
    "additionalProperties": False,
}


def build_audio_prompt(row: dict[str, Any], profile: dict[str, Any]) -> str:
    """Build a target-free prompt; the transcript and future evidence are excluded."""

    duration_s = float(row["window_end_s"]) - float(row["window_start_s"])
    return "\n".join(
        [
            "You are a strict audio turn-taking event classifier.",
            (
                f"The attached {duration_s:.3f}-second mono conversation audio contains only "
                "information available before the prediction boundary and ends exactly at time t."
            ),
            "Predict the event in the next 40 milliseconds, [t, t+40 ms].",
            "",
            "Labels:",
            "- C: the current speaker continues speaking.",
            "- BC: the listener begins a short backchannel without taking the floor.",
            "- T: the floor transfers to the other participant.",
            "- I: non-backchannel overlapping speech or an interruption begins.",
            "- NA: neither participant speaks.",
            "",
            "Use audible speech activity, pauses, overlap, turn-final prosody, and backchannel cues.",
            "The recording is mono and may contain both participants; do not invent a speaker identity.",
            "Use the profile only as contextual evidence, never as a substitute for the audio.",
            "",
            "Profile condition:",
            profile_to_prompt(profile),
            "",
            'Return exactly one JSON object: {"label":"C"}.',
            "Replace C with exactly one of C, BC, T, I, NA.",
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


def prepare_mllm_prompt_run(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    max_per_class: int = 1,
    seed: int = 13,
    context_seconds: float = 30.0,
    sample_rate: int = 16_000,
) -> dict[str, Any]:
    """Create identical causal audio clips for hidden/given/shuffled prompts."""

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
    split_rows = [row for row in all_rows if row.get("split") == split]
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
        profiles = {
            "hidden": UNKNOWN_PROFILE,
            "given": row["profile"],
            "shuffled": shuffled[str(row["conversation_id"])],
        }
        for profile_mode in PROFILE_MODES:
            prompt = build_audio_prompt(prepared_row, profiles[profile_mode])
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
                    "prompt": prompt,
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
                }
            )

    write_jsonl(destination / "requests.jsonl", requests)
    write_jsonl(destination / "gold.jsonl", gold)
    summary = {
        "task": "inference_only_audio_profile_mllm_baseline",
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
        "request_file_contains_targets": False,
        "model_inputs": ["causal_mono_audio", "fixed_template_profile_text"],
        "transcript_is_model_input": False,
        "paired_invariant": "Each sample uses the same audio SHA-256 in all profile modes.",
        "limitations": [
            "This is zero-shot prompting; no model training or fine-tuning is performed.",
            "SBCSAE MVP labels are weak labels, not frame-accurate human gold labels.",
            "A general MLLM was not specifically trained to forecast a 40-ms event horizon.",
        ],
    }
    write_json(destination / "run_config.json", summary)
    return summary


def parse_mllm_cli_label(raw_output: str) -> str | None:
    """Read the final schema-constrained JSON object from llama.cpp output."""

    matches = [match.upper() for match in _JSON_LABEL_PATTERN.findall(raw_output)]
    return matches[-1] if matches else None


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


__all__ = [
    "OUTPUT_SCHEMA",
    "build_audio_prompt",
    "parse_mllm_cli_label",
    "prepare_mllm_prompt_run",
    "run_mllm_prompt_requests",
    "score_prompt_run",
]
