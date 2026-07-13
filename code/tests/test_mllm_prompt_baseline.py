from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

import profile_turntaking.mllm_prompt_baseline as mllm
from profile_turntaking.constants import LABELS
from profile_turntaking.mllm_prompt_baseline import (
    build_audio_prompt,
    parse_mllm_cli_label,
    prepare_mllm_prompt_run,
    run_mllm_prompt_requests,
)
from profile_turntaking.utils import read_jsonl, write_jsonl


def _write_audio(path: Path) -> None:
    samples = (0.05 * np.sin(2 * np.pi * 180 * np.arange(32_000) / 16_000) * 32767).astype(
        "<i2"
    )
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(samples.tobytes())


def _profile(role: str) -> dict:
    return {
        "speaker_A": {
            "age_group": "25-34",
            "gender": "female",
            "social_role": role,
            "background": "synthetic A",
        },
        "speaker_B": {
            "age_group": "35-44",
            "gender": "male",
            "social_role": "listener",
            "background": "synthetic B",
        },
        "relationship": "friends",
        "situation": "casual_conversation",
    }


def _rows(audio_path: Path) -> list[dict]:
    rows = []
    for index, label in enumerate(LABELS):
        conversation = "A" if index < 3 else "B"
        rows.append(
            {
                "sample_id": f"sample-{label}",
                "conversation_id": conversation,
                "split": "test",
                "prediction_time_s": 2.0,
                "window_start_s": 1.0,
                "window_end_s": 2.0,
                "audio_path": str(audio_path),
                "transcript_prefix": "LEAK_TRANSCRIPT",
                "profile": _profile("teacher" if conversation == "A" else "doctor"),
                "label": label,
                "training_target": {"next_40ms_label": "LEAK_TARGET"},
                "annotation_only_not_model_input": {"reason": "LEAK_REASON"},
            }
        )
    return rows


def test_audio_prompt_describes_causal_task_without_transcript() -> None:
    prompt = build_audio_prompt(
        {"window_start_s": 0.0, "window_end_s": 30.0}, _profile("teacher")
    )
    assert "30.000-second mono" in prompt
    assert "next 40 milliseconds" in prompt
    assert "teacher" in prompt
    assert "transcript" not in prompt.lower()


def test_prepare_uses_identical_audio_and_separates_gold(tmp_path: Path) -> None:
    audio_path = tmp_path / "source.wav"
    _write_audio(audio_path)
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, _rows(audio_path))
    summary = prepare_mllm_prompt_run(
        manifest,
        tmp_path / "run",
        max_per_class=1,
        context_seconds=1.0,
    )
    requests = list(read_jsonl(tmp_path / "run" / "requests.jsonl"))
    gold = list(read_jsonl(tmp_path / "run" / "gold.jsonl"))
    assert summary["selected_samples"] == 5
    assert summary["mllm_requests"] == 15
    assert len(requests) == len(gold) == 15
    serialized = json.dumps(requests, ensure_ascii=False)
    assert "LEAK_TRANSCRIPT" not in serialized
    assert "LEAK_TARGET" not in serialized
    assert "LEAK_REASON" not in serialized
    assert '"target"' not in serialized

    sample = [row for row in requests if row["sample_id"] == "sample-C"]
    assert len({row["audio_sha256"] for row in sample}) == 1
    assert len({row["audio_path"] for row in sample}) == 1
    prompts = {row["profile_mode"]: row["prompt"] for row in sample}
    assert "Profile information is unavailable" in prompts["hidden"]
    assert "teacher" in prompts["given"]
    assert "doctor" in prompts["shuffled"]

    clip_path = tmp_path / "run" / sample[0]["audio_path"]
    with wave.open(str(clip_path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 16_000
        assert wav.getnframes() == 16_000


def test_cli_label_parser_uses_final_json() -> None:
    assert parse_mllm_cli_label('{"label":"BC"}') == "BC"
    assert parse_mllm_cli_label('prompt {"label":"C"}\n{"label":"NA"}') == "NA"
    assert parse_mllm_cli_label("C") is None


def test_runner_resolves_relative_audio_and_resumes(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    clip = run_dir / "audio_clips" / "sample.wav"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"RIFF-fixture")
    runtime_files = []
    for name in ("runner.exe", "model.gguf", "mmproj.gguf"):
        path = tmp_path / name
        path.write_bytes(b"fixture")
        runtime_files.append(path)
    request = {
        "request_id": "sample::hidden",
        "sample_id": "sample",
        "conversation_id": "conversation",
        "profile_mode": "hidden",
        "audio_path": "audio_clips/sample.wav",
        "audio_sha256": "abc",
        "prompt": "target-free prompt",
        "request_sha256": "def",
    }
    write_jsonl(run_dir / "requests.jsonl", [request])
    observed_audio_paths = []

    def fake_run(row, **kwargs):
        observed_audio_paths.append(row["audio_path"])
        return '{"label":"T"}', "runtime log", 0

    monkeypatch.setattr(mllm, "_run_llama_request", fake_run)
    arguments = {
        "executable": runtime_files[0],
        "model_path": runtime_files[1],
        "mmproj_path": runtime_files[2],
    }
    first = run_mllm_prompt_requests(
        run_dir / "requests.jsonl", run_dir / "responses.jsonl", **arguments
    )
    second = run_mllm_prompt_requests(
        run_dir / "requests.jsonl", run_dir / "responses.jsonl", **arguments
    )
    assert first["newly_written"] == 1
    assert second["already_completed"] == 1
    assert observed_audio_paths == [str(clip.resolve())]
    response = list(read_jsonl(run_dir / "responses.jsonl"))[0]
    assert response["prediction"] == "T"
    assert response["valid"] is True
