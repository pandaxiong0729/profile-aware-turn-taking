from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

import profile_turntaking.mllm_prompt_baseline as mllm
from profile_turntaking.constants import LABELS
from profile_turntaking.paper_binary_hierarchy import (
    aggregate_binary_hierarchy,
    parse_binary_answer,
    prepare_binary_hierarchy_eval,
)
from profile_turntaking.qwen25_omni_event_eval import (
    aggregate_candidate_scores,
    apply_causal_asr,
    audit_event_eval,
    prepare_candidate_score_eval,
    prepare_event_eval,
    score_event_eval,
)
from profile_turntaking.utils import read_jsonl, write_jsonl


def _write_audio(path: Path, seconds: float = 8.0) -> None:
    times = np.arange(int(seconds * 16_000)) / 16_000
    samples = (0.08 * np.sin(2 * np.pi * 190 * times) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(samples.tobytes())


def _participants(role: str) -> dict:
    return {
        "speaker_00": {
            "display_name": "SHOULD_NOT_APPEAR",
            "profile": {
                "age_group": "25-34",
                "gender": "female",
                "social_role": role,
                "background": f"background {role}",
            },
        },
        "speaker_01": {
            "display_name": "ALSO_PRIVATE",
            "profile": {
                "age_group": "35-44",
                "gender": "male",
                "social_role": "listener",
                "background": "background listener",
            },
        },
    }


def _rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for conversation_id, role in (("A", "teacher"), ("B", "doctor")):
        audio = root / f"{conversation_id}.wav"
        _write_audio(audio)
        for index, label in enumerate(LABELS):
            rows.append(
                {
                    "candidate_label": label,
                    "structure": "LEAK_STRUCTURE",
                    "anchor_s": 10.0 + index,
                    "event_id": f"{conversation_id}-event-{label}",
                    "conversation_id": conversation_id,
                    "candidate_confidence": "high",
                    "evidence": {"answer": label},
                    "human_label": None,
                    "review_status": "unreviewed",
                    "conversation_context": {
                        "relationship": f"friends {role}",
                        "situation": "casual conversation",
                    },
                    "participants": _participants(role),
                    "audio_path": str(audio),
                    "audio_sha256": "source-review-audio",
                    "target_start_in_clip_s": 2.0,
                    "context_transcript": [
                        {
                            "speaker": "speaker_00",
                            "start_in_clip_s": 0.4,
                            "end_in_clip_s": 1.4,
                            "text": "PAST_WORDS",
                        },
                        {
                            "speaker": "speaker_01",
                            "start_in_clip_s": 1.8,
                            "end_in_clip_s": 2.2,
                            "text": "LEAK_CROSSING_WORDS",
                        },
                        {
                            "speaker": "speaker_01",
                            "start_in_clip_s": 2.3,
                            "end_in_clip_s": 3.0,
                            "text": "LEAK_FUTURE_WORDS",
                        },
                    ],
                }
            )
    return rows


def _prepare(tmp_path: Path) -> Path:
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, _rows(tmp_path))
    run_dir = tmp_path / "run"
    prepare_event_eval(
        manifest,
        run_dir,
        conversations=["A", "B"],
        per_class=1,
        context_seconds=2.0,
        min_boundary_separation_s=0.0,
        max_per_conversation_class=1,
    )
    return run_dir


def test_prepare_is_strictly_causal_and_profile_paired(tmp_path: Path) -> None:
    run_dir = _prepare(tmp_path)
    audit = audit_event_eval(run_dir, expected_samples=5, expected_per_class=1)
    assert audit["passed"] is True
    requests = list(read_jsonl(run_dir / "requests.jsonl"))
    references = list(read_jsonl(run_dir / "reference_labels.jsonl"))
    assert len(requests) == 15
    assert len(references) == 5
    assert {row["reference_label"] for row in references} == set(LABELS)
    serialized = json.dumps(requests)
    assert "LEAK_CROSSING_WORDS" not in serialized
    assert "LEAK_FUTURE_WORDS" not in serialized
    assert "LEAK_STRUCTURE" not in serialized
    assert "SHOULD_NOT_APPEAR" not in serialized
    assert "PAST_WORDS" in serialized
    assert "speaker_01" in serialized
    assert "observed_start_s" in serialized
    assert "candidate_label" not in serialized
    assert "event_time_in_conversation_s" not in serialized
    assert "event_offset_ms" not in serialized
    for reference in references:
        assert reference["event_offset_ms"] == 100
        assert round(
            reference["event_time_in_conversation_s"]
            - reference["prediction_boundary_in_conversation_s"],
            3,
        ) == 0.1

    by_sample: dict[str, list[dict]] = {}
    for row in requests:
        by_sample.setdefault(row["sample_id"], []).append(row)
    for rows in by_sample.values():
        assert {row["profile_mode"] for row in rows} == {
            "hidden",
            "given",
            "shuffled",
        }
        assert len({row["audio_sha256"] for row in rows}) == 1
        assert len({row["transcript_sha256"] for row in rows}) == 1
        assert len({row["prompt_template_sha256"] for row in rows}) == 1
        assert len({row["profile_text"] for row in rows}) == 3
        assert {row["horizon_ms"] for row in rows} == {140}
        assert {row["forecast_offset_ms"] for row in rows} == {100}
        assert {row["evaluation_window_ms"] for row in rows} == {40}
        assert {row["source_clip_boundary_s"] for row in rows} == {1.9}
        assert "<PREDICTION_BOUNDARY t=END_OF_AUDIO=1.900s>" in rows[0]["prompt"]
        assert "[t+100 ms, t+140 ms)" in rows[0]["prompt"]
        with wave.open(str(run_dir / rows[0]["audio_path"]), "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getframerate() == 16_000
            assert wav.getnframes() == 30_400


def test_causal_asr_is_shared_across_profile_conditions(tmp_path: Path) -> None:
    run_dir = _prepare(tmp_path)
    unique = {}
    for row in read_jsonl(run_dir / "requests.jsonl"):
        unique.setdefault(
            (row["sample_id"], row["audio_sha256"]),
            {
                "sample_id": row["sample_id"],
                "audio_sha256": row["audio_sha256"],
                "transcript": f"causal words for {row['sample_id']}",
                "error": "",
            },
        )
    write_jsonl(run_dir / "asr.jsonl", unique.values())
    result = apply_causal_asr(run_dir)
    assert result["audit_passed"] is True
    assert result["samples"] == 5
    by_sample: dict[str, list[dict]] = {}
    for row in read_jsonl(run_dir / "requests.jsonl"):
        by_sample.setdefault(row["sample_id"], []).append(row)
    for sample_id, rows in by_sample.items():
        expected = f"causal words for {sample_id}"
        assert {row["causal_asr_transcript"] for row in rows} == {expected}
        assert len({row["causal_asr_sha256"] for row in rows}) == 1
        assert all(expected in row["prompt"] for row in rows)
        assert len({row["prompt_template_sha256"] for row in rows}) == 1


def test_candidate_scoring_preserves_inputs_and_aggregates(tmp_path: Path) -> None:
    source = _prepare(tmp_path)
    unique = {}
    for row in read_jsonl(source / "requests.jsonl"):
        unique.setdefault(
            (row["sample_id"], row["audio_sha256"]),
            {
                "sample_id": row["sample_id"],
                "audio_sha256": row["audio_sha256"],
                "transcript": f"causal words for {row['sample_id']}",
                "error": "",
            },
        )
    write_jsonl(source / "asr.jsonl", unique.values())
    apply_causal_asr(source)
    run_dir = tmp_path / "candidate"
    prepared = prepare_candidate_score_eval(source, run_dir)
    assert prepared["candidate_requests"] == 75
    candidates = list(read_jsonl(run_dir / "candidate_requests.jsonl"))
    assert len(candidates) == 75
    reference = {
        row["sample_id"]: row["reference_label"]
        for row in read_jsonl(run_dir / "reference_labels.jsonl")
    }
    responses = []
    for row in candidates:
        score = 90 if row["hypothesis_label"] == reference[row["sample_id"]] else 10
        responses.append(
            {
                "request_id": row["request_id"],
                "base_request_id": row["base_request_id"],
                "sample_id": row["sample_id"],
                "conversation_id": row["conversation_id"],
                "profile_mode": row["profile_mode"],
                "hypothesis_label": row["hypothesis_label"],
                "score": score,
                "valid": True,
                "latency_ms": 1.0,
            }
        )
    write_jsonl(run_dir / "candidate_responses.jsonl", responses)
    summary = aggregate_candidate_scores(run_dir)
    assert summary["ties"] == 0
    assert summary["paired_inputs"] == 15
    predictions = list(read_jsonl(run_dir / "responses.jsonl"))
    assert all(
        row["prediction"] == reference[row["sample_id"]] for row in predictions
    )


def test_paper_binary_hierarchy_maps_all_five_classes(tmp_path: Path) -> None:
    assert parse_binary_answer("A") == "A"
    assert parse_binary_answer("The answer is (b)") == "B"
    source = _prepare(tmp_path)
    unique = {}
    for row in read_jsonl(source / "requests.jsonl"):
        unique.setdefault(
            (row["sample_id"], row["audio_sha256"]),
            {
                "sample_id": row["sample_id"],
                "audio_sha256": row["audio_sha256"],
                "transcript": f"causal words for {row['sample_id']}",
                "error": "",
            },
        )
    write_jsonl(source / "asr.jsonl", unique.values())
    apply_causal_asr(source)
    run_dir = tmp_path / "binary"
    prepared = prepare_binary_hierarchy_eval(source, run_dir)
    assert prepared["binary_requests"] == 120
    reference = {
        row["sample_id"]: row["reference_label"]
        for row in read_jsonl(run_dir / "reference_labels.jsonl")
    }
    answer_plan = {
        "C": {"silence": "B", "listener_onset": "A", "brief_response": "B", "yield": "B"},
        "BC": {"silence": "B", "listener_onset": "B", "brief_response": "A", "yield": "B"},
        "T": {"silence": "B", "listener_onset": "B", "brief_response": "B", "yield": "A"},
        "I": {"silence": "B", "listener_onset": "B", "brief_response": "B", "yield": "B"},
        "NA": {"silence": "A", "listener_onset": "A", "brief_response": "B", "yield": "B"},
    }
    responses = []
    for row in read_jsonl(run_dir / "binary_requests.jsonl"):
        semantic_answer = answer_plan[reference[row["sample_id"]]][row["binary_stage"]]
        answer = (
            semantic_answer
            if row["binary_order"] == "ab"
            else ("B" if semantic_answer == "A" else "A")
        )
        responses.append(
            {
                "request_id": row["request_id"],
                "base_request_id": row["base_request_id"],
                "sample_id": row["sample_id"],
                "conversation_id": row["conversation_id"],
                "profile_mode": row["profile_mode"],
                "binary_stage": row["binary_stage"],
                "binary_order": row["binary_order"],
                "answer": answer,
                "semantic_A_log_odds": 1.0 if semantic_answer == "A" else -1.0,
                "valid": True,
                "latency_ms": 1.0,
            }
        )
    write_jsonl(run_dir / "binary_responses.jsonl", responses)
    summary = aggregate_binary_hierarchy(run_dir)
    assert summary["paired_inputs"] == 15
    predictions = list(read_jsonl(run_dir / "responses.jsonl"))
    assert all(
        row["prediction"] == reference[row["sample_id"]] for row in predictions
    )


def test_score_detects_collapse_and_accepts_noncollapsed_output(tmp_path: Path) -> None:
    run_dir = _prepare(tmp_path)
    requests = list(read_jsonl(run_dir / "requests.jsonl"))
    collapsed = [
        {
            "request_id": row["request_id"],
            "sample_id": row["sample_id"],
            "conversation_id": row["conversation_id"],
            "profile_mode": row["profile_mode"],
            "prediction": "C",
            "valid": True,
            "latency_ms": 5.0,
        }
        for row in requests
    ]
    write_jsonl(run_dir / "responses.jsonl", collapsed)
    collapsed_report = score_event_eval(run_dir, bootstrap_resamples=20)
    assert collapsed_report["diagnostics"]["collapse_gate"]["passed"] is False

    reference_by_sample = {
        row["sample_id"]: row["reference_label"]
        for row in read_jsonl(run_dir / "reference_labels.jsonl")
    }
    noncollapsed = [
        {
            "request_id": row["request_id"],
            "sample_id": row["sample_id"],
            "conversation_id": row["conversation_id"],
            "profile_mode": row["profile_mode"],
            "prediction": reference_by_sample[row["sample_id"]],
            "valid": True,
            "latency_ms": 5.0,
        }
        for row in requests
    ]
    write_jsonl(run_dir / "responses.jsonl", noncollapsed)
    report = score_event_eval(run_dir, bootstrap_resamples=20)
    assert report["diagnostics"]["collapse_gate"]["passed"] is True
    assert report["metrics"]["hidden"]["macro_f1"] == 1.0


def test_server_places_instruction_before_audio(tmp_path: Path, monkeypatch) -> None:
    audio_path = tmp_path / "audio.wav"
    _write_audio(audio_path, seconds=0.2)
    captured: dict = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"content":"{\\"label\\":\\"C\\"}"}}]}'

    def _urlopen(request, timeout):
        captured.update(json.loads(request.data.decode("utf-8")))
        return _Response()

    monkeypatch.setattr(mllm.urllib.request, "urlopen", _urlopen)
    result = mllm._server_chat_completion(
        "http://local.test/v1/chat/completions",
        {"audio_path": str(audio_path), "prompt": "INSTRUCTION"},
        model="qwen2.5-omni-3b-q4_k_m",
        timeout_s=5.0,
        seed=13,
    )
    content = captured["messages"][0]["content"]
    assert [part["type"] for part in content] == ["text", "input_audio"]
    assert content[0]["text"] == "INSTRUCTION"
    assert result == '{"label":"C"}'
