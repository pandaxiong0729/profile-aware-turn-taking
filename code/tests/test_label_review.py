from __future__ import annotations

import json
import wave
from pathlib import Path

from profile_turntaking.label_review import apply_reviewed_labels, build_review_page
from profile_turntaking.utils import read_jsonl, write_jsonl


def test_build_and_apply_review(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    write_jsonl(
        run / "requests.jsonl",
        [
            {
                "sample_id": "s1",
                "conversation_id": "c1",
                "prediction_time_s": 10.0,
                "profile_mode": "hidden",
                "audio_path": "audio_clips/s1.wav",
                "transcript_prefix": "hello",
            }
        ],
    )
    write_jsonl(run / "gold.jsonl", [{"sample_id": "s1", "target": "I"}])
    report = build_review_page(run)
    assert report["review_samples"] == 1
    assert "SBCSAE 500-event label review" in (run / "review.html").read_text(encoding="utf-8")

    source = tmp_path / "source.jsonl"
    write_jsonl(source, [{"sample_id": "s1", "label": "I", "gold_label": False}])
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps({"reviews": [{"sample_id": "s1", "human_label": "BC", "note": "heard feedback"}]}),
        encoding="utf-8",
    )
    output = tmp_path / "reviewed.jsonl"
    applied = apply_reviewed_labels(source, review, output, reviewer_id="r1")
    row = next(read_jsonl(output))
    assert applied["changed_labels"] == 1
    assert row["label"] == "BC"
    assert row["gold_label"] is True


def test_build_review_can_include_annotation_only_future_audio(tmp_path: Path) -> None:
    source_audio = tmp_path / "source.wav"
    with wave.open(str(source_audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\0\0" * 64_000)
    run = tmp_path / "run"
    run.mkdir()
    write_jsonl(
        run / "requests.jsonl",
        [
            {
                "sample_id": "s1",
                "conversation_id": "c1",
                "prediction_time_s": 2.0,
                "audio_duration_s": 2.0,
                "profile_mode": "hidden",
                "audio_path": "audio_clips/s1.wav",
                "transcript_prefix": "causal only",
            }
        ],
    )
    write_jsonl(run / "gold.jsonl", [{"sample_id": "s1", "target": "C"}])
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "sample_id": "s1",
                "conversation_id": "c1",
                "prediction_time_s": 2.0,
                "audio_path": str(source_audio),
                "label": "C",
                "weak_event_start_s": 2.0,
                "event_representative_policy": "onset",
            }
        ],
    )
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    write_jsonl(
        catalog / "conversations.jsonl",
        [
            {
                "conversation_id": "c1",
                "participants": [
                    {"local_speaker_id": "speaker_00", "is_person": True},
                    {"local_speaker_id": "speaker_01", "is_person": True},
                ],
            }
        ],
    )
    write_jsonl(
        catalog / "utterances.jsonl",
        [
            {
                "conversation_id": "c1",
                "start_s": 1.95,
                "end_s": 2.10,
                "speaker": "speaker_00",
                "is_person": True,
                "text": "(H)",
                "clean_text": "",
            },
            {
                "conversation_id": "c1",
                "start_s": 2.10,
                "end_s": 2.50,
                "speaker": "speaker_01",
                "is_person": True,
                "text": "hello",
                "clean_text": "hello",
            },
        ],
    )
    report = build_review_page(
        run, source_manifest=manifest, catalog_dir=catalog
    )
    item = json.loads((run / "review_items.json").read_text(encoding="utf-8"))[0]
    assert report["annotation_only_future_audio"] is True
    assert report["annotation_only_boundary_transcript"] is True
    assert report["risk_flags"] == {"target_has_nonlexical_human_unit": 1}
    assert item["annotation_only_future_audio"] is True
    assert "speaker_A" in item["annotation_only_boundary_transcript"]
    assert "after t" in item["annotation_only_boundary_transcript"]
    assert item["risk_flags"] == ["target_has_nonlexical_human_unit"]
    assert (run / item["audio_path"]).is_file()
