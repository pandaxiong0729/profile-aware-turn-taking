from __future__ import annotations

from pathlib import Path

from profile_turntaking.data import (
    assign_splits,
    canonicalize_speakers,
    label_at,
    parse_trn,
    transcript_prefix,
)
from profile_turntaking.schemas import Sample, Utterance


def test_parse_trn_carries_forward_speaker() -> None:
    rows = parse_trn(Path("examples/smoke.trn"))
    assert rows[0].speaker == "KRISTIN"
    assert len(rows) == 18


def test_canonicalize_speaker_mapping() -> None:
    rows = [Utterance(0.0, 1.0, "KRISTIN", "hello"), Utterance(1.0, 2.0, "PAIGE", "hi")]
    mapped = canonicalize_speakers(rows, {"KRISTIN": "speaker_A", "PAIGE": "speaker_B"})
    assert [row.speaker for row in mapped] == ["speaker_A", "speaker_B"]


def test_five_class_label_rules() -> None:
    rows = [
        Utterance(0.0, 1.0, "speaker_A", "a longer statement"),
        Utterance(1.0, 1.4, "speaker_B", "Mhm"),
        Utterance(1.2, 2.0, "speaker_A", "continuing after feedback"),
        Utterance(2.4, 3.2, "speaker_B", "a new turn"),
        Utterance(3.0, 3.8, "speaker_A", "overlapping disagreement"),
    ]
    assert label_at(rows, 0.5) == "C"
    assert label_at(rows, 1.0) == "BC"
    assert label_at(rows, 2.1) == "NA"
    assert label_at(rows, 2.4) == "T"
    assert label_at(rows, 3.1) == "I"


def test_transcript_prefix_excludes_partial_future_text() -> None:
    rows = [
        Utterance(0.0, 1.0, "speaker_A", "visible"),
        Utterance(1.0, 2.0, "speaker_B", "must stay hidden until complete"),
    ]
    prefix = transcript_prefix(rows, 0.0, 1.5)
    assert "visible" in prefix
    assert "must stay hidden" not in prefix


def test_smoke_split_keeps_each_supported_label_in_test() -> None:
    samples = [
        Sample(
            sample_id=f"T-{index}",
            conversation_id="one",
            split_group="one",
            split="unassigned",
            prediction_time_s=float(index),
            horizon_ms=40,
            window_start_s=0.0,
            window_end_s=float(index),
            audio_path="unused.wav",
            transcript_prefix="",
            profile={},
            label="T",
        )
        for index in range(6)
    ]
    assigned = assign_splits(samples)
    assert {sample.split for sample in assigned} == {"train", "val", "test"}


def test_three_groups_always_create_three_splits() -> None:
    samples = [
        Sample(
            sample_id=f"sample-{index}",
            conversation_id=f"conv-{index}",
            split_group=f"group-{index}",
            split="unassigned",
            prediction_time_s=1.0,
            horizon_ms=40,
            window_start_s=0.0,
            window_end_s=1.0,
            audio_path="unused.wav",
            transcript_prefix="",
            profile={},
            label="C",
        )
        for index in range(3)
    ]
    assigned = assign_splits(samples)
    assert {sample.split for sample in assigned} == {"train", "val", "test"}
