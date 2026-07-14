from __future__ import annotations

from pathlib import Path

import torch

from profile_turntaking.constants import PROFILE_FIELDS, UNKNOWN_PROFILE
from profile_turntaking.data import (
    assign_splits,
    canonicalize_speakers,
    clean_transcript_text,
    is_backchannel,
    label_at,
    parse_trn,
    transcript_prefix,
)
from profile_turntaking.dataset import ManifestDataset
from profile_turntaking.features import profile_bucket_ids
from profile_turntaking.schemas import Sample, Utterance
from profile_turntaking.training import TrainConfig, _apply_profile_dropout
from profile_turntaking.utils import write_jsonl


def test_parse_trn_carries_forward_speaker() -> None:
    rows = parse_trn(Path(__file__).parents[1] / "examples" / "smoke.trn")
    assert rows[0].speaker == "KRISTIN"
    assert len(rows) == 18


def test_canonicalize_speaker_mapping() -> None:
    rows = [Utterance(0.0, 1.0, "KRISTIN", "hello"), Utterance(1.0, 2.0, "PAIGE", "hi")]
    mapped = canonicalize_speakers(rows, {"KRISTIN": "speaker_A", "PAIGE": "speaker_B"})
    assert [row.speaker for row in mapped] == ["speaker_A", "speaker_B"]


def test_canonicalize_explicit_mapping_skips_environment() -> None:
    rows = [
        Utterance(0.0, 0.5, ">ENV", "drawer"),
        Utterance(0.5, 1.0, "KRISTIN", "hello"),
        Utterance(1.0, 1.5, "PAIGE", "hi"),
    ]
    mapped = canonicalize_speakers(rows, {"KRISTIN": "speaker_A", "PAIGE": "speaker_B"})
    assert [row.speaker for row in mapped] == ["speaker_A", "speaker_B"]


def test_parse_part1_space_separated_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "part1.trn"
    path.write_text(
        "0.00 1.25\tALICE:\tHello.\n1.25 2.00\t\tStill Alice.\n",
        encoding="utf-8",
    )
    rows = parse_trn(path)
    assert [(row.start_s, row.end_s, row.speaker) for row in rows] == [
        (0.0, 1.25, "ALICE"),
        (1.25, 2.0, "ALICE"),
    ]


def test_parse_part1_inline_speaker(tmp_path: Path) -> None:
    path = tmp_path / "inline-speaker.trn"
    path.write_text(
        "0.00 1.25 ALICE:\tHello.\n1.25 2.00\tStill Alice.\n",
        encoding="utf-8",
    )
    rows = parse_trn(path)
    assert [row.speaker for row in rows] == ["ALICE", "ALICE"]
    assert [row.text for row in rows] == ["Hello.", "Still Alice."]


def test_parse_non_strict_records_bad_interval(tmp_path: Path) -> None:
    path = tmp_path / "bad.trn"
    path.write_text(
        "0.0\t1.0\tALICE:\tHello.\n1.0\t1.0\tALICE:\tBad.\n",
        encoding="utf-8",
    )
    diagnostics: list[dict[str, object]] = []
    rows = parse_trn(path, strict=False, diagnostics=diagnostics)
    assert len(rows) == 1
    assert diagnostics[0]["reason"] == "non_positive_interval"


def test_parse_recovers_embedded_trn_row(tmp_path: Path) -> None:
    path = tmp_path / "embedded.trn"
    path.write_text(
        "1.0 2.0\tALICE:\tFirst.\\000000000 000000000 BOB: 2.0 3.0\tSecond.\n",
        encoding="utf-8",
    )
    diagnostics: list[dict[str, object]] = []
    rows = parse_trn(path, diagnostics=diagnostics)
    assert [(row.start_s, row.end_s, row.speaker, row.text) for row in rows] == [
        (1.0, 2.0, "ALICE", "First."),
        (2.0, 3.0, "BOB", "Second."),
    ]
    assert diagnostics[0]["reason"] == "recovered_embedded_row"


def test_overlap_brackets_preserve_backchannel_words() -> None:
    row = Utterance(1.0, 1.3, "speaker_B", "[Mhm]")
    assert clean_transcript_text(row.text) == "mhm"
    assert is_backchannel(row)


def test_parse_shifted_timestamp_and_blank_speaker_marker(tmp_path: Path) -> None:
    path = tmp_path / "shifted.trn"
    path.write_text(
        "0.0\t1.0\tALICE:\tHello.\n"
        "\t1.0 2.0\t\tShifted but recoverable.\n"
        "2.0\t3.0 :\tBlank speaker marker.\n",
        encoding="utf-8",
    )
    rows = parse_trn(path)
    assert len(rows) == 3
    assert [row.speaker for row in rows] == ["ALICE", "ALICE", "ALICE"]


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


def test_sequential_turns_inside_one_chunk_are_not_interruption() -> None:
    rows = [
        Utterance(0.0, 1.01, "speaker_A", "ending"),
        Utterance(1.02, 2.0, "speaker_B", "starting"),
    ]
    assert label_at(rows, 1.0) == "T"


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


def test_shuffled_profiles_are_stable_and_cross_conversation(tmp_path: Path) -> None:
    profile_a = {"name": "profile-a"}
    profile_b = {"name": "profile-b"}
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(
        manifest,
        [
            {"conversation_id": "A", "split": "test", "profile": profile_a},
            {"conversation_id": "A", "split": "test", "profile": profile_a},
            {"conversation_id": "B", "split": "test", "profile": profile_b},
        ],
    )
    dataset = ManifestDataset(str(manifest), split="test", profile_mode="shuffled")
    assert dataset._profile_for_index(0) == profile_b
    assert dataset._profile_for_index(1) == profile_b
    assert dataset._profile_for_index(2) == profile_a


def test_profile_dropout_uses_same_unknown_encoding_as_hidden_evaluation() -> None:
    profile_ids = torch.full((3, len(PROFILE_FIELDS)), 7, dtype=torch.long)
    unknown = torch.from_numpy(profile_bucket_ids(UNKNOWN_PROFILE, buckets=512))
    dropped = _apply_profile_dropout(
        profile_ids,
        probability=1.0,
        unknown_profile_ids=unknown,
    )
    assert torch.equal(dropped, unknown.expand_as(profile_ids))


def test_train_config_from_dict_keeps_profile_dropout_and_seed() -> None:
    config = TrainConfig.from_dict({"profile_dropout": 0.25, "seed": 7, "ignored": 1})
    assert config.profile_dropout == 0.25
    assert config.seed == 7
