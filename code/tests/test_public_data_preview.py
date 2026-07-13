from __future__ import annotations

import csv
import json
import wave
from pathlib import Path


PREVIEW = Path(__file__).parents[1] / "examples" / "data_preview"


def _json(name: str):
    return json.loads((PREVIEW / name).read_text(encoding="utf-8"))


def test_public_sbcsae_preview_is_synthetic_and_schema_faithful() -> None:
    manifest = _json("sbcsae_manifest_sample.json")
    profile = _json("profile_sample.json")
    assert manifest["contains_original_sbcsae_audio"] is False
    assert manifest["contains_original_sbcsae_transcript"] is False
    assert manifest["profile"] == profile["profile"]
    assert manifest["label"] in {"C", "BC", "T", "I", "NA"}
    assert set(manifest["profile"]) == {
        "speaker_A",
        "speaker_B",
        "relationship",
        "situation",
    }


def test_public_preview_audio_is_small_mono_pcm() -> None:
    with wave.open(str(PREVIEW / "synthetic_input.wav"), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 16_000
        assert wav.getsampwidth() == 2
        assert 3.9 <= wav.getnframes() / wav.getframerate() <= 4.1


def test_public_pachat_preview_has_no_audio_or_turntaking_claim() -> None:
    demo = _json("pachat_demo_sample.json")
    assert demo["audio_included"] is False
    assert demo["turn"]["turntaking_label_eligible"] is False
    assert demo["turntaking_output"] is None
    assert demo["license_status"] == "not_specified_in_official_demo_repository"


def test_public_smoke_outputs_are_complete_and_marked_non_research() -> None:
    history = _json("smoke_training_history.json")
    assert history["result_status"] == "functional_smoke_not_research_result"
    assert history["checkpoint"] == "not_distributed_smoke_checkpoint"
    assert len(history["history"]) == 6
    with (PREVIEW / "smoke_predictions.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        predictions = list(csv.DictReader(handle))
    assert len(predictions) == 17
    assert set(predictions[0]) == {
        "sample_id",
        "target",
        "hidden_prediction",
        "given_prediction",
        "shuffled_prediction",
    }
    with (PREVIEW / "profile_comparison.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        comparison = list(csv.DictReader(handle))
    assert [row["profile_mode"] for row in comparison] == [
        "hidden",
        "given",
        "shuffled",
        "given_minus_hidden",
    ]
