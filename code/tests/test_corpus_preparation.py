from __future__ import annotations

import random
import wave
from pathlib import Path

from profile_turntaking.data import label_at
from profile_turntaking.pachat_demo import parse_pachat_demo, prepare_pachat_demo
from profile_turntaking.quality import valid_profile
from profile_turntaking.sbcsae_corpus import (
    age_group,
    infer_context,
    metadata_aliases,
    normalize_speaker_name,
    resolve_context,
)
from profile_turntaking.sbcsae_manifest import (
    MonotonicWeakLabeler,
    _event_representative_time,
)
from profile_turntaking.schemas import Utterance


def _write_silence(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 1600)


def test_metadata_normalization_and_context_rules() -> None:
    assert normalize_speaker_name("Montoya") == "MONTOYO"
    assert "ANNETTE" in metadata_aliases("Anette")
    assert age_group("37") == "35-44"
    relationship, situation, confidence = infer_context(
        "A patient is consulting a dietician about diabetes management."
    )
    assert relationship == "professional_client"
    assert situation == "healthcare_consultation"
    assert confidence == "high"


def test_core_context_overrides_use_reviewed_chat_comments() -> None:
    relationship, situation, confidence, method = resolve_context(
        "SBC029", "business meeting for an estimate"
    )
    assert relationship == "professional_client"
    assert situation == "workplace_or_business"
    assert confidence == "manual_high"
    assert method == "manual_core_chat_comment_review_v1"
    assert resolve_context("SBC043", "mother and daughter")[:2] == (
        "family",
        "family_or_home_conversation",
    )


def test_profile_schema_requires_both_speakers_and_context() -> None:
    speaker = {
        "age_group": "35-44",
        "gender": "female",
        "social_role": "teacher",
        "background": "unknown",
    }
    profile = {
        "speaker_A": speaker,
        "speaker_B": {**speaker, "social_role": "student"},
        "relationship": "teacher_student",
        "situation": "classroom",
    }
    assert valid_profile(profile)
    assert not valid_profile({**profile, "extra": "leak"})


def test_monotonic_labeler_matches_reference_examples() -> None:
    rows = [
        Utterance(0.0, 1.0, "speaker_A", "a longer statement"),
        Utterance(1.0, 1.4, "speaker_B", "Mhm"),
        Utterance(1.2, 2.0, "speaker_A", "continuing after feedback"),
        Utterance(2.4, 3.2, "speaker_B", "a new turn"),
        Utterance(3.0, 3.8, "speaker_A", "overlapping disagreement"),
    ]
    times = [0.5, 1.0, 2.1, 2.4, 3.1]
    labeler = MonotonicWeakLabeler(rows, 40)
    assert [labeler.label(time_s) for time_s in times] == [
        label_at(rows, time_s) for time_s in times
    ]


def test_monotonic_labeler_does_not_turn_sequential_speech_into_overlap() -> None:
    rows = [
        Utterance(0.0, 1.01, "speaker_A", "ending"),
        Utterance(1.02, 2.0, "speaker_B", "starting"),
    ]
    labeler = MonotonicWeakLabeler(rows, 40)
    assert labeler.label(1.0) == label_at(rows, 1.0) == "T"


def test_event_representative_is_an_observed_grid_frame() -> None:
    event = {"start_s": 10.0, "end_s": 10.2}
    assert _event_representative_time(event, frame_stride_ms=40) == 10.08
    assert _event_representative_time(
        event, frame_stride_ms=40, policy="onset"
    ) == 10.0


def test_monotonic_labeler_matches_reference_on_random_sequences() -> None:
    rng = random.Random(7)
    rows = []
    for index in range(80):
        start = round(rng.uniform(0.0, 12.0), 3)
        duration = rng.uniform(0.08, 1.6)
        text = "Mhm" if index % 13 == 0 else f"words {index}"
        rows.append(
            Utterance(start, round(start + duration, 3), f"speaker_{'A' if index % 2 else 'B'}", text)
        )
    times = [round(index * 0.04, 3) for index in range(1, 300)]
    labeler = MonotonicWeakLabeler(rows, 40)
    assert [labeler.label(time_s) for time_s in times] == [
        label_at(rows, time_s) for time_s in times
    ]


def test_pachat_demo_parser_marks_isolated_audio_ineligible(tmp_path: Path) -> None:
    site = tmp_path / "site"
    site.mkdir()
    _write_silence(site / "res" / "a" / "1.wav")
    (site / "index.html").write_text(
        """
        <div class="case-card">
          <h2>Case 1</h2>
          <span><strong>Scenario：</strong> Family life</span>
          <span><strong>Theme：</strong> Shopping</span>
          <div style="background: #f5f5f5;">
            <div><span>George Hanson</span><span>(Father)</span>
              <span>George is a 44-year-old engineer.</span></div>
          </div>
          <div class="dialogue-case left">
            <div class="speaker-name">George</div>
            <source src="res/a/1.wav" type="audio/wav">
            <div class="text">Hello there.</div>
          </div>
        </div>
        """,
        encoding="utf-8",
    )
    cases, profiles, turns, issues = parse_pachat_demo(site)
    assert len(cases) == len(profiles) == len(turns) == 1
    assert profiles[0]["structured_profile"]["age_group"] == "35-44"
    assert turns[0]["profile_id"] == profiles[0]["profile_id"]
    assert turns[0]["audio_info"]["sample_rate_hz"] == 16000
    assert turns[0]["turntaking_label_eligible"] is False
    assert not issues

    summary = prepare_pachat_demo(site_dir=site, output_dir=tmp_path / "processed")
    assert summary["dataset_scope"] == "official_project_page_demo_only"
    assert summary["turntaking_label_eligible"] is False
    assert summary["issues"] == 3
