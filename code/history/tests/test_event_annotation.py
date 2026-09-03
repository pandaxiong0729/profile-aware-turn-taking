from __future__ import annotations

import json

from profile_turntaking.event_annotation import (
    build_event_candidates,
    build_ipus,
    build_static_review_site,
)


def _utterance(uid: str, start: float, end: float, speaker: str, text: str) -> dict:
    return {
        "utterance_id": uid,
        "start_s": start,
        "end_s": end,
        "speaker": speaker,
        "clean_text": text,
        "is_person": True,
    }


def test_ipus_use_vad_to_remove_long_transcript_silence() -> None:
    utterances = [_utterance("u1", 0.0, 3.0, "speaker_A", "hello")]
    ipus, diagnostics = build_ipus(
        utterances,
        [(0.2, 0.8), (2.0, 2.5)],
        ipu_silence_s=0.2,
    )
    assert [(row["start_s"], row["end_s"]) for row in ipus] == [(0.2, 0.8), (2.0, 2.5)]
    assert diagnostics["transcript_rows_without_vad"] == 0


def test_event_candidates_cover_pause_shift_overlap_bc_and_silence() -> None:
    ipus = [
        {"ipu_id": "a1", "speaker": "speaker_A", "start_s": 0.0, "end_s": 2.0, "text": "long speech"},
        # Isolated listener feedback while A holds the floor.
        {"ipu_id": "b_bc", "speaker": "speaker_B", "start_s": 1.0, "end_s": 1.3, "text": "mhm"},
        {"ipu_id": "a2", "speaker": "speaker_A", "start_s": 2.4, "end_s": 3.5, "text": "continues"},
        # B overlaps A and continues after A, producing I then T.
        {"ipu_id": "b_i", "speaker": "speaker_B", "start_s": 3.2, "end_s": 4.5, "text": "takes the floor"},
        # Clean change back to A.
        {"ipu_id": "a3", "speaker": "speaker_A", "start_s": 4.9, "end_s": 5.6, "text": "new turn"},
    ]
    vad = [(0.0, 2.0), (2.4, 4.5), (4.9, 5.6)]
    events, _ = build_event_candidates(ipus, vad, duration_s=6.0)
    labels = [row["candidate_label"] for row in events]
    structures = {row["structure"] for row in events}
    assert set(labels) == {"C", "BC", "T", "I", "NA"}
    assert "backchannel_candidate" in structures
    assert "hold_after_pause" in structures
    assert "interruption_candidate" in structures
    assert "shift_after_successful_interruption" in structures
    assert "natural_turn_shift" in structures


def test_short_nonlexical_floor_taking_overlap_is_not_forced_to_bc() -> None:
    ipus = [
        {"ipu_id": "a1", "speaker": "speaker_A", "start_s": 0.0, "end_s": 2.0, "text": "statement"},
        {"ipu_id": "b1", "speaker": "speaker_B", "start_s": 1.5, "end_s": 3.0, "text": "but listen"},
    ]
    events, _ = build_event_candidates(ipus, [(0.0, 3.0)], duration_s=3.0)
    at_onset = [row for row in events if abs(row["anchor_s"] - 1.5) < 1e-9]
    assert any(row["candidate_label"] == "I" for row in at_onset)
    assert not any(row["candidate_label"] == "BC" for row in at_onset)


def test_static_review_site_is_relative_and_shows_machine_event_for_audit(tmp_path) -> None:
    row = {
        "event_id": "SBC005_E000001",
        "conversation_id": "SBC005",
        "candidate_label": "BC",
        "structure": "backchannel_candidate",
        "anchor_s": 16.0,
        "audio_path": "audio_clips/SBC005_E000001.wav",
        "clip_start_in_conversation_s": 10.0,
        "clip_end_in_conversation_s": 20.0,
        "target_start_in_clip_s": 6.0,
        "target_end_in_clip_s": 6.3,
        "context_transcript": [],
    }
    (tmp_path / "annotation_manifest.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )
    result = build_static_review_site(tmp_path)
    page = (tmp_path / "review.html").read_text(encoding="utf-8")
    data = (tmp_path / "review_data" / "SBC005.js").read_text(encoding="utf-8")

    assert result["portable"] is True
    assert "review_data/index.js" in page
    assert "http://" not in page and "https://" not in page
    assert "C:\\" not in page
    assert 'id="importFile"' in page
    assert "applyImportedReviews" in page
    assert "LAST_REVIEWER_KEY" in page
    assert "已保存到结果目录" in page
    assert "serverSaveQueue" in page
    assert "reviewerLoadTimer" in page
    assert "保存本会话结果文件" not in page
    assert '"candidate_label":"BC"' in data
    assert "audio_clips/SBC005_E000001.wav" in data


def test_turn_change_is_a_point_and_does_not_cover_later_interruption() -> None:
    ipus = [
        {"ipu_id": "b1", "speaker": "speaker_B", "start_s": 0.0, "end_s": 1.0, "text": "first"},
        {"ipu_id": "a1", "speaker": "speaker_A", "start_s": 1.0, "end_s": 2.5, "text": "new turn"},
        {"ipu_id": "b2", "speaker": "speaker_B", "start_s": 1.5, "end_s": 1.8, "text": "but"},
    ]
    events, _ = build_event_candidates(ipus, [(0.0, 2.5)], duration_s=2.5)
    turn = next(row for row in events if row["structure"] == "natural_turn_shift")
    interruption = next(row for row in events if row["structure"] == "interruption_candidate")

    assert turn["event_start_s"] == turn["event_end_s"] == 1.0
    assert turn["event_end_s"] < interruption["event_start_s"]


def test_same_onset_short_ipus_do_not_remove_content_floor() -> None:
    ipus = [
        {"ipu_id": "a1", "speaker": "speaker_A", "start_s": 0.0, "end_s": 1.0, "text": "setup"},
        {"ipu_id": "a2", "speaker": "speaker_A", "start_s": 2.0, "end_s": 3.0, "text": "has to happen"},
        {"ipu_id": "b_bc", "speaker": "speaker_B", "start_s": 2.0, "end_s": 3.0, "text": ""},
        {"ipu_id": "b2", "speaker": "speaker_B", "start_s": 3.4, "end_s": 4.0, "text": "whoa"},
    ]
    events, _ = build_event_candidates(ipus, [(0.0, 1.0), (2.0, 4.0)], duration_s=4.0)
    event = next(row for row in events if abs(row["anchor_s"] - 3.4) < 1e-9)
    assert event["candidate_label"] == "T"
    assert event["evidence"]["previous_ipu_id"] == "a2"
    assert event["evidence"]["silence_or_latch_ms"] == 400.0


def test_sub_200ms_empty_sound_is_not_a_floor_turn() -> None:
    ipus = [
        {"ipu_id": "a1", "speaker": "speaker_A", "start_s": 0.0, "end_s": 2.0, "text": "because"},
        {"ipu_id": "b_noise", "speaker": "speaker_B", "start_s": 2.0, "end_s": 2.083, "text": ""},
        {"ipu_id": "a2", "speaker": "speaker_A", "start_s": 3.0, "end_s": 4.0, "text": "continuing"},
    ]
    events, _ = build_event_candidates(ipus, [(0.0, 2.083), (3.0, 4.0)], duration_s=4.0)
    event = next(row for row in events if abs(row["anchor_s"] - 3.0) < 1e-9)
    assert event["candidate_label"] == "C"
    assert event["evidence"]["previous_ipu_id"] == "a1"
