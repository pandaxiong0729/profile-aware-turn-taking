from __future__ import annotations

import numpy as np

from profile_turntaking.vad_annotation import VadConfig, derive_frame_annotations


def test_vad_assigns_only_safe_c_and_na_and_queues_short_semantics() -> None:
    config = VadConfig(
        frame_ms=40,
        vad_boundary_margin_ms=0,
        transcript_boundary_margin_ms=0,
        review_join_gap_ms=0,
    )
    # Three-source agreement: silence, long A speech, short B response, silence.
    votes = np.array([0] * 5 + [3] * 50 + [3] * 5 + [0] * 5, dtype=np.uint8)
    utterances = [
        {
            "start_s": 0.20,
            "end_s": 2.20,
            "speaker": "speaker_A",
            "is_person": True,
        },
        {
            "start_s": 2.20,
            "end_s": 2.40,
            "speaker": "speaker_B",
            "is_person": True,
        },
    ]
    frames, _, events = derive_frame_annotations(
        conversation_id="TEST",
        duration_s=2.6,
        source_names=["channel_0", "channel_1", "channel_mean"],
        vad_votes=votes,
        utterances=utterances,
        config=config,
    )
    assert frames[2]["automatic_label"] == "NA"
    assert any(row["automatic_label"] == "C" for row in frames[8:20])
    assert all(
        row["automatic_label"] not in {"BC", "T", "I"} for row in frames
    )
    assert events
    assert any("short_utterance_semantics" in event["review_reasons"] for event in events)
