from __future__ import annotations

from profile_turntaking.schemas import Utterance
from profile_turntaking.vad_fiveclass import force_frame_labels


def _vad_rows(frames: int, *, silence: set[int] | None = None) -> list[dict]:
    silence = silence or set()
    return [
        {
            "conversation_id": "demo",
            "frame_index": index,
            "start_s": round(index * 0.04, 6),
            "end_s": round((index + 1) * 0.04, 6),
            "vad_state": "silence" if index in silence else "speech",
            "vad_votes": 0 if index in silence else 3,
            "vad_source_count": 3,
            "trn_active_speakers": [],
        }
        for index in range(frames)
    ]


def test_forced_rules_cover_all_five_labels() -> None:
    utterances = [
        Utterance(0.00, 1.00, "speaker_A", "a longer statement"),
        Utterance(0.40, 0.72, "speaker_B", "Mhm"),
        Utterance(1.20, 1.80, "speaker_B", "a clean new turn"),
        Utterance(1.68, 2.20, "speaker_A", "overlapping interruption"),
    ]
    rows = force_frame_labels(_vad_rows(60, silence={55}), utterances)
    labels = [row["label"] for row in rows]
    assert set(labels) == {"C", "BC", "T", "I", "NA"}
    assert labels[10] == "BC"
    assert labels[30] == "T"
    assert labels[42] == "I"
    assert labels[55] == "NA"
    assert all(row["label"] for row in rows)


def test_overlap_shorter_than_one_frame_is_turn_change_not_interruption() -> None:
    utterances = [
        Utterance(0.00, 1.01, "speaker_A", "ending"),
        Utterance(1.00, 1.50, "speaker_B", "new turn"),
    ]
    rows = force_frame_labels(_vad_rows(40), utterances)
    assert rows[25]["label"] == "T"
    assert "I" not in {row["label"] for row in rows}


def test_vad_silence_overrides_transcript_event() -> None:
    utterances = [
        Utterance(0.00, 0.80, "speaker_A", "statement"),
        Utterance(0.40, 0.70, "speaker_B", "yeah"),
    ]
    rows = force_frame_labels(_vad_rows(20, silence={10}), utterances)
    assert rows[10]["label"] == "NA"
