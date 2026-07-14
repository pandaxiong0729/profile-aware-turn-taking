from __future__ import annotations

import json
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
