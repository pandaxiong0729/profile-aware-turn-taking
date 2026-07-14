from __future__ import annotations

import json
from pathlib import Path

from profile_turntaking.constants import LABELS
from profile_turntaking.prompt_baseline import (
    parse_label,
    prepare_prompt_run,
    profile_to_prompt,
    score_prompt_run,
)
from profile_turntaking.utils import read_jsonl, write_jsonl


def _profile(role_a: str, relationship: str) -> dict:
    return {
        "speaker_A": {
            "age_group": "25-34",
            "gender": "female",
            "social_role": role_a,
            "background": "synthetic background A",
        },
        "speaker_B": {
            "age_group": "35-44",
            "gender": "male",
            "social_role": "listener",
            "background": "synthetic background B",
        },
        "relationship": relationship,
        "situation": "casual_conversation",
    }


def _manifest_rows() -> list[dict]:
    rows = []
    for index, label in enumerate(LABELS):
        conversation_id = "conversation-A" if index < 3 else "conversation-B"
        rows.append(
            {
                "sample_id": f"sample-{label}",
                "conversation_id": conversation_id,
                "split": "test",
                "prediction_time_s": 10.0 + index,
                "transcript_prefix": f"[speaker_A] causal history for {index}",
                "profile": _profile(
                    "teacher" if conversation_id == "conversation-A" else "doctor",
                    "friends" if conversation_id == "conversation-A" else "professional_client",
                ),
                "label": label,
                "training_target": {"next_40ms_label": "LEAK_ME"},
                "annotation_only_not_model_input": {"reason": "LEAK_ME"},
            }
        )
    return rows


def test_fixed_profile_prompt_is_readable() -> None:
    prompt = profile_to_prompt(_profile("teacher", "friends"))
    assert "Speaker A" in prompt
    assert "teacher" in prompt
    assert "Relationship: friends" in prompt
    assert "professional_client" not in prompt


def test_prepare_separates_target_free_requests_from_gold(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, _manifest_rows())
    summary = prepare_prompt_run(manifest, tmp_path / "run", max_per_class=1)
    requests = list(read_jsonl(tmp_path / "run" / "requests.jsonl"))
    gold = list(read_jsonl(tmp_path / "run" / "gold.jsonl"))
    assert summary["selected_samples"] == 5
    assert summary["api_requests"] == 15
    assert len(requests) == len(gold) == 15
    assert all("target" not in request for request in requests)
    serialized_requests = json.dumps(requests, ensure_ascii=False)
    assert "LEAK_ME" not in serialized_requests
    assert "training_target" not in serialized_requests
    assert "annotation_only_not_model_input" not in serialized_requests
    sample_requests = {
        request["profile_mode"]: request
        for request in requests
        if request["sample_id"] == "sample-C"
    }
    assert "Profile information is unavailable" in json.dumps(sample_requests["hidden"])
    assert "teacher" in json.dumps(sample_requests["given"])
    assert "doctor" in json.dumps(sample_requests["shuffled"])


def test_parse_label_accepts_json_and_rejects_ambiguous_text() -> None:
    assert parse_label('{"label":"BC"}') == "BC"
    assert parse_label("NA") == "NA"
    assert parse_label("I predict C, but BC is possible") is None
    assert parse_label('{"label":"other"}') is None


def test_score_uses_only_samples_valid_in_all_three_modes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    gold = []
    responses = []
    fixtures = {
        "sample-1": {"target": "C", "hidden": "NA", "given": "C", "shuffled": "NA"},
        "sample-2": {"target": "BC", "hidden": "BC", "given": "BC", "shuffled": "C"},
    }
    for sample_id, fixture in fixtures.items():
        for mode in ("hidden", "given", "shuffled"):
            request_id = f"{sample_id}::{mode}"
            gold.append(
                {
                    "request_id": request_id,
                    "sample_id": sample_id,
                    "conversation_id": "conversation",
                    "profile_mode": mode,
                    "target": fixture["target"],
                }
            )
            responses.append(
                {
                    "request_id": request_id,
                    "sample_id": sample_id,
                    "conversation_id": "conversation",
                    "profile_mode": mode,
                    "prediction": fixture[mode],
                    "valid": True,
                }
            )
    write_jsonl(run_dir / "gold.jsonl", gold)
    write_jsonl(run_dir / "responses.jsonl", responses)
    report = score_prompt_run(run_dir)
    assert report["validity"]["paired_valid_samples"] == 2
    assert report["paired_changes"]["given_fixes_hidden_error"] == 1
    assert report["metrics"]["given"]["accuracy"] == 1.0
    assert report["metrics"]["hidden"]["accuracy"] == 0.5
    assert report["bootstrap_95ci"]["cluster_unit"] == "conversation_id"
    assert report["bootstrap_95ci"]["clusters"] == 1
    assert (run_dir / "profile_comparison.csv").is_file()
    assert (run_dir / "predictions.csv").is_file()
    assert (run_dir / "bootstrap_95ci.json").is_file()
