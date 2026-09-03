from __future__ import annotations

import json
from pathlib import Path

import pytest

from profile_turntaking.protocol_lock import (
    prompt_sentinel_sha256,
    verify_prompt_protocol,
)


@pytest.mark.parametrize(
    "config_name",
    ["mllm_prompt_pilot50_locked.json", "mllm_prompt_pilot_locked.json"],
)
def test_locked_prompt_fingerprint_matches_implementation(config_name: str) -> None:
    protocol_path = Path(__file__).parents[1] / "configs" / config_name
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert prompt_sentinel_sha256() == protocol["prompt"]["sentinel_sha256"]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_protocol_verifier_separates_consistency_from_review_readiness(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    protocol_path = root / "code" / "configs" / "protocol.json"
    candidate_manifest = root / "data" / "pilot50.jsonl"
    run_dir = root / "artifacts" / "pilot50"
    labels = ["C", "BC", "T", "I", "NA"]
    class_counts = {label: 10 for label in labels}
    protocol = {
        "protocol_id": "test-pilot50",
        "data": {
            "candidate_manifest": "data/pilot50.jsonl",
            "candidate_run_dir": "artifacts/pilot50",
            "selection": {
                "policy": "conversation_balanced_class_targets_v1",
                "seed": 13,
                "samples": 50,
                "class_targets": class_counts,
                "max_per_conversation_class": 1,
                "min_boundary_separation_s": 5.0,
                "minimum_conversations_per_class": 8,
                "maximum_single_conversation_class_share": 0.25,
            },
        },
        "prompt": {"sentinel_sha256": prompt_sentinel_sha256()},
        "target": {"horizon_ms": 40},
        "run_gates": {
            "human_reviewed_samples": 50,
            "weak_label_samples": 0,
            "first_batch_samples": 50,
            "full_batch_samples": 500,
        },
    }
    _write_json(protocol_path, protocol)
    _write_json(
        candidate_manifest.with_suffix(".selection.json"),
        {
            "selection_policy": "conversation_balanced_class_targets_v1",
            "seed": 13,
            "samples": 50,
            "class_counts": class_counts,
            "max_per_conversation_class": 1,
            "observed_min_boundary_separation_s": 9.24,
            "class_conversation_coverage": {label: 10 for label in labels},
            "class_max_conversation_share": {label: 0.1 for label in labels},
        },
    )
    _write_json(
        run_dir / "run_config.json",
        {
            "selected_samples": 50,
            "profile_modes": ["hidden", "given", "shuffled"],
            "horizon_ms": 40,
            "label_quality": {
                "human_gold_samples": 50,
                "weak_label_samples": 0,
                "formal_claim_allowed": True,
            },
        },
    )
    _write_json(run_dir / "input_audit.json", {"passed": True})
    _write_json(
        run_dir / "preflight_audit.json",
        {
            "expected_samples": 50,
            "implementation_integrity_passed": True,
            "selection_distribution_passed": True,
            "prompt_pilot_ready": True,
        },
    )

    ready = verify_prompt_protocol(protocol_path)
    assert ready["protocol_consistent"] is True
    assert ready["stage"] == "pilot_50"
    assert ready["ready_for_inference"] is True
    assert ready["ready_for_50_sample_pilot"] is True
    assert ready["ready_for_full_500"] is False

    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    config["label_quality"] = {
        "human_gold_samples": 0,
        "weak_label_samples": 50,
        "formal_claim_allowed": False,
    }
    _write_json(run_dir / "run_config.json", config)
    preflight = json.loads(
        (run_dir / "preflight_audit.json").read_text(encoding="utf-8")
    )
    preflight["prompt_pilot_ready"] = False
    _write_json(run_dir / "preflight_audit.json", preflight)

    blocked = verify_prompt_protocol(protocol_path)
    assert blocked["protocol_consistent"] is True
    assert blocked["ready_for_inference"] is False
    assert blocked["ready_for_50_sample_pilot"] is False
    assert any("human-reviewed labels: 0/50" in item for item in blocked["readiness_blockers"])
