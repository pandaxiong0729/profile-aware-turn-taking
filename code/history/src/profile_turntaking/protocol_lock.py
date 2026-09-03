"""Machine-readable lock and verifier for the reviewed MLLM prompt pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .constants import LABELS
from .mllm_prompt_baseline import build_audio_prompt
from .utils import write_json


_SENTINEL_PROFILE = {
    "speaker_A": {
        "age_group": "25-34",
        "gender": "female",
        "social_role": "teacher",
        "background": "sentinel A",
    },
    "speaker_B": {
        "age_group": "35-44",
        "gender": "male",
        "social_role": "student",
        "background": "sentinel B",
    },
    "relationship": "teacher_student",
    "situation": "classroom",
}
_SENTINEL_ROW = {
    "prediction_time_s": 30.0,
    "window_start_s": 0.0,
    "window_end_s": 30.0,
    "transcript_prefix": "[speaker_A 0.00-1.00] protocol sentinel",
}


def prompt_sentinel_sha256() -> str:
    prompt = build_audio_prompt(
        _SENTINEL_ROW, _SENTINEL_PROFILE, max_transcript_chars=6000
    )
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _load_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"missing required file: {path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def verify_prompt_protocol(
    protocol_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify protocol drift separately from the reviewed-label readiness gate."""

    protocol_file = Path(protocol_path).resolve()
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    repository_root = protocol_file.parents[2]
    data = protocol["data"]
    selection_spec = data["selection"]
    candidate_manifest = repository_root / data["candidate_manifest"]
    selection_path = candidate_manifest.with_suffix(".selection.json")
    resolved_run = (
        Path(run_dir).resolve()
        if run_dir is not None
        else repository_root / data["candidate_run_dir"]
    )
    errors: list[str] = []
    blockers: list[str] = []

    observed_prompt_hash = prompt_sentinel_sha256()
    expected_prompt_hash = str(protocol["prompt"]["sentinel_sha256"])
    if observed_prompt_hash != expected_prompt_hash:
        errors.append(
            "prompt implementation drifted from the locked sentinel fingerprint"
        )

    selection = _load_json(selection_path, errors)
    if selection:
        exact_checks = {
            "selection_policy": selection_spec["policy"],
            "seed": selection_spec["seed"],
            "samples": selection_spec["samples"],
            "class_counts": selection_spec["class_targets"],
            "max_per_conversation_class": selection_spec[
                "max_per_conversation_class"
            ],
        }
        for key, expected in exact_checks.items():
            if selection.get(key) != expected:
                errors.append(
                    f"selection {key} differs: {selection.get(key)!r} != {expected!r}"
                )
        observed_separation = selection.get("observed_min_boundary_separation_s")
        if (
            observed_separation is None
            or float(observed_separation)
            < float(selection_spec["min_boundary_separation_s"])
        ):
            errors.append("selection boundary-separation contract failed")
        for label in LABELS:
            coverage = int(selection.get("class_conversation_coverage", {}).get(label, 0))
            share = float(
                selection.get("class_max_conversation_share", {}).get(label, 1.0)
            )
            if coverage < int(selection_spec["minimum_conversations_per_class"]):
                errors.append(f"{label} conversation coverage is below the lock")
            if share > float(
                selection_spec["maximum_single_conversation_class_share"]
            ):
                errors.append(f"{label} single-conversation share exceeds the lock")

    run_config = _load_json(resolved_run / "run_config.json", errors)
    input_audit = _load_json(resolved_run / "input_audit.json", errors)
    preflight = _load_json(resolved_run / "preflight_audit.json", errors)
    if run_config:
        if int(run_config.get("selected_samples", -1)) != int(
            selection_spec["samples"]
        ):
            errors.append("run selected-sample count differs from the protocol")
        if run_config.get("profile_modes") != ["hidden", "given", "shuffled"]:
            errors.append("run profile modes differ from the protocol")
        if int(run_config.get("horizon_ms", -1)) != int(protocol["target"]["horizon_ms"]):
            errors.append("run horizon differs from the protocol")
        quality = run_config.get("label_quality", {})
        expected_reviewed = int(protocol["run_gates"]["human_reviewed_samples"])
        if int(quality.get("human_gold_samples", 0)) != expected_reviewed:
            blockers.append(
                f"human-reviewed labels: {quality.get('human_gold_samples', 0)}/"
                f"{expected_reviewed}"
            )
        if int(quality.get("weak_label_samples", -1)) != int(
            protocol["run_gates"]["weak_label_samples"]
        ):
            blockers.append(
                f"weak-label rows remain: {quality.get('weak_label_samples', 'unknown')}"
            )
        if not quality.get("formal_claim_allowed", False):
            blockers.append("run_config formal_claim_allowed is false")
    if input_audit and not input_audit.get("passed", False):
        errors.append("paired request input audit failed")
    if preflight:
        if int(preflight.get("expected_samples", -1)) != int(
            selection_spec["samples"]
        ):
            errors.append("preflight expected-sample count differs from the protocol")
        if not preflight.get("implementation_integrity_passed", False):
            errors.append("implementation-integrity preflight failed")
        if not preflight.get("selection_distribution_passed", False):
            errors.append("selection-distribution preflight failed")
        if not preflight.get("prompt_pilot_ready", False):
            blockers.append("preflight prompt_pilot_ready is false")

    ready = not errors and not blockers
    selected_samples = int(selection_spec["samples"])
    first_batch_samples = int(protocol["run_gates"].get("first_batch_samples", -1))
    full_batch_samples = int(protocol["run_gates"].get("full_batch_samples", -1))
    stage = (
        "pilot_50"
        if selected_samples == first_batch_samples
        else "full_500"
        if selected_samples == full_batch_samples
        else f"custom_{selected_samples}"
    )
    report = {
        "protocol_id": protocol["protocol_id"],
        "stage": stage,
        "selected_samples": selected_samples,
        "protocol_consistent": not errors,
        "ready_for_inference": ready,
        "ready_for_50_sample_pilot": ready and stage == "pilot_50",
        "ready_for_full_500": ready and stage == "full_500",
        "errors": errors,
        "readiness_blockers": blockers,
        "protocol_path": str(protocol_file),
        "run_dir": str(resolved_run),
        "prompt_sentinel_sha256": observed_prompt_hash,
    }
    if output_path is not None:
        write_json(output_path, report)
    return report


__all__ = ["prompt_sentinel_sha256", "verify_prompt_protocol"]
