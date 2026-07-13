"""Cross-artifact quality audit for the prepared SBCSAE and PAChat data."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .constants import LABELS
from .utils import read_jsonl, write_json

_PROFILE_KEYS = {"age_group", "gender", "social_role", "background"}
_PREFIX_TIME_RE = re.compile(r"\[[^\]]+\s+(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\]")


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    severity: str = "error",
    evidence: Any = None,
) -> None:
    checks.append(
        {
            "name": name,
            "status": "pass" if passed else "fail",
            "severity": severity,
            "evidence": evidence,
        }
    )


def valid_profile(profile: dict[str, Any]) -> bool:
    return (
        isinstance(profile, dict)
        and set(profile) == {"speaker_A", "speaker_B", "relationship", "situation"}
        and isinstance(profile["speaker_A"], dict)
        and isinstance(profile["speaker_B"], dict)
        and set(profile["speaker_A"]) == _PROFILE_KEYS
        and set(profile["speaker_B"]) == _PROFILE_KEYS
    )


def audit_preprocessed_data(
    *,
    sbcsae_catalog_dir: str | Path,
    sbcsae_manifest: str | Path,
    pachat_demo_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    catalog = Path(sbcsae_catalog_dir)
    pachat = Path(pachat_demo_dir)
    checks: list[dict[str, Any]] = []
    conversations = list(read_jsonl(catalog / "conversations.jsonl"))
    utterances = list(read_jsonl(catalog / "utterances.jsonl"))
    sbc_issues = list(read_jsonl(catalog / "issues.jsonl"))
    conversation_by_id = {row["conversation_id"]: row for row in conversations}

    _check(
        checks,
        "sbcsae_60_unique_conversations",
        len(conversations) == len(conversation_by_id) == 60,
        evidence={"rows": len(conversations), "unique_ids": len(conversation_by_id)},
    )
    _check(
        checks,
        "sbcsae_all_audio_present_and_valid",
        all(row.get("audio_path") and row.get("audio_info") for row in conversations),
        evidence={
            "audio_paths": sum(bool(row.get("audio_path")) for row in conversations),
            "valid_headers": sum(bool(row.get("audio_info")) for row in conversations),
        },
    )
    _check(
        checks,
        "sbcsae_all_audio_paths_exist",
        all(Path(row["audio_path"]).is_file() for row in conversations if row.get("audio_path")),
        evidence={
            "missing": [
                row["conversation_id"]
                for row in conversations
                if row.get("audio_path") and not Path(row["audio_path"]).is_file()
            ]
        },
    )
    utterance_counts = Counter(row["conversation_id"] for row in utterances)
    mismatched_counts = {
        row["conversation_id"]: {
            "declared": row["utterance_count"],
            "catalog": utterance_counts[row["conversation_id"]],
        }
        for row in conversations
        if row["utterance_count"] != utterance_counts[row["conversation_id"]]
    }
    _check(
        checks,
        "sbcsae_utterance_counts_match",
        not mismatched_counts,
        evidence=mismatched_counts,
    )
    participant_profiles = [
        participant
        for row in conversations
        for participant in row["participants"]
        if participant["is_person"]
    ]
    unresolved_profiles = [
        {
            "conversation_id": row["conversation_id"],
            "speaker": participant["display_name"],
            "status": participant["metadata_match_status"],
        }
        for row in conversations
        for participant in row["participants"]
        if participant["is_person"]
        and participant["metadata_match_status"] in {"not_found", "ambiguous_name"}
    ]
    _check(
        checks,
        "sbcsae_participant_profiles_have_fixed_fields",
        all(set(row["profile"]) == _PROFILE_KEYS for row in participant_profiles),
        evidence={"human_participant_rows": len(participant_profiles)},
    )
    _check(
        checks,
        "sbcsae_unresolved_profiles_documented",
        all(
            any(
                issue.get("scope") == "profile"
                and issue.get("conversation_id") == row["conversation_id"]
                and issue.get("speaker") == row["speaker"]
                for issue in sbc_issues
            )
            for row in unresolved_profiles
        ),
        severity="warning",
        evidence={"unresolved": unresolved_profiles},
    )

    sample_ids: set[str] = set()
    duplicate_sample_ids: list[str] = []
    manifest_counts: Counter[tuple[str, str]] = Counter()
    labels_invalid: Counter[str] = Counter()
    invalid_profiles = 0
    missing_sample_audio: set[str] = set()
    future_text_rows = 0
    split_by_group: dict[str, set[str]] = defaultdict(set)
    split_by_conversation: dict[str, set[str]] = defaultdict(set)
    manifest_conversations: set[str] = set()
    manifest_rows = 0
    for row in read_jsonl(sbcsae_manifest):
        manifest_rows += 1
        sample_id = row["sample_id"]
        if sample_id in sample_ids:
            duplicate_sample_ids.append(sample_id)
        sample_ids.add(sample_id)
        split = row["split"]
        label = row["label"]
        manifest_counts[(split, label)] += 1
        if label not in LABELS:
            labels_invalid[label] += 1
        if not valid_profile(row["profile"]):
            invalid_profiles += 1
        if not Path(row["audio_path"]).is_file():
            missing_sample_audio.add(row["audio_path"])
        prediction_time = float(row["window_end_s"])
        if any(float(end_s) > prediction_time + 1e-6 for _, end_s in _PREFIX_TIME_RE.findall(row["transcript_prefix"])):
            future_text_rows += 1
        split_by_group[row["split_group"]].add(split)
        split_by_conversation[row["conversation_id"]].add(split)
        manifest_conversations.add(row["conversation_id"])

    _check(
        checks,
        "manifest_nonempty_unique_sample_ids",
        manifest_rows > 0 and not duplicate_sample_ids,
        evidence={"rows": manifest_rows, "duplicates": duplicate_sample_ids[:20]},
    )
    _check(
        checks,
        "manifest_labels_are_five_class",
        not labels_invalid,
        evidence=dict(labels_invalid),
    )
    _check(
        checks,
        "manifest_profile_schema_valid",
        invalid_profiles == 0,
        evidence={"invalid_rows": invalid_profiles},
    )
    _check(
        checks,
        "manifest_audio_paths_exist",
        not missing_sample_audio,
        evidence={"missing": sorted(missing_sample_audio)},
    )
    _check(
        checks,
        "manifest_transcript_prefix_is_causal",
        future_text_rows == 0,
        evidence={"rows_with_future_end_time": future_text_rows},
    )
    leaking_groups = {group: sorted(splits) for group, splits in split_by_group.items() if len(splits) > 1}
    _check(
        checks,
        "manifest_split_groups_do_not_cross_splits",
        not leaking_groups,
        evidence=leaking_groups,
    )
    leaking_conversations = {
        conversation_id: sorted(splits)
        for conversation_id, splits in split_by_conversation.items()
        if len(splits) > 1
    }
    _check(
        checks,
        "manifest_conversations_do_not_cross_splits",
        not leaking_conversations,
        evidence=leaking_conversations,
    )
    uid_splits: dict[str, set[str]] = defaultdict(set)
    for conversation_id in manifest_conversations:
        split = next(iter(split_by_conversation[conversation_id]))
        for participant in conversation_by_id[conversation_id]["participants"]:
            if participant["is_person"]:
                uid_splits[participant["speaker_uid"]].add(split)
    leaking_speakers = {
        uid: sorted(splits) for uid, splits in uid_splits.items() if len(splits) > 1
    }
    _check(
        checks,
        "manifest_speaker_uids_do_not_cross_splits",
        not leaking_speakers,
        evidence=leaking_speakers,
    )
    missing_label_cells = [
        f"{split}/{label}"
        for split in ("train", "val", "test")
        for label in LABELS
        if manifest_counts[(split, label)] == 0
    ]
    _check(
        checks,
        "manifest_all_labels_present_in_every_split",
        not missing_label_cells,
        evidence={"missing": missing_label_cells},
    )

    pachat_cases = list(read_jsonl(pachat / "cases.jsonl"))
    pachat_profiles = list(read_jsonl(pachat / "profiles.jsonl"))
    pachat_turns = list(read_jsonl(pachat / "turns.jsonl"))
    pachat_issues = list(read_jsonl(pachat / "issues.jsonl"))
    _check(
        checks,
        "pachat_demo_inventory_matches_official_page",
        (len(pachat_cases), len(pachat_profiles), len(pachat_turns)) == (4, 14, 29),
        evidence={
            "cases": len(pachat_cases),
            "profiles": len(pachat_profiles),
            "turns": len(pachat_turns),
        },
    )
    _check(
        checks,
        "pachat_all_demo_audio_valid",
        all(row.get("audio_info") and Path(row["audio_path"]).is_file() for row in pachat_turns),
        evidence={"valid": sum(bool(row.get("audio_info")) for row in pachat_turns)},
    )
    _check(
        checks,
        "pachat_not_misrepresented_as_turntaking_gold",
        all(not row["turntaking_label_eligible"] for row in pachat_turns),
        evidence={"ineligible_turns": sum(not row["turntaking_label_eligible"] for row in pachat_turns)},
    )
    required_pachat_issues = {
        "full_persona_dialogue_release_not_found",
        "license_not_specified_in_official_demo_repository",
        "isolated_turn_audio_has_no_continuous_turn_timing",
    }
    actual_pachat_issues = {row["reason"] for row in pachat_issues}
    _check(
        checks,
        "pachat_release_limitations_documented",
        required_pachat_issues <= actual_pachat_issues,
        severity="warning",
        evidence={"reasons": sorted(actual_pachat_issues)},
    )

    failures = [row for row in checks if row["status"] == "fail" and row["severity"] == "error"]
    warnings = [row for row in checks if row["status"] == "fail" and row["severity"] == "warning"]
    report = {
        "schema_version": "1.0",
        "status": "pass" if not failures else "fail",
        "checks_total": len(checks),
        "checks_passed": sum(row["status"] == "pass" for row in checks),
        "error_failures": len(failures),
        "warning_failures": len(warnings),
        "checks": checks,
        "statistics": {
            "sbcsae_conversations": len(conversations),
            "sbcsae_utterances": len(utterances),
            "sbcsae_human_participant_rows": len(participant_profiles),
            "sbcsae_unresolved_profile_rows": len(unresolved_profiles),
            "manifest_rows": manifest_rows,
            "manifest_conversations": len(manifest_conversations),
            "manifest_counts": {
                split: {label: manifest_counts[(split, label)] for label in LABELS}
                for split in ("train", "val", "test")
            },
            "pachat_demo_cases": len(pachat_cases),
            "pachat_demo_profiles": len(pachat_profiles),
            "pachat_demo_turns": len(pachat_turns),
        },
    }
    write_json(output_path, report)
    return report
