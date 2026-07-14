"""Deep preflight audit for the zero-shot 500-event MLLM prompt pilot."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .constants import LABELS
from .data import _previous_floor, clean_transcript_text, is_backchannel
from .mllm_prompt_baseline import audit_mllm_prompt_run
from .sbcsae_manifest import MonotonicWeakLabeler, _canonical_conversation
from .utils import read_jsonl, write_json


def audit_prompt_pilot_data(
    *,
    catalog_dir: str | Path,
    event_manifest: str | Path,
    run_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    catalog = Path(catalog_dir)
    run = Path(run_dir)
    conversations = {
        row["conversation_id"]: row for row in read_jsonl(catalog / "conversations.jsonl")
    }
    raw_by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(catalog / "utterances.jsonl"):
        raw_by_conversation[row["conversation_id"]].append(row)
    prepared = {
        conversation_id: _canonical_conversation(
            conversation, raw_by_conversation[conversation_id]
        )
        for conversation_id, conversation in conversations.items()
        if conversation.get("observed_dyadic")
    }
    events = list(read_jsonl(event_manifest))
    event_by_id = {str(row["sample_id"]): row for row in events}
    gold_rows = list(read_jsonl(run / "gold.jsonl"))
    request_rows = list(read_jsonl(run / "requests.jsonl"))
    selected_ids = sorted({str(row["sample_id"]) for row in gold_rows})
    selected = [event_by_id[sample_id] for sample_id in selected_ids if sample_id in event_by_id]

    errors: list[str] = []
    warnings: list[str] = []
    if len(event_by_id) != len(events):
        errors.append("event manifest contains duplicate sample_id values")
    if len({str(row.get("weak_event_id", "")) for row in events}) != len(events):
        errors.append("event manifest contains duplicate weak_event_id values")
    missing_selected = sorted(set(selected_ids) - set(event_by_id))
    if missing_selected:
        errors.append(f"{len(missing_selected)} selected samples are absent from event manifest")

    label_mismatches = 0
    profile_mismatches = 0
    audio_mismatches = 0
    representative_errors = 0
    rows_by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        rows_by_conversation[str(row["conversation_id"])].append(row)
    for conversation_id, rows in rows_by_conversation.items():
        utterances, expected_profile, expected_provenance = prepared[conversation_id]
        labeler = MonotonicWeakLabeler(utterances, 40)
        source_audio = Path(str(conversations[conversation_id]["audio_path"])).resolve()
        for row in sorted(rows, key=lambda item: float(item["prediction_time_s"])):
            if labeler.label(float(row["prediction_time_s"])) != row["label"]:
                label_mismatches += 1
            if (
                row.get("profile") != expected_profile
                or row.get("profile_provenance") != expected_provenance
            ):
                profile_mismatches += 1
            if Path(str(row["audio_path"])).resolve() != source_audio:
                audio_mismatches += 1
            prediction_time = float(row["prediction_time_s"])
            if not (
                float(row["weak_event_start_s"])
                <= prediction_time
                < float(row["weak_event_end_s"]) + 1e-8
            ):
                representative_errors += 1
    for name, count in (
        ("recomputed label", label_mismatches),
        ("speaker/profile mapping", profile_mismatches),
        ("source audio mapping", audio_mismatches),
        ("event representative boundary", representative_errors),
    ):
        if count:
            errors.append(f"{count} {name} mismatches")

    group_splits: dict[str, set[str]] = defaultdict(set)
    conversation_splits: dict[str, set[str]] = defaultdict(set)
    for row in events:
        group_splits[str(row["split_group"])].add(str(row["split"]))
        conversation_splits[str(row["conversation_id"])].add(str(row["split"]))
    split_group_leaks = sum(len(splits) > 1 for splits in group_splits.values())
    conversation_leaks = sum(len(splits) > 1 for splits in conversation_splits.values())
    if split_group_leaks or conversation_leaks:
        errors.append(
            f"split leakage: groups={split_group_leaks}, conversations={conversation_leaks}"
        )

    gold_targets: dict[str, set[str]] = defaultdict(set)
    for row in gold_rows:
        gold_targets[str(row["sample_id"])].add(str(row["target"]))
    target_manifest_mismatches = sum(
        len(gold_targets[sample_id]) != 1
        or next(iter(gold_targets[sample_id])) != event_by_id[sample_id]["label"]
        for sample_id in selected_ids
        if sample_id in event_by_id
    )
    if target_manifest_mismatches:
        errors.append(f"{target_manifest_mismatches} request targets differ from manifest labels")

    request_audit = audit_mllm_prompt_run(
        run,
        expected_samples=len(selected_ids),
        expected_per_class=None,
    )
    requests_by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in request_rows:
        requests_by_sample[str(row["sample_id"])][str(row["profile_mode"])] = row
    given_equals_shuffled = sum(
        modes.get("given", {}).get("profile_text")
        == modes.get("shuffled", {}).get("profile_text")
        for modes in requests_by_sample.values()
    )
    if given_equals_shuffled:
        errors.append(f"{given_equals_shuffled} samples have identical given/shuffled profile text")

    ongoing_units: Counter[str] = Counter()
    nonlexical_active: Counter[str] = Counter()
    nonperson_active: Counter[str] = Counter()
    already_active: Counter[str] = Counter()
    event_durations: dict[str, list[float]] = defaultdict(list)
    for row in selected:
        conversation_id = str(row["conversation_id"])
        prediction_time = float(row["prediction_time_s"])
        horizon_end = prediction_time + 0.04
        utterances = prepared[conversation_id][0]
        active = [
            utterance
            for utterance in utterances
            if utterance.start_s < horizon_end and utterance.end_s > prediction_time
        ]
        if any(
            utterance.start_s < prediction_time < utterance.end_s
            for utterance in utterances
        ):
            ongoing_units[str(row["label"])] += 1
        if any(not clean_transcript_text(utterance.text) for utterance in active):
            nonlexical_active[str(row["label"])] += 1
        if any(
            not utterance["is_person"]
            and float(utterance["start_s"]) < horizon_end
            and float(utterance["end_s"]) > prediction_time
            for utterance in raw_by_conversation[conversation_id]
        ):
            nonperson_active[str(row["label"])] += 1
        if row["label"] == "BC":
            backchannels = [
                utterance
                for utterance in active
                if is_backchannel(utterance)
                and _previous_floor(utterances, utterance.start_s)
                not in (None, utterance.speaker)
            ]
            if any(utterance.start_s < prediction_time for utterance in backchannels):
                already_active["BC"] += 1
        elif row["label"] == "I":
            for index, first in enumerate(active):
                if any(
                    first.speaker != second.speaker
                    and max(first.start_s, second.start_s) < prediction_time
                    and max(prediction_time, first.start_s, second.start_s)
                    < min(horizon_end, first.end_s, second.end_s)
                    for second in active[index + 1 :]
                ):
                    already_active["I"] += 1
                    break
        event_durations[str(row["label"])].append(
            float(row["weak_event_end_s"]) - float(row["weak_event_start_s"])
        )

    reviewed = sum(bool(row.get("gold_label", False)) for row in selected)
    weak = len(selected) - reviewed
    if weak:
        warnings.append(f"{weak} selected targets are unreviewed weak labels")
    transcript_sources = Counter(str(row.get("text_source", "unknown")) for row in selected)
    if any("not_streaming_asr" in source for source in transcript_sources):
        warnings.append("text input is completed-unit TRN, not a streaming-ASR prefix")
    if nonlexical_active:
        warnings.append(
            f"{sum(nonlexical_active.values())} selected windows contain an active non-lexical person unit"
        )
    if nonperson_active:
        warnings.append(
            f"{sum(nonperson_active.values())} selected windows contain an active non-person/environment unit"
        )

    selected_conversations = sorted({str(row["conversation_id"]) for row in selected})
    context_methods = Counter(
        str(conversations[conversation_id].get("context_mapping_method", "unknown"))
        for conversation_id in selected_conversations
    )
    profile_provenance = {
        speaker: dict(Counter(str(row["profile_provenance"][speaker]) for row in selected))
        for speaker in ("speaker_A", "speaker_B")
    }
    class_counts = Counter(str(row["label"]) for row in selected)
    class_conversation_counts = {
        label: Counter(
            str(row["conversation_id"])
            for row in selected
            if str(row["label"]) == label
        )
        for label in LABELS
    }
    class_conversation_coverage = {
        label: len(class_conversation_counts[label]) for label in LABELS
    }
    class_max_conversation_share = {
        label: (
            max(class_conversation_counts[label].values()) / class_counts[label]
            if class_counts[label]
            else 1.0
        )
        for label in LABELS
    }
    selected_times: dict[str, list[float]] = defaultdict(list)
    for row in selected:
        selected_times[str(row["conversation_id"])].append(
            float(row["prediction_time_s"])
        )
    observed_min_boundary_separation_s = min(
        (
            right - left
            for values in selected_times.values()
            for left, right in zip(sorted(values), sorted(values)[1:])
        ),
        default=None,
    )
    distribution_failures = []
    for label in LABELS:
        if class_conversation_coverage[label] < 8:
            distribution_failures.append(
                f"{label} covers only {class_conversation_coverage[label]} conversations"
            )
        if class_max_conversation_share[label] > 0.25:
            distribution_failures.append(
                f"{label} max conversation share is "
                f"{class_max_conversation_share[label]:.1%}"
            )
    if (
        observed_min_boundary_separation_s is not None
        and observed_min_boundary_separation_s < 5.0 - 1e-8
    ):
        distribution_failures.append(
            "selected boundaries are less than 5 seconds apart within a conversation"
        )
    selection_distribution_passed = not distribution_failures
    if distribution_failures:
        warnings.append(
            "selection/profile confounding gate failed: "
            + "; ".join(distribution_failures)
        )
    implementation_passed = not errors and bool(request_audit.get("passed"))
    prompt_pilot_ready = (
        implementation_passed
        and selection_distribution_passed
        and weak == 0
        and len(selected) == 500
    )
    report = {
        "scope": "zero_shot_audio_transcript_profile_prompt_pilot",
        "implementation_integrity_passed": implementation_passed,
        "selection_distribution_passed": selection_distribution_passed,
        "prompt_pilot_ready": prompt_pilot_ready,
        "formal_adapter_experiment_ready": False,
        "formal_adapter_blockers": [
            "frame-accurate speaker-aware VAD/diarization labels are not present",
            "streaming-ASR prefixes are not present",
            "this run is zero-shot prompting and does not train the profile adapter",
        ],
        "errors": errors,
        "warnings": warnings,
        "catalog": {
            "conversations": len(conversations),
            "event_manifest_rows": len(events),
            "unique_event_ids": len({str(row.get("weak_event_id", "")) for row in events}),
            "label_recompute_mismatches": label_mismatches,
            "profile_mapping_mismatches": profile_mismatches,
            "audio_mapping_mismatches": audio_mismatches,
            "split_group_leaks": split_group_leaks,
            "conversation_leaks": conversation_leaks,
        },
        "selected_set": {
            "samples": len(selected),
            "conversations": len(selected_conversations),
            "conversation_ids": selected_conversations,
            "proposed_class_counts": {label: class_counts[label] for label in LABELS},
            "reviewed_labels": reviewed,
            "weak_labels": weak,
            "ongoing_person_unit_at_t": dict(ongoing_units),
            "active_nonlexical_person_unit": dict(nonlexical_active),
            "active_nonperson_or_environment_unit": dict(nonperson_active),
            "BC_or_I_already_active_at_t": dict(already_active),
            "event_duration_median_s": {
                label: statistics.median(event_durations[label])
                for label in LABELS
                if event_durations[label]
            },
            "class_conversation_coverage": class_conversation_coverage,
            "class_max_conversation_share": class_max_conversation_share,
            "observed_min_boundary_separation_s": observed_min_boundary_separation_s,
            "selection_distribution_failures": distribution_failures,
        },
        "profile": {
            "context_mapping_methods_by_conversation": dict(context_methods),
            "speaker_metadata_provenance_by_sample": profile_provenance,
            "given_equals_shuffled_samples": given_equals_shuffled,
        },
        "transcript_sources": dict(transcript_sources),
        "request_audit": request_audit,
    }
    write_json(output_path, report)
    return report


__all__ = ["audit_prompt_pilot_data"]
