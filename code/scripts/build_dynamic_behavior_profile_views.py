from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("train", "val", "test")
SPEAKERS = ("speaker_00", "speaker_01")
UNKNOWN = "<UNK>"
HISTORY_END_S = 25.0  # The final 5 s stay exclusively in the base context branch.
FILLERS = {"uh", "um", "erm", "mm", "mhm", "uh huh", "yeah", "okay", "right"}


def parse_profile(profile_text: str) -> tuple[str, str]:
    relationship = UNKNOWN
    situation = UNKNOWN
    for raw_line in profile_text.splitlines():
        line = raw_line.strip()
        if line.startswith("Their relationship is "):
            relationship = line.removeprefix("Their relationship is ").rstrip(".").strip().lower()
        elif line.startswith("The conversation situation is "):
            situation = (
                line.removeprefix("The conversation situation is ").rstrip(".").strip().lower()
            )
    return relationship or UNKNOWN, situation or UNKNOWN


def vocabulary(values: set[str]) -> dict[str, int]:
    return {value: index for index, value in enumerate([UNKNOWN, *sorted(values - {UNKNOWN})])}


def clip_units_until(
    units: list[dict[str, Any]], history_end_s: float
) -> list[dict[str, Any]]:
    clipped: list[dict[str, Any]] = []
    for unit in units:
        original_end = float(unit["end_s"])
        if original_end > history_end_s:
            # Do not use the text or eventual duration of an utterance that
            # was still in progress at the dynamic-profile cutoff.
            continue
        start = max(0.0, float(unit["start_s"]))
        end = original_end
        speaker = str(unit["speaker"])
        if speaker in SPEAKERS and end > start:
            clipped.append(
                {"speaker": speaker, "start_s": start, "end_s": end, "text": str(unit.get("text", ""))}
            )
    return sorted(clipped, key=lambda item: (item["start_s"], item["end_s"], item["speaker"]))


def clip_units(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return clip_units_until(units, HISTORY_END_S)


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def interval_total(intervals: list[tuple[float, float]]) -> float:
    return float(sum(end - start for start, end in intervals))


def intersection_total(
    first: list[tuple[float, float]], second: list[tuple[float, float]]
) -> float:
    i = j = 0
    total = 0.0
    while i < len(first) and j < len(second):
        start = max(first[i][0], second[j][0])
        end = min(first[i][1], second[j][1])
        total += max(0.0, end - start)
        if first[i][1] <= second[j][1]:
            i += 1
        else:
            j += 1
    return float(total)


def speaker_features(
    units: list[dict[str, Any]], speaker: str, history_end_s: float = HISTORY_END_S
) -> tuple[list[float], list[str]]:
    own = [unit for unit in units if unit["speaker"] == speaker]
    other = [unit for unit in units if unit["speaker"] != speaker]
    durations = np.asarray([unit["end_s"] - unit["start_s"] for unit in own], dtype=np.float64)
    intervals = merge_intervals([(unit["start_s"], unit["end_s"]) for unit in own])
    speech_seconds = interval_total(intervals)
    words = sum(len(re.findall(r"[A-Za-z']+", unit["text"])) for unit in own)
    normalized_texts = [re.sub(r"\s+", " ", unit["text"].strip().lower()) for unit in own]

    response_gaps: list[float] = []
    overlap_responses = 0
    overlapping_units = 0
    overlap_attempts = 0
    successful_overlap_attempts = 0
    incumbent_overlap_cases = 0
    incumbent_yields = 0
    attempt_overlap_seconds: list[float] = []
    for unit in own:
        prior_other_ends = [
            candidate["end_s"]
            for candidate in other
            if candidate["start_s"] <= unit["start_s"]
        ]
        if prior_other_ends:
            gap = float(unit["start_s"] - max(prior_other_ends))
            response_gaps.append(float(np.clip(gap, -2.0, 2.0)))
            overlap_responses += int(gap < 0.0)
        if any(
            min(unit["end_s"], candidate["end_s"])
            > max(unit["start_s"], candidate["start_s"])
            for candidate in other
        ):
            overlapping_units += 1
        active_others = [
            candidate
            for candidate in other
            if candidate["start_s"] < unit["start_s"] < candidate["end_s"]
        ]
        if active_others:
            incumbent = max(active_others, key=lambda candidate: candidate["end_s"])
            overlap_attempts += 1
            successful_overlap_attempts += int(unit["end_s"] > incumbent["end_s"])
            attempt_overlap_seconds.append(
                max(0.0, min(unit["end_s"], incumbent["end_s"]) - unit["start_s"])
            )
        later_others = [
            candidate
            for candidate in other
            if unit["start_s"] < candidate["start_s"] < unit["end_s"]
        ]
        if later_others:
            newcomer = min(later_others, key=lambda candidate: candidate["start_s"])
            incumbent_overlap_cases += 1
            incumbent_yields += int(newcomer["end_s"] > unit["end_s"])

    count = len(own)
    features = [
        speech_seconds / history_end_s,
        count / history_end_s,
        float(durations.mean()) if count else 0.0,
        float(np.median(durations)) if count else 0.0,
        float(durations.std()) if count else 0.0,
        float(np.mean(durations <= 1.0)) if count else 0.0,
        float(np.mean(durations <= 0.5)) if count else 0.0,
        words / max(speech_seconds, 1e-6),
        float(np.mean([text in FILLERS for text in normalized_texts])) if count else 0.0,
        overlapping_units / max(count, 1),
        float(np.mean(response_gaps)) if response_gaps else 0.0,
        overlap_responses / max(len(response_gaps), 1),
        overlap_attempts / max(count, 1),
        successful_overlap_attempts / max(overlap_attempts, 1),
        incumbent_yields / max(incumbent_overlap_cases, 1),
        float(np.mean(attempt_overlap_seconds)) if attempt_overlap_seconds else 0.0,
    ]
    names = [
        "speech_fraction",
        "units_per_second",
        "mean_unit_duration",
        "median_unit_duration",
        "std_unit_duration",
        "short_unit_ratio_le_1s",
        "very_short_unit_ratio_le_0.5s",
        "words_per_speaking_second",
        "filler_unit_ratio",
        "overlapping_unit_ratio",
        "mean_response_gap_clipped",
        "overlap_response_ratio",
        "overlap_attempt_ratio",
        "overlap_attempt_success_ratio",
        "incumbent_overlap_yield_ratio",
        "mean_attempt_overlap_seconds",
    ]
    return features, [f"{speaker}.{name}" for name in names]


def behavior_features_until(
    units: list[dict[str, Any]], history_end_s: float
) -> tuple[np.ndarray, list[str]]:
    if history_end_s <= 0.0:
        raise ValueError("history_end_s must be positive")
    units = clip_units_until(units, history_end_s)
    values: list[float] = []
    names: list[str] = []
    merged_by_speaker: dict[str, list[tuple[float, float]]] = {}
    for speaker in SPEAKERS:
        speaker_values, speaker_names = speaker_features(units, speaker, history_end_s)
        values.extend(speaker_values)
        names.extend(speaker_names)
        merged_by_speaker[speaker] = merge_intervals(
            [(unit["start_s"], unit["end_s"]) for unit in units if unit["speaker"] == speaker]
        )

    union = merge_intervals(
        [(unit["start_s"], unit["end_s"]) for unit in units]
    )
    overlap = intersection_total(merged_by_speaker[SPEAKERS[0]], merged_by_speaker[SPEAKERS[1]])
    speaker_seconds = [interval_total(merged_by_speaker[speaker]) for speaker in SPEAKERS]
    switches = sum(
        first["speaker"] != second["speaker"] for first, second in zip(units, units[1:])
    )
    cross_speaker_gaps = [
        float(np.clip(second["start_s"] - first["end_s"], -2.0, 2.0))
        for first, second in zip(units, units[1:])
        if first["speaker"] != second["speaker"]
    ]
    values.extend(
        [
            interval_total(union) / history_end_s,
            max(0.0, history_end_s - interval_total(union)) / history_end_s,
            overlap / history_end_s,
            abs(speaker_seconds[0] - speaker_seconds[1]) / max(sum(speaker_seconds), 1e-6),
            switches / history_end_s,
            float(np.mean(cross_speaker_gaps)) if cross_speaker_gaps else 0.0,
        ]
    )
    names.extend(
        [
            "global.speech_union_fraction",
            "global.silence_fraction",
            "global.overlap_fraction",
            "global.speaker_time_imbalance",
            "global.speaker_switches_per_second",
            "global.mean_cross_speaker_gap_clipped",
        ]
    )
    return np.asarray(values, dtype=np.float32), names


def behavior_features(units: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    return behavior_features_until(units, HISTORY_END_S)


def latest_completed_speaker(units: list[dict[str, Any]]) -> str | None:
    """Infer the current floor role using causal completed transcript only."""

    eligible = [
        unit
        for unit in units
        if str(unit.get("speaker", "")) in SPEAKERS
        and float(unit.get("end_s", 0.0)) <= 30.0 + 1e-6
    ]
    if not eligible:
        return None
    latest = max(
        eligible,
        key=lambda unit: (
            float(unit.get("end_s", 0.0)),
            float(unit.get("start_s", 0.0)),
            str(unit.get("speaker", "")),
        ),
    )
    return str(latest["speaker"])


def role_normalize_behavior(
    values: np.ndarray,
    names: list[str],
    units: list[dict[str, Any]],
) -> tuple[np.ndarray, list[str], str | None]:
    """Put the causal current speaker first and the other participant second."""

    if (len(values) - 6) % 2 != 0 or len(names) != len(values):
        raise ValueError("Unexpected behavior feature layout")
    per_speaker = (len(values) - 6) // 2
    current = latest_completed_speaker(units)
    if current is None:
        current = SPEAKERS[0]
        inferred: str | None = None
    else:
        inferred = current
    other = SPEAKERS[1] if current == SPEAKERS[0] else SPEAKERS[0]
    speaker_slice = {
        SPEAKERS[0]: slice(0, per_speaker),
        SPEAKERS[1]: slice(per_speaker, per_speaker * 2),
    }
    ordered = np.concatenate(
        [values[speaker_slice[current]], values[speaker_slice[other]], values[per_speaker * 2 :]]
    ).astype(np.float32)
    base_names = [name.split(".", 1)[1] for name in names[:per_speaker]]
    ordered_names = [
        *[f"current_speaker.{name}" for name in base_names],
        *[f"other_participant.{name}" for name in base_names],
        *names[per_speaker * 2 :],
    ]
    return ordered, ordered_names, inferred


def read_given_requests(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["profile_mode"] == "given":
                result[str(record["sample_id"])] = record
    return result


def read_catalog_utterances(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            conversation_id = str(row.get("conversation_id", ""))
            speaker = str(row.get("speaker", ""))
            if conversation_id and speaker in SPEAKERS and bool(row.get("is_person", True)):
                grouped.setdefault(conversation_id, []).append(
                    {
                        "speaker": speaker,
                        "start_s": float(row["start_s"]),
                        "end_s": float(row["end_s"]),
                        "text": str(row.get("clean_text") or row.get("text") or ""),
                    }
                )
    for rows in grouped.values():
        rows.sort(key=lambda item: (item["start_s"], item["end_s"], item["speaker"]))
    return grouped


def category_vector(
    profile_text: str,
    relationship_vocab: dict[str, int],
    situation_vocab: dict[str, int],
    pair_vocab: dict[str, int],
) -> np.ndarray:
    relationship, situation = parse_profile(profile_text)
    pair = f"{relationship} || {situation}"
    vector = np.zeros(
        len(relationship_vocab) + len(situation_vocab) + len(pair_vocab), dtype=np.float32
    )
    vector[relationship_vocab.get(relationship, relationship_vocab[UNKNOWN])] = 1.0
    vector[
        len(relationship_vocab) + situation_vocab.get(situation, situation_vocab[UNKNOWN])
    ] = 1.0
    vector[
        len(relationship_vocab)
        + len(situation_vocab)
        + pair_vocab.get(pair, pair_vocab[UNKNOWN])
    ] = 1.0
    return vector


def deterministic_shuffled_indices(sample_ids: np.ndarray, conversation_ids: np.ndarray) -> np.ndarray:
    result = np.empty(len(sample_ids), dtype=np.int64)
    for index, (sample_id, conversation_id) in enumerate(zip(sample_ids, conversation_ids, strict=True)):
        candidates = np.flatnonzero(conversation_ids != conversation_id)
        if not len(candidates):
            raise ValueError("Cannot build a different-conversation shuffled dynamic profile")
        digest = int(hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()[:16], 16)
        result[index] = int(candidates[digest % len(candidates)])
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build causal dynamic behavior profile sidecars.")
    parser.add_argument(
        "--data-dir", default="data/processed/sbcsae_qwen_shared_ab_30s_causal_v1"
    )
    parser.add_argument(
        "--cache-dir", default="artifacts/main_experiment/qwen_feature_cache"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/main_experiment/profile_features_rebuilt"
    )
    parser.add_argument(
        "--speaker-order",
        choices=["absolute", "causal-role"],
        default="absolute",
        help="causal-role orders per-speaker behavior as current speaker then other participant.",
    )
    parser.add_argument(
        "--history-scope",
        choices=["window25", "conversation-to-t-minus5"],
        default="window25",
        help="Use either the first 25 s of each window or all conversation history ending 5 s before t.",
    )
    parser.add_argument(
        "--catalog-utterances",
        default="data/processed/sbcsae_catalog_v2/utterances.jsonl",
    )
    parser.add_argument(
        "--history-lookback-seconds",
        type=float,
        default=0.0,
        help="For conversation history, keep this many seconds before t-5; 0 keeps all history.",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = (repo_root / args.data_dir).resolve()
    cache_dir = (repo_root / args.cache_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    catalog_by_conversation = (
        read_catalog_utterances((repo_root / args.catalog_utterances).resolve())
        if args.history_scope == "conversation-to-t-minus5"
        else {}
    )

    requests = {
        split: read_given_requests(data_dir / split / "requests.jsonl") for split in SPLITS
    }
    train_profiles = [str(record["profile_text"]) for record in requests["train"].values()]
    parsed_train = [parse_profile(text) for text in train_profiles]
    relationship_vocab = vocabulary({relationship for relationship, _ in parsed_train})
    situation_vocab = vocabulary({situation for _, situation in parsed_train})
    pair_vocab = vocabulary(
        {f"{relationship} || {situation}" for relationship, situation in parsed_train}
    )

    raw_behavior: dict[str, np.ndarray] = {}
    categories: dict[str, np.ndarray] = {}
    sample_ids_by_split: dict[str, np.ndarray] = {}
    conversation_ids_by_split: dict[str, np.ndarray] = {}
    feature_names: list[str] | None = None
    role_inference_counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        with np.load(cache_dir / f"{split}.qwen-hidden.npz", allow_pickle=False) as base:
            sample_ids = base["sample_ids"].astype(str)
            conversation_ids = base["conversation_ids"].astype(str)
        missing = [sample_id for sample_id in sample_ids if sample_id not in requests[split]]
        if missing:
            raise ValueError(f"{split}: {len(missing)} cache rows lack a causal request")
        rows: list[np.ndarray] = []
        category_rows: list[np.ndarray] = []
        inferred_roles = {"speaker_00": 0, "speaker_01": 0, "unknown": 0}
        for sample_id in sample_ids:
            record = requests[split][sample_id]
            transcript_units = list(record["transcript_units"])
            if args.history_scope == "conversation-to-t-minus5":
                conversation_id = str(record["conversation_id"])
                cutoff_s = float(record["decision_time_in_conversation_s"]) - 5.0
                if cutoff_s <= 0.0:
                    raise ValueError(f"{sample_id}: non-positive causal history boundary")
                lookback_s = float(args.history_lookback_seconds)
                history_start_s = max(0.0, cutoff_s - lookback_s) if lookback_s > 0.0 else 0.0
                history_end_s = cutoff_s - history_start_s
                history_units = [
                    {
                        **unit,
                        "start_s": float(unit["start_s"]) - history_start_s,
                        "end_s": float(unit["end_s"]) - history_start_s,
                    }
                    for unit in catalog_by_conversation.get(conversation_id, [])
                    if float(unit["end_s"]) > history_start_s
                    and float(unit["end_s"]) <= cutoff_s + 1e-6
                ]
                values, names = behavior_features_until(
                    history_units, history_end_s
                )
            else:
                values, names = behavior_features(transcript_units)
            if args.speaker_order == "causal-role":
                values, names, inferred_role = role_normalize_behavior(
                    values, names, transcript_units
                )
                inferred_roles[inferred_role or "unknown"] += 1
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise AssertionError("Behavior feature order changed")
            rows.append(values)
            category_rows.append(
                category_vector(
                    str(record["profile_text"]), relationship_vocab, situation_vocab, pair_vocab
                )
            )
        raw_behavior[split] = np.stack(rows)
        categories[split] = np.stack(category_rows)
        sample_ids_by_split[split] = sample_ids
        conversation_ids_by_split[split] = conversation_ids
        role_inference_counts[split] = inferred_roles

    mean = raw_behavior["train"].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(
        raw_behavior["train"].std(axis=0, dtype=np.float64).astype(np.float32), 1e-5
    )
    reports: dict[str, Any] = {}
    for split in SPLITS:
        normalized = (raw_behavior[split] - mean) / std
        given = np.concatenate([normalized, categories[split]], axis=1).astype(np.float32)
        shuffled_indices = deterministic_shuffled_indices(
            sample_ids_by_split[split], conversation_ids_by_split[split]
        )
        shuffled = given[shuffled_indices]
        if bool(np.any(conversation_ids_by_split[split][shuffled_indices] == conversation_ids_by_split[split])):
            raise AssertionError("Shuffled dynamic profile came from the same conversation")
        destination = output_dir / f"{split}.profile-view.npz"
        write_npz_atomic(
            destination,
            {
                "sample_ids": sample_ids_by_split[split],
                "profile_given": given,
                "profile_shuffled": shuffled,
                "shuffled_source_indices": shuffled_indices,
            },
        )
        reports[split] = {
            "samples": int(len(given)),
            "dimension": int(given.shape[1]),
            "sidecar": str(destination),
            "sidecar_sha256": sha256_file(destination),
            "sample_ids_aligned": True,
            "shuffled_different_conversation": True,
            "history_scope": args.history_scope,
            "history_lookback_seconds": float(args.history_lookback_seconds),
            "profile_cutoff_before_prediction_s": 30.0 - HISTORY_END_S,
        }

    metadata = {
        "name": (
            "dynamic_behavior_causal_role_relationship_situation_v1"
            if args.speaker_order == "causal-role"
            else "dynamic_behavior_relationship_situation_v1"
        ),
        "description": (
            (
                "Train-normalized causal behavior statistics from the rolling "
                f"{float(args.history_lookback_seconds):g}-second history ending 5 seconds before the "
                "prediction boundary, concatenated with relationship/situation one-hot fields. "
                "Speaker statistics are ordered as current speaker/other participant using only "
                "transcript units completed by the causal boundary."
                if args.speaker_order == "causal-role"
                else "Train-normalized causal behavior statistics for speaker_00/speaker_01 from the rolling "
                f"{float(args.history_lookback_seconds):g}-second history ending 5 seconds before the "
                "prediction boundary, concatenated with relationship/situation one-hot fields."
            )
            if args.history_scope == "conversation-to-t-minus5"
            else (
                "Train-normalized causal behavior statistics ordered as current speaker/other participant "
                "over audio-relative [0,25] s, concatenated with relationship/situation one-hot fields. "
                "The current role is inferred only from transcript units completed by the causal boundary."
                if args.speaker_order == "causal-role"
                else "Train-normalized causal behavior statistics for speaker_00/speaker_01 over audio-relative "
                "[0,25] s, concatenated with relationship/situation one-hot fields. The final 5 s before "
                "the prediction boundary are excluded from the profile and remain in the Qwen context branch."
            )
        ),
        "input_contract": (
            "Only profile vectors change across hidden/given/shuffled; base sample IDs, 30 s causal audio, "
            "causal transcript, prediction boundary, task targets, and decoding are unchanged."
        ),
        "future_information_used": False,
        "history_scope": args.history_scope,
        "history_definition": (
            (
                f"catalog transcript units in the {float(args.history_lookback_seconds):g} seconds "
                "ending 5 seconds before prediction time"
                if float(args.history_lookback_seconds) > 0.0
                else "all catalog transcript units completed by prediction time minus 5 seconds"
            )
            if args.history_scope == "conversation-to-t-minus5"
            else "audio-relative [0,25] seconds of the 30-second causal window"
        ),
        "history_lookback_seconds": float(args.history_lookback_seconds),
        "speaker_order": args.speaker_order,
        "speaker_role_source": (
            "latest transcript unit completed by the 30 s causal prediction boundary"
            if args.speaker_order == "causal-role"
            else "fixed speaker_00 then speaker_01"
        ),
        "role_inference_counts": role_inference_counts,
        "behavior_feature_names": feature_names,
        "behavior_mean_train": mean.tolist(),
        "behavior_std_train": std.tolist(),
        "relationship_vocab": relationship_vocab,
        "situation_vocab": situation_vocab,
        "pair_vocab": pair_vocab,
        "dimension": reports["train"]["dimension"],
        "splits": reports,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
