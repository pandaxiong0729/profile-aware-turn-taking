"""SBCSAE parsing, five-class heuristic labeling, and manifest construction."""

from __future__ import annotations

import copy
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .audio import write_synthetic_conversation
from .constants import BACKCHANNEL_WORDS, LABELS, UNKNOWN_PROFILE
from .schemas import Sample, Utterance
from .utils import write_json, write_jsonl

_ANNOTATION_RE = re.compile(r"\[[^\]]*\]|<[^>]*>|\([^)]*\)|[@%=~]|\d")
_SPACE_RE = re.compile(r"\s+")
_NUMBER_PREFIX_RE = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


def clean_transcript_text(text: str) -> str:
    text = text.replace("--", " ").replace("=", " ")
    text = _ANNOTATION_RE.sub(" ", text.lower())
    text = re.sub(r"[^a-z'\s-]", " ", text)
    return _SPACE_RE.sub(" ", text).strip(" .,-")


def is_backchannel(utterance: Utterance) -> bool:
    cleaned = clean_transcript_text(utterance.text)
    return bool(cleaned) and utterance.end_s - utterance.start_s <= 1.5 and cleaned in BACKCHANNEL_WORDS


def parse_trn(
    path: str | Path,
    *,
    strict: bool = True,
    diagnostics: list[dict[str, Any]] | None = None,
) -> list[Utterance]:
    """Parse both SBCSAE TRN layouts, carrying forward blank speaker cells.

    Part 1 stores start and end timestamps in the first tab-delimited cell,
    whereas later parts normally use separate cells. In non-strict mode,
    malformed rows are appended to ``diagnostics`` and skipped.
    """
    utterances: list[Utterance] = []
    current_speaker = ""
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not raw_line.strip():
            continue
        parts = raw_line.split("\t")
        while parts and not parts[0].strip():
            parts.pop(0)
        if not parts:
            continue
        first_cell = parts[0].strip().split()
        if len(first_cell) >= 2:
            timestamp_cells = first_cell[:2]
            inline_tail = " ".join(first_cell[2:]).strip()
            remainder = ([inline_tail] if inline_tail else []) + [
                part.strip() for part in parts[1:]
            ]
        elif len(parts) >= 3:
            timestamp_cells = [parts[0].strip(), parts[1].strip()]
            remainder = [part.strip() for part in parts[2:]]
        else:
            if diagnostics is not None:
                diagnostics.append(
                    {"line_number": line_number, "reason": "too_few_columns", "raw": raw_line}
                )
            continue
        try:
            start_match = _NUMBER_PREFIX_RE.match(timestamp_cells[0])
            end_match = _NUMBER_PREFIX_RE.match(timestamp_cells[1])
            if start_match is None or end_match is None:
                raise ValueError
            start_s = float(start_match.group())
            end_s = float(end_match.group())
        except (AttributeError, ValueError):
            if diagnostics is not None:
                diagnostics.append(
                    {"line_number": line_number, "reason": "invalid_timestamp", "raw": raw_line}
                )
            continue
        speaker_cell = ""
        text_parts: list[str] = []
        for index, cell in enumerate(remainder):
            if index == 0 and cell.endswith(":") and cell != ":":
                speaker_cell = cell[:-1].strip()
            elif cell:
                text_parts.append(cell)
        if speaker_cell:
            current_speaker = speaker_cell
        if not current_speaker:
            if diagnostics is not None:
                diagnostics.append(
                    {"line_number": line_number, "reason": "missing_speaker", "raw": raw_line}
                )
            continue
        text = " ".join(text_parts).strip()
        if end_s <= start_s:
            if diagnostics is not None:
                diagnostics.append(
                    {"line_number": line_number, "reason": "non_positive_interval", "raw": raw_line}
                )
            if strict:
                raise ValueError(f"Invalid interval at {path}:{line_number}")
            continue
        utterances.append(Utterance(start_s, end_s, current_speaker, text))
    if not utterances and Path(path).stat().st_size:
        raise ValueError(f"No utterances parsed from non-empty TRN: {path}")
    return utterances


def load_profile_bundle(path: str | Path) -> tuple[dict[str, Any], dict[str, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    profile = payload.get("model_input", {}).get("profile", payload.get("profile", UNKNOWN_PROFILE))
    mapping = payload.get("annotation_only_not_model_input", {}).get("speaker_mapping", {})
    normalized_mapping = {str(name).upper(): key for key, name in mapping.items()}
    return profile, normalized_mapping


def canonicalize_speakers(
    utterances: Sequence[Utterance], speaker_mapping: dict[str, str] | None = None
) -> list[Utterance]:
    """Map corpus names to speaker_A/B, falling back to first-seen order."""
    provided = {key.upper(): value for key, value in (speaker_mapping or {}).items()}
    order: dict[str, str] = {}
    output: list[Utterance] = []
    for utterance in utterances:
        raw = utterance.speaker.upper()
        if raw in provided:
            canonical = provided[raw]
        elif provided:
            continue
        else:
            if raw not in order:
                if len(order) >= 2:
                    continue
                order[raw] = "speaker_A" if not order else "speaker_B"
            canonical = order[raw]
        if canonical not in {"speaker_A", "speaker_B"}:
            continue
        output.append(Utterance(utterance.start_s, utterance.end_s, canonical, utterance.text))
    return output


def label_at(utterances: Sequence[Utterance], prediction_time_s: float, horizon_ms: int = 40) -> str:
    horizon_end = prediction_time_s + horizon_ms / 1000.0
    active = [item for item in utterances if item.overlaps(prediction_time_s, horizon_end)]
    if not active:
        return "NA"
    backchannels = [item for item in active if is_backchannel(item)]
    if backchannels:
        for item in backchannels:
            previous = _previous_floor(utterances, item.start_s)
            if previous and previous != item.speaker:
                return "BC"
    for index, first in enumerate(active):
        for second in active[index + 1 :]:
            if first.speaker == second.speaker:
                continue
            overlap_start = max(prediction_time_s, first.start_s, second.start_s)
            overlap_end = min(horizon_end, first.end_s, second.end_s)
            if overlap_start < overlap_end:
                return "I"
    onsets = sorted(
        [
            item
            for item in active
            if prediction_time_s - 1e-9 <= item.start_s < horizon_end and not is_backchannel(item)
        ],
        key=lambda item: item.start_s,
    )
    for current in onsets:
        previous = _previous_floor(utterances, current.start_s)
        if previous and previous != current.speaker:
            return "T"
    return "C"


def _previous_floor(utterances: Sequence[Utterance], time_s: float) -> str | None:
    candidates = [
        item
        for item in utterances
        if item.start_s < time_s and not is_backchannel(item)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: min(item.end_s, time_s)).speaker


def transcript_prefix(utterances: Sequence[Utterance], start_s: float, end_s: float) -> str:
    """Return only completed utterances, preventing future text leakage."""
    lines = [
        f"[{item.speaker} {item.start_s:.2f}-{item.end_s:.2f}] {clean_transcript_text(item.text)}"
        for item in utterances
        if item.end_s <= end_s + 1e-9 and item.end_s >= start_s
    ]
    return " ".join(lines)


def assign_splits(samples: list[Sample], seed: int = 13) -> list[Sample]:
    conversations = sorted({sample.split_group for sample in samples})
    rng = random.Random(seed)
    rng.shuffle(conversations)
    if len(conversations) >= 3:
        count = len(conversations)
        train_end = min(max(1, int(round(count * 0.7))), count - 2)
        val_count = min(max(1, int(round(count * 0.15))), count - train_end - 1)
        val_end = train_end + val_count
        mapping = {cid: "train" for cid in conversations[:train_end]}
        mapping.update({cid: "val" for cid in conversations[train_end:val_end]})
        mapping.update({cid: "test" for cid in conversations[val_end:]})
        return [Sample(**{**sample.to_dict(), "split": mapping[sample.split_group]}) for sample in samples]
    # Smoke-only fallback: stratify one conversation so every label reaches every split.
    # Scientific experiments must use three or more speaker-disjoint conversations.
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)
    result: list[Sample] = []
    for label_samples in grouped.values():
        rng.shuffle(label_samples)
        count = len(label_samples)
        if count >= 3:
            train_count = min(max(1, int(round(count * 0.7))), count - 2)
            val_count = min(max(1, int(round(count * 0.15))), count - train_count - 1)
        else:
            train_count = count
            val_count = 0
        for index, sample in enumerate(label_samples):
            split = "train" if index < train_count else "val" if index < train_count + val_count else "test"
            result.append(Sample(**{**sample.to_dict(), "split": split}))
    rng.shuffle(result)
    return result


def build_samples(
    *,
    conversation_id: str,
    utterances: Sequence[Utterance],
    audio_path: str | Path,
    profile: dict[str, Any],
    split_group: str | None = None,
    context_seconds: float = 3.0,
    horizon_ms: int = 40,
    stride_ms: int = 40,
    max_per_class: int = 96,
    seed: int = 13,
    max_time_s: float | None = None,
) -> list[Sample]:
    if not utterances:
        raise ValueError("No utterances available")
    end_time = min(max(item.end_s for item in utterances), max_time_s or float("inf"))
    by_label: dict[str, list[Sample]] = defaultdict(list)
    step = stride_ms / 1000.0
    index = 0
    time_s = context_seconds
    while time_s + horizon_ms / 1000.0 <= end_time:
        label = label_at(utterances, time_s, horizon_ms)
        sample = Sample(
            sample_id=f"{conversation_id}-{index:07d}",
            conversation_id=conversation_id,
            split_group=split_group or conversation_id,
            split="unassigned",
            prediction_time_s=round(time_s, 6),
            horizon_ms=horizon_ms,
            window_start_s=round(time_s - context_seconds, 6),
            window_end_s=round(time_s, 6),
            audio_path=str(Path(audio_path).resolve()),
            transcript_prefix=transcript_prefix(utterances, time_s - context_seconds, time_s),
            profile=copy.deepcopy(profile),
            label=label,
        )
        by_label[label].append(sample)
        index += 1
        time_s += step
    rng = random.Random(seed)
    balanced: list[Sample] = []
    for label in LABELS:
        candidates = by_label[label]
        rng.shuffle(candidates)
        balanced.extend(candidates[:max_per_class])
    rng.shuffle(balanced)
    return assign_splits(balanced, seed=seed)


def prepare_sbcsae(
    *,
    trn_path: str | Path,
    profile_path: str | Path,
    output_manifest: str | Path,
    audio_path: str | Path | None = None,
    synthetic_audio_path: str | Path | None = None,
    conversation_id: str | None = None,
    split_group: str | None = None,
    context_seconds: float = 3.0,
    horizon_ms: int = 40,
    stride_ms: int = 40,
    max_per_class: int = 96,
    max_time_s: float | None = 120.0,
    seed: int = 13,
) -> dict[str, Any]:
    trn = Path(trn_path)
    conversation = conversation_id or trn.stem
    profile, mapping = load_profile_bundle(profile_path)
    utterances = canonicalize_speakers(parse_trn(trn), mapping)
    if max_time_s is not None:
        utterances = [item for item in utterances if item.start_s < max_time_s]
        utterances = [
            Utterance(item.start_s, min(item.end_s, max_time_s), item.speaker, item.text)
            for item in utterances
        ]
    selected_audio: Path
    if audio_path and Path(audio_path).is_file():
        selected_audio = Path(audio_path)
        audio_source = "real"
    else:
        selected_audio = Path(synthetic_audio_path or Path(output_manifest).with_suffix(".wav"))
        duration = min(max(item.end_s for item in utterances) + 0.2, max_time_s or float("inf"))
        write_synthetic_conversation(selected_audio, utterances, duration_s=duration, seed=seed)
        audio_source = "synthetic_from_real_timestamps"
    samples = build_samples(
        conversation_id=conversation,
        utterances=utterances,
        audio_path=selected_audio,
        profile=profile,
        split_group=split_group,
        context_seconds=context_seconds,
        horizon_ms=horizon_ms,
        stride_ms=stride_ms,
        max_per_class=max_per_class,
        seed=seed,
        max_time_s=max_time_s,
    )
    write_jsonl(output_manifest, (sample.to_dict() for sample in samples))
    summary = {
        "conversation_id": conversation,
        "audio_source": audio_source,
        "audio_path": str(selected_audio.resolve()),
        "manifest_path": str(Path(output_manifest).resolve()),
        "utterances": len(utterances),
        "samples": len(samples),
        "labels": dict(Counter(sample.label for sample in samples)),
        "splits": dict(Counter(sample.split for sample in samples)),
        "context_seconds": context_seconds,
        "horizon_ms": horizon_ms,
        "scientific_split": len({sample.split_group for sample in samples}) >= 3,
    }
    write_json(Path(output_manifest).with_suffix(".summary.json"), summary)
    return summary


def merge_manifests(
    manifest_paths: Sequence[str | Path], output_manifest: str | Path, *, seed: int = 13
) -> dict[str, Any]:
    from .utils import read_jsonl

    samples = [Sample(**row) for path in manifest_paths for row in read_jsonl(path)]
    groups = {sample.split_group for sample in samples}
    if len(groups) < 3:
        raise ValueError("Scientific group splitting requires at least three split_group values")
    assigned = assign_splits(samples, seed=seed)
    write_jsonl(output_manifest, (sample.to_dict() for sample in assigned))
    summary = {
        "manifest_path": str(Path(output_manifest).resolve()),
        "input_manifests": [str(Path(path).resolve()) for path in manifest_paths],
        "samples": len(assigned),
        "split_groups": len(groups),
        "labels": dict(Counter(sample.label for sample in assigned)),
        "splits": dict(Counter(sample.split for sample in assigned)),
        "scientific_split": True,
    }
    write_json(Path(output_manifest).with_suffix(".summary.json"), summary)
    return summary
