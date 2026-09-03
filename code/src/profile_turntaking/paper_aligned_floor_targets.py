"""Repair the floor-taking benchmark boundary without changing model inputs.

Talking Turns defines floor-taking after an interruption has already been
observed: the next decision is whether the interrupting speaker takes the
floor (T) or the incumbent keeps it (C).  SBCSAE event candidates already
contain causal outcome-boundary events for those two cases.  This module maps
those events to the fourth paper-style A/B target while leaving the cached
audio, transcript, profile, sample IDs, and the first three targets untouched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PAPER_TASK_ORDER = ("turn_change", "backchannel", "interruption", "floor_taking")
FLOOR_TASK_INDEX = PAPER_TASK_ORDER.index("floor_taking")
FLOOR_SOURCE_TO_TARGET = {
    "hold_after_unsuccessful_interruption": 0,
    "shift_after_successful_interruption": 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def outcome_boundary_targets(
    sample_ids: np.ndarray,
    original_targets: np.ndarray,
    references: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, int]]:
    """Return targets using causal post-interruption outcome boundaries.

    The reference file is used only to build targets.  No reference field is
    copied into any model feature array.
    """

    ids = np.asarray(sample_ids).astype(str)
    targets = np.asarray(original_targets, dtype=np.int64).copy()
    if targets.shape != (len(ids), len(PAPER_TASK_ORDER)):
        raise ValueError(
            f"paper_targets shape {targets.shape} does not match ({len(ids)}, {len(PAPER_TASK_ORDER)})"
        )

    by_id: dict[str, dict[str, Any]] = {}
    for row in references:
        sample_id = str(row["sample_id"])
        if sample_id in by_id:
            raise ValueError(f"Duplicate reference sample_id: {sample_id}")
        by_id[sample_id] = row
    missing = [sample_id for sample_id in ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"Reference labels missing {len(missing)} cache sample IDs; first={missing[0]}")

    # Remove the old pre-interruption floor-outcome target, then assign the
    # target only to the event whose boundary is immediately before T or C.
    targets[:, FLOOR_TASK_INDEX] = -100
    counts = {"A": 0, "B": 0, "ignored": 0}
    for index, sample_id in enumerate(ids):
        source = str(by_id[sample_id].get("source_kind", ""))
        target = FLOOR_SOURCE_TO_TARGET.get(source)
        if target is None:
            counts["ignored"] += 1
            continue
        targets[index, FLOOR_TASK_INDEX] = target
        counts["B" if target == 1 else "A"] += 1
    if counts["A"] == 0 or counts["B"] == 0:
        raise ValueError(f"Outcome-boundary floor task is not binary: {counts}")
    return targets, counts


def rewrite_cache_split(
    source_cache: Path,
    reference_path: Path,
    destination_cache: Path,
) -> dict[str, Any]:
    with np.load(source_cache, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    required = {"sample_ids", "paper_targets"}
    missing = required - arrays.keys()
    if missing:
        raise ValueError(f"Cache missing required arrays: {sorted(missing)}")

    original_hashes = {
        key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for key, value in arrays.items()
        if key != "paper_targets"
    }
    targets, counts = outcome_boundary_targets(
        arrays["sample_ids"], arrays["paper_targets"], read_jsonl(reference_path)
    )
    arrays["paper_targets"] = targets

    destination_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_cache.with_suffix(destination_cache.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(destination_cache)

    with np.load(destination_cache, allow_pickle=False) as saved:
        for key, expected_hash in original_hashes.items():
            actual_hash = hashlib.sha256(np.ascontiguousarray(saved[key]).tobytes()).hexdigest()
            if actual_hash != expected_hash:
                raise RuntimeError(f"Non-target array changed while rewriting cache: {key}")

    report = {
        "source_cache": str(source_cache.resolve()),
        "source_cache_sha256": sha256_file(source_cache),
        "reference_labels": str(reference_path.resolve()),
        "reference_labels_sha256": sha256_file(reference_path),
        "destination_cache": str(destination_cache.resolve()),
        "destination_cache_sha256": sha256_file(destination_cache),
        "samples": int(len(arrays["sample_ids"])),
        "floor_target_counts": counts,
        "unchanged_arrays": sorted(original_hashes),
        "target_definition": {
            "A": "hold_after_unsuccessful_interruption (incumbent keeps floor)",
            "B": "shift_after_successful_interruption (interrupting speaker takes floor)",
            "input_boundary": "100 ms before the post-overlap C/T outcome event",
            "future_information_in_model_input": False,
        },
    }
    source_meta_path = source_cache.with_suffix(".meta.json")
    destination_meta_path = destination_cache.with_suffix(".meta.json")
    source_meta = (
        json.loads(source_meta_path.read_text(encoding="utf-8"))
        if source_meta_path.is_file()
        else {}
    )
    derived_meta = {
        **source_meta,
        "schema_version": "qwen-paper-aligned-floor-boundary-cache-v1",
        "cache_path": str(destination_cache.resolve()),
        "cache_sha256": report["destination_cache_sha256"],
        "derived_from_cache": str(source_cache.resolve()),
        "derived_from_cache_sha256": report["source_cache_sha256"],
        "paper_floor_target_rewrite": report["target_definition"],
        "paper_floor_target_counts": counts,
        "non_target_arrays_byte_identical": True,
    }
    destination_meta_path.write_text(
        json.dumps(derived_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report["destination_meta"] = str(destination_meta_path.resolve())
    report["destination_meta_sha256"] = sha256_file(destination_meta_path)
    return report


__all__ = [
    "FLOOR_SOURCE_TO_TARGET",
    "outcome_boundary_targets",
    "rewrite_cache_split",
]
