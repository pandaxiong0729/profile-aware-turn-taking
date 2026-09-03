#!/usr/bin/env python
"""Merge train and validation rows for a fixed-epoch final refit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def concatenate(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if first.keys() != second.keys():
        raise ValueError("Train and validation cache keys differ")
    result = {}
    for key in first:
        if first[key].shape[1:] != second[key].shape[1:]:
            raise ValueError(f"Array shape mismatch for {key}")
        result[key] = np.concatenate([first[key], second[key]], axis=0)
    ids = result["sample_ids"].astype(str)
    if len(set(ids)) != len(ids):
        raise ValueError("Merged train+validation sample IDs are not unique")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--profile-view-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cache_output = args.output_dir / "cache"
    view_output = args.output_dir / "profile-view"
    train_cache = load_npz(args.cache_dir / "train.qwen-hidden.npz")
    val_cache = load_npz(args.cache_dir / "val.qwen-hidden.npz")
    test_cache = load_npz(args.cache_dir / "test.qwen-hidden.npz")
    merged_cache = concatenate(train_cache, val_cache)
    write_npz(cache_output / "train.qwen-hidden.npz", merged_cache)
    write_npz(cache_output / "val.qwen-hidden.npz", val_cache)
    write_npz(cache_output / "test.qwen-hidden.npz", test_cache)

    train_view = load_npz(args.profile_view_dir / "train.profile-view.npz")
    val_view = load_npz(args.profile_view_dir / "val.profile-view.npz")
    test_view = load_npz(args.profile_view_dir / "test.profile-view.npz")
    merged_view = concatenate(train_view, val_view)
    if "shuffled_source_indices" in merged_view:
        merged_view["shuffled_source_indices"] = np.concatenate(
            [
                train_view["shuffled_source_indices"].astype(np.int64),
                val_view["shuffled_source_indices"].astype(np.int64) + len(train_view["sample_ids"]),
            ]
        )
    write_npz(view_output / "train.profile-view.npz", merged_view)
    write_npz(view_output / "val.profile-view.npz", val_view)
    write_npz(view_output / "test.profile-view.npz", test_view)

    if not np.array_equal(
        merged_cache["sample_ids"].astype(str), merged_view["sample_ids"].astype(str)
    ):
        raise RuntimeError("Merged cache/profile IDs are not aligned")
    if not np.array_equal(
        test_cache["sample_ids"].astype(str), test_view["sample_ids"].astype(str)
    ):
        raise RuntimeError("Test cache/profile IDs are not aligned")

    source_profile_metadata = args.profile_view_dir / "metadata.json"
    metadata = {
        "schema": "qwen-train-val-fixed-epoch-refit-v1",
        "purpose": "Refit a validation-selected architecture on train+validation; test remains unchanged.",
        "train_rows": int(len(merged_cache["sample_ids"])),
        "original_train_rows": int(len(train_cache["sample_ids"])),
        "added_validation_rows": int(len(val_cache["sample_ids"])),
        "test_rows": int(len(test_cache["sample_ids"])),
        "test_cache_unchanged_sha256": sha256_file(args.cache_dir / "test.qwen-hidden.npz"),
        "profile_source_metadata": (
            json.loads(source_profile_metadata.read_text(encoding="utf-8"))
            if source_profile_metadata.is_file()
            else None
        ),
        "fixed_epoch_required": True,
        "validation_checkpoint_selection_allowed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # The runner requires cache metadata files only in the validation-grid
    # orchestrator, but keeping them here makes the derived data inspectable.
    for split in ("train", "val", "test"):
        cache_path = cache_output / f"{split}.qwen-hidden.npz"
        (cache_output / f"{split}.qwen-hidden.meta.json").write_text(
            json.dumps(
                {
                    "schema_version": metadata["schema"],
                    "split": split,
                    "cache_path": str(cache_path.resolve()),
                    "cache_sha256": sha256_file(cache_path),
                    "test_unchanged": split == "test",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    (view_output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
