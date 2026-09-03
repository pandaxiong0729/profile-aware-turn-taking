#!/usr/bin/env python
"""Inspect one real SBCSAE sample and compare the two experiment inputs.

This command is deliberately read-only.  It resolves a sample ID across the
stored JSONL, NPZ and prediction files, then prints one self-contained record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data/processed/sbcsae_qwen_shared_ab_30s_causal_v1"
PROFILE_ROOT = ROOT / "artifacts/main_experiment/profile_features"
CACHE_ROOT = ROOT / "artifacts/main_experiment/qwen_feature_cache"
MAIN_PREDICTIONS = ROOT / "artifacts/main_experiment/results/test_predictions.jsonl"
TT_PREDICTIONS = ROOT / "artifacts/talking_turns/sbcsae_test/test_predictions.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def resolve_audio_path(row: dict[str, Any]) -> Path:
    recorded = Path(str(row.get("source_audio_path") or row.get("audio_path") or ""))
    if recorded.is_file():
        return recorded
    portable = ROOT / "data/sbcsae/openslr/WAV" / f"{row['conversation_id']}.wav"
    if portable.is_file():
        return portable
    raise FileNotFoundError(
        f"Audio not found. Tried the recorded path and {portable.relative_to(ROOT)}"
    )


def select_sample(split: str, sample_id: str | None, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = read_jsonl(DATA_ROOT / split / "selected_inputs.jsonl")
    references = {row["sample_id"]: row for row in read_jsonl(DATA_ROOT / split / "reference_labels.jsonl")}
    if sample_id:
        matches = [row for row in selected if row["sample_id"] == sample_id]
        if not matches:
            raise KeyError(f"sample_id {sample_id!r} is not in split {split!r}")
        row = matches[0]
    else:
        if index < 0 or index >= len(selected):
            raise IndexError(f"index must be between 0 and {len(selected) - 1}")
        row = selected[index]
    return row, references[row["sample_id"]]


def request_for_sample(split: str, sample_id: str) -> dict[str, Any]:
    for row in read_jsonl(DATA_ROOT / split / "requests.jsonl"):
        if row["sample_id"] == sample_id and row["profile_mode"] == "given":
            return row
    raise KeyError(f"given request not found for {sample_id}")


def aligned_npz_row(path: Path, sample_id: str) -> tuple[Any, int]:
    archive = np.load(path)
    ids = archive["sample_ids"].astype(str)
    matches = np.flatnonzero(ids == sample_id)
    if len(matches) != 1:
        raise KeyError(f"Expected one NPZ row for {sample_id}, found {len(matches)} in {path}")
    return archive, int(matches[0])


def predictions_for(path: Path, sample_id: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [row for row in read_jsonl(path) if row.get("sample_id") == sample_id]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--sample-id")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--extract-audio",
        type=Path,
        help="Optional destination WAV. The extracted clip is exactly the causal 30 s input.",
    )
    args = parser.parse_args()

    selected, reference = select_sample(args.split, args.sample_id, args.index)
    request = request_for_sample(args.split, selected["sample_id"])
    profile_npz, profile_index = aligned_npz_row(
        PROFILE_ROOT / f"{args.split}.profile-view.npz", selected["sample_id"]
    )
    cache_npz, cache_index = aligned_npz_row(
        CACHE_ROOT / f"{args.split}.qwen-hidden.npz", selected["sample_id"]
    )
    profile_meta = json.loads((PROFILE_ROOT / "metadata.json").read_text(encoding="utf-8"))
    profile_names = list(profile_meta["behavior_feature_names"])
    profile_names += [f"relationship.one_hot[{name}]" for name in profile_meta["relationship_vocab"]]
    profile_names += [f"situation.one_hot[{name}]" for name in profile_meta["situation_vocab"]]
    profile_names += [f"relationship_situation.one_hot[{name}]" for name in profile_meta["pair_vocab"]]
    profile_values = profile_npz["profile_given"][profile_index].astype(float).tolist()

    audio_path = resolve_audio_path(selected)
    extracted = None
    if args.extract_audio:
        import sys

        src = ROOT / "code/src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from profile_turntaking.audio import read_wav_window_robust_mix, write_wav_mono

        destination = args.extract_audio if args.extract_audio.is_absolute() else ROOT / args.extract_audio
        samples = read_wav_window_robust_mix(
            audio_path,
            float(selected["audio_window_start_s"]),
            float(selected["audio_window_end_s"]),
        )
        write_wav_mono(destination, samples)
        extracted = destination.relative_to(ROOT).as_posix()

    result = {
        "sample_identity": {
            "split": args.split,
            "sample_id": selected["sample_id"],
            "conversation_id": selected["conversation_id"],
        },
        "shared_time_definition": {
            "source_audio": audio_path.relative_to(ROOT).as_posix(),
            "window_start_s": selected["audio_window_start_s"],
            "prediction_boundary_s": selected["audio_window_end_s"],
            "duration_s": float(selected["audio_window_end_s"]) - float(selected["audio_window_start_s"]),
            "predict_event_from_ms_after_boundary": reference["event_offset_ms"],
            "extracted_audio": extracted,
        },
        "our_main_experiment_input": {
            "audio": "the causal audio window above",
            "causal_transcript": request["transcript_prefix"],
            "profile_vector_dimension": len(profile_values),
            "profile_vector": dict(zip(profile_names, profile_values, strict=True)),
            "cached_qwen_context_shape": list(cache_npz["qwen_context"][cache_index].shape),
            "cached_qwen_audio_layers_shape": list(cache_npz["qwen_audio_layers"][cache_index].shape),
        },
        "our_main_experiment_target": {
            "five_way_reference": reference["reference_label"],
            "four_binary_targets": reference["paper_binary_targets"],
            "null_means": "this sample is not scored for that binary task",
        },
        "talking_turns_input_on_our_benchmark": {
            "audio": "the exact same causal audio window",
            "transcript": None,
            "profile": None,
            "model_output": "five probabilities C/BC/T/I/NA, also converted to four A/B tasks for comparison",
        },
        "saved_predictions": {
            "our_main_model": predictions_for(MAIN_PREDICTIONS, selected["sample_id"]),
            "talking_turns_checkpoint": predictions_for(TT_PREDICTIONS, selected["sample_id"]),
        },
        "source_files": {
            "sample": (DATA_ROOT / args.split / "selected_inputs.jsonl").relative_to(ROOT).as_posix(),
            "transcript_request": (DATA_ROOT / args.split / "requests.jsonl").relative_to(ROOT).as_posix(),
            "reference": (DATA_ROOT / args.split / "reference_labels.jsonl").relative_to(ROOT).as_posix(),
            "profile": (PROFILE_ROOT / f"{args.split}.profile-view.npz").relative_to(ROOT).as_posix(),
            "qwen_cache": (CACHE_ROOT / f"{args.split}.qwen-hidden.npz").relative_to(ROOT).as_posix(),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
