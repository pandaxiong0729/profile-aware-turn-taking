from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("train", "val", "test")
PROFILE_MODES = ("given", "shuffled")
UNKNOWN = "<UNK>"


def parse_profile(profile_text: str) -> tuple[str, str]:
    relationship_prefix = "Their relationship is "
    situation_prefix = "The conversation situation is "
    relationship = UNKNOWN
    situation = UNKNOWN
    for raw_line in profile_text.splitlines():
        line = raw_line.strip()
        if line.startswith(relationship_prefix):
            relationship = line[len(relationship_prefix) :].rstrip(".").strip().lower()
        elif line.startswith(situation_prefix):
            situation = line[len(situation_prefix) :].rstrip(".").strip().lower()
    return relationship or UNKNOWN, situation or UNKNOWN


def read_profiles(requests_path: Path) -> dict[str, dict[str, str]]:
    by_sample: dict[str, dict[str, str]] = {}
    with requests_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            mode = str(record["profile_mode"])
            if mode not in PROFILE_MODES:
                continue
            sample_id = str(record["sample_id"])
            by_sample.setdefault(sample_id, {})[mode] = str(record["profile_text"])
    incomplete = [sample_id for sample_id, modes in by_sample.items() if set(modes) != set(PROFILE_MODES)]
    if incomplete:
        raise ValueError(f"Missing given/shuffled profile rows for {len(incomplete)} samples")
    return by_sample


def vocabulary(values: set[str]) -> dict[str, int]:
    ordered = [UNKNOWN, *sorted(value for value in values if value != UNKNOWN)]
    return {value: index for index, value in enumerate(ordered)}


def encode_profile(
    profile_text: str,
    relationship_vocab: dict[str, int],
    situation_vocab: dict[str, int],
    pair_vocab: dict[str, int],
) -> np.ndarray:
    relationship, situation = parse_profile(profile_text)
    pair = f"{relationship} || {situation}"
    result = np.zeros(
        len(relationship_vocab) + len(situation_vocab) + len(pair_vocab), dtype=np.float32
    )
    relationship_index = relationship_vocab.get(relationship, relationship_vocab[UNKNOWN])
    situation_index = situation_vocab.get(situation, situation_vocab[UNKNOWN])
    pair_index = pair_vocab.get(pair, pair_vocab[UNKNOWN])
    result[relationship_index] = 1.0
    result[len(relationship_vocab) + situation_index] = 1.0
    result[len(relationship_vocab) + len(situation_vocab) + pair_index] = 1.0
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build aligned relationship+situation profile vectors without changing base caches."
    )
    parser.add_argument(
        "--data-dir", default="data/processed/sbcsae_qwen_shared_ab_30s_causal_v1"
    )
    parser.add_argument(
        "--cache-dir",
        default="artifacts/qwen-shared-ab-30s-causal/layer-weighted-search/cache",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/qwen-shared-ab-30s-causal/profile-views/relationship-situation",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = (repo_root / args.data_dir).resolve()
    cache_dir = (repo_root / args.cache_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    profiles = {
        split: read_profiles(data_dir / split / "requests.jsonl") for split in SPLITS
    }
    train_values = [
        parse_profile(text)
        for modes in profiles["train"].values()
        for text in modes.values()
    ]
    relationship_vocab = vocabulary({relationship for relationship, _ in train_values})
    situation_vocab = vocabulary({situation for _, situation in train_values})
    pair_vocab = vocabulary(
        {f"{relationship} || {situation}" for relationship, situation in train_values}
    )

    split_reports: dict[str, Any] = {}
    for split in SPLITS:
        base_cache_path = cache_dir / f"{split}.qwen-hidden.npz"
        with np.load(base_cache_path, allow_pickle=False) as base:
            sample_ids = base["sample_ids"].astype(str)
        missing = [sample_id for sample_id in sample_ids if sample_id not in profiles[split]]
        if missing:
            raise ValueError(f"{split}: {len(missing)} cache sample IDs are missing from requests")
        encoded: dict[str, np.ndarray] = {}
        parsed_counts: dict[str, dict[str, int]] = {}
        for mode in PROFILE_MODES:
            encoded[mode] = np.stack(
                [
                    encode_profile(
                        profiles[split][sample_id][mode],
                        relationship_vocab,
                        situation_vocab,
                        pair_vocab,
                    )
                    for sample_id in sample_ids
                ]
            )
            counts: dict[str, int] = {}
            for sample_id in sample_ids:
                relationship, situation = parse_profile(profiles[split][sample_id][mode])
                key = f"{relationship} || {situation}"
                counts[key] = counts.get(key, 0) + 1
            parsed_counts[mode] = counts

        destination = output_dir / f"{split}.profile-view.npz"
        write_npz_atomic(
            destination,
            {
                "sample_ids": sample_ids,
                "profile_given": encoded["given"],
                "profile_shuffled": encoded["shuffled"],
            },
        )
        split_reports[split] = {
            "samples": int(len(sample_ids)),
            "dimension": int(encoded["given"].shape[1]),
            "base_cache": str(base_cache_path),
            "base_cache_sha256": sha256_file(base_cache_path),
            "sidecar": str(destination),
            "sidecar_sha256": sha256_file(destination),
            "sample_ids_aligned": True,
            "given_counts": parsed_counts["given"],
            "shuffled_counts": parsed_counts["shuffled"],
        }

    metadata = {
        "name": "relationship_situation_structured_v1",
        "description": (
            "Three concatenated one-hot fields: relationship, situation, and their pair. "
            "Vocabularies are fitted on train profiles only; unknown val/test values map to <UNK>."
        ),
        "input_contract": "Only the profile representation changes; sample/audio/transcript/targets remain in the audited base caches.",
        "relationship_vocab": relationship_vocab,
        "situation_vocab": situation_vocab,
        "pair_vocab": pair_vocab,
        "dimension": len(relationship_vocab) + len(situation_vocab) + len(pair_vocab),
        "splits": split_reports,
    }
    write_json(output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
