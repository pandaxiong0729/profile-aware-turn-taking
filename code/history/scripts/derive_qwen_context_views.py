from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


VIEW_SLICES: dict[str, tuple[int, int]] = {
    "prompt_last": (0, 1),
    "paper_audio_last": (1, 2),
    "prompt_audio_last": (0, 2),
    "audio_boundary": (1, 3),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_split(source: Path, destination: Path, *, start: int, stop: int, view: str) -> None:
    with np.load(source, allow_pickle=False) as payload:
        arrays: dict[str, np.ndarray] = {key: payload[key] for key in payload.files}
    profile_dim = int(arrays["profile_given"].shape[1])
    context = arrays["qwen_context"]
    if context.ndim != 2 or context.shape[1] != profile_dim * 4:
        raise ValueError(
            f"Expected rich context dimension 4*{profile_dim}, got {context.shape} in {source}"
        )
    arrays["qwen_context"] = context[:, start * profile_dim : stop * profile_dim].astype(
        np.float32
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)

    source_meta_path = source.with_suffix(".meta.json")
    meta: dict[str, Any] = json.loads(source_meta_path.read_text(encoding="utf-8"))
    meta.update(
        {
            "derived_from_cache": str(source.resolve()),
            "derived_from_cache_sha256": file_sha256(source),
            "cache_path": str(destination.resolve()),
            "cache_sha256": file_sha256(destination),
            "context_pooling": view,
            "context_pooling_definition": {
                "prompt_last": "last non-padding Thinker token after causal audio+text prompt",
                "paper_audio_last": "final Thinker hidden state at the last causal audio token",
                "prompt_audio_last": "concatenate prompt_last and paper_audio_last",
                "audio_boundary": "concatenate last audio token and last-8-audio-token mean",
            }[view],
            "qwen_context_dimension": int(arrays["qwen_context"].shape[1]),
        }
    )
    destination.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    source_profiles = source.with_suffix(".profiles.jsonl")
    if source_profiles.is_file():
        shutil.copyfile(source_profiles, destination.with_suffix(".profiles.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive zero-recompute Qwen context views from one rich 4-block cache."
    )
    parser.add_argument("--rich-cache-dir", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    rich_dir = Path(args.rich_cache_dir).resolve()
    output_root = Path(args.output_root).resolve()
    mapping: dict[str, str] = {"rich": str(rich_dir)}
    for view, (start, stop) in VIEW_SLICES.items():
        view_dir = output_root / view
        for split in ("train", "val", "test"):
            source = rich_dir / f"{split}.qwen-hidden.npz"
            if not source.is_file() or not source.with_suffix(".meta.json").is_file():
                raise FileNotFoundError(f"Incomplete rich cache: {source}")
            derive_split(
                source,
                view_dir / f"{split}.qwen-hidden.npz",
                start=start,
                stop=stop,
                view=view,
            )
        mapping[view] = str(view_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "context_views.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(mapping, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
