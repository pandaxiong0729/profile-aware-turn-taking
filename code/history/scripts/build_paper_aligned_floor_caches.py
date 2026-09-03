#!/usr/bin/env python
"""Create cache copies with paper-aligned floor-taking decision boundaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "code" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from profile_turntaking.paper_aligned_floor_targets import rewrite_cache_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-cache-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "qwen-shared-ab-30s-causal" / "layer-weighted-search" / "cache",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "sbcsae_qwen_shared_ab_30s_causal_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "qwen-shared-ab-30s-causal" / "paper-aligned-floor" / "cache",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = {}
    for split in ("train", "val", "test"):
        reports[split] = rewrite_cache_split(
            args.source_cache_dir / f"{split}.qwen-hidden.npz",
            args.dataset_dir / split / "reference_labels.jsonl",
            args.output_dir / f"{split}.qwen-hidden.npz",
        )
    metadata = {
        "schema": "qwen-paper-aligned-floor-boundary-cache-v1",
        "purpose": "Replace only the fourth A/B target with post-interruption outcome-boundary targets.",
        "splits": reports,
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
