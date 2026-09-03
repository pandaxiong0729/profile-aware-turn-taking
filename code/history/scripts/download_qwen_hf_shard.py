"""Download selected HuggingFace files for Qwen2.5-Omni-3B."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--cache-dir", default="models/huggingface")
    parser.add_argument("--allow-pattern", action="append", required=True)
    parser.add_argument("--single-file", action="store_true")
    parser.add_argument("--disable-xet", action="store_true")
    args = parser.parse_args()
    if args.disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    if args.single_file:
        if len(args.allow_pattern) != 1:
            raise ValueError("--single-file expects exactly one --allow-pattern")
        path = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.allow_pattern[0],
            cache_dir=str(Path(args.cache_dir)),
            resume_download=True,
        )
    else:
        path = snapshot_download(
            args.repo_id,
            cache_dir=str(Path(args.cache_dir)),
            allow_patterns=args.allow_pattern,
            resume_download=True,
        )
    print(path)


if __name__ == "__main__":
    main()
