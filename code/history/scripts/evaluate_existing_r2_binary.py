#!/usr/bin/env python
"""Evaluate the unchanged trained R2 checkpoint as the prior four A/B probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from profile_turntaking.semantic_profile_binary_eval import (
    evaluate_existing_r2_as_binary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        default="artifacts/semantic-profile-embedding/minilm-additive-raw-v2",
    )
    parser.add_argument(
        "--data-dir", default="data/processed/sbcsae_semantic_profile_v1"
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/semantic-profile-embedding/minilm-r2-existing-checkpoint-prompt-matched-binary",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 37, 71])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    data = Path(args.data_dir)
    summary = evaluate_existing_r2_as_binary(
        args.checkpoint_dir,
        {
            "prompt_seed137": data / "prompt_seed137.semantic.npz",
            "prompt_seed237": data / "prompt_seed237.semantic.npz",
        },
        args.output_dir,
        seeds=tuple(args.seeds),
        device=args.device,
    )
    print(json.dumps(summary["ensemble_reports"], indent=2))
    print(json.dumps(summary["training_overlap_audit"], indent=2))


if __name__ == "__main__":
    main()
