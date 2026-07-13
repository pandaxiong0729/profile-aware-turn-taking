"""Prepare, execute, and score the audio-plus-profile MLLM pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from profile_turntaking.mllm_prompt_baseline import (
    prepare_mllm_prompt_run,
    run_mllm_prompt_requests,
    score_prompt_run,
)


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _prepare(args: argparse.Namespace) -> None:
    _print(
        prepare_mllm_prompt_run(
            args.manifest,
            args.output_dir,
            split=args.split,
            max_per_class=args.max_per_class,
            seed=args.seed,
            context_seconds=args.context_seconds,
        )
    )


def _run(args: argparse.Namespace) -> None:
    root = Path(args.run_dir)
    _print(
        run_mllm_prompt_requests(
            root / "requests.jsonl",
            root / "responses.jsonl",
            executable=args.executable,
            model_path=args.model,
            mmproj_path=args.mmproj,
            timeout_s=args.timeout,
            seed=args.seed,
            context_size=args.context_size,
            gpu_layers=args.gpu_layers,
            limit=args.limit,
        )
    )


def _score(args: argparse.Namespace) -> None:
    _print(score_prompt_run(args.run_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zero-training audio+profile MLLM hidden/given/shuffled pilot"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="make causal audio and target-free prompts")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--split", default="test")
    prepare.add_argument("--max-per-class", type=int, default=1)
    prepare.add_argument("--context-seconds", type=float, default=30.0)
    prepare.add_argument("--seed", type=int, default=13)
    prepare.set_defaults(func=_prepare)

    run = commands.add_parser("run", help="run local llama.cpp audio inference")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--executable", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--mmproj", required=True)
    run.add_argument("--timeout", type=float, default=180.0)
    run.add_argument("--seed", type=int, default=13)
    run.add_argument("--context-size", type=int, default=4096)
    run.add_argument("--gpu-layers", default="all")
    run.add_argument("--limit", type=int)
    run.set_defaults(func=_run)

    score = commands.add_parser("score", help="score paired valid predictions")
    score.add_argument("--run-dir", required=True)
    score.set_defaults(func=_score)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
