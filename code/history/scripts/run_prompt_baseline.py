"""Prepare, execute, and score an inference-only profile prompt baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from profile_turntaking.prompt_baseline import (
    prepare_prompt_run,
    run_prompt_requests,
    score_prompt_run,
)


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _prepare(args: argparse.Namespace) -> None:
    _print(
        prepare_prompt_run(
            args.manifest,
            args.output_dir,
            split=args.split,
            max_per_class=args.max_per_class,
            seed=args.seed,
            max_transcript_chars=args.max_transcript_chars,
        )
    )


def _run(args: argparse.Namespace) -> None:
    root = Path(args.run_dir)
    _print(
        run_prompt_requests(
            root / "requests.jsonl",
            root / "responses.jsonl",
            endpoint=args.endpoint,
            model=args.model,
            api_key_env=args.api_key_env,
            timeout_s=args.timeout,
            retries=args.retries,
            delay_s=args.delay,
            limit=args.limit,
        )
    )


def _score(args: argparse.Namespace) -> None:
    _print(score_prompt_run(args.run_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zero-training hidden/given/shuffled profile prompt baseline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="write requests and separate gold files")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--split", default="test")
    prepare.add_argument("--max-per-class", type=int, default=20)
    prepare.add_argument("--max-transcript-chars", type=int, default=6000)
    prepare.add_argument("--seed", type=int, default=13)
    prepare.set_defaults(func=_prepare)

    run = subparsers.add_parser("run", help="call an OpenAI-compatible chat endpoint")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--endpoint", required=True, help="full .../chat/completions URL")
    run.add_argument("--model", required=True)
    run.add_argument("--api-key-env", default="PROMPT_API_KEY")
    run.add_argument("--timeout", type=float, default=60.0)
    run.add_argument("--retries", type=int, default=2)
    run.add_argument("--delay", type=float, default=0.0)
    run.add_argument("--limit", type=int)
    run.set_defaults(func=_run)

    score = subparsers.add_parser("score", help="score valid paired responses")
    score.add_argument("--run-dir", required=True)
    score.set_defaults(func=_score)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
