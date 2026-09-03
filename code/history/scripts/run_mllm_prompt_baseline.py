"""Prepare, audit, execute, and score the three-input MLLM experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from profile_turntaking.mllm_prompt_baseline import (
    audit_mllm_prompt_run,
    prepare_mllm_prompt_run,
    prepare_silenced_audio_control,
    require_reviewed_labels,
    run_mllm_prompt_requests,
    run_mllm_server_requests,
    score_silenced_audio_control,
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
            max_transcript_chars=args.max_transcript_chars,
        )
    )


def _audit(args: argparse.Namespace) -> None:
    _print(
        audit_mllm_prompt_run(
            args.run_dir,
            expected_samples=args.expected_samples,
            expected_per_class=args.expected_per_class,
        )
    )


def _run(args: argparse.Namespace) -> None:
    root = Path(args.run_dir)
    audit_mllm_prompt_run(root)
    require_reviewed_labels(root, allow_weak_labels=args.allow_weak_labels)
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


def _run_server(args: argparse.Namespace) -> None:
    root = Path(args.run_dir)
    audit_mllm_prompt_run(root)
    require_reviewed_labels(root, allow_weak_labels=args.allow_weak_labels)
    _print(
        run_mllm_server_requests(
            root / "requests.jsonl",
            root / "responses.jsonl",
            endpoint=args.endpoint,
            model=args.model,
            timeout_s=args.timeout,
            retries=args.retries,
            seed=args.seed,
            limit=args.limit,
        )
    )


def _score(args: argparse.Namespace) -> None:
    require_reviewed_labels(args.run_dir, allow_weak_labels=args.allow_weak_labels)
    _print(score_prompt_run(args.run_dir))


def _prepare_audio_control(args: argparse.Namespace) -> None:
    _print(
        prepare_silenced_audio_control(
            args.main_run_dir,
            args.output_dir,
            samples=args.samples,
            seed=args.seed,
        )
    )


def _score_audio_control(args: argparse.Namespace) -> None:
    _print(score_silenced_audio_control(args.main_run_dir, args.control_run_dir))


def _run_audio_control(args: argparse.Namespace) -> None:
    root = Path(args.control_run_dir)
    _print(
        run_mllm_server_requests(
            root / "requests.jsonl",
            root / "responses.jsonl",
            endpoint=args.endpoint,
            model=args.model,
            timeout_s=args.timeout,
            retries=args.retries,
            seed=args.seed,
            limit=args.limit,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-training audio+causal-transcript+profile MLLM "
            "hidden/given/shuffled experiment"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare", help="make causal audio+transcript+profile target-free prompts"
    )
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--split", default="test")
    prepare.add_argument("--max-per-class", type=int, default=1)
    prepare.add_argument("--context-seconds", type=float, default=30.0)
    prepare.add_argument("--max-transcript-chars", type=int, default=6000)
    prepare.add_argument("--seed", type=int, default=13)
    prepare.set_defaults(func=_prepare)

    audit = commands.add_parser("audit", help="verify the paired three-input contract")
    audit.add_argument("--run-dir", required=True)
    audit.add_argument("--expected-samples", type=int)
    audit.add_argument("--expected-per-class", type=int)
    audit.set_defaults(func=_audit)

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
    run.add_argument(
        "--allow-weak-labels",
        action="store_true",
        help="diagnostic only: bypass the reviewed-label gate",
    )
    run.set_defaults(func=_run)

    run_server = commands.add_parser(
        "run-server", help="run against a persistent llama.cpp multimodal server"
    )
    run_server.add_argument("--run-dir", required=True)
    run_server.add_argument(
        "--endpoint", default="http://127.0.0.1:8091/v1/chat/completions"
    )
    run_server.add_argument("--model", default="qwen2.5-omni-3b-q4_k_m")
    run_server.add_argument("--timeout", type=float, default=180.0)
    run_server.add_argument("--retries", type=int, default=2)
    run_server.add_argument("--seed", type=int, default=13)
    run_server.add_argument("--limit", type=int)
    run_server.add_argument(
        "--allow-weak-labels",
        action="store_true",
        help="diagnostic only: bypass the reviewed-label gate",
    )
    run_server.set_defaults(func=_run_server)

    score = commands.add_parser("score", help="score paired valid predictions")
    score.add_argument("--run-dir", required=True)
    score.add_argument(
        "--allow-weak-labels",
        action="store_true",
        help="diagnostic only: bypass the reviewed-label gate",
    )
    score.set_defaults(func=_score)

    prepare_control = commands.add_parser(
        "prepare-audio-control", help="make a hidden-profile silenced-audio diagnostic"
    )
    prepare_control.add_argument("--main-run-dir", required=True)
    prepare_control.add_argument("--output-dir", required=True)
    prepare_control.add_argument("--samples", type=int, default=50)
    prepare_control.add_argument("--seed", type=int, default=29)
    prepare_control.set_defaults(func=_prepare_audio_control)

    run_control = commands.add_parser(
        "run-audio-control", help="run the prepared audio-sensitivity diagnostic"
    )
    run_control.add_argument("--control-run-dir", required=True)
    run_control.add_argument(
        "--endpoint", default="http://127.0.0.1:8091/v1/chat/completions"
    )
    run_control.add_argument("--model", default="qwen2.5-omni-3b-q4_k_m")
    run_control.add_argument("--timeout", type=float, default=180.0)
    run_control.add_argument("--retries", type=int, default=2)
    run_control.add_argument("--seed", type=int, default=13)
    run_control.add_argument("--limit", type=int)
    run_control.set_defaults(func=_run_audio_control)

    score_control = commands.add_parser(
        "score-audio-control", help="compare original and silenced-audio predictions"
    )
    score_control.add_argument("--main-run-dir", required=True)
    score_control.add_argument("--control-run-dir", required=True)
    score_control.set_defaults(func=_score_audio_control)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
