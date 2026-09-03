"""Prepare, audit, run, and score the strict-causal 3B event evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from profile_turntaking.mllm_prompt_baseline import (
    prepare_silenced_audio_control,
    run_mllm_server_requests,
    score_silenced_audio_control,
)
from profile_turntaking.paper_binary_hierarchy import (
    aggregate_binary_hierarchy,
    audit_binary_hierarchy_eval,
    prepare_binary_hierarchy_eval,
    run_binary_hierarchy_server,
)
from profile_turntaking.qwen25_omni_event_eval import (
    DEFAULT_CONFIG_PATH,
    aggregate_candidate_scores,
    apply_causal_asr,
    audit_candidate_score_eval,
    audit_event_eval,
    prepare_candidate_score_eval,
    prepare_phase,
    run_event_server,
    run_candidate_score_server,
    score_event_eval,
    verify_formal_readiness,
)


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _prepare(args: argparse.Namespace) -> None:
    if args.phase == "formal500":
        if not args.gate_run_dir or not args.audio_control_dir:
            raise ValueError(
                "formal500 requires --gate-run-dir and --audio-control-dir"
            )
        verify_formal_readiness(
            args.gate_run_dir,
            args.audio_control_dir,
            config_path=args.config,
        )
    _print(
        prepare_phase(
            args.manifest,
            args.output_dir,
            phase=args.phase,
            config_path=args.config,
            prompt_style=args.prompt_style,
            selection_seed=args.selection_seed,
        )
    )


def _audit(args: argparse.Namespace) -> None:
    _print(audit_event_eval(args.run_dir))


def _apply_asr(args: argparse.Namespace) -> None:
    _print(apply_causal_asr(args.run_dir, asr_path=args.asr_path))


def _prepare_candidate(args: argparse.Namespace) -> None:
    _print(prepare_candidate_score_eval(args.source_run_dir, args.output_dir))


def _audit_candidate(args: argparse.Namespace) -> None:
    _print(audit_candidate_score_eval(args.run_dir))


def _run_candidate(args: argparse.Namespace) -> None:
    _print(
        run_candidate_score_server(
            args.run_dir,
            endpoint=args.endpoint,
            model=args.model,
            timeout_s=args.timeout,
            retries=args.retries,
            seed=args.seed,
            limit=args.limit,
        )
    )


def _aggregate_candidate(args: argparse.Namespace) -> None:
    _print(aggregate_candidate_scores(args.run_dir))


def _prepare_binary(args: argparse.Namespace) -> None:
    _print(prepare_binary_hierarchy_eval(args.source_run_dir, args.output_dir))


def _audit_binary(args: argparse.Namespace) -> None:
    _print(audit_binary_hierarchy_eval(args.run_dir))


def _run_binary(args: argparse.Namespace) -> None:
    _print(
        run_binary_hierarchy_server(
            args.run_dir,
            endpoint=args.endpoint,
            model=args.model,
            timeout_s=args.timeout,
            retries=args.retries,
            seed=args.seed,
            limit=args.limit,
        )
    )


def _aggregate_binary(args: argparse.Namespace) -> None:
    _print(aggregate_binary_hierarchy(args.run_dir))


def _run_server(args: argparse.Namespace) -> None:
    _print(
        run_event_server(
            args.run_dir,
            endpoint=args.endpoint,
            model=args.model,
            timeout_s=args.timeout,
            retries=args.retries,
            seed=args.seed,
            limit=args.limit,
            structured_output=not args.unstructured_output,
            max_tokens=args.max_tokens,
        )
    )


def _score(args: argparse.Namespace) -> None:
    _print(
        score_event_eval(
            args.run_dir,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )
    )


def _prepare_control(args: argparse.Namespace) -> None:
    _print(
        prepare_silenced_audio_control(
            args.main_run_dir,
            args.output_dir,
            samples=args.samples,
            seed=args.seed,
        )
    )


def _run_control(args: argparse.Namespace) -> None:
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


def _score_control(args: argparse.Namespace) -> None:
    _print(score_silenced_audio_control(args.main_run_dir, args.control_run_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict-causal Qwen2.5-Omni-3B SBCSAE event evaluation"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument(
        "--manifest",
        default="data/processed/sbcsae_turn_events_v1/annotation_manifest.jsonl",
    )
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--phase", choices=("micro", "gate50", "formal500"), required=True)
    prepare.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    prepare.add_argument(
        "--prompt-style",
        choices=("direct", "decision_tree", "reasoned", "hierarchical"),
    )
    prepare.add_argument("--selection-seed", type=int)
    prepare.add_argument("--gate-run-dir")
    prepare.add_argument("--audio-control-dir")
    prepare.set_defaults(func=_prepare)

    audit = commands.add_parser("audit")
    audit.add_argument("--run-dir", required=True)
    audit.set_defaults(func=_audit)

    apply_asr = commands.add_parser("apply-causal-asr")
    apply_asr.add_argument("--run-dir", required=True)
    apply_asr.add_argument("--asr-path")
    apply_asr.set_defaults(func=_apply_asr)

    prepare_candidate = commands.add_parser("prepare-candidate-scores")
    prepare_candidate.add_argument("--source-run-dir", required=True)
    prepare_candidate.add_argument("--output-dir", required=True)
    prepare_candidate.set_defaults(func=_prepare_candidate)

    audit_candidate = commands.add_parser("audit-candidate-scores")
    audit_candidate.add_argument("--run-dir", required=True)
    audit_candidate.set_defaults(func=_audit_candidate)

    run_candidate = commands.add_parser("run-candidate-scores")
    run_candidate.add_argument("--run-dir", required=True)
    run_candidate.add_argument(
        "--endpoint", default="http://127.0.0.1:8091/v1/chat/completions"
    )
    run_candidate.add_argument("--model", default="Qwen2.5-Omni-3B-Q8_0")
    run_candidate.add_argument("--timeout", type=float, default=180.0)
    run_candidate.add_argument("--retries", type=int, default=2)
    run_candidate.add_argument("--seed", type=int, default=13)
    run_candidate.add_argument("--limit", type=int)
    run_candidate.set_defaults(func=_run_candidate)

    aggregate_candidate = commands.add_parser("aggregate-candidate-scores")
    aggregate_candidate.add_argument("--run-dir", required=True)
    aggregate_candidate.set_defaults(func=_aggregate_candidate)

    prepare_binary = commands.add_parser("prepare-binary-hierarchy")
    prepare_binary.add_argument("--source-run-dir", required=True)
    prepare_binary.add_argument("--output-dir", required=True)
    prepare_binary.set_defaults(func=_prepare_binary)

    audit_binary = commands.add_parser("audit-binary-hierarchy")
    audit_binary.add_argument("--run-dir", required=True)
    audit_binary.set_defaults(func=_audit_binary)

    run_binary = commands.add_parser("run-binary-hierarchy")
    run_binary.add_argument("--run-dir", required=True)
    run_binary.add_argument(
        "--endpoint", default="http://127.0.0.1:8091/v1/chat/completions"
    )
    run_binary.add_argument("--model", default="Qwen2.5-Omni-3B-Q8_0")
    run_binary.add_argument("--timeout", type=float, default=180.0)
    run_binary.add_argument("--retries", type=int, default=2)
    run_binary.add_argument("--seed", type=int, default=13)
    run_binary.add_argument("--limit", type=int)
    run_binary.set_defaults(func=_run_binary)

    aggregate_binary = commands.add_parser("aggregate-binary-hierarchy")
    aggregate_binary.add_argument("--run-dir", required=True)
    aggregate_binary.set_defaults(func=_aggregate_binary)

    run_server = commands.add_parser("run-server")
    run_server.add_argument("--run-dir", required=True)
    run_server.add_argument(
        "--endpoint", default="http://127.0.0.1:8091/v1/chat/completions"
    )
    run_server.add_argument("--model", default="Qwen2.5-Omni-3B-Q8_0")
    run_server.add_argument("--timeout", type=float, default=180.0)
    run_server.add_argument("--retries", type=int, default=2)
    run_server.add_argument("--seed", type=int, default=13)
    run_server.add_argument("--limit", type=int)
    run_server.add_argument("--unstructured-output", action="store_true")
    run_server.add_argument("--max-tokens", type=int, default=16)
    run_server.set_defaults(func=_run_server)

    score = commands.add_parser("score")
    score.add_argument("--run-dir", required=True)
    score.add_argument("--bootstrap-resamples", type=int, default=2000)
    score.add_argument("--seed", type=int, default=13)
    score.set_defaults(func=_score)

    prepare_control = commands.add_parser("prepare-audio-control")
    prepare_control.add_argument("--main-run-dir", required=True)
    prepare_control.add_argument("--output-dir", required=True)
    prepare_control.add_argument("--samples", type=int, default=10)
    prepare_control.add_argument("--seed", type=int, default=29)
    prepare_control.set_defaults(func=_prepare_control)

    run_control = commands.add_parser("run-audio-control")
    run_control.add_argument("--control-run-dir", required=True)
    run_control.add_argument(
        "--endpoint", default="http://127.0.0.1:8091/v1/chat/completions"
    )
    run_control.add_argument("--model", default="qwen2.5-omni-3b-q4_k_m")
    run_control.add_argument("--timeout", type=float, default=180.0)
    run_control.add_argument("--retries", type=int, default=2)
    run_control.add_argument("--seed", type=int, default=13)
    run_control.add_argument("--limit", type=int)
    run_control.set_defaults(func=_run_control)

    score_control = commands.add_parser("score-audio-control")
    score_control.add_argument("--main-run-dir", required=True)
    score_control.add_argument("--control-run-dir", required=True)
    score_control.set_defaults(func=_score_control)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
