from __future__ import annotations

import argparse
import json
from pathlib import Path

from profile_turntaking.mllm_annotation import (
    analyze_annotation_pilot,
    prepare_annotation_pilot,
    run_annotation_requests,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrospective audio-MLLM event annotation")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument(
        "--events", default="data/processed/sbcsae_vad_v1/semantic_review_queue.jsonl"
    )
    prepare.add_argument(
        "--conversations",
        default="data/processed/sbcsae_catalog_v2/conversations.jsonl",
    )
    prepare.add_argument(
        "--utterances",
        default="data/processed/sbcsae_catalog_v2/utterances.jsonl",
    )
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--per-group", type=int, default=20)
    prepare.add_argument("--seed", type=int, default=41)

    run = commands.add_parser("run")
    run.add_argument("--run-dir", required=True)
    run.add_argument(
        "--endpoint", default="http://127.0.0.1:8091/v1/chat/completions"
    )
    run.add_argument("--model", default="qwen2.5-omni-3b-q4_k_m")
    run.add_argument("--timeout", type=float, default=180.0)
    run.add_argument("--retries", type=int, default=2)
    run.add_argument("--seed", type=int, default=41)
    run.add_argument("--limit", type=int)

    analyze = commands.add_parser("analyze")
    analyze.add_argument("--run-dir", required=True)
    analyze.add_argument("--confidence-threshold", type=float, default=0.75)

    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_annotation_pilot(
            events_path=args.events,
            conversations_path=args.conversations,
            utterances_path=args.utterances,
            output_dir=args.output_dir,
            per_group=args.per_group,
            seed=args.seed,
        )
    elif args.command == "run":
        root = Path(args.run_dir)
        result = run_annotation_requests(
            requests_path=root / "requests.jsonl",
            responses_path=root / "responses.jsonl",
            endpoint=args.endpoint,
            model=args.model,
            timeout_s=args.timeout,
            retries=args.retries,
            seed=args.seed,
            limit=args.limit,
        )
    else:
        result = analyze_annotation_pilot(
            args.run_dir, confidence_threshold=args.confidence_threshold
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
