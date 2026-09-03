from __future__ import annotations

import argparse
import json

from profile_turntaking.talking_turns_prompt_pilot import (
    audit_semantic_requests,
    prepare_probability_requests,
    prepare_semantic_requests,
    run_probability_requests,
    run_semantic_requests,
    score_probability_requests,
    score_semantic_requests,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the paper-aligned semantic-label prompt diagnostic."
    )
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "audit",
            "run",
            "score",
            "prepare-probability",
            "run-probability",
            "score-probability",
        ),
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:8091/v1/chat/completions"
    )
    parser.add_argument("--model", default="Qwen2.5-Omni-3B-Q8_0")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_semantic_requests(args.run_dir)
    elif args.command == "audit":
        result = audit_semantic_requests(args.run_dir)
    elif args.command == "run":
        result = run_semantic_requests(
            args.run_dir, endpoint=args.endpoint, model=args.model
        )
    elif args.command == "score":
        result = score_semantic_requests(args.run_dir)
    elif args.command == "prepare-probability":
        result = prepare_probability_requests(args.run_dir)
    elif args.command == "run-probability":
        result = run_probability_requests(
            args.run_dir, endpoint=args.endpoint, model=args.model
        )
    else:
        result = score_probability_requests(args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
