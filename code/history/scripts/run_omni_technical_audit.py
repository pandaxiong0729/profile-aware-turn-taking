from __future__ import annotations

import argparse
import json

from profile_turntaking.omni_technical_audit import (
    audit_requests,
    prepare_audit,
    render_frontend,
    run_audit,
    score_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Qwen2.5-Omni-3B turn-taking technical audit."
    )
    parser.add_argument(
        "command", choices=("prepare", "audit", "run", "score", "render", "all")
    )
    parser.add_argument(
        "--source-run-dir",
        default=(
            "artifacts/talking-turns-paper-aligned/qwen2.5-omni-3b-q8/"
            "hidden50-v3-reasoned"
        ),
    )
    parser.add_argument(
        "--catalog-dir", default="data/processed/sbcsae_catalog_v2"
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/omni-technical-audit/qwen2.5-omni-3b-q8/"
            "audit8-20260731"
        ),
    )
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:8091/v1/chat/completions"
    )
    parser.add_argument("--model", default="Qwen2.5-Omni-3B-Q8_0")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    args = parser.parse_args()

    result: object
    if args.command in {"prepare", "all"}:
        result = prepare_audit(
            source_run_dir=args.source_run_dir,
            catalog_dir=args.catalog_dir,
            output_dir=args.output_dir,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if args.command == "prepare":
            return
    if args.command == "audit":
        result = audit_requests(args.output_dir)
    elif args.command == "run":
        result = run_audit(
            args.output_dir,
            endpoint=args.endpoint,
            model=args.model,
            timeout_s=args.timeout_s,
        )
    elif args.command == "score":
        result = score_audit(args.output_dir)
    elif args.command == "render":
        result = {"review_html": str(render_frontend(args.output_dir))}
    elif args.command == "all":
        inference = run_audit(
            args.output_dir,
            endpoint=args.endpoint,
            model=args.model,
            timeout_s=args.timeout_s,
        )
        metrics = score_audit(args.output_dir)
        review_html = render_frontend(args.output_dir)
        result = {
            "inference": inference,
            "metrics": metrics,
            "review_html": str(review_html),
        }
    else:
        result = {}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
