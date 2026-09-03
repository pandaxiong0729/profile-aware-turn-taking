from __future__ import annotations

import argparse
import json

from profile_turntaking.vad_annotation import VadConfig, build_vad_annotations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build acoustic-first SBCSAE C/NA labels and semantic review events"
    )
    parser.add_argument(
        "--catalog-dir", default="data/processed/sbcsae_catalog_v2"
    )
    parser.add_argument(
        "--scope-summary", default="data/processed/sbcsae_mvp_v2/summary.json"
    )
    parser.add_argument(
        "--output-dir", default="data/processed/sbcsae_vad_v1"
    )
    parser.add_argument("--max-conversations", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    summary = build_vad_annotations(
        catalog_dir=args.catalog_dir,
        scope_summary=args.scope_summary,
        output_dir=args.output_dir,
        config=VadConfig(threshold=args.threshold),
        max_conversations=args.max_conversations,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
