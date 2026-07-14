from __future__ import annotations

import argparse
import json

from profile_turntaking.label_review import apply_reviewed_labels, build_review_page


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/apply the SBCSAE event-label review")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--run-dir", required=True)
    build.add_argument("--source-manifest")
    build.add_argument("--catalog-dir")
    apply = subparsers.add_parser("apply")
    apply.add_argument("--source-manifest", required=True)
    apply.add_argument("--review-json", required=True)
    apply.add_argument("--output-manifest", required=True)
    apply.add_argument("--reviewer-id", required=True)
    args = parser.parse_args()
    if args.command == "build":
        report = build_review_page(
            args.run_dir,
            source_manifest=args.source_manifest,
            catalog_dir=args.catalog_dir,
        )
    else:
        report = apply_reviewed_labels(
            args.source_manifest,
            args.review_json,
            args.output_manifest,
            reviewer_id=args.reviewer_id,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
