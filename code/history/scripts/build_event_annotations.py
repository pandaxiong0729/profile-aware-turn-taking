from __future__ import annotations

import argparse
import json
from pathlib import Path

from profile_turntaking.event_annotation import (
    build_sbcsae_event_annotations,
    build_static_review_site,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build event-centred SBCSAE C/BC/T/I/NA candidates and human-review audio"
    )
    parser.add_argument(
        "--catalog-dir", default="data/processed/sbcsae_catalog_v2"
    )
    parser.add_argument("--vad-dir", default="data/processed/sbcsae_vad_v1")
    parser.add_argument(
        "--output-dir", default="data/processed/sbcsae_turn_events_v1"
    )
    parser.add_argument("--context-before-s", type=float, default=6.0)
    parser.add_argument("--context-after-s", type=float, default=4.0)
    parser.add_argument("--ipu-silence-ms", type=int, default=200)
    parser.add_argument("--minimum-silence-event-ms", type=int, default=200)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument(
        "--overwrite-audio",
        action="store_true",
        help="Regenerate every review WAV even when a file with the same event id exists",
    )
    args = parser.parse_args()
    summary = build_sbcsae_event_annotations(
        catalog_dir=Path(args.catalog_dir),
        vad_dir=Path(args.vad_dir),
        output_dir=Path(args.output_dir),
        context_before_s=args.context_before_s,
        context_after_s=args.context_after_s,
        ipu_silence_ms=args.ipu_silence_ms,
        minimum_silence_event_ms=args.minimum_silence_event_ms,
        generate_audio=not args.no_audio,
        overwrite_audio=args.overwrite_audio,
    )
    if not args.no_audio:
        summary["review_site"] = build_static_review_site(Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
