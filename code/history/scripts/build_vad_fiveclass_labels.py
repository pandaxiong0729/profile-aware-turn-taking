from __future__ import annotations

import argparse
import json

from profile_turntaking.vad_fiveclass import build_forced_fiveclass_labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Force C/BC/T/I/NA labels for every existing SBCSAE VAD frame"
    )
    parser.add_argument(
        "--vad-frames",
        default="data/processed/sbcsae_vad_v1/frame_annotations.jsonl.gz",
    )
    parser.add_argument(
        "--vad-summary", default="data/processed/sbcsae_vad_v1/summary.json"
    )
    parser.add_argument(
        "--utterances",
        default="data/processed/sbcsae_catalog_v2/utterances.jsonl",
    )
    parser.add_argument(
        "--output-dir", default="data/processed/sbcsae_vad_fiveclass_v2"
    )
    parser.add_argument("--minimum-overlap-ms", type=int, default=40)
    args = parser.parse_args()
    summary = build_forced_fiveclass_labels(
        vad_frames_path=args.vad_frames,
        vad_summary_path=args.vad_summary,
        utterances_path=args.utterances,
        output_dir=args.output_dir,
        minimum_overlap_ms=args.minimum_overlap_ms,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
