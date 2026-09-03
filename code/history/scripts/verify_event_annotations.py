from __future__ import annotations

import argparse
import json

from profile_turntaking.event_annotation import verify_event_annotation_package


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify event manifest, target positions, audio files, and hashes"
    )
    parser.add_argument(
        "--output-dir", default="data/processed/sbcsae_turn_events_v1"
    )
    args = parser.parse_args()
    result = verify_event_annotation_package(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
