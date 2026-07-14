from __future__ import annotations

import argparse
import json

from profile_turntaking.experiment_preflight import audit_prompt_pilot_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep-audit the 500-event MLLM prompt pilot")
    parser.add_argument("--catalog-dir", required=True)
    parser.add_argument("--event-manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_prompt_pilot_data(
        catalog_dir=args.catalog_dir,
        event_manifest=args.event_manifest,
        run_dir=args.run_dir,
        output_path=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
