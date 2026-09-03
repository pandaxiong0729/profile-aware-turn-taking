from __future__ import annotations

import argparse
import json

from profile_turntaking.protocol_lock import verify_prompt_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the locked MLLM pilot protocol")
    parser.add_argument(
        "--protocol",
        default="code/configs/mllm_prompt_pilot_locked.json",
    )
    parser.add_argument("--run-dir")
    parser.add_argument("--output")
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    report = verify_prompt_protocol(
        args.protocol,
        run_dir=args.run_dir,
        output_path=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["protocol_consistent"]:
        raise SystemExit(2)
    if args.require_ready and not report["ready_for_inference"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
