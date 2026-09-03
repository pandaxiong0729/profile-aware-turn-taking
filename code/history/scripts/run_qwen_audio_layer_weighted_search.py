"""End-to-end Qwen multi-layer audio boundary + profile adapter experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(command: list[str], *, repo_root: Path, log_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root / "code" / "src")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        subprocess.run(
            command,
            cwd=repo_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--base-cache-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument("--adapter-device", default="cuda")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_root = (repo_root / args.output_root).resolve()
    cache_dir = output_root / "cache"
    grid_dir = output_root / "adapter-search"
    status_path = output_root / "status.json"
    output_root.mkdir(parents=True, exist_ok=True)

    write_json(status_path, {"phase": "extract_qwen_audio_layers"})
    run(
        [
            sys.executable,
            str(repo_root / "code" / "scripts" / "build_qwen_audio_layer_caches.py"),
            "--data-dir",
            str((repo_root / args.data_dir).resolve()),
            "--base-cache-dir",
            str((repo_root / args.base_cache_dir).resolve()),
            "--output-dir",
            str(cache_dir),
            "--model-name",
            str((repo_root / args.model_name).resolve()),
            "--device",
            "cuda",
            "--encoder-batch-size",
            str(args.encoder_batch_size),
            "--checkpoint-every",
            "100",
        ],
        repo_root=repo_root,
        log_path=output_root / "logs" / "01-audio-layer-cache.log",
    )

    write_json(status_path, {"phase": "validation_search_and_structural_test"})
    run(
        [
            sys.executable,
            str(repo_root / "code" / "scripts" / "run_qwen_shared_ab_validation_grid.py"),
            "--cache-dir",
            str(cache_dir),
            "--output-dir",
            str(grid_dir),
            "--adapter-device",
            args.adapter_device,
            "--validation-seeds",
            "13",
            "37",
            "71",
            "--final-seeds",
            "3",
            "13",
            "37",
            "71",
            "101",
        ],
        repo_root=repo_root,
        log_path=output_root / "logs" / "02-adapter-search.log",
    )
    grid_summary = json.loads((grid_dir / "grid_and_final_summary.json").read_text(encoding="utf-8"))
    final_summary = {
        "experiment": "qwen_audio_layer_weighted_shared_ab_adapter",
        "input_contract": "30 s causal audio + matching causal partial transcript + profile",
        "architecture": (
            "Frozen Qwen prompt-last transcript/context vector + trainable softmax weighting over "
            "33 frozen Qwen audio-tower boundary layers + shared profile adapter + four A/B heads"
        ),
        "selection": "adapter family and epoch selected on validation only",
        "test_disclosure": (
            "This is a second-stage structural evaluation on the same held-out conversations after "
            "the prior last-layer family failed; report it as confirmatory/iterative, not as an untouched first test."
        ),
        "result": grid_summary,
        "goal_met": bool(grid_summary.get("goal_met")),
    }
    write_json(output_root / "FINAL_SUMMARY.json", final_summary)
    write_json(
        status_path,
        {
            "phase": "complete" if final_summary["goal_met"] else "complete_unmet",
            "summary": str(output_root / "FINAL_SUMMARY.json"),
        },
    )
    print(json.dumps(final_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
