from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from run_qwen_shared_ab_validation_grid import (
    command_for_config,
    result_row,
    run_command,
    selection_key,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_logged(command: list[str], *, log_path: Path, repo_root: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root / "code" / "src")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        subprocess.run(
            command,
            cwd=repo_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )


def complete_cache_mapping(cache_dir: Path) -> dict[str, Path]:
    mapping = {split: cache_dir / f"{split}.qwen-hidden.npz" for split in ("train", "val", "test")}
    for path in mapping.values():
        if not path.is_file() or not path.with_suffix(".meta.json").is_file():
            raise FileNotFoundError(f"Incomplete cache view: {path}")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one rich Qwen cache, compare paper-aligned context views on validation, "
            "then evaluate one selected model on test."
        )
    )
    parser.add_argument("--data-dir", default="data/processed/sbcsae_qwen_shared_ab_30s_causal_v1")
    parser.add_argument("--model-name", default="models/qwen2.5-omni-3b-local")
    parser.add_argument("--output-root", default="artifacts/qwen-shared-ab-30s-causal/paper-aligned-search")
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument("--adapter-device", default="cuda")
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=[13, 37, 71])
    parser.add_argument("--final-seeds", type=int, nargs="+", default=[3, 13, 37, 71, 101])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_root = (repo_root / args.output_root).resolve()
    rich_cache_dir = output_root / "rich-cache"
    view_root = output_root / "cache-views"
    validation_root = output_root / "validation"
    final_root = output_root / "final"
    status_path = output_root / "status.json"
    qwen_script = repo_root / "code" / "scripts" / "run_qwen_hidden_profile_experiment.py"
    derive_script = repo_root / "code" / "scripts" / "derive_qwen_context_views.py"
    grid_script = repo_root / "code" / "scripts" / "run_qwen_shared_ab_validation_grid.py"
    adapter_script = repo_root / "code" / "scripts" / "run_qwen_shared_binary_multitask_adapter.py"

    write_json(status_path, {"phase": "extract_rich_qwen_cache"})
    cache_command = [
        sys.executable,
        str(qwen_script),
        "full",
        "--data-dir",
        str((repo_root / args.data_dir).resolve()),
        "--cache-dir",
        str(rich_cache_dir),
        "--output-dir",
        str(output_root / "unused-5way-output"),
        "--model-name",
        str((repo_root / args.model_name).resolve()),
        "--context-mode",
        "audio_transcript",
        "--context-pooling",
        "rich",
        "--encoder-batch-size",
        str(args.encoder_batch_size),
        "--checkpoint-every",
        "100",
        "--cache-only",
        "--local-files-only",
        "--torch-dtype",
        "auto",
        "--device-map",
        "auto",
        "--offload-folder",
        str(output_root / "offload"),
    ]
    run_logged(cache_command, log_path=output_root / "logs" / "01-rich-cache.log", repo_root=repo_root)

    write_json(status_path, {"phase": "derive_context_views"})
    run_logged(
        [
            sys.executable,
            str(derive_script),
            "--rich-cache-dir",
            str(rich_cache_dir),
            "--output-root",
            str(view_root),
        ],
        log_path=output_root / "logs" / "02-derive-views.log",
        repo_root=repo_root,
    )
    view_mapping: dict[str, str] = json.loads(
        (view_root / "context_views.json").read_text(encoding="utf-8")
    )

    candidates: list[dict[str, Any]] = []
    for index, (view, cache_dir_text) in enumerate(view_mapping.items(), start=1):
        write_json(
            status_path,
            {
                "phase": "validation_search",
                "view": view,
                "view_index": index,
                "view_total": len(view_mapping),
            },
        )
        view_output = validation_root / view
        run_logged(
            [
                sys.executable,
                str(grid_script),
                "--cache-dir",
                cache_dir_text,
                "--output-dir",
                str(view_output),
                "--skip-final-test",
                "--adapter-device",
                args.adapter_device,
                "--validation-seeds",
                *[str(seed) for seed in args.validation_seeds],
            ],
            log_path=output_root / "logs" / f"validation-{view}.log",
            repo_root=repo_root,
        )
        report = json.loads(
            (view_output / "grid_and_final_summary.json").read_text(encoding="utf-8")
        )
        selected = dict(report["selected_from_validation"])
        selected["context_view"] = view
        selected["cache_dir"] = cache_dir_text
        candidates.append(selected)
        write_json(output_root / "validation_candidates.json", candidates)

    selected = max(candidates, key=selection_key)
    selected_caches = complete_cache_mapping(Path(selected["cache_dir"]))
    selected_config = dict(selected["config"])
    final_dir = final_root / str(selected["context_view"]) / str(selected["name"])
    write_json(
        status_path,
        {
            "phase": "final_test",
            "selected_context_view": selected["context_view"],
            "selected_adapter": selected["name"],
        },
    )
    final_command = command_for_config(
        script=adapter_script,
        caches=selected_caches,
        output_dir=final_dir,
        config=selected_config,
        seeds=list(args.final_seeds),
        validation_only=False,
        device=args.adapter_device,
    )
    run_command(final_command, log_path=final_dir / "run.log", repo_root=repo_root)
    final_summary = json.loads((final_dir / "summary.json").read_text(encoding="utf-8"))
    final_row = result_row(str(selected["name"]), final_summary)
    final_row["context_view"] = selected["context_view"]

    report = {
        "protocol": {
            "input": "30 s causal audio + matching causal transcript + profile",
            "profile_control": "hidden/given/shuffled change only profile vectors",
            "representation_and_adapter_selection": "validation only",
            "test_access": "one selected context view and adapter configuration evaluated once",
            "metric": "mean accuracy over four deterministic balanced A/B subsets",
            "target": "given > hidden, given > shuffled, and given accuracy >= or near 0.73",
        },
        "validation_candidates": candidates,
        "selected_from_validation": selected,
        "final_test": final_row,
        "goal_met_strict": bool(
            final_row["given_accuracy"] >= 0.73
            and final_row["given_accuracy"] > final_row["hidden_accuracy"]
            and final_row["given_accuracy"] > final_row["shuffled_accuracy"]
        ),
        "goal_near_73": bool(
            final_row["given_accuracy"] >= 0.70
            and final_row["given_accuracy"] > final_row["hidden_accuracy"]
            and final_row["given_accuracy"] > final_row["shuffled_accuracy"]
        ),
        "final_result_dir": str(final_dir),
    }
    write_json(output_root / "FINAL_SUMMARY.json", report)
    write_json(status_path, {"phase": "complete", "summary": str(output_root / "FINAL_SUMMARY.json")})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
