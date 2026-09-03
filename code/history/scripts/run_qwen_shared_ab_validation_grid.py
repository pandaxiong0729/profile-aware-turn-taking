from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


CONFIGS: list[dict[str, Any]] = [
    {
        "name": "gate_base",
        "fusion": "gate",
        "profile_dropout": 0.25,
    },
    {
        "name": "concat_base",
        "fusion": "concat",
        "profile_dropout": 0.25,
    },
    {
        "name": "film_base",
        "fusion": "film",
        "profile_dropout": 0.25,
    },
    {
        "name": "gate_margin",
        "fusion": "gate",
        "profile_dropout": 0.25,
        "hidden_ce_weight": 0.50,
        "hidden_margin_weight": 0.10,
        "control_margin_weight": 0.10,
        "margin": 0.05,
        "selection_delta_weight": 0.25,
        "selection_min_delta_weight": 0.10,
    },
    {
        "name": "concat_margin",
        "fusion": "concat",
        "profile_dropout": 0.25,
        "hidden_ce_weight": 0.50,
        "hidden_margin_weight": 0.10,
        "control_margin_weight": 0.10,
        "margin": 0.05,
        "selection_delta_weight": 0.25,
        "selection_min_delta_weight": 0.10,
    },
    {
        "name": "film_margin",
        "fusion": "film",
        "profile_dropout": 0.25,
        "hidden_ce_weight": 0.50,
        "hidden_margin_weight": 0.10,
        "control_margin_weight": 0.10,
        "margin": 0.05,
        "selection_delta_weight": 0.25,
        "selection_min_delta_weight": 0.10,
    },
]

TASKWISE_CONFIGS: list[dict[str, Any]] = [
    {
        **config,
        "name": f"taskwise_{config['name']}",
        "task_specific_branches": True,
    }
    for config in CONFIGS
    if str(config["name"]).endswith("_margin")
]

# The base margin grid uses a deliberately weak shuffled-profile constraint.
# These configurations test whether explicit correct-vs-wrong profile pairing
# is underweighted, while leaving the model, data, and test protocol unchanged.
PROFILE_MARGIN_CONFIGS: list[dict[str, Any]] = [
    {
        "name": f"gate_profile_margin_{str(weight).replace('.', 'p')}",
        "fusion": "gate",
        "profile_dropout": 0.25,
        "hidden_ce_weight": 0.50,
        "hidden_margin_weight": 0.10,
        "control_margin_weight": weight,
        "margin": 0.05,
        "selection_delta_weight": 0.25,
        "selection_min_delta_weight": 0.10,
    }
    for weight in (0.25, 0.50, 1.00)
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def complete_caches(cache_dir: Path) -> dict[str, Path] | None:
    caches = {split: cache_dir / f"{split}.qwen-hidden.npz" for split in ("train", "val", "test")}
    if all(path.is_file() and path.with_suffix(".meta.json").is_file() for path in caches.values()):
        return caches
    return None


def wait_for_caches(cache_dir: Path, *, timeout_hours: float, poll_seconds: int) -> dict[str, Path]:
    deadline = time.monotonic() + timeout_hours * 3600.0
    while time.monotonic() < deadline:
        caches = complete_caches(cache_dir)
        if caches is not None:
            return caches
        time.sleep(max(5, poll_seconds))
    raise TimeoutError(f"Timed out waiting for complete Qwen caches in {cache_dir}")


def command_for_config(
    *,
    script: Path,
    caches: dict[str, Path],
    output_dir: Path,
    config: dict[str, Any],
    seeds: list[int],
    validation_only: bool,
    device: str = "cpu",
) -> list[str]:
    command = [
        sys.executable,
        str(script),
        "--train-cache",
        str(caches["train"]),
        "--val-cache",
        str(caches["val"]),
        "--test-cache",
        str(caches["val"] if validation_only else caches["test"]),
        "--output-dir",
        str(output_dir),
        "--task-scheme",
        "paper",
        "--device",
        device,
        "--seeds",
        *[str(seed) for seed in seeds],
        "--fusion",
        str(config["fusion"]),
        "--profile-dropout",
        str(config.get("profile_dropout", 0.25)),
        "--hidden-dim",
        "256",
        "--dropout",
        "0.15",
        "--epochs",
        "100",
        "--patience",
        "14",
        "--batch-size",
        "128",
        "--lr",
        "0.0008",
        "--weight-decay",
        "0.0001",
        "--shuffled-strategy",
        "random",
    ]
    optional = {
        "hidden_ce_weight": "--hidden-ce-weight",
        "control_ce_weight": "--control-ce-weight",
        "hidden_margin_weight": "--hidden-margin-weight",
        "control_margin_weight": "--control-margin-weight",
        "margin": "--margin",
        "selection_delta_weight": "--selection-delta-weight",
        "selection_min_delta_weight": "--selection-min-delta-weight",
    }
    for key, flag in optional.items():
        if key in config:
            command.extend([flag, str(config[key])])
    if bool(config.get("task_specific_branches", False)):
        command.append("--task-specific-branches")
    if config.get("profile_view_dir"):
        command.extend(["--profile-view-dir", str(config["profile_view_dir"])])
    return command


def run_command(command: list[str], *, log_path: Path, repo_root: Path) -> None:
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


def result_row(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    overall = summary["aggregate"]["overall"]
    given = float(overall["given"]["paper_balanced_accuracy_mean"])
    hidden = float(overall["hidden"]["paper_balanced_accuracy_mean"])
    shuffled = float(overall["shuffled"]["paper_balanced_accuracy_mean"])
    return {
        "name": name,
        "given_accuracy": given,
        "hidden_accuracy": hidden,
        "shuffled_accuracy": shuffled,
        "given_minus_hidden": given - hidden,
        "given_minus_shuffled": given - shuffled,
        "profile_order_pass": given > hidden and given > shuffled,
        "accuracy_target_pass": given >= 0.73,
    }


def selection_key(row: dict[str, Any]) -> tuple[float, float, float]:
    minimum_delta = min(float(row["given_minus_hidden"]), float(row["given_minus_shuffled"]))
    tier = (
        2.0
        if bool(row["profile_order_pass"] and row["accuracy_target_pass"])
        else (1.0 if bool(row["profile_order_pass"]) else 0.0)
    )
    return (
        tier,
        float(row["given_accuracy"]),
        minimum_delta,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only grid and one final Qwen shared A/B test.")
    parser.add_argument("--cache-dir", default="artifacts/qwen-shared-ab-30s-causal/full-cache")
    parser.add_argument("--output-dir", default="artifacts/qwen-shared-ab-30s-causal/experiments")
    parser.add_argument("--wait-for-caches", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=8.0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--skip-final-test",
        action="store_true",
        help="Select on validation and stop, leaving the test cache untouched.",
    )
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=[13, 37, 71])
    parser.add_argument("--final-seeds", type=int, nargs="+", default=[3, 13, 37, 71, 101])
    parser.add_argument("--adapter-device", default="cpu")
    parser.add_argument(
        "--config-family",
        choices=["shared", "shared-margin", "taskwise", "profile-margin"],
        default="shared",
        help="Choose the original shared branches or task-specific audio/profile branches.",
    )
    parser.add_argument(
        "--profile-view-dir",
        default="",
        help="Optional aligned profile-view sidecar directory passed to every configuration.",
    )
    args = parser.parse_args()

    if args.config_family == "taskwise":
        configs = TASKWISE_CONFIGS
    elif args.config_family == "profile-margin":
        configs = PROFILE_MARGIN_CONFIGS
    elif args.config_family == "shared-margin":
        configs = [config for config in CONFIGS if str(config["name"]).endswith("_margin")]
    else:
        configs = CONFIGS
    if args.profile_view_dir:
        profile_view_dir = (Path(__file__).resolve().parents[2] / args.profile_view_dir).resolve()
        configs = [{**config, "profile_view_dir": str(profile_view_dir)} for config in configs]

    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = (repo_root / args.cache_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    script = repo_root / "code" / "scripts" / "run_qwen_shared_binary_multitask_adapter.py"
    caches = (
        wait_for_caches(cache_dir, timeout_hours=args.timeout_hours, poll_seconds=args.poll_seconds)
        if args.wait_for_caches
        else complete_caches(cache_dir)
    )
    if caches is None:
        raise FileNotFoundError(f"Incomplete caches in {cache_dir}")

    rows: list[dict[str, Any]] = []
    for config in configs:
        config_dir = output_dir / "validation-grid" / str(config["name"])
        command = command_for_config(
            script=script,
            caches=caches,
            output_dir=config_dir,
            config=config,
            seeds=list(args.validation_seeds),
            validation_only=True,
            device=args.adapter_device,
        )
        run_command(command, log_path=config_dir / "run.log", repo_root=repo_root)
        summary = json.loads((config_dir / "summary.json").read_text(encoding="utf-8"))
        row = result_row(str(config["name"]), summary)
        row["config"] = config
        rows.append(row)
        write_json(output_dir / "validation-grid-progress.json", rows)

    selected = max(rows, key=selection_key)
    selected_config = next(config for config in configs if config["name"] == selected["name"])
    final_dir: Path | None = None
    final_row: dict[str, Any] | None = None
    if not args.skip_final_test:
        final_dir = output_dir / "final" / str(selected["name"])
        final_command = command_for_config(
            script=script,
            caches=caches,
            output_dir=final_dir,
            config=selected_config,
            seeds=list(args.final_seeds),
            validation_only=False,
            device=args.adapter_device,
        )
        run_command(final_command, log_path=final_dir / "run.log", repo_root=repo_root)
        final_summary = json.loads((final_dir / "summary.json").read_text(encoding="utf-8"))
        final_row = result_row(str(selected["name"]), final_summary)

    report = {
        "protocol": {
            "config_family": args.config_family,
            "architecture_selection": "validation only; the val cache is passed as the grid evaluation split",
            "final_test": (
                "not run; this invocation leaves test untouched"
                if args.skip_final_test
                else "one selected configuration is retrained and evaluated on the untouched test cache"
            ),
            "paper_comparable_metric": "accuracy on deterministic per-task 50/50 A/B subsets",
            "target": {
                "given_greater_than_hidden": True,
                "given_greater_than_shuffled": True,
                "mean_binary_accuracy_at_least_or_near": 0.73,
            },
        },
        "validation_rows": rows,
        "selected_from_validation": selected,
        "final_test": final_row,
        "goal_met": (
            None
            if final_row is None
            else bool(final_row["profile_order_pass"] and final_row["accuracy_target_pass"])
        ),
        "next_action_if_unmet": "Run paper-inspired learned layer weighting / stronger acoustic representation; do not reinterpret validation as test.",
        "final_result_dir": str(final_dir) if final_dir is not None else None,
    }
    write_json(output_dir / "grid_and_final_summary.json", report)
    with (output_dir / "validation_grid.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "given_accuracy",
                "hidden_accuracy",
                "shuffled_accuracy",
                "given_minus_hidden",
                "given_minus_shuffled",
                "profile_order_pass",
                "accuracy_target_pass",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
