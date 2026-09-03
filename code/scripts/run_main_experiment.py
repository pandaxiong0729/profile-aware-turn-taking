"""Single entry point for the collaborator-facing main experiment.

Commands:
  check        Verify required files, paired input invariants, and five-class labels.
  train        Re-run the validation-selected shared gated A/B adapter.
  show-results Print the canonical result table from the saved summary.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data/processed/sbcsae_qwen_shared_ab_30s_causal_v1"
CACHE_DIR = REPO_ROOT / "artifacts/main_experiment/qwen_feature_cache"
PROFILE_VIEW_DIR = REPO_ROOT / "artifacts/main_experiment/profile_features"
CANONICAL_RESULT_DIR = REPO_ROOT / "artifacts/main_experiment/results"

PROFILE_ONLY_FIELDS = {
    "request_id",
    "profile_mode",
    "profile_text",
    "profile_sha256",
    "prompt",
    "request_sha256",
}
EXPECTED_MODES = {"hidden", "given", "shuffled"}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def paired_input_audit() -> dict[str, Any]:
    report: dict[str, Any] = {"splits": {}}
    all_passed = True
    split_conversations: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in read_jsonl(DATA_DIR / split / "requests.jsonl"):
            grouped[str(row["sample_id"])].append(row)
        mode_errors = 0
        non_profile_differences: Counter[str] = Counter()
        target_leakage = 0
        future_transcript = 0
        conversations: set[str] = set()
        labels = Counter(
            str(row["reference_label"])
            for row in read_jsonl(DATA_DIR / split / "reference_labels.jsonl")
        )
        for rows in grouped.values():
            modes = {str(row["profile_mode"]) for row in rows}
            if len(rows) != 3 or modes != EXPECTED_MODES:
                mode_errors += 1
                continue
            baseline = rows[0]
            conversations.add(str(baseline["conversation_id"]))
            for row in rows[1:]:
                for key in baseline:
                    if key in PROFILE_ONLY_FIELDS:
                        continue
                    if row.get(key) != baseline.get(key):
                        non_profile_differences[key] += 1
            for row in rows:
                prompt_lower = str(row.get("prompt", "")).lower()
                target_leakage += int(
                    any(key in row for key in ("reference_label", "candidate_label", "label_evidence"))
                    or "reference_label" in prompt_lower
                    or "candidate_label" in prompt_lower
                )
                boundary = float(row["decision_time_in_conversation_s"])
                for unit in row.get("transcript_units", []):
                    if float(unit["end_s"]) > 30.0 + 1e-6:
                        future_transcript += 1
        passed = not mode_errors and not non_profile_differences and not target_leakage and not future_transcript
        all_passed &= passed
        split_conversations[split] = conversations
        report["splits"][split] = {
            "passed": passed,
            "samples": len(grouped),
            "class_counts": dict(labels),
            "mode_group_errors": mode_errors,
            "non_profile_difference_fields": dict(non_profile_differences),
            "target_leakage_rows": target_leakage,
            "future_transcript_units": future_transcript,
            "conversations": sorted(conversations),
        }
    overlap = {
        "train_val": sorted(split_conversations["train"] & split_conversations["val"]),
        "train_test": sorted(split_conversations["train"] & split_conversations["test"]),
        "val_test": sorted(split_conversations["val"] & split_conversations["test"]),
    }
    report["conversation_overlap"] = overlap
    report["passed"] = all_passed and not any(overlap.values())
    return report


def required_files() -> list[Path]:
    paths = [
        DATA_DIR / "summary.json",
        PROFILE_VIEW_DIR / "metadata.json",
        CANONICAL_RESULT_DIR / "summary.json",
        CANONICAL_RESULT_DIR / "aggregate.csv",
        CANONICAL_RESULT_DIR / "test_predictions.jsonl",
    ]
    for split in ("train", "val", "test"):
        paths.extend(
            [
                DATA_DIR / split / "requests.jsonl",
                DATA_DIR / split / "reference_labels.jsonl",
                CACHE_DIR / f"{split}.qwen-hidden.npz",
                PROFILE_VIEW_DIR / f"{split}.profile-view.npz",
            ]
        )
    return paths


def check() -> None:
    missing = [str(path.relative_to(REPO_ROOT)) for path in required_files() if not path.exists()]
    paired = paired_input_audit()
    audit_command = [
        sys.executable,
        str(REPO_ROOT / "code/scripts/audit_fiveclass_event_labels.py"),
    ]
    labels_completed = subprocess.run(audit_command, cwd=REPO_ROOT, check=False).returncode == 0
    result = {
        "passed": not missing and paired["passed"] and labels_completed,
        "missing_files": missing,
        "paired_input_audit": paired,
        "fiveclass_label_audit_passed": labels_completed,
    }
    output = REPO_ROOT / "collaboration/audits/main_experiment_preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


def train(device: str, output_dir: str) -> None:
    check()
    command = [
        sys.executable,
        str(REPO_ROOT / "code/scripts/run_qwen_shared_binary_multitask_adapter.py"),
        "--train-cache", str(CACHE_DIR / "train.qwen-hidden.npz"),
        "--val-cache", str(CACHE_DIR / "val.qwen-hidden.npz"),
        "--test-cache", str(CACHE_DIR / "test.qwen-hidden.npz"),
        "--profile-view-dir", str(PROFILE_VIEW_DIR),
        "--output-dir", str((REPO_ROOT / output_dir).resolve()),
        "--task-scheme", "paper",
        "--seeds", "3", "13", "37", "71", "101",
        "--device", device,
        "--hidden-dim", "256",
        "--dropout", "0.15",
        "--profile-dropout", "0.25",
        "--epochs", "100",
        "--patience", "14",
        "--batch-size", "128",
        "--lr", "0.0008",
        "--weight-decay", "0.0001",
        "--hidden-ce-weight", "0.5",
        "--hidden-margin-weight", "0.1",
        "--control-margin-weight", "0.25",
        "--margin", "0.05",
        "--balanced-margin",
        "--selection-delta-weight", "0.25",
        "--selection-min-delta-weight", "0.1",
        "--fusion", "gate",
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def show_results() -> None:
    summary = json.loads((CANONICAL_RESULT_DIR / "summary.json").read_text(encoding="utf-8"))
    aggregate = summary["aggregate"]
    print("task\thidden\tgiven\tshuffled\tgiven-hidden\tgiven-shuffled")
    for task in ("turn_change", "backchannel", "interruption", "floor_taking"):
        row = aggregate[task]
        h = row["hidden"]["paper_balanced_accuracy_mean"]
        g = row["given"]["paper_balanced_accuracy_mean"]
        s = row["shuffled"]["paper_balanced_accuracy_mean"]
        print(f"{task}\t{h:.4f}\t{g:.4f}\t{s:.4f}\t{g-h:+.4f}\t{g-s:+.4f}")
    row = aggregate["overall"]
    h = row["hidden"]["paper_balanced_accuracy_mean"]
    g = row["given"]["paper_balanced_accuracy_mean"]
    s = row["shuffled"]["paper_balanced_accuracy_mean"]
    print(f"overall\t{h:.4f}\t{g:.4f}\t{s:.4f}\t{g-h:+.4f}\t{g-s:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or inspect the main collaborator experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    subparsers.add_parser("show-results")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument(
        "--output-dir", default="artifacts/main_experiment/rerun"
    )
    args = parser.parse_args()
    if args.command == "check":
        check()
    elif args.command == "show-results":
        show_results()
    else:
        train(args.device, args.output_dir)


if __name__ == "__main__":
    main()
