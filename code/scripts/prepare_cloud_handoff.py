#!/usr/bin/env python
"""Inventory the workspace and package the cache-based main experiment for cloud handoff."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = ROOT / "collaboration" / "PROJECT_FULL_FILE_INVENTORY.csv"
DEFAULT_ARCHIVE = ROOT / "artifacts" / "main_experiment" / "turn-taking-main-cloud.zip"

EXCLUDED_PARTS = {".git", ".venv", "__pycache__"}
EXCLUDED_PREFIXES = (
    ".pytest_cache/",
    ".pytest_tmp/",
    ".pytest-tmp/",
    ".pytest_tmp_",
)

BUNDLE_EXCLUDED_PREFIXES = (
    "code/history/",
    "intro/history/",
    "artifacts/history/",
    "data/processed/history/",
    "models/history/",
)

MAIN_BUNDLE_PATHS = (
    "README.md",
    "PROJECT_STRUCTURE.md",
    "LICENSE",
    "AGENTS.md",
    "code",
    "intro",
    "collaboration",
    "data/README.md",
    "data/processed/sbcsae_turn_events_v3",
    "data/processed/sbcsae_vad_fiveclass_v2",
    "data/processed/sbcsae_qwen_shared_ab_30s_causal_v1",
    "artifacts/main_experiment/qwen_feature_cache",
    "artifacts/main_experiment/profile_features",
    "artifacts/main_experiment/results",
    "artifacts/README.md",
    "models/README.md",
)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def excluded(path: Path) -> bool:
    rel = relative(path)
    if rel in {
        "artifacts/main_experiment/turn-taking-main-cloud.zip",
        "artifacts/main_experiment/turn-taking-main-cloud.zip.sha256",
    }:
        return True
    return any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts) or any(
        rel.startswith(prefix) for prefix in EXCLUDED_PREFIXES
    )


def classify(rel: str) -> tuple[str, str, str]:
    """Return category, main-experiment status, and human-readable purpose."""
    if rel.startswith(("history/", "intro/history/", "code/history/", "data/processed/history/", "artifacts/history/", "models/history/")):
        return "history", "history", "历史实验或旧版本；不参与当前主实验和Talking Turns复现"
    if rel.startswith("collaboration/"):
        return "handoff", "required", "云端接手说明、审计或文件清单"
    if rel in {"README.md", "LICENSE", "AGENTS.md", "WORKING_STATE.md"}:
        return "project-root", "required", "项目入口、许可和实验约束"
    if rel.startswith("code/tests/"):
        return "test", "recommended", "自动测试；验证代码和数据契约"
    if rel.startswith("code/scripts/run_main_experiment.py"):
        return "main-code", "required", "主实验统一检查、训练和结果查看入口"
    if rel.startswith("code/scripts/inspect_experiment_data.py"):
        return "main-code", "required", "逐条查看主实验与Talking Turns输入、标签和预测"
    if rel.startswith("code/scripts/run_qwen_shared_binary_multitask_adapter.py"):
        return "main-code", "required", "Qwen shared gated A/B adapter训练与评测"
    if rel.startswith("code/scripts/audit_fiveclass_event_labels.py"):
        return "main-code", "required", "主实验事件标签和split审计"
    if rel.startswith("code/scripts/build_dynamic_behavior_profile_views.py"):
        return "main-code", "required", "构建59维动态交互profile向量"
    if rel.startswith("code/src/profile_turntaking/event_annotation.py"):
        return "main-code", "required", "IPU和五分类事件定义"
    if rel.startswith("code/src/profile_turntaking/paper_aligned_floor_targets.py"):
        return "main-code", "required", "Talking Turns风格floor-taking目标定义"
    if rel.startswith("code/"):
        return "code-or-doc", "supporting-or-history", "代码、配置、说明、报告或历史实验入口"
    if rel.startswith("intro/") or rel == "profile_aware_turn_taking_intro_experiments.md":
        return "paper", "recommended", "论文思路、实验设计和写作草稿"
    if rel.startswith("data/processed/sbcsae_qwen_shared_ab_30s_causal_v1/"):
        return "main-data", "required", "主实验10,804条事件的请求、标签和输入审计"
    if rel.startswith("data/processed/sbcsae_turn_events_v3/"):
        return "event-source", "required", "16段会话的IPU与16,144个事件标签源"
    if rel.startswith("data/processed/sbcsae_vad_fiveclass_v2/"):
        return "dense-labels", "optional", "早期全时间轴578,686个40ms五分类标签"
    if rel.startswith("data/sbcsae/openslr/"):
        return "raw-sbcsae", "feature-rebuild-only", "SBCSAE原始WAV、TRN、CHAT或语料文档"
    if rel.startswith("data/"):
        return "other-data", "history-or-optional", "其他处理版本、PaChat演示或历史数据"
    if rel.startswith("artifacts/main_experiment/qwen_feature_cache/"):
        return "main-cache", "required", "冻结Qwen音频层和上下文向量；云端快速训练直接读取"
    if rel.startswith(
        "artifacts/main_experiment/profile_features/"
    ):
        return "main-profile", "required", "主实验59维hidden/given/shuffled profile向量"
    if rel.startswith(
        "artifacts/main_experiment/results/"
    ):
        return "main-result", "required", "当前固定主结果、逐样本预测和随机种子输出"
    if rel.startswith("artifacts/main_experiment/audio_only_baseline/"):
        return "audio-baseline", "recommended", "同任务的Qwen纯音频对照结果"
    if rel.startswith("artifacts/talking_turns/"):
        return "talking-turns-result", "recommended", "官方Talking Turns checkpoint在SBCSAE test上的结果"
    if rel.startswith("artifacts/"):
        return "supporting-artifact", "optional", "数据预览或其他当前辅助产物"
    if rel.startswith("models/talking_turns/"):
        return "external-model", "optional", "Talking Turns官方ESPnet外部基线"
    if rel.startswith("models/qwen2.5-omni-3b-local/"):
        return "qwen-model", "feature-rebuild-only", "用于重新提取Qwen特征；用缓存训练时不需要"
    if rel.startswith("models/"):
        return "local-model", "feature-rebuild-only", "本地Qwen、量化模型或下载缓存"
    if rel.startswith("tmp/") or rel.startswith(".pytest"):
        return "temporary", "discard", "临时下载、测试缓存或中间文件"
    return "other", "review", "其他项目文件"


def workspace_files() -> list[Path]:
    files: list[Path] = []
    for current, directories, filenames in os.walk(ROOT):
        current_path = Path(current)
        directories[:] = [
            name
            for name in directories
            if name not in EXCLUDED_PARTS
            and not any(
                relative(current_path / name).startswith(prefix)
                for prefix in EXCLUDED_PREFIXES
            )
        ]
        for filename in filenames:
            path = current_path / filename
            if not excluded(path):
                files.append(path)
    return sorted(files, key=relative)


def build_inventory(output: Path) -> dict[str, object]:
    rows = []
    totals: dict[str, dict[str, int]] = {}
    for path in workspace_files():
        rel = relative(path)
        category, status, purpose = classify(rel)
        size = path.stat().st_size
        rows.append(
            {
                "path": rel,
                "bytes": size,
                "category": category,
                "main_experiment_status": status,
                "purpose": purpose,
            }
        )
        entry = totals.setdefault(status, {"files": 0, "bytes": 0})
        entry["files"] += 1
        entry["bytes"] += size
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "bytes", "category", "main_experiment_status", "purpose"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return {"output": relative(output), "files": len(rows), "status_totals": totals}


def bundle_files(output: Path) -> list[Path]:
    selected: dict[str, Path] = {}
    for item in MAIN_BUNDLE_PATHS:
        path = ROOT / item
        if not path.exists():
            raise FileNotFoundError(f"required handoff path is missing: {item}")
        candidates = [path] if path.is_file() else path.rglob("*")
        for candidate in candidates:
            if not candidate.is_file() or excluded(candidate):
                continue
            if candidate.resolve() == output.resolve():
                continue
            rel = relative(candidate)
            if any(rel.startswith(prefix) for prefix in BUNDLE_EXCLUDED_PREFIXES):
                continue
            selected[rel] = candidate
    return [selected[key] for key in sorted(selected)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(output: Path) -> dict[str, object]:
    files = bundle_files(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
        for index, path in enumerate(files, start=1):
            suffix = path.suffix.lower()
            compression = zipfile.ZIP_STORED if suffix in {".npz", ".wav", ".pt"} else zipfile.ZIP_DEFLATED
            archive.write(path, relative(path), compress_type=compression)
            if index % 100 == 0 or index == len(files):
                print(f"[package] {index}/{len(files)}", flush=True)
    digest = sha256(output)
    checksum = output.with_suffix(output.suffix + ".sha256")
    checksum.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    return {
        "archive": relative(output),
        "bytes": output.stat().st_size,
        "sha256": digest,
        "checksum_file": relative(checksum),
        "files": len(files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("--output", type=Path, default=DEFAULT_INVENTORY)
    package_parser = subparsers.add_parser("package")
    package_parser.add_argument("--output", type=Path, default=DEFAULT_ARCHIVE)
    package_parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.command == "inventory":
        result = build_inventory(output)
    elif args.dry_run:
        files = bundle_files(output)
        result = {
            "archive": relative(output),
            "dry_run": True,
            "files": len(files),
            "source_bytes": sum(path.stat().st_size for path in files),
        }
    else:
        result = package(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
