"""Build a portable inventory of files required for collaborator handoff."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CORE_CONVERSATIONS = (
    "SBC005", "SBC006", "SBC007", "SBC009", "SBC010", "SBC017",
    "SBC024", "SBC029", "SBC034", "SBC041", "SBC043", "SBC044",
    "SBC045", "SBC047", "SBC058", "SBC060",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path: Path, role: str, *, hash_file: bool = True) -> dict[str, object]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "role": role,
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path) if hash_file and path.is_file() else None,
    }


def main() -> None:
    items: list[dict[str, object]] = []
    fixed = {
        "code/scripts/run_main_experiment.py": "唯一主实验入口",
        "code/scripts/inspect_experiment_data.py": "逐样本输入和预测查看器",
        "code/scripts/audit_fiveclass_event_labels.py": "五分类标签复核",
        "code/scripts/run_qwen_shared_binary_multitask_adapter.py": "主模型实现",
        "code/scripts/build_dynamic_behavior_profile_views.py": "59维动态交互状态构建",
        "code/scripts/run_espnet_talking_turns_baseline.py": "Talking Turns官方checkpoint运行与评分",
        "code/src/profile_turntaking/event_annotation.py": "事件与五分类规则",
        "code/src/profile_turntaking/paper_aligned_floor_targets.py": "Floor-taking目标",
        "data/processed/sbcsae_turn_events_v3/event_candidates.jsonl": "完整事件标签源",
        "data/processed/sbcsae_turn_events_v3/ipus.jsonl": "IPU源数据",
        "data/processed/sbcsae_qwen_shared_ab_30s_causal_v1/summary.json": "主实验数据统计",
        "artifacts/main_experiment/profile_features/metadata.json": "profile字段定义",
        "artifacts/main_experiment/results/summary.json": "主实验汇总结果",
        "artifacts/main_experiment/results/test_predictions.jsonl": "逐样本测试预测",
        "artifacts/main_experiment/rerun/summary.json": "整理后完整重训验证",
        "artifacts/talking_turns/sbcsae_test/metrics.json": "Talking Turns在SBCSAE上的指标",
        "artifacts/talking_turns/sbcsae_test/test_predictions.jsonl": "Talking Turns逐样本五分类预测",
        "models/talking_turns/checkpoint/exp/asr_train_asr_whisper_turn_taking_raw_en_word/config.yaml": "Talking Turns官方配置",
        "models/talking_turns/checkpoint/exp/asr_train_asr_whisper_turn_taking_raw_en_word/valid.loss.ave.pth": "Talking Turns官方checkpoint",
        "collaboration/README.md": "合作者总说明",
    }
    for relative, role in fixed.items():
        items.append(describe(ROOT / relative, role))
    for split in ("train", "val", "test"):
        items.extend(
            [
                describe(
                    ROOT / f"data/processed/sbcsae_qwen_shared_ab_30s_causal_v1/{split}/reference_labels.jsonl",
                    f"{split}五分类与四任务标签",
                ),
                describe(
                    ROOT / f"artifacts/main_experiment/qwen_feature_cache/{split}.qwen-hidden.npz",
                    f"{split}冻结Qwen音频/上下文特征缓存",
                ),
                describe(
                    ROOT / f"artifacts/main_experiment/profile_features/{split}.profile-view.npz",
                    f"{split} hidden/given/shuffled profile向量",
                ),
            ]
        )
    for conversation_id in CORE_CONVERSATIONS:
        items.append(
            describe(
                ROOT / f"data/sbcsae/openslr/WAV/{conversation_id}.wav",
                "原始SBCSAE音频；重建Qwen缓存时需要",
                hash_file=False,
            )
        )
    missing = [item["path"] for item in items if not item["exists"]]
    result = {
        "schema_version": "1.0",
        "portable_paths": True,
        "path_base": "repository root",
        "main_experiment_can_train_from_cached_features_without_raw_audio_or_qwen_weights": True,
        "items": items,
        "missing": missing,
        "complete": not missing,
    }
    output = ROOT / "collaboration/HANDOFF_MANIFEST.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": not missing, "items": len(items), "missing": missing}, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
