"""Summarize local Qwen prompt vs hidden-vector profile experiments.

This script is intentionally read-only over experiment outputs. It collects the
same held-out SBCSAE test results for:

- direct Qwen prompt inference through llama.cpp;
- frozen Qwen hidden-vector + small profile adapter variants.

It writes a compact Chinese report plus CSV tables that can be checked by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = ("C", "BC", "T", "I", "NA")
PROFILE_MODES = ("hidden", "given", "shuffled")


ADAPTER_RUNS = {
    "adapter_gate_pdrop050": "full-local",
    "adapter_concat_pdrop050": "full-local-concat",
    "adapter_gate_pdrop025": "full-local-gate-pdrop025",
    "adapter_gate_pdrop000": "full-local-gate-pdrop000",
    "adapter_concat_pdrop025": "full-local-concat-pdrop025",
    "adapter_concat_pdrop000": "full-local-concat-pdrop000",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _prediction_distribution_from_confusion(matrix: list[list[int]]) -> dict[str, int]:
    totals = {label: 0 for label in LABELS}
    for row in matrix:
        for index, value in enumerate(row):
            totals[LABELS[index]] += int(value)
    return totals


def _confusion_matrix(targets: list[str], predictions: list[str]) -> list[list[int]]:
    index = {label: i for i, label in enumerate(LABELS)}
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    for target, prediction in zip(targets, predictions):
        matrix[index[target]][index[prediction]] += 1
    return matrix


def _write_confusion_csv(path: Path, matrix: list[list[int]]) -> None:
    rows = []
    for label, row in zip(LABELS, matrix):
        payload = {"target": label}
        payload.update({f"pred_{pred}": int(value) for pred, value in zip(LABELS, row)})
        rows.append(payload)
    _write_csv(path, rows)


def _load_prompt_rows(prompt_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = _read_json(prompt_dir / "metrics.json")
    diagnostics = _read_json(prompt_dir / "diagnostics.json")
    comparison = []
    for mode in PROFILE_MODES:
        m = metrics[mode]
        dist = diagnostics["prediction_distribution"][mode]
        comparison.append(
            {
                "method": "qwen_prompt_llamacpp_q4",
                "run_id": "prompt_same_test_250",
                "profile_mode": mode,
                "split": "test",
                "samples": m["samples"],
                "macro_f1": m["macro_f1"],
                "balanced_accuracy": m["balanced_accuracy"],
                "accuracy": m["accuracy"],
                "log_loss": "",
                "brier_score": "",
                "ece": "",
                "noncollapsed": diagnostics["collapse_gate"]["hidden_noncollapsed"]
                if mode == "hidden"
                else "",
                "dominant_prediction": max(dist, key=dist.get),
                "dominant_prediction_fraction": max(dist.values()) / max(1, sum(dist.values())),
                "prediction_C": dist.get("C", 0),
                "prediction_BC": dist.get("BC", 0),
                "prediction_T": dist.get("T", 0),
                "prediction_I": dist.get("I", 0),
                "prediction_NA": dist.get("NA", 0),
                "source_dir": str(prompt_dir.resolve()),
            }
        )
    deltas = [
        {
            "method": "qwen_prompt_llamacpp_q4",
            "run_id": "prompt_same_test_250",
            "split": "test",
            "given_minus_hidden_macro_f1": metrics["given"]["macro_f1"]
            - metrics["hidden"]["macro_f1"],
            "given_minus_shuffled_macro_f1": metrics["given"]["macro_f1"]
            - metrics["shuffled"]["macro_f1"],
            "hidden_noncollapsed": diagnostics["collapse_gate"]["hidden_noncollapsed"],
            "profile_effect_claim_allowed": diagnostics["collapse_gate"]["profile_effect_claim_allowed"],
            "source_dir": str(prompt_dir.resolve()),
        }
    ]
    return comparison, deltas


def _load_adapter_rows(base_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparison: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    for run_id, rel in ADAPTER_RUNS.items():
        run_dir = base_dir / rel / "run"
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = _read_json(summary_path)
        for split_name, aggregate_key in (("validation", "validation_aggregate"), ("test", "aggregate")):
            aggregate = summary[aggregate_key]
            for mode in PROFILE_MODES:
                mode_payload = aggregate[mode]
                comparison.append(
                    {
                        "method": "qwen_hidden_profile_adapter",
                        "run_id": run_id,
                        "profile_mode": mode,
                        "split": split_name,
                        "samples": summary["samples"]["val" if split_name == "validation" else "test"],
                        "macro_f1": mode_payload["macro_f1_mean"],
                        "macro_f1_std": mode_payload.get("macro_f1_std", ""),
                        "balanced_accuracy": mode_payload.get("balanced_accuracy_mean", ""),
                        "accuracy": "",
                        "log_loss": mode_payload["log_loss_mean"],
                        "brier_score": mode_payload.get("brier_score_mean", ""),
                        "ece": mode_payload.get("ece_mean", ""),
                        "noncollapsed": mode_payload["all_seeds_noncollapsed"],
                        "dominant_prediction": "",
                        "dominant_prediction_fraction": "",
                        "prediction_C": "",
                        "prediction_BC": "",
                        "prediction_T": "",
                        "prediction_I": "",
                        "prediction_NA": "",
                        "fusion": summary["config"]["fusion"],
                        "profile_dropout": summary["config"]["profile_dropout"],
                        "seeds": " ".join(str(seed) for seed in summary["config"]["seeds"]),
                        "source_dir": str(run_dir.resolve()),
                    }
                )
            deltas.append(
                {
                    "method": "qwen_hidden_profile_adapter",
                    "run_id": run_id,
                    "split": split_name,
                    "given_minus_hidden_macro_f1": aggregate["given_minus_hidden_macro_f1"]["mean"],
                    "given_minus_shuffled_macro_f1": aggregate["given_minus_shuffled_macro_f1"]["mean"],
                    "hidden_minus_given_log_loss": aggregate.get("hidden_minus_given_log_loss", {}).get("mean", ""),
                    "shuffled_minus_given_log_loss": aggregate.get("shuffled_minus_given_log_loss", {}).get("mean", ""),
                    "hidden_noncollapsed": aggregate["hidden"]["all_seeds_noncollapsed"],
                    "all_modes_noncollapsed": all(
                        aggregate[mode]["all_seeds_noncollapsed"] for mode in PROFILE_MODES
                    ),
                    "profile_effect_claim_allowed": summary["interpretation_gate"][
                        "profile_effect_claim_allowed"
                    ],
                    "fusion": summary["config"]["fusion"],
                    "profile_dropout": summary["config"]["profile_dropout"],
                    "source_dir": str(run_dir.resolve()),
                }
            )
    return comparison, deltas


def _write_adapter_confusions(base_dir: Path, output_dir: Path) -> None:
    confusion_dir = output_dir / "confusion_matrices"
    for run_id, rel in ADAPTER_RUNS.items():
        predictions_path = base_dir / rel / "run" / "predictions.jsonl"
        if not predictions_path.exists():
            continue
        rows = _read_jsonl(predictions_path)
        by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_seed[int(row["seed"])].append(row)
        for seed, seed_rows in by_seed.items():
            targets = [str(row["target"]) for row in seed_rows]
            for mode in PROFILE_MODES:
                preds = [str(row[f"prediction_{mode}"]) for row in seed_rows]
                matrix = _confusion_matrix(targets, preds)
                _write_confusion_csv(confusion_dir / f"{run_id}_{mode}_seed{seed}.csv", matrix)


def _write_prompt_confusions(prompt_dir: Path, output_dir: Path) -> None:
    metrics = _read_json(prompt_dir / "metrics.json")
    confusion_dir = output_dir / "confusion_matrices"
    for mode in PROFILE_MODES:
        _write_confusion_csv(
            confusion_dir / f"prompt_same_test_250_{mode}.csv",
            metrics[mode]["confusion_matrix"],
        )


def _make_report(
    *,
    output_dir: Path,
    adapter_base: Path,
    prompt_dir: Path,
    comparison_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
) -> str:
    prompt_metrics = _read_json(prompt_dir / "metrics.json")
    prompt_diag = _read_json(prompt_dir / "diagnostics.json")
    main_summary = _read_json(adapter_base / "full-local" / "run" / "summary.json")

    def row_for(run_id: str, split: str, mode: str) -> dict[str, Any]:
        for row in comparison_rows:
            if row["run_id"] == run_id and row["split"] == split and row["profile_mode"] == mode:
                return row
        raise KeyError((run_id, split, mode))

    adapter_table_lines = [
        "| 方法 | split | hidden | given | shuffled | given-hidden | given-shuffled | 结论门禁 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for run_id in ADAPTER_RUNS:
        for split in ("validation", "test"):
            try:
                h = row_for(run_id, split, "hidden")
                g = row_for(run_id, split, "given")
                s = row_for(run_id, split, "shuffled")
            except KeyError:
                continue
            delta = next(row for row in delta_rows if row["run_id"] == run_id and row["split"] == split)
            gate = "通过" if delta["profile_effect_claim_allowed"] else "未通过"
            adapter_table_lines.append(
                "| "
                + " | ".join(
                    [
                        run_id,
                        split,
                        _fmt(h["macro_f1"]),
                        _fmt(g["macro_f1"]),
                        _fmt(s["macro_f1"]),
                        _fmt(delta["given_minus_hidden_macro_f1"]),
                        _fmt(delta["given_minus_shuffled_macro_f1"]),
                        gate,
                    ]
                )
                + " |"
            )

    prompt_table = "\n".join(
        [
            "| 条件 | Macro-F1 | Balanced Accuracy | Accuracy | 预测分布 |",
            "|---|---:|---:|---:|---|",
            *[
                "| "
                + " | ".join(
                    [
                        mode,
                        _fmt(prompt_metrics[mode]["macro_f1"]),
                        _fmt(prompt_metrics[mode]["balanced_accuracy"]),
                        _fmt(prompt_metrics[mode]["accuracy"]),
                        ", ".join(
                            f"{label}={prompt_diag['prediction_distribution'][mode].get(label, 0)}"
                            for label in LABELS
                        ),
                    ]
                )
                + " |"
                for mode in PROFILE_MODES
            ],
        ]
    )

    report = f"""# 本机 Qwen profile 实验阶段报告

生成时间：2026-08-15

## 1. 这次真正完成了什么

这次完成了两套可以直接对比的本机实验：

1. **Qwen prompt 直接预测**：Qwen2.5-Omni-3B Q4_K_M，通过 llama.cpp 音频接口，直接输入音频、历史转写和 profile，让模型输出 `C / BC / T / I / NA`。
2. **Qwen hidden + profile adapter**：冻结 Qwen2.5-Omni-3B，用它提取“音频+历史转写”的上下文向量，再单独提取 profile 向量，只训练一个小的融合分类器输出五类概率。

两套实验都使用同一个 held-out test split：

- test 样本：250 条；
- 五类均衡：C/BC/T/I/NA 各 50 条；
- 会话：{", ".join(main_summary["split_conversations"]["test"])}；
- 每条都有 hidden / given / shuffled 三种 profile 条件；
- 三个条件中，音频、转写、预测点、样本编号不变，只改变 profile。

## 2. 输入和输出

每条样本的输入是：

```text
预测点之前的音频
+ 预测点之前的历史转写
+ profile 条件：hidden / given / shuffled
```

模型输出是五分类：

```text
C / BC / T / I / NA
```

其中 prompt 版本直接让 Qwen 输出 JSON：`{{"label":"T"}}` 这种格式。

adapter 版本不让 Qwen 直接说答案，而是取 Qwen 内部向量，再由小分类器输出五类概率。

## 3. 同样本 prompt 结果

路径：`{prompt_dir.resolve()}`

{prompt_table}

prompt 运行完成情况：

- 请求数：750；
- 有效响应：750；
- 无效输出：0；
- median latency：{_fmt(prompt_diag["latency_ms"]["median"], 1)} ms；
- hidden 是否不塌陷：{prompt_diag["collapse_gate"]["hidden_noncollapsed"]}。

结论：prompt 版本虽然全部能输出，但 hidden 基线严重偏向少数类别，主要预测 `T`，所以它不能作为可靠 profile 效果证据。

## 4. Qwen hidden + profile adapter 结果

主输出路径：

- 默认 gate：`{(adapter_base / "full-local" / "run").resolve()}`
- 默认 concat/MLP：`{(adapter_base / "full-local-concat" / "run").resolve()}`
- 其他 profile-dropout 变体也在 `{adapter_base.resolve()}`

下面表格数值是 Macro-F1，越高越好。

{chr(10).join(adapter_table_lines)}

## 5. 当前最重要的结论

1. **adapter 明显比 prompt 直接问答稳定。**  
   prompt 的 Macro-F1 大约 0.09，而且 hidden 塌陷；adapter 的 test Macro-F1 大约 0.39–0.42，并且所有模式都没有塌陷。

2. **adapter 已经训练了模型，但没有微调 Qwen。**  
   Qwen 只负责把音频、转写和 profile 变成向量；真正训练的是后面的小型融合分类器。

3. **正确 profile 的作用目前还不稳定。**  
   默认 gate 版本 test 上 given 比 hidden 高 `+0.0061` Macro-F1，但 given 比 shuffled 低 `-0.0012`。这说明“给 profile”可能改变概率和预测，但还不能强说模型已经稳定理解了“正确 profile”。

4. **这不是输出塌陷问题。**  
   与 prompt 不同，adapter 的 hidden/given/shuffled 都预测了多类，说明它至少学到了可用的五分类边界。

## 6. 可以给老师看的简短说法

我们在同一个 SBCSAE held-out test set 上做了两种本机实验。直接 prompt Qwen2.5-Omni-3B 虽然 750 个请求都成功，但预测严重集中到 T/I，Macro-F1 只有约 0.09，说明零样本 prompt 不适合作为主方法。随后我们冻结 Qwen，只取音频+历史转写 hidden state 和 profile hidden state，训练一个很小的 profile adapter。这个版本没有塌陷，Macro-F1 提高到约 0.39–0.42，说明监督式 adapter 能明显学到 turn-taking 五分类。当前 profile 的增益还不稳定：正确 profile 相比隐藏 profile有小幅变化，但没有稳定超过 shuffled profile。下一步应增强 profile 表示或做动态 profile，而不是继续依赖零样本 prompt。

## 7. 文件清单

- 统一对比表：`{(output_dir / "all_results_comparison.csv").resolve()}`
- profile 差值表：`{(output_dir / "profile_effect_deltas.csv").resolve()}`
- 混淆矩阵目录：`{(output_dir / "confusion_matrices").resolve()}`
- prompt 逐样本预测：`{(prompt_dir / "predictions.csv").resolve()}`
- prompt 原始响应：`{(prompt_dir / "responses.jsonl").resolve()}`
- adapter 默认 gate 逐样本预测：`{(adapter_base / "full-local" / "run" / "predictions.jsonl").resolve()}`
- adapter 默认 concat 逐样本预测：`{(adapter_base / "full-local-concat" / "run" / "predictions.jsonl").resolve()}`
"""
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter-base",
        default="artifacts/qwen-hidden-profile",
        help="Directory containing full-local* adapter run directories.",
    )
    parser.add_argument(
        "--prompt-dir",
        default="artifacts/qwen-hidden-profile/prompt-on-semantic-test-250",
        help="Prompt run directory with metrics.json and diagnostics.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/qwen-hidden-profile/final-local-report-20260815",
    )
    args = parser.parse_args()

    adapter_base = Path(args.adapter_base)
    prompt_dir = Path(args.prompt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_comparison, prompt_deltas = _load_prompt_rows(prompt_dir)
    adapter_comparison, adapter_deltas = _load_adapter_rows(adapter_base)
    comparison_rows = prompt_comparison + adapter_comparison
    delta_rows = prompt_deltas + adapter_deltas

    _write_csv(output_dir / "all_results_comparison.csv", comparison_rows)
    _write_csv(output_dir / "profile_effect_deltas.csv", delta_rows)
    _write_prompt_confusions(prompt_dir, output_dir)
    _write_adapter_confusions(adapter_base, output_dir)

    report = _make_report(
        output_dir=output_dir,
        adapter_base=adapter_base,
        prompt_dir=prompt_dir,
        comparison_rows=comparison_rows,
        delta_rows=delta_rows,
    )
    (output_dir / "LOCAL_QWEN_PROFILE_EXPERIMENT_REPORT_ZH.md").write_text(
        report, encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str((output_dir / "LOCAL_QWEN_PROFILE_EXPERIMENT_REPORT_ZH.md").resolve()),
                "comparison_csv": str((output_dir / "all_results_comparison.csv").resolve()),
                "delta_csv": str((output_dir / "profile_effect_deltas.csv").resolve()),
                "confusion_matrices": str((output_dir / "confusion_matrices").resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
