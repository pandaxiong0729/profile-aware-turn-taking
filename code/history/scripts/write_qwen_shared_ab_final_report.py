from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


PAPER_TOPLINE = {
    "turn_change": 0.786,
    "backchannel": 0.751,
    "interruption": 0.749,
    "floor_taking": 0.656,
}
TASK_ZH = {
    "turn_change": "Turn Change",
    "backchannel": "Backchannel",
    "interruption": "Interruption",
    "floor_taking": "Floor-taking Interruption",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def signed_pct(value: float) -> str:
    return f"{100.0 * float(value):+.2f} pp"


def wait_for(path: Path, *, timeout_hours: float, poll_seconds: int) -> None:
    deadline = time.monotonic() + timeout_hours * 3600.0
    while time.monotonic() < deadline:
        if path.is_file():
            return
        time.sleep(max(5, poll_seconds))
    raise TimeoutError(f"Timed out waiting for {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the audited Chinese Qwen shared A/B report.")
    parser.add_argument(
        "--search-root",
        default="artifacts/qwen-shared-ab-30s-causal/paper-aligned-search",
    )
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    search_root = (repo_root / args.search_root).resolve()
    final_path = search_root / "FINAL_SUMMARY.json"
    if args.wait:
        wait_for(final_path, timeout_hours=args.timeout_hours, poll_seconds=args.poll_seconds)
    if not final_path.is_file():
        raise FileNotFoundError(final_path)

    search = load_json(final_path)
    result_dir = Path(search["final_result_dir"])
    adapter = load_json(result_dir / "summary.json")
    aggregate = adapter["aggregate"]
    final = search["final_test"]
    validation = search["validation_candidates"]

    lines = [
        "# Qwen shared A/B adapter：30 秒因果输入最终实验报告",
        "",
        "## 1. 实验问题",
        "",
        "本实验检验：在音频、转写、样本、预测边界和模型设置完全相同的条件下，正确 profile（given）是否同时优于隐藏 profile（hidden）和错误 profile（shuffled），以及四个二分类任务的平均准确率能否达到或逼近 73%。",
        "",
        "## 2. 数据与输入",
        "",
        "- 数据：SBCSAE 事件样本，共 10,804 条；train / validation / test 按完整会话划分为 6,623 / 2,243 / 1,938 条。",
        "- 模型输入：预测点前 30 秒单声道混合音频、同一范围内的因果转写、profile。",
        "- 请求中不包含未来音频、未来转写、目标标签或标注证据。",
        "- hidden / given / shuffled 只改变 profile；样本 ID、音频、转写、预测位置和问题定义保持不变。",
        "- 训练使用全部有效二分类样本；论文可比指标使用每个任务固定的 50/50 A/B 测试子集，因此随机基线为 50%。",
        "",
        "## 3. 模型与选择方法",
        "",
        "Qwen2.5-Omni-3B Thinker 冻结，只提取因果音频和转写的 hidden representation；profile 由同一个冻结 Qwen 文本路径单独编码。可训练部分是一个共享 profile adapter 和四个二分类输出头。",
        "",
        "一次 Qwen 前向同时保存：完整提示后的最后 token、预测边界最后音频 token、边界前 8 个音频 token 均值、整段音频 token 均值。由此派生五种 context view，并在 validation 上比较 gate、concat、FiLM 及其 margin 版本。test 在表示和结构选定后只运行一次。",
        "",
        "## 4. Validation 选型结果",
        "",
        "| Context view | Adapter | hidden | given | shuffled | given-hidden | given-shuffled |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in validation:
        lines.append(
            "| {context_view} | {name} | {hidden} | {given} | {shuffled} | {dh} | {ds} |".format(
                context_view=row["context_view"],
                name=row["name"],
                hidden=pct(row["hidden_accuracy"]),
                given=pct(row["given_accuracy"]),
                shuffled=pct(row["shuffled_accuracy"]),
                dh=signed_pct(row["given_minus_hidden"]),
                ds=signed_pct(row["given_minus_shuffled"]),
            )
        )
    selected = search["selected_from_validation"]
    lines.extend(
        [
            "",
            f"验证集选中：`{selected['context_view']}` context view＋`{selected['name']}` adapter。",
            "",
            "## 5. 最终 Test 结果",
            "",
            "主指标是固定 50/50 A/B 子集上的 accuracy。",
            "",
            "| 任务 | Talking Turns supervised topline | hidden | given | shuffled | given-hidden | given-shuffled |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for task in PAPER_TOPLINE:
        h = float(aggregate[task]["hidden"]["paper_balanced_accuracy_mean"])
        g = float(aggregate[task]["given"]["paper_balanced_accuracy_mean"])
        s = float(aggregate[task]["shuffled"]["paper_balanced_accuracy_mean"])
        lines.append(
            f"| {TASK_ZH[task]} | {pct(PAPER_TOPLINE[task])} | {pct(h)} | {pct(g)} | {pct(s)} | {signed_pct(g-h)} | {signed_pct(g-s)} |"
        )
    lines.append(
        "| 四任务平均 | {paper} | {hidden} | {given} | {shuffled} | {dh} | {ds} |".format(
            paper=pct(sum(PAPER_TOPLINE.values()) / len(PAPER_TOPLINE)),
            hidden=pct(final["hidden_accuracy"]),
            given=pct(final["given_accuracy"]),
            shuffled=pct(final["shuffled_accuracy"]),
            dh=signed_pct(final["given_minus_hidden"]),
            ds=signed_pct(final["given_minus_shuffled"]),
        )
    )
    order_pass = final["given_accuracy"] > final["hidden_accuracy"] and final["given_accuracy"] > final["shuffled_accuracy"]
    lines.extend(
        [
            "",
            "注意：Talking Turns 数字来自其数据集和监督 topline，只作为任务难度参照；本实验使用 SBCSAE，不能把两者差值直接声明为同数据集 SOTA 提升。",
            "",
            "## 6. 目标核验",
            "",
            f"- `given > hidden`：{'通过' if final['given_accuracy'] > final['hidden_accuracy'] else '未通过'}。",
            f"- `given > shuffled`：{'通过' if final['given_accuracy'] > final['shuffled_accuracy'] else '未通过'}。",
            f"- given 平均 accuracy ≥ 73%：{'通过' if final['given_accuracy'] >= 0.73 else '未通过'}（实际 {pct(final['given_accuracy'])}）。",
            f"- given 平均 accuracy ≥ 70%（逼近 73%）：{'通过' if final['given_accuracy'] >= 0.70 else '未通过'}。",
            f"- 两项 profile 顺序同时成立：{'是' if order_pass else '否'}。",
            "",
            "## 7. 可核查文件",
            "",
            f"- 总结：`{final_path}`",
            f"- 最终 adapter 结果：`{result_dir / 'summary.json'}`",
            f"- 逐样本预测：`{result_dir / 'test_predictions.jsonl'}`",
            f"- 逐任务汇总：`{result_dir / 'aggregate.csv'}`",
            f"- profile 差值：`{result_dir / 'profile_deltas.csv'}`",
            f"- 数据报告：`{repo_root / 'artifacts/qwen-shared-ab-30s-causal/DATA_REPORT.md'}`",
            "",
        ]
    )
    destination = search_root / "FINAL_REPORT_ZH.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
