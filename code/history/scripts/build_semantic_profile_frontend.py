#!/usr/bin/env python
"""Build a self-contained, non-technical review page for the R2 pilot."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from urllib.parse import quote


LABELS = ("C", "BC", "T", "I", "NA")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def number(value: float) -> str:
    return f"{value:.4f}"


def build(artifact_dir: Path, data_dir: Path, output: Path) -> None:
    summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
    requests = read_jsonl(data_dir / "requests.jsonl")
    predictions = read_jsonl(artifact_dir / "predictions.jsonl")
    example_request = next(row for row in requests if row["profile_mode"] == "given")
    seed = int(summary["config"]["seeds"][0])
    example_prediction = next(
        row
        for row in predictions
        if row["seed"] == seed and row["sample_id"] == example_request["sample_id"]
    )
    audio_path = (data_dir / example_request["audio_path"]).resolve()
    audio_href = quote(os.path.relpath(audio_path, output.parent).replace("\\", "/"))
    aggregate = summary["aggregate"]
    validation = summary["validation_aggregate"]
    modes = (("hidden", "不提供 profile"), ("given", "正确 profile"), ("shuffled", "错误 profile"))
    result_rows = "".join(
        "<tr>"
        f"<td>{label}</td>"
        f"<td>{number(aggregate[mode]['macro_f1_mean'])}</td>"
        f"<td>{number(aggregate[mode]['balanced_accuracy_mean'])}</td>"
        f"<td>{number(aggregate[mode]['log_loss_mean'])}</td>"
        f"<td>{number(validation[mode]['macro_f1_mean'])}</td>"
        "</tr>"
        for mode, label in modes
    )
    per_class_rows = "".join(
        "<tr>"
        f"<td>{label}</td>"
        + "".join(
            f"<td>{number(aggregate[mode]['per_class_f1_mean'][label])}</td>"
            for mode, _ in modes
        )
        + "</tr>"
        for label in LABELS
    )
    seed_rows = "".join(
        "<tr>"
        f"<td>{row['seed']}</td>"
        f"<td>{number(row['reports']['hidden']['macro_f1'])}</td>"
        f"<td>{number(row['reports']['given']['macro_f1'])}</td>"
        f"<td>{number(row['reports']['shuffled']['macro_f1'])}</td>"
        f"<td>{number(row['reports']['controls']['audio_changed_fraction'])}</td>"
        "</tr>"
        for row in summary["seed_reports"]
    )
    probabilities = "".join(
        f"<li>{mode_name}：<strong>{example_prediction['prediction_' + mode]}</strong> — "
        + ", ".join(
            f"{label} {example_prediction['probabilities_' + mode][label]:.3f}"
            for label in LABELS
        )
        + "</li>"
        for mode, mode_name in modes
    )
    claim = summary["interpretation_gate"]["profile_effect_claim_allowed"]
    verdict = "可以形成 profile 效果结论" if claim else "暂不能形成 profile 效果结论"
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>R2 Profile 语义向量实验</title>
<style>
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;background:#f4f7fb;color:#172033}}
main{{max-width:1080px;margin:auto;padding:32px 20px 64px}}
h1{{margin-bottom:8px}} h2{{margin-top:32px}} p{{line-height:1.65}}
.card{{background:white;border:1px solid #dce3ee;border-radius:14px;padding:20px;margin:16px 0;box-shadow:0 4px 14px #1720330d}}
.verdict{{border-left:7px solid #d97706;background:#fff8e7}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}
.metric{{background:#eef4ff;border-radius:10px;padding:16px}} .metric b{{font-size:1.65rem;display:block;color:#1447a6}}
table{{width:100%;border-collapse:collapse;background:white}} th,td{{padding:10px;border-bottom:1px solid #e4e9f1;text-align:left}} th{{background:#edf3fc}}
pre{{white-space:pre-wrap;background:#f1f4f8;padding:14px;border-radius:9px;line-height:1.5}}
audio{{width:100%}} .small{{color:#5c687a;font-size:.93rem}} li{{margin:8px 0}}
</style>
</head>
<body><main>
<h1>自然语言 Profile 语义向量实验（R2）</h1>
<p class="small">同一批因果音频与转写，只改变 profile；五分类 C / BC / T / I / NA。</p>
<section class="card verdict"><h2 style="margin-top:0">结论：{verdict}</h2>
<p>测试集上正确 profile 的 Macro-F1 为 <b>0.4072</b>，略高于不提供 profile 的 <b>0.4037</b> 和错误 profile 的 <b>0.4048</b>；但验证集上错误 profile 为 <b>0.4285</b>，高于正确 profile 的 <b>0.4260</b>。因此只能报告“小幅正向趋势，尚未稳定复现”。</p></section>
<div class="grid">
<div class="metric"><b>2,000</b>总样本</div><div class="metric"><b>16</b>会话 / profile</div>
<div class="metric"><b>3</b>随机种子</div><div class="metric"><b>384</b>维冻结 profile 向量</div>
</div>
<h2>主结果</h2><div class="card"><table><thead><tr><th>条件</th><th>测试 Macro-F1</th><th>测试 Balanced Acc.</th><th>测试 Log Loss</th><th>验证 Macro-F1</th></tr></thead><tbody>{result_rows}</tbody></table></div>
<h2>测试集逐类 F1</h2><div class="card"><table><thead><tr><th>类别</th><th>不提供 profile</th><th>正确 profile</th><th>错误 profile</th></tr></thead><tbody>{per_class_rows}</tbody></table></div>
<h2>三次训练与音频敏感性</h2><div class="card"><table><thead><tr><th>Seed</th><th>Hidden</th><th>Given</th><th>Shuffled</th><th>音频置零后改变比例</th></tr></thead><tbody>{seed_rows}</tbody></table><p class="small">输出没有塌缩。音频置零会改变 63.2%–73.6% 的预测，说明模型确实使用音频。</p></div>
<h2>一条真实输入示例</h2><div class="card">
<p><b>样本：</b>{html.escape(example_request['sample_id'])}　<b>目标：</b>{html.escape(example_prediction['target'])}</p>
<audio controls src="{audio_href}"></audio>
<h3>截止预测点的转写</h3><pre>{html.escape(example_request['transcript_prefix'])}</pre>
<h3>正确 Profile 原文</h3><pre>{html.escape(example_request['profile_text'])}</pre>
<h3>同一 checkpoint 的三种输出（seed {seed}）</h3><ul>{probabilities}</ul>
</div>
<h2>模型做了什么</h2><div class="card"><p>冻结的 <code>all-MiniLM-L6-v2</code> 把原来的四行自然语言 profile 转成 384 维向量。训练的只是小型音频/转写融合器、profile 投影与五分类头。三种条件中的音频、转写、样本、预测边界和设置完全相同。</p></div>
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/semantic-profile-embedding/minilm-additive-raw-v2",
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed/sbcsae_semantic_profile_v1/test",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    artifact_dir = Path(args.artifact_dir)
    output = Path(args.output) if args.output else artifact_dir / "review.html"
    build(artifact_dir, Path(args.data_dir), output)
    print(output.resolve())


if __name__ == "__main__":
    main()
