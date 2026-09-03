"""Build a static, audio-enabled review page for calibrated binary runs."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


LABELS = ("C", "BC", "T", "I", "NA")
MODES = ("hidden", "given", "shuffled")
MODE_NAMES = {"hidden": "隐藏 profile", "given": "真实 profile", "shuffled": "错误 profile"}
STAGE_NAMES = {
    "silence": "是否静音",
    "listener_onset": "听者是否开始回应",
    "brief_response": "是否只是短回应",
    "yield": "是否先让出话轮",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def load_run(path: Path) -> dict[str, Any]:
    base_requests: dict[str, dict[str, dict[str, Any]]] = {}
    for row in read_jsonl(path / "requests.jsonl"):
        base_requests.setdefault(str(row["sample_id"]), {})[
            str(row["profile_mode"])
        ] = row
    binary_predictions: dict[str, dict[str, dict[str, Any]]] = {}
    for row in read_jsonl(path / "binary_predictions.jsonl"):
        binary_predictions.setdefault(str(row["sample_id"]), {})[
            str(row["profile_mode"])
        ] = row
    prompts: dict[tuple[str, str], str] = {}
    for row in read_jsonl(path / "binary_requests.jsonl"):
        if row["profile_mode"] == "given" and row["binary_order"] == "ab":
            prompts[(str(row["sample_id"]), str(row["binary_stage"]))] = str(
                row["prompt"]
            )
    return {
        "name": path.name,
        "path": path,
        "metrics": read_json(path / "metrics.json"),
        "diagnostics": read_json(path / "diagnostics.json"),
        "aggregation": read_json(path / "binary_aggregation.json"),
        "predictions": read_json(path / "predictions.json"),
        "base_requests": base_requests,
        "binary_predictions": binary_predictions,
        "prompts": prompts,
    }


def metrics_table(runs: list[dict[str, Any]]) -> str:
    rows = []
    for run in runs:
        metrics = run["metrics"]
        diagnostics = run["diagnostics"]
        n = int(metrics["hidden"]["samples"])
        rows.append(
            "<tr>"
            f"<td>{esc(run['name'])}</td><td>{n}</td>"
            + "".join(
                f"<td>{100*metrics[mode]['accuracy']:.1f}%</td>"
                f"<td>{metrics[mode]['macro_f1']:.3f}</td>"
                for mode in MODES
            )
            + f"<td>{diagnostics['profile_pair_changes']['hidden_vs_given']['changed']}/{n}</td>"
            + "</tr>"
        )
    return (
        "<table><thead><tr><th>样本组</th><th>条数</th>"
        "<th>隐藏准确率</th><th>隐藏 Macro-F1</th>"
        "<th>真实准确率</th><th>真实 Macro-F1</th>"
        "<th>错误准确率</th><th>错误 Macro-F1</th>"
        "<th>真实 profile 改变答案</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def distribution_table(runs: list[dict[str, Any]]) -> str:
    rows = []
    for run in runs:
        distribution = run["diagnostics"]["prediction_distribution"]
        for mode in MODES:
            rows.append(
                f"<tr><td>{esc(run['name'])}</td><td>{MODE_NAMES[mode]}</td>"
                + "".join(f"<td>{int(distribution[mode][label])}</td>" for label in LABELS)
                + "</tr>"
            )
    return (
        "<table><thead><tr><th>样本组</th><th>条件</th>"
        + "".join(f"<th>{label}</th>" for label in LABELS)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def stage_table(run: dict[str, Any], sample_id: str) -> str:
    rows = []
    binary = run["binary_predictions"][sample_id]
    for stage, stage_name in STAGE_NAMES.items():
        cells = [f"<td>{esc(stage_name)}</td>"]
        for mode in MODES:
            row = binary[mode]
            answer = row["answers"][stage]
            odds = float(row["semantic_A_log_odds"][stage])
            cells.append(f"<td>{answer} <small>log-odds {odds:+.3f}</small></td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<table class='stages'><thead><tr><th>二问分支</th>"
        + "".join(f"<th>{MODE_NAMES[mode]}</th>" for mode in MODES)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def sample_card(output_parent: Path, run: dict[str, Any], row: dict[str, Any]) -> str:
    sample_id = str(row["sample_id"])
    base = run["base_requests"][sample_id]
    given = base["given"]
    audio = (run["path"] / str(given["audio_path"])).resolve()
    audio_url = audio.relative_to(output_parent.resolve()).as_posix()
    hidden = str(row["hidden_prediction"])
    profile = str(row["given_prediction"])
    shuffled = str(row["shuffled_prediction"])
    changed = hidden != profile
    prompts = "\n\n".join(
        f"===== {STAGE_NAMES[stage]} =====\n{run['prompts'][(sample_id, stage)]}"
        for stage in STAGE_NAMES
    )
    return f"""
<section class="sample" data-run="{esc(run['name'])}" data-label="{esc(row['reference_label'])}" data-changed="{'yes' if changed else 'no'}">
  <div class="head"><div><h3>{esc(sample_id)}</h3><span>{esc(run['name'])}</span></div><b>参考标签：{esc(row['reference_label'])}</b></div>
  <audio controls preload="metadata" src="{esc(audio_url)}"></audio>
  <div class="pred"><div>隐藏 profile<strong>{esc(hidden)}</strong></div><div>真实 profile<strong>{esc(profile)}</strong></div><div>错误 profile<strong>{esc(shuffled)}</strong></div></div>
  <details open><summary>四个二问分支的答案与校准分数</summary>{stage_table(run, sample_id)}</details>
  <details><summary>因果 ASR（音频只到预测点）</summary><pre>{esc(given['causal_asr_transcript'])}</pre></details>
  <details><summary>预测点前的说话人转写</summary><pre>{esc(given['transcript_prefix'])}</pre></details>
  <details><summary>真实 profile</summary><pre>{esc(given['profile_text'])}</pre></details>
  <details><summary>错误 profile</summary><pre>{esc(base['shuffled']['profile_text'])}</pre></details>
  <details><summary>实际送入模型的四个正序问题</summary><pre>{esc(prompts)}</pre></details>
</section>"""


def build(run_paths: list[Path], output: Path) -> Path:
    runs = [load_run(path.resolve()) for path in run_paths]
    output.parent.mkdir(parents=True, exist_ok=True)
    cards = "".join(
        sample_card(output.parent, run, row)
        for run in runs
        for row in run["predictions"]
    )
    run_options = "".join(
        f'<option value="{esc(run["name"])}">{esc(run["name"])}</option>' for run in runs
    )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen Profile 五分类复核</title>
<style>
body{{margin:0;background:#f5f7fb;color:#182033;font-family:system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1500px;margin:auto;padding:26px}}.notice,.panel,.sample{{background:#fff;border:1px solid #dce3ee;border-radius:14px;padding:18px;margin:16px 0;box-shadow:0 4px 14px #17315d0d}}.notice{{border-left:5px solid #2563eb}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #e2e8f0;text-align:left}}.panel{{overflow:auto}}.head{{display:flex;justify-content:space-between;gap:15px}}h3{{margin:0 0 7px}}.head span{{color:#2563eb}}audio{{width:100%;margin:14px 0}}.pred{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}.pred div{{background:#f7f9fc;padding:10px;border-radius:9px;color:#667085}}.pred strong{{display:block;font-size:24px;color:#182033}}details{{margin-top:10px}}summary{{cursor:pointer;font-weight:650}}pre{{white-space:pre-wrap;background:#f7f9fc;padding:12px;border-radius:9px}}small{{display:block;color:#667085}}.filters{{position:sticky;top:0;padding:10px 0;background:#f5f7fbe6;z-index:3}}select{{padding:9px;margin-right:8px;border:1px solid #ccd5e3;border-radius:8px}}.hidden{{display:none}}@media(max-width:800px){{main{{padding:12px}}.pred{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Qwen2.5-Omni-3B：Profile 五分类实验复核</h1>
<div class="notice">每条输入都包含同一段因果音频、同一份部分转写和同一预测点。hidden / given / shuffled 只改变 profile。五分类由四个论文式 A/B 问题得到；每个问题正反顺序各问一次，并用 token 概率消除选项位置偏差。</div>
<div class="panel"><h2>主要结果</h2>{metrics_table(runs)}</div>
<div class="panel"><h2>输出类别分布</h2>{distribution_table(runs)}</div>
<div class="filters"><select id="run"><option value="all">全部样本组</option>{run_options}</select><select id="label"><option value="all">全部参考标签</option>{''.join(f'<option>{label}</option>' for label in LABELS)}</select><select id="changed"><option value="all">全部 profile 变化</option><option value="yes">真实 profile 改变答案</option><option value="no">答案未变</option></select></div>
<div id="samples">{cards}</div>
<script>function apply(){{const r=document.querySelector('#run').value,l=document.querySelector('#label').value,c=document.querySelector('#changed').value;document.querySelectorAll('.sample').forEach(x=>x.classList.toggle('hidden',!((r==='all'||x.dataset.run===r)&&(l==='all'||x.dataset.label===l)&&(c==='all'||x.dataset.changed===c))))}}document.querySelectorAll('select').forEach(x=>x.addEventListener('change',apply));</script>
</main></body></html>"""
    output.write_text(page, encoding="utf-8")
    return output.resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(build([Path(path) for path in args.run], Path(args.output)))


if __name__ == "__main__":
    main()
