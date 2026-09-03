"""Build one static review page for the Qwen profile pilot runs."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def metric_card(title: str, value: str, note: str) -> str:
    return (
        '<div class="metric">'
        f'<div class="metric-title">{esc(title)}</div>'
        f'<div class="metric-value">{esc(value)}</div>'
        f'<div class="metric-note">{esc(note)}</div>'
        '</div>'
    )


def load_run(root: Path, name: str) -> dict[str, Any]:
    run = root / name
    requests = read_jsonl(run / "requests.jsonl")
    predictions = read_json(run / "predictions.json")
    by_sample: dict[str, dict[str, dict[str, Any]]] = {}
    for row in requests:
        by_sample.setdefault(str(row["sample_id"]), {})[str(row["profile_mode"])] = row
    return {
        "name": name,
        "root": run,
        "metrics": read_json(run / "metrics.json"),
        "diagnostics": read_json(run / "diagnostics.json"),
        "paired": read_json(run / "paired_changes.json"),
        "predictions": predictions,
        "requests": by_sample,
    }


def run_table(runs: list[dict[str, Any]]) -> str:
    rows = []
    for run in runs:
        m = run["metrics"]
        d = run["diagnostics"]
        p = run["paired"]
        rows.append(
            "<tr>"
            f"<td>{esc(run['name'])}</td>"
            f"<td>{pct(m['hidden']['accuracy'])}</td>"
            f"<td>{pct(m['given']['accuracy'])}</td>"
            f"<td>{pct(m['shuffled']['accuracy'])}</td>"
            f"<td>{m['hidden']['macro_f1']:.3f}</td>"
            f"<td>{m['given']['macro_f1']:.3f}</td>"
            f"<td>{m['shuffled']['macro_f1']:.3f}</td>"
            f"<td>{d['profile_pair_changes']['hidden_vs_given']['changed']}/50</td>"
            f"<td>{p['given_fixes_hidden_error']}</td>"
            f"<td>{p['given_breaks_hidden_correct']}</td>"
            "</tr>"
        )
    return (
        '<table><thead><tr>'
        '<th>样本组</th><th>隐藏准确率</th><th>真实准确率</th><th>错误准确率</th>'
        '<th>隐藏 Macro-F1</th><th>真实 Macro-F1</th><th>错误 Macro-F1</th>'
        '<th>真实 profile 改变答案</th><th>改对</th><th>改错</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>'
    )


def sample_card(base: Path, run: dict[str, Any], row: dict[str, Any]) -> str:
    sample_id = str(row["sample_id"])
    requests = run["requests"][sample_id]
    given = requests["given"]
    audio_path = (run["root"] / str(given["audio_path"])).resolve()
    audio_rel = audio_path.relative_to(base.resolve()).as_posix()
    changed = row["hidden_prediction"] != row["given_prediction"]
    fixed = (
        row["hidden_prediction"] != row["reference_label"]
        and row["given_prediction"] == row["reference_label"]
    )
    broken = (
        row["hidden_prediction"] == row["reference_label"]
        and row["given_prediction"] != row["reference_label"]
    )
    change_kind = "fixed" if fixed else "broken" if broken else "changed" if changed else "same"
    profile_text = given["profile_text"]
    shuffled_text = requests["shuffled"]["profile_text"]
    transcript = given.get("transcript_prefix") or "（预测点前没有完整转写单元）"
    return f"""
    <section class="sample" data-run="{esc(run['name'])}" data-label="{esc(row['reference_label'])}" data-change="{change_kind}">
      <div class="sample-head">
        <div><h3>{esc(sample_id)}</h3><span class="tag">{esc(run['name'])}</span></div>
        <div class="reference">参考：{esc(row['reference_label'])}</div>
      </div>
      <audio controls preload="metadata" src="{esc(audio_rel)}"></audio>
      <div class="predictions">
        <div><span>隐藏 profile</span><strong>{esc(row['hidden_prediction'])}</strong></div>
        <div><span>真实 profile</span><strong>{esc(row['given_prediction'])}</strong></div>
        <div><span>错误 profile</span><strong>{esc(row['shuffled_prediction'])}</strong></div>
        <div><span>变化</span><strong>{esc(change_kind)}</strong></div>
      </div>
      <details><summary>预测点前的部分转写</summary><pre>{esc(transcript)}</pre></details>
      <details><summary>真实 profile</summary><pre>{esc(profile_text)}</pre></details>
      <details><summary>错误 profile</summary><pre>{esc(shuffled_text)}</pre></details>
      <details><summary>实际提示词</summary><pre>{esc(given['prompt'])}</pre></details>
    </section>
    """


def build(root: Path) -> Path:
    runs = [
        load_run(root, "gate50-decision"),
        load_run(root, "gate50-decision-seed237"),
    ]
    pilot_rows = []
    for name in ("micro-main-10", "micro-decision-10", "micro-reasoned-10"):
        run = load_run(root, name)
        m = run["metrics"]
        d = run["diagnostics"]
        pilot_rows.append(
            "<tr>"
            f"<td>{esc(name)}</td>"
            f"<td>{pct(m['hidden']['accuracy'])}</td>"
            f"<td>{pct(m['given']['accuracy'])}</td>"
            f"<td>{esc(json.dumps(d['prediction_distribution']['hidden'], ensure_ascii=False))}</td>"
            f"<td>{d['profile_pair_changes']['hidden_vs_given']['changed']}/10</td>"
            "</tr>"
        )
    silence = read_json(root / "gate50-decision-silence" / "audio_sensitivity.json")
    cards = "".join(
        sample_card(root, run, row)
        for run in runs
        for row in run["predictions"]
    )
    metrics = "".join(
        [
            metric_card("第1批真实 profile", "28%", "隐藏 26%，错误 26%"),
            metric_card("第2批真实 profile", "26%", "隐藏 18%，错误 20%"),
            metric_card("两批样本", "99条不重复", "两批各50条，仅重合1条"),
            metric_card(
                "静音后答案改变",
                f"{silence['changed_predictions']}/50",
                "音频会影响部分判断，转写仍提供大量信息",
            ),
        ]
    )
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen2.5-Omni-3B Profile 对照实验</title>
<style>
:root{{--bg:#f5f7fb;--card:#fff;--text:#172033;--muted:#667085;--blue:#2563eb;--line:#dce3ee}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1500px;margin:auto;padding:28px}} h1{{margin:0 0 8px}} .lead{{color:var(--muted);font-size:17px}}
.notice{{background:#eaf2ff;border-left:5px solid var(--blue);padding:14px 18px;border-radius:10px;margin:18px 0}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:20px 0}}
.metric,.sample,.panel{{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:0 4px 14px #17315d0d}}
.metric{{padding:18px}} .metric-title,.metric-note{{color:var(--muted)}} .metric-value{{font-size:30px;font-weight:750;margin:8px 0}}
.panel{{padding:18px;margin:18px 0;overflow:auto}} table{{width:100%;border-collapse:collapse;min-width:900px}} th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:left}}
.filters{{position:sticky;top:0;background:#f5f7fbdd;backdrop-filter:blur(8px);padding:12px 0;z-index:2;display:flex;gap:10px;flex-wrap:wrap}}
select{{padding:9px 12px;border:1px solid var(--line);border-radius:9px;background:#fff}}
.sample{{padding:18px;margin:14px 0}} .sample-head{{display:flex;justify-content:space-between;gap:12px}} h3{{margin:0 0 8px}}
.tag{{background:#eaf2ff;color:#174ea6;border-radius:999px;padding:4px 9px}} .reference{{font-weight:700;color:#0f7a47}}
audio{{width:100%;margin:14px 0}} .predictions{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}
.predictions div{{background:#f7f9fc;border:1px solid var(--line);padding:10px;border-radius:9px}} .predictions span{{display:block;color:var(--muted);font-size:13px}}
.predictions strong{{font-size:21px}} details{{margin-top:10px}} summary{{cursor:pointer;font-weight:650}} pre{{white-space:pre-wrap;background:#f7f9fc;padding:12px;border-radius:9px;overflow:auto}}
.hidden{{display:none}} @media(max-width:800px){{main{{padding:14px}}.predictions{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main>
<h1>Qwen2.5-Omni-3B：Profile 对照实验</h1>
<p class="lead">同一个本地3B模型。每条主实验输入同时包含：预测点前的音频、匹配的部分转写、profile。hidden / given / shuffled 三组只改变 profile。</p>
<div class="notice"><strong>目前最直接的结果：</strong>两批几乎独立的50条中，真实 profile 都比隐藏 profile 准确；第二批也优于错误 profile。但模型目前只输出 C/I 两类，所以这是正向 pilot，不是最终五分类结果。</div>
<div class="metrics">{metrics}</div>
<div class="panel"><h2>两批50条结果</h2>{run_table(runs)}</div>
<div class="panel"><h2>三种提示方式的10条试验</h2><table><thead><tr><th>提示组</th><th>隐藏准确率</th><th>真实准确率</th><th>隐藏输出分布</th><th>真实 profile 改变答案</th></tr></thead><tbody>{''.join(pilot_rows)}</tbody></table></div>
<div class="filters">
  <select id="run"><option value="all">全部样本组</option><option value="gate50-decision">第1批50条</option><option value="gate50-decision-seed237">第2批50条</option></select>
  <select id="label"><option value="all">全部参考标签</option>{''.join(f'<option>{x}</option>' for x in ('C','BC','T','I','NA'))}</select>
  <select id="change"><option value="all">全部变化</option><option value="fixed">真实 profile 改对</option><option value="broken">真实 profile 改错</option><option value="changed">改变但未改对/改错</option><option value="same">答案不变</option></select>
</div>
<div id="samples">{cards}</div>
<script>
const controls=[...document.querySelectorAll('select')];
function apply(){{const run=document.querySelector('#run').value,label=document.querySelector('#label').value,change=document.querySelector('#change').value;document.querySelectorAll('.sample').forEach(x=>{{const ok=(run==='all'||x.dataset.run===run)&&(label==='all'||x.dataset.label===label)&&(change==='all'||x.dataset.change===change);x.classList.toggle('hidden',!ok)}})}}
controls.forEach(x=>x.addEventListener('change',apply));
</script></main></body></html>"""
    output = root / "review.html"
    output.write_text(page, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="artifacts/qwen25-omni-profile/q8-v10",
    )
    args = parser.parse_args()
    print(build(Path(args.root)).resolve())


if __name__ == "__main__":
    main()
