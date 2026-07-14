"""Small dependency-free review UI for candidate SBCSAE event labels."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .constants import LABELS
from .utils import read_jsonl, write_json, write_jsonl


def build_review_page(run_dir: str | Path) -> dict[str, Any]:
    """Create a local HTML reviewer for the unique samples in an MLLM run."""

    root = Path(run_dir)
    requests = list(read_jsonl(root / "requests.jsonl"))
    gold = list(read_jsonl(root / "gold.jsonl"))
    target_by_sample: dict[str, str] = {}
    for row in gold:
        target_by_sample.setdefault(str(row["sample_id"]), str(row["target"]))
    hidden_by_sample = {
        str(row["sample_id"]): row
        for row in requests
        if row.get("profile_mode") == "hidden"
    }
    missing = sorted(set(target_by_sample) - set(hidden_by_sample))
    if missing:
        raise ValueError(f"Missing hidden requests for {len(missing)} review samples")
    items = []
    for sample_id in sorted(target_by_sample):
        request = hidden_by_sample[sample_id]
        items.append(
            {
                "sample_id": sample_id,
                "conversation_id": request["conversation_id"],
                "prediction_time_s": request["prediction_time_s"],
                "weak_label": target_by_sample[sample_id],
                "audio_path": str(request["audio_path"]).replace("\\", "/"),
                "transcript_prefix": request.get("transcript_prefix", ""),
            }
        )
    page = _review_html(items)
    (root / "review.html").write_text(page, encoding="utf-8")
    write_json(root / "review_items.json", items)
    return {
        "review_samples": len(items),
        "labels": {label: sum(row["weak_label"] == label for row in items) for label in LABELS},
        "review_page": str((root / "review.html").resolve()),
        "instructions": "Open review.html, label every item, then export reviewed_labels.json.",
    }


def apply_reviewed_labels(
    source_manifest: str | Path,
    review_json: str | Path,
    output_manifest: str | Path,
    *,
    reviewer_id: str,
) -> dict[str, Any]:
    """Apply a complete exported review and promote selected rows to reviewed labels."""

    payload = json.loads(Path(review_json).read_text(encoding="utf-8"))
    reviews = payload.get("reviews", payload) if isinstance(payload, dict) else payload
    if not isinstance(reviews, list):
        raise ValueError("Review JSON must contain a list or a reviews list")
    by_sample: dict[str, dict[str, Any]] = {}
    for row in reviews:
        sample_id = str(row.get("sample_id", ""))
        label = str(row.get("human_label", "")).upper()
        if not sample_id or label not in LABELS:
            raise ValueError(f"Incomplete or invalid review row: {row!r}")
        if sample_id in by_sample:
            raise ValueError(f"Duplicate review for {sample_id}")
        by_sample[sample_id] = row
    source_rows = list(read_jsonl(source_manifest))
    source_by_id = {str(row["sample_id"]): row for row in source_rows}
    missing = sorted(set(by_sample) - set(source_by_id))
    if missing:
        raise ValueError(f"{len(missing)} reviewed sample IDs are absent from the source manifest")
    selected = []
    changes = 0
    for sample_id, review in by_sample.items():
        row = dict(source_by_id[sample_id])
        old_label = str(row["label"])
        new_label = str(review["human_label"]).upper()
        changes += old_label != new_label
        row.update(
            {
                "label": new_label,
                "gold_label": True,
                "label_source": "single_annotator_human_review_v1",
                "weak_label_before_review": old_label,
                "reviewer_id": reviewer_id,
                "review_note": str(review.get("note", "")),
            }
        )
        selected.append(row)
    selected.sort(key=lambda row: str(row["sample_id"]))
    write_jsonl(output_manifest, selected)
    report = {
        "reviewer_id": reviewer_id,
        "reviewed_samples": len(selected),
        "changed_labels": changes,
        "labels": {label: sum(row["label"] == label for row in selected) for label in LABELS},
        "output_manifest": str(Path(output_manifest).resolve()),
        "limitation": "single-annotator reviewed labels; use a second annotator on the ambiguity subset for agreement",
    }
    write_json(Path(output_manifest).with_suffix(".review.json"), report)
    return report


def _review_html(items: list[dict[str, Any]]) -> str:
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    buttons = "".join(
        f'<button onclick="mark(\'{label}\')">{index + 1}: {label}</button>'
        for index, label in enumerate(LABELS)
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>SBCSAE label review</title>
<style>
body{{font:16px system-ui;max-width:1000px;margin:24px auto;padding:0 16px;background:#f6f7f9}}
.card{{background:white;padding:20px;border-radius:12px;box-shadow:0 2px 10px #0001}}
button{{margin:5px;padding:10px 16px}} audio{{width:100%;margin:12px 0}}
pre{{white-space:pre-wrap;max-height:260px;overflow:auto;background:#f3f3f3;padding:12px}}
#weak{{display:none}} textarea{{width:100%;height:70px}} .meta{{color:#555}}
</style></head><body><h1>SBCSAE 500-event label review</h1>
<p>只根据音频和截止 t 的转写判断未来 40 ms 状态。快捷键 1–5 对应 C/BC/T/I/NA，U=不确定。</p>
<div class="card"><div id="progress"></div><h2 id="id"></h2><div class="meta" id="meta"></div>
<audio id="audio" controls preload="metadata"></audio>
<div>{buttons}<button onclick="mark('UNSURE')">U: 不确定</button></div>
<button onclick="toggleWeak()">显示/隐藏弱标签</button><b id="weak"></b>
<h3>截止 t 已完成的转写单元</h3><pre id="transcript"></pre>
<label>备注</label><textarea id="note" oninput="saveNote()"></textarea>
<div><button onclick="move(-1)">上一条</button><button onclick="move(1)">下一条</button>
<button onclick="nextPending()">下一条未标</button><button onclick="exportReviews()">导出 JSON</button></div></div>
<script>
const items={data}; const key='sbcsae-review-v2'; let state=JSON.parse(localStorage.getItem(key)||'{{}}'); let i=0;
function persist(){{localStorage.setItem(key,JSON.stringify(state))}}
function render(){{let x=items[i],r=state[x.sample_id]||{{}}; progress.textContent=`${{i+1}}/${{items.length}}；已完成 ${{Object.values(state).filter(v=>v.human_label&&v.human_label!='UNSURE').length}}`;
id.textContent=x.sample_id; meta.textContent=`${{x.conversation_id}} | t=${{x.prediction_time_s}} s | 当前选择=${{r.human_label||'未标'}}`;
audio.src=x.audio_path; audio.onloadedmetadata=()=>{{audio.currentTime=Math.max(0,audio.duration-6)}};
transcript.textContent=x.transcript_prefix; weak.textContent='弱标签：'+x.weak_label; note.value=r.note||'';}}
function mark(label){{let x=items[i]; state[x.sample_id]={{...(state[x.sample_id]||{{}}),sample_id:x.sample_id,human_label:label,note:note.value}};persist();render();if(label!='UNSURE')move(1)}}
function saveNote(){{let x=items[i];state[x.sample_id]={{...(state[x.sample_id]||{{}}),sample_id:x.sample_id,note:note.value}};persist()}}
function move(d){{i=Math.max(0,Math.min(items.length-1,i+d));render()}}
function nextPending(){{let n=items.findIndex((x,j)=>j>i&&(!state[x.sample_id]||!state[x.sample_id].human_label||state[x.sample_id].human_label=='UNSURE'));i=n<0?i:n;render()}}
function toggleWeak(){{weak.style.display=weak.style.display=='none'?'inline':'none'}}
function exportReviews(){{let reviews=items.map(x=>state[x.sample_id]||{{sample_id:x.sample_id,human_label:'',note:''}});let blob=new Blob([JSON.stringify({{schema_version:'1.0',reviews}},null,2)],{{type:'application/json'}});let a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='reviewed_labels.json';a.click()}}
document.addEventListener('keydown',e=>{{if(document.activeElement===note)return;let labels=['C','BC','T','I','NA'];if(e.key>='1'&&e.key<='5')mark(labels[+e.key-1]);if(e.key.toLowerCase()=='u')mark('UNSURE')}});render();
</script></body></html>"""


__all__ = ["apply_reviewed_labels", "build_review_page"]
