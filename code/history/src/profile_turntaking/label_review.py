"""Small dependency-free review UI for candidate SBCSAE event labels."""

from __future__ import annotations

import hashlib
import html
import json
import re
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .audio import read_wav_window
from .constants import LABELS
from .utils import read_jsonl, write_json, write_jsonl


_SAFE_FILE_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def _write_pcm16_wav(path: Path, samples: np.ndarray, sample_rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def _load_catalog_review_context(
    catalog_dir: str | Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, str]]]:
    """Load rows and reproduce the manifest's first-seen A/B mapping."""

    catalog = Path(catalog_dir)
    rows_by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(catalog / "utterances.jsonl"):
        rows_by_conversation[str(row["conversation_id"])].append(row)
    speaker_maps: dict[str, dict[str, str]] = {}
    for conversation_id, rows in rows_by_conversation.items():
        first_seen: list[str] = []
        for row in rows:
            speaker = str(row.get("speaker"))
            if row.get("is_person") and speaker not in first_seen:
                first_seen.append(speaker)
        speaker_maps[conversation_id] = {
            speaker: f"speaker_{chr(ord('A') + index)}"
            for index, speaker in enumerate(first_seen[:2])
        }
    return dict(rows_by_conversation), speaker_maps


def _annotation_evidence(
    *,
    rows: list[dict[str, Any]],
    speaker_map: dict[str, str],
    prediction_time_s: float,
    review_start_s: float,
    review_end_s: float,
) -> tuple[str, list[str]]:
    """Format boundary evidence kept strictly outside all model requests."""

    evidence: list[str] = []
    risk_flags: set[str] = set()
    horizon_end = prediction_time_s + 0.04
    for row in rows:
        start = float(row["start_s"])
        end = float(row["end_s"])
        if start >= review_end_s or end <= review_start_s:
            continue
        is_person = bool(row.get("is_person"))
        speaker = speaker_map.get(str(row.get("speaker")), "environment")
        position = (
            "before t"
            if end <= prediction_time_s
            else "after t"
            if start >= prediction_time_s
            else "crosses t"
        )
        text = str(row.get("text", "")).strip() or "<non-lexical/empty>"
        evidence.append(
            f"[{speaker} {start - prediction_time_s:+.3f}→"
            f"{end - prediction_time_s:+.3f}s | {position}] {text}"
        )
        overlaps_target = start < horizon_end and end > prediction_time_s
        if overlaps_target and is_person and not str(row.get("clean_text", "")).strip():
            risk_flags.add("target_has_nonlexical_human_unit")
        if overlaps_target and not is_person:
            risk_flags.add("target_has_environment_unit")
    return "\n".join(evidence) or "<no catalog units in review window>", sorted(risk_flags)


def build_review_page(
    run_dir: str | Path,
    *,
    source_manifest: str | Path | None = None,
    catalog_dir: str | Path | None = None,
    seconds_before_t: float = 3.0,
    seconds_after_t: float = 2.0,
) -> dict[str, Any]:
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
    source_by_sample = (
        {str(row["sample_id"]): row for row in read_jsonl(source_manifest)}
        if source_manifest is not None
        else {}
    )
    if catalog_dir is not None and source_manifest is None:
        raise ValueError("catalog_dir requires source_manifest")
    catalog_rows, speaker_maps = (
        _load_catalog_review_context(catalog_dir)
        if catalog_dir is not None
        else ({}, {})
    )
    items = []
    for sample_id in sorted(target_by_sample):
        request = hidden_by_sample[sample_id]
        review_audio_path = str(request["audio_path"]).replace("\\", "/")
        review_boundary_s = float(
            request.get("audio_duration_s", request.get("prediction_time_s", 0.0))
        )
        contains_future = False
        annotation_evidence = "<catalog evidence not requested>"
        risk_flags: list[str] = []
        if source_manifest is not None:
            source_row = source_by_sample.get(sample_id)
            if source_row is None:
                raise ValueError(f"Review sample {sample_id} is absent from source manifest")
            prediction_time = float(source_row["prediction_time_s"])
            review_start = max(0.0, prediction_time - seconds_before_t)
            review_end = prediction_time + seconds_after_t
            review_audio = read_wav_window(
                source_row["audio_path"], review_start, review_end, target_rate=16_000
            )
            safe_sample_id = _SAFE_FILE_PATTERN.sub("_", sample_id)
            review_clip = root / "review_clips" / f"{safe_sample_id}.wav"
            _write_pcm16_wav(review_clip, review_audio)
            review_audio_path = str(Path("review_clips") / review_clip.name).replace(
                "\\", "/"
            )
            review_boundary_s = prediction_time - review_start
            contains_future = True
            if catalog_dir is not None:
                annotation_evidence, risk_flags = _annotation_evidence(
                    rows=catalog_rows.get(str(source_row["conversation_id"]), []),
                    speaker_map=speaker_maps.get(str(source_row["conversation_id"]), {}),
                    prediction_time_s=prediction_time,
                    review_start_s=review_start,
                    review_end_s=review_end,
                )
            if source_row.get("event_representative_policy") != "onset":
                risk_flags.append("representative_is_not_event_onset")
            if (
                str(source_row.get("label")) in {"BC", "I"}
                and float(source_row.get("weak_event_start_s", prediction_time))
                < prediction_time - 1e-8
            ):
                risk_flags.append("BC_or_I_already_active_before_t")
            risk_flags = sorted(set(risk_flags))
        items.append(
            {
                "sample_id": sample_id,
                "conversation_id": request["conversation_id"],
                "prediction_time_s": request["prediction_time_s"],
                "weak_label": target_by_sample[sample_id],
                "audio_path": review_audio_path,
                "prediction_boundary_in_review_audio_s": round(review_boundary_s, 3),
                "annotation_only_future_audio": contains_future,
                "annotation_only_boundary_transcript": annotation_evidence,
                "risk_flags": risk_flags,
                "risk_score": len(risk_flags),
                "transcript_prefix": request.get("transcript_prefix", ""),
            }
        )
    items.sort(
        key=lambda item: (
            -int(item["risk_score"]),
            str(item["conversation_id"]),
            float(item["prediction_time_s"]),
        )
    )
    page = _review_html(items)
    (root / "review.html").write_text(page, encoding="utf-8")
    write_json(root / "review_items.json", items)
    return {
        "review_samples": len(items),
        "labels": {label: sum(row["weak_label"] == label for row in items) for label in LABELS},
        "review_page": str((root / "review.html").resolve()),
        "annotation_only_future_audio": source_manifest is not None,
        "annotation_only_boundary_transcript": catalog_dir is not None,
        "risk_flags": dict(
            Counter(flag for item in items for flag in item.get("risk_flags", []))
        ),
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
    item_count = len(items)
    page_fingerprint = hashlib.sha256(
        "\n".join(str(item["sample_id"]) for item in items).encode("utf-8")
    ).hexdigest()[:12]
    storage_key = f"sbcsae-review-v2-{page_fingerprint}"
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
</style></head><body><h1>SBCSAE {item_count}-event label review</h1>
<p>快捷键 1–5 对应 C/BC/T/I/NA，U=不确定。</p>
<p style="color:#a00"><b>标注专用音频包含 t 之后的证据，只用于确定 gold label，绝不是模型输入。</b></p>
<div class="card"><div id="progress"></div><h2 id="id"></h2><div class="meta" id="meta"></div>
<audio id="audio" controls preload="metadata"></audio>
<button onclick="playBoundary()">播放边界前后 0.8 秒</button>
<div>{buttons}<button onclick="mark('UNSURE')">U: 不确定</button></div>
<button onclick="toggleWeak()">显示/隐藏弱标签</button><b id="weak"></b>
<button onclick="nextRisk()">下一个高风险未标</button>
<h3>截止 t 已完成的转写单元</h3><pre id="transcript"></pre>
<h3 style="color:#a00">标注专用：边界前后转写证据（绝不进入模型）</h3><pre id="evidence"></pre>
<label>备注</label><textarea id="note" oninput="saveNote()"></textarea>
<div><button onclick="move(-1)">上一条</button><button onclick="move(1)">下一条</button>
<button onclick="nextPending()">下一条未标</button><button onclick="exportReviews()">导出 JSON</button>
<label><button type="button" onclick="importFile.click()">导入已有复核 JSON</button><input id="importFile" type="file" accept="application/json,.json" hidden onchange="importReviews(this.files[0])"></label></div></div>
<script>
const items={data}; const key='{storage_key}'; let state=JSON.parse(localStorage.getItem(key)||'{{}}'); let i=0;
function persist(){{localStorage.setItem(key,JSON.stringify(state))}}
function render(){{let x=items[i],r=state[x.sample_id]||{{}}; progress.textContent=`${{i+1}}/${{items.length}}；已完成 ${{Object.values(state).filter(v=>v.human_label&&v.human_label!='UNSURE').length}}`;
id.textContent=x.sample_id; meta.textContent=`${{x.conversation_id}} | 原录音 t=${{x.prediction_time_s}} s | 复核音频中的边界=${{x.prediction_boundary_in_review_audio_s}} s | 风险=${{x.risk_flags.length?x.risk_flags.join(', '):'无自动风险标记'}} | 当前选择=${{r.human_label||'未标'}}`;
audio.src=x.audio_path; audio.onloadedmetadata=()=>{{audio.currentTime=Math.max(0,x.prediction_boundary_in_review_audio_s-1.5)}};
transcript.textContent=x.transcript_prefix; evidence.textContent=x.annotation_only_boundary_transcript; weak.textContent='弱标签：'+x.weak_label; note.value=r.note||'';}}
function playBoundary(){{let x=items[i],end=x.prediction_boundary_in_review_audio_s+0.4;audio.currentTime=Math.max(0,x.prediction_boundary_in_review_audio_s-0.4);audio.play();let timer=setInterval(()=>{{if(audio.currentTime>=end||audio.paused){{audio.pause();clearInterval(timer)}}}},20)}}
function mark(label){{let x=items[i]; state[x.sample_id]={{...(state[x.sample_id]||{{}}),sample_id:x.sample_id,human_label:label,note:note.value}};persist();render();if(label!='UNSURE')move(1)}}
function saveNote(){{let x=items[i];state[x.sample_id]={{...(state[x.sample_id]||{{}}),sample_id:x.sample_id,note:note.value}};persist()}}
function move(d){{i=Math.max(0,Math.min(items.length-1,i+d));render()}}
function nextPending(){{let n=items.findIndex((x,j)=>j>i&&(!state[x.sample_id]||!state[x.sample_id].human_label||state[x.sample_id].human_label=='UNSURE'));i=n<0?i:n;render()}}
function nextRisk(){{let n=items.findIndex((x,j)=>j>i&&x.risk_score>0&&(!state[x.sample_id]||!state[x.sample_id].human_label||state[x.sample_id].human_label=='UNSURE'));if(n<0)n=items.findIndex(x=>x.risk_score>0&&(!state[x.sample_id]||!state[x.sample_id].human_label||state[x.sample_id].human_label=='UNSURE'));i=n<0?i:n;render()}}
function toggleWeak(){{weak.style.display=weak.style.display=='none'?'inline':'none'}}
function exportReviews(){{let reviews=items.map(x=>state[x.sample_id]||{{sample_id:x.sample_id,human_label:'',note:''}});let blob=new Blob([JSON.stringify({{schema_version:'1.0',reviews}},null,2)],{{type:'application/json'}});let a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='reviewed_labels.json';a.click()}}
function importReviews(file){{if(!file)return;let reader=new FileReader();reader.onload=()=>{{try{{let payload=JSON.parse(reader.result),reviews=Array.isArray(payload)?payload:payload.reviews;if(!Array.isArray(reviews))throw new Error('缺少 reviews 数组');let allowed=new Set(['C','BC','T','I','NA']),ids=new Set(items.map(x=>x.sample_id)),loaded=0;for(let row of reviews){{let label=String(row.human_label||'').toUpperCase();if(ids.has(String(row.sample_id))&&allowed.has(label)){{state[String(row.sample_id)]={{sample_id:String(row.sample_id),human_label:label,note:String(row.note||'')}};loaded++}}}}persist();render();alert(`已导入 ${{loaded}} 条与本页面重合的已复核标签。`)}}catch(error){{alert('导入失败：'+error.message)}}}};reader.readAsText(file);importFile.value=''}}
document.addEventListener('keydown',e=>{{if(document.activeElement===note)return;let labels=['C','BC','T','I','NA'];if(e.key>='1'&&e.key<='5')mark(labels[+e.key-1]);if(e.key.toLowerCase()=='u')mark('UNSURE')}});render();
</script></body></html>"""


__all__ = ["apply_reviewed_labels", "build_review_page"]
