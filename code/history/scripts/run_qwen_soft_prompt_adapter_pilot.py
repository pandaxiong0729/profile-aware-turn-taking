"""Train the opt-in Qwen soft-prompt adapter on a bounded, leakage-audited pilot.

The default is intentionally tiny.  Increase per-class limits only after a
pilot run succeeds; Qwen is frozen but each 30-second audio prompt must still
be encoded once per sample.
"""
from __future__ import annotations

import argparse, json, random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from profile_turntaking.audio import read_wav_window_robust_mix
from profile_turntaking.qwen_hidden_profile_experiment import QwenThinkerHiddenEncoder, _move_tensor_inputs
from profile_turntaking.qwen_soft_prompt_adapter import (
    PAPER_TASKS, FrozenQwenSoftPromptAB, ProfileSoftTokenAdapter,
    build_paper_ab_prompt, exact_single_token_id,
)


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def targets(cache: dict[str, np.ndarray], task: str) -> np.ndarray:
    labels = cache["labels"]
    y = np.full(len(labels), -100, dtype=np.int64)
    if task == "turn_change": y[labels == 0], y[labels == 2] = 0, 1
    elif task == "backchannel": y[labels == 0], y[labels == 1] = 0, 1
    elif task == "interruption": y[labels == 0], y[labels == 3] = 0, 1
    else: y = cache["paper_targets"][:, 3].astype(np.int64)
    return y


def selected(cache: dict[str, np.ndarray], limit: int) -> list[tuple[int, str, int]]:
    result: list[tuple[int, str, int]] = []
    for task in PAPER_TASKS:
        y = targets(cache, task)
        for value in (0, 1):
            idx = np.flatnonzero(y == value)
            # Stable selection independent of model outputs.
            idx = sorted(idx.tolist(), key=lambda i: str(cache["sample_ids"][i]))[:limit]
            result += [(i, task, value) for i in idx]
    return result


def audit_requests(requests: list[dict[str, Any]], cache: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in requests: grouped[row["sample_id"]][row["profile_mode"]] = row
    wanted = set(cache["sample_ids"].astype(str).tolist())
    chosen: dict[str, dict[str, Any]] = {}
    for sample_id in wanted:
        group = grouped[sample_id]
        if set(group) != {"hidden", "given", "shuffled"}: raise ValueError(f"profile triplet missing: {sample_id}")
        common = ("audio_window_sha256", "transcript_sha256", "causal_asr_sha256", "audio_window_end_s")
        if any(len({str(group[m].get(k, "")) for m in group}) != 1 for k in common):
            raise ValueError(f"non-profile input changed: {sample_id}")
        if any("reference_label" in group[m] for m in group): raise ValueError("label leaked into request")
        chosen[sample_id] = group["given"]
    return chosen


def make_inputs(encoder: QwenThinkerHiddenEncoder, row: dict[str, Any], task: str) -> dict[str, Any]:
    text = encoder.processor.apply_chat_template([{"role":"user","content":[
        {"type":"audio", "audio":"causal_audio.wav"}, {"type":"text", "text":build_paper_ab_prompt(row, task)}]}], add_generation_prompt=True, tokenize=False)
    audio = read_wav_window_robust_mix(row["audio_path"], float(row["audio_window_start_s"]), float(row["audio_window_end_s"]))
    return _move_tensor_inputs(encoder.processor(text=[text], audio=[audio], sampling_rate=16000, return_tensors="pt", padding=True), encoder.device)


def evaluate(model: FrozenQwenSoftPromptAB, encoder: QwenThinkerHiddenEncoder, records: list[tuple[int,str,int]], request_map: dict[str,dict[str,Any]], cache: dict[str,np.ndarray]) -> dict[str,float]:
    outputs: dict[str, list[int]] = {m: [] for m in ("hidden","given","shuffled")}; gold: list[int] = []
    with torch.no_grad():
        for index, task, target in records:
            prepared = model.prepare_multimodal_prefix(make_inputs(encoder, request_map[str(cache["sample_ids"][index])], task))
            for mode, field in (("hidden", None), ("given", "profile_given"), ("shuffled", "profile_shuffled")):
                profile = torch.zeros(1, cache["profile_given"].shape[1], device=encoder.device) if field is None else torch.from_numpy(cache[field][index:index+1].astype(np.float32)).to(encoder.device)
                outputs[mode].append(int(model.score_prepared(prepared, profile, task).argmax(-1).item()))
            gold.append(target)
            del prepared
    return {mode: float(np.mean(np.asarray(pred) == np.asarray(gold))) for mode, pred in outputs.items()}


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--model-dir",required=True); p.add_argument("--train-requests",required=True); p.add_argument("--val-requests",required=True); p.add_argument("--test-requests",required=True); p.add_argument("--train-cache",required=True); p.add_argument("--val-cache",required=True); p.add_argument("--test-cache",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--train-per-class",type=int,default=1); p.add_argument("--eval-per-class",type=int,default=1); p.add_argument("--epochs",type=int,default=1); p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--seed",type=int,default=13); a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    caches=[]
    for path in (a.train_cache,a.val_cache,a.test_cache):
        with np.load(path,allow_pickle=False) as z: caches.append({k:z[k] for k in z.files})
    maps=[audit_requests(rows(Path(path)), cache) for path,cache in zip((a.train_requests,a.val_requests,a.test_requests),caches)]
    train_set, val_set, test_set = selected(caches[0],a.train_per_class), selected(caches[1],a.eval_per_class), selected(caches[2],a.eval_per_class)
    encoder=QwenThinkerHiddenEncoder(a.model_dir,torch_dtype="float16",device_map="auto",local_files_only=True,offload_folder=Path(a.output_dir)/"offload")
    adapter=ProfileSoftTokenAdapter(profile_dim=int(caches[0]["profile_given"].shape[1]),qwen_dim=int(encoder.model.config.get_text_config().hidden_size)).to(encoder.device)
    tok=encoder.processor.tokenizer; model=FrozenQwenSoftPromptAB(encoder.model,adapter,answer_token_ids=(exact_single_token_id(tok,"A"),exact_single_token_id(tok,"B")))
    opt=torch.optim.AdamW(adapter.parameters(),lr=a.lr)
    for _ in range(a.epochs):
        random.shuffle(train_set)
        for index,task,target in train_set:
            prepared=model.prepare_multimodal_prefix(make_inputs(encoder,maps[0][str(caches[0]["sample_ids"][index])],task))
            profile=torch.from_numpy(caches[0]["profile_given"][index:index+1].astype(np.float32)).to(encoder.device)
            opt.zero_grad(set_to_none=True); loss=torch.nn.functional.cross_entropy(model.score_prepared(prepared,profile,task),torch.tensor([target],device=encoder.device)); loss.backward(); opt.step(); del prepared
    report={"scope":"bounded soft-prompt pilot; do not treat a small pilot as final evidence","input":"causal audio + causal transcript + profile soft tokens","qwen_frozen":not any(x.requires_grad for x in encoder.model.parameters()),"train_records":len(train_set),"val_records":len(val_set),"test_records":len(test_set),"val_accuracy":evaluate(model,encoder,val_set,maps[1],caches[1]),"test_accuracy":evaluate(model,encoder,test_set,maps[2],caches[2])}
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); torch.save(adapter.state_dict(),out/"adapter.pt"); (out/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
