"""One-sample compatibility smoke test for the true Qwen soft-prompt adapter.

This is deliberately not an accuracy experiment.  It verifies that a real
Qwen2.5-Omni multimodal A/B prompt can be cached, continued by profile soft
tokens, scored by the original LM head, and trained without updating Qwen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from profile_turntaking.audio import read_wav_window_robust_mix
from profile_turntaking.qwen_hidden_profile_experiment import (
    QwenThinkerHiddenEncoder,
    _move_tensor_inputs,
)
from profile_turntaking.qwen_soft_prompt_adapter import (
    FrozenQwenSoftPromptAB,
    ProfileSoftTokenAdapter,
    build_paper_ab_prompt,
    exact_single_token_id,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = read_jsonl(Path(args.requests))
    with np.load(args.cache, allow_pickle=False) as payload:
        sample_ids = payload["sample_ids"].astype(str)
        labels = payload["labels"].astype(np.int64)
        given_profiles = payload["profile_given"].astype(np.float32)
        shuffled_profiles = payload["profile_shuffled"].astype(np.float32)
    # A turn-change A/B record must have C(0) or T(2) as its reference.  The
    # reference selects an eligible smoke sample but is never inserted in prompt.
    index = int(np.flatnonzero(np.isin(labels, [0, 2]))[0])
    sample_id = sample_ids[index]
    matches = [
        row for row in records
        if row["sample_id"] == sample_id and row["profile_mode"] == "given"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one given request for {sample_id}, got {len(matches)}")
    record = matches[0]

    encoder = QwenThinkerHiddenEncoder(
        args.model_dir,
        torch_dtype="float16",
        device_map="auto",
        local_files_only=True,
        offload_folder=Path(args.output).parent / "offload",
    )
    prompt = build_paper_ab_prompt(record, "turn_change")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": "causal_audio.wav"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = encoder.processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    sample_rate = 16_000
    audio = read_wav_window_robust_mix(
        record["audio_path"],
        float(record["audio_window_start_s"]),
        float(record["audio_window_end_s"]),
        target_rate=sample_rate,
    )
    inputs = encoder.processor(
        text=[text],
        audio=[audio],
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = _move_tensor_inputs(inputs, encoder.device)
    qwen_dim = int(encoder.model.config.get_text_config().hidden_size)
    adapter = ProfileSoftTokenAdapter(
        profile_dim=int(given_profiles.shape[1]),
        qwen_dim=qwen_dim,
        prefix_length=4,
        bottleneck_dim=256,
        dropout=0.0,
    ).to(encoder.device)
    tokenizer = encoder.processor.tokenizer
    answer_ids = (
        exact_single_token_id(tokenizer, "A"),
        exact_single_token_id(tokenizer, "B"),
    )
    model = FrozenQwenSoftPromptAB(
        encoder.model,
        adapter,
        answer_token_ids=answer_ids,
    )
    prepared = model.prepare_multimodal_prefix(inputs)
    profile_tensors = {
        "hidden": torch.zeros(1, given_profiles.shape[1], device=encoder.device),
        "given": torch.from_numpy(given_profiles[index : index + 1]).to(encoder.device),
        "shuffled": torch.from_numpy(shuffled_profiles[index : index + 1]).to(encoder.device),
    }
    before = {
        mode: model.score_prepared(prepared, profile, "turn_change").detach().cpu()
        for mode, profile in profile_tensors.items()
    }
    target = torch.tensor([0 if labels[index] == 0 else 1], device=encoder.device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    train_logits = model.score_prepared(prepared, profile_tensors["given"], "turn_change")
    loss = torch.nn.functional.cross_entropy(train_logits, target)
    loss.backward()
    base_has_gradient = any(parameter.grad is not None for parameter in encoder.model.parameters())
    adapter_has_gradient = any(parameter.grad is not None for parameter in adapter.parameters())
    optimizer.step()
    after_given = model.score_prepared(
        prepared, profile_tensors["given"], "turn_change"
    ).detach().cpu()

    report = {
        "status": "passed",
        "scope": "one-sample architecture/hardware smoke test; not profile-effect evidence",
        "sample_id": sample_id,
        "input": "30 s causal audio + matching causal partial transcript + profile soft tokens",
        "question": "Talking-Turns-style turn-change A/B prompt",
        "answer_token_ids": {"A": answer_ids[0], "B": answer_ids[1]},
        "profile_soft_tokens": [1, adapter.prefix_length, adapter.qwen_dim],
        "qwen_frozen": not any(parameter.requires_grad for parameter in encoder.model.parameters()),
        "qwen_received_gradient": base_has_gradient,
        "adapter_received_gradient": adapter_has_gradient,
        "training_loss": float(loss.detach().cpu()),
        "logits_before": {mode: value.tolist()[0] for mode, value in before.items()},
        "given_logits_after_one_step": after_given.tolist()[0],
        "ordinary_qwen_path": "wrapper delegates directly to thinker when adapter_enabled=False",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
