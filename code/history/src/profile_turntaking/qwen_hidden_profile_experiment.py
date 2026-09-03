"""Qwen-Omni hidden-state profile adapter experiment.

This module is the direct local follow-up to the prompt baseline:

* Qwen2.5-Omni Thinker reads the causal audio and matching causal transcript.
* The correct/shuffled profile is not written into that context prompt.
* The same frozen Qwen Thinker encodes profile text separately.
* A small trainable classifier learns whether the profile vector improves
  five-class turn-taking prediction.

The paired invariant is strict: within hidden/given/shuffled evaluation, the
Qwen audio+transcript vector is identical and only the profile vector changes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from .audio import read_wav_window, read_wav_window_robust_mix
from .constants import LABELS, LABEL_TO_ID
from .metrics import classification_metrics
from .semantic_profile_experiment import (
    PROFILE_MODES,
    load_and_audit_paired_requests,
)
from .utils import write_json, write_jsonl


DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-Omni-3B"
QWEN_CACHE_SCHEMA = "qwen-omni-thinker-hidden-profile-cache-v1"
CONTEXT_POOLING_MODES = ("prompt_last", "paper_audio_last", "rich")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_audio_array(samples: np.ndarray) -> str:
    audio = np.asarray(samples, dtype="<f4")
    return hashlib.sha256(audio.tobytes()).hexdigest()


def _label_counts(labels: np.ndarray) -> dict[str, int]:
    counts = Counter(int(value) for value in labels.tolist())
    return {label: int(counts.get(index, 0)) for index, label in enumerate(LABELS)}


def _prediction_distribution(predictions: Sequence[int]) -> dict[str, int]:
    counts = Counter(LABELS[int(index)] for index in predictions)
    return {label: int(counts.get(label, 0)) for label in LABELS}


def _pool_masked_hidden(
    hidden: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    mode: str,
    tail_tokens: int = 8,
) -> np.ndarray:
    """Pool one hidden vector per sample over a two-dimensional token mask."""

    if hidden.ndim != 3:
        raise ValueError(f"hidden must be [batch, sequence, dimension], got {hidden.shape}")
    mask = token_mask.to(hidden.device).bool()
    if mask.shape != hidden.shape[:2]:
        raise ValueError(f"token mask {mask.shape} does not match hidden {hidden.shape[:2]}")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("Every sample must contain at least one selected token")
    pooled: list[torch.Tensor] = []
    for sample_hidden, sample_mask in zip(hidden, mask):
        selected = sample_hidden[sample_mask]
        if mode == "last":
            vector = selected[-1]
        elif mode == "mean":
            vector = selected.mean(dim=0)
        elif mode == "tail_mean":
            vector = selected[-max(1, int(tail_tokens)) :].mean(dim=0)
        else:
            raise ValueError(f"Unknown masked pooling mode: {mode}")
        pooled.append(vector.detach().float().cpu())
    return torch.stack(pooled).numpy().astype(np.float32)


def build_qwen_context_prompt(
    record: dict[str, Any], *, context_mode: str = "audio_transcript"
) -> str:
    """Prompt used only to obtain a Qwen hidden vector for causal context.

    The prompt names the task and label space but deliberately excludes the
    profile text.  Profile enters later through a separate vector branch.
    """

    if context_mode not in {"audio_transcript", "audio_only"}:
        raise ValueError(f"Unknown Qwen context mode: {context_mode}")
    offset = int(record["forecast_offset_ms"])
    horizon = offset + int(record["evaluation_window_ms"])
    if context_mode == "audio_only":
        duration = float(record["audio_duration_s"])
        return "\n".join(
            [
                "You are encoding causal audio evidence for a two-person turn-taking prediction task.",
                f"The attached mono audio is {duration:.3f} seconds long and ends exactly at prediction boundary t.",
                "No transcript, ASR text, speaker-activity text, speaker profile, future audio, future transcript, or target label is provided.",
                f"Prediction target: estimate the event beginning in [t+{offset} ms, t+{horizon} ms).",
                "Possible labels are C, BC, T, I, and NA:",
                "C = current floor holder continues; no new listener response takes priority.",
                "BC = listener gives a short acknowledgement while the current speaker keeps the floor.",
                "T = current speaker yields and the other participant takes the floor.",
                "I = other participant starts a substantive contribution before the current speaker yields.",
                "NA = nobody is speaking at the decision point; silence begins or continues.",
                "",
                "Do not answer. Produce only an internal representation of the causal audio evidence.",
            ]
        )
    transcript = record["transcript_prefix"] or "No completed transcript unit is available."
    boundary = record["boundary_state_text"] or "No speaker activity summary is available."
    causal_asr = record["causal_asr_transcript"] or "No separate causal ASR is available."
    return "\n".join(
        [
            "You are encoding evidence for a two-person turn-taking prediction task.",
            "The attached mono audio ends exactly at prediction boundary t.",
            "Only audio and transcript before t are available. There is no future audio or future transcript.",
            f"Prediction target: classify the event beginning in [t+{offset} ms, t+{horizon} ms).",
            "Possible labels are C, BC, T, I, and NA:",
            "C = current floor holder continues; no new listener response takes priority.",
            "BC = listener gives a short acknowledgement while the current speaker keeps the floor.",
            "T = current speaker yields and the other participant takes the floor.",
            "I = other participant starts a substantive contribution before the current speaker yields.",
            "NA = nobody is speaking at the decision point; silence begins or continues.",
            "",
            "Completed transcript before t:",
            transcript,
            "",
            "Speaker activity exactly at t:",
            boundary,
            "",
            "Causal ASR of the same audio, including unfinished words nearest t if available:",
            causal_asr,
            "",
            "Do not answer. Produce only an internal representation of the causal evidence.",
        ]
    )


def build_qwen_profile_prompt(profile_text: str) -> str:
    """Prompt used to encode profile text as a separate frozen Qwen vector."""

    return "\n".join(
        [
            "Encode this speaker profile for a turn-taking prediction adapter.",
            "Focus on response style, listener behavior, floor-taking tendency, relationship, and situation.",
            "Do not answer the prediction task.",
            "",
            profile_text,
        ]
    )


def _move_tensor_inputs(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


class QwenThinkerHiddenEncoder:
    """Thin wrapper around Qwen2.5-Omni Thinker forward hidden states."""

    def __init__(
        self,
        model_name: str = DEFAULT_QWEN_MODEL,
        *,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        local_files_only: bool = False,
        cache_dir: str | Path | None = None,
        offload_folder: str | Path | None = None,
    ) -> None:
        try:
            from transformers import (  # type: ignore
                Qwen2_5OmniProcessor,
                Qwen2_5OmniThinkerForConditionalGeneration,
            )
        except Exception as exc:  # pragma: no cover - dependency path
            raise RuntimeError(
                "Qwen2.5-Omni hidden extraction requires a Transformers build "
                "with Qwen2_5Omni support."
            ) from exc

        dtype: str | torch.dtype
        if torch_dtype == "auto":
            dtype = "auto"
        elif torch_dtype in {"float16", "fp16"}:
            dtype = torch.float16
        elif torch_dtype in {"bfloat16", "bf16"}:
            dtype = torch.bfloat16
        elif torch_dtype in {"float32", "fp32"}:
            dtype = torch.float32
        else:
            raise ValueError(f"Unknown torch_dtype: {torch_dtype}")

        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": device_map,
            "local_files_only": local_files_only,
        }
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        if offload_folder is not None:
            kwargs["offload_folder"] = str(offload_folder)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_name,
            **kwargs,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.model_name = model_name
        # Qwen emits the same audio-output warning for every non-default system
        # prompt. This cache path never generates speech, so repeated warnings
        # are noise and can overwhelm multi-hour extraction logs.
        logging.getLogger().setLevel(logging.ERROR)

    def _pool_last_token(self, hidden: torch.Tensor, inputs: dict[str, Any]) -> np.ndarray:
        attention = inputs.get("attention_mask")
        if attention is None:
            indices = torch.full(
                (hidden.shape[0],),
                hidden.shape[1] - 1,
                dtype=torch.long,
                device=hidden.device,
            )
        else:
            mask = attention.to(hidden.device).bool()
            positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
            indices = positions.expand(mask.shape[0], -1).masked_fill(~mask, -1).max(dim=1).values
            indices = torch.clamp(indices, min=0, max=hidden.shape[1] - 1)
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        vectors = hidden[batch, indices].detach().float().cpu().numpy()
        return vectors.astype(np.float32)

    def _forward_hidden_with_audio_mask(
        self, inputs: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the Qwen Thinker backbone without the large vocabulary LM head.

        The public conditional-generation forward always materializes logits for
        every input token. Hidden-cache extraction only needs the final backbone
        state, so reproducing the documented multimodal merge and calling the
        backbone directly is both equivalent and substantially cheaper.
        """

        thinker = self.model
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        inputs_embeds = thinker.get_input_embeddings()(input_ids)
        input_features = inputs.get("input_features")
        feature_attention_mask = inputs.get("feature_attention_mask")
        audio_feature_lengths = inputs.get("audio_feature_lengths")
        audio_token_mask: torch.Tensor | None = None
        if input_features is not None:
            audio_features = thinker.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
                return_dict=True,
            ).last_hidden_state
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            _, _, audio_mask = thinker.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)
            audio_token_index = int(getattr(thinker.config, "audio_token_index"))
            audio_token_mask = input_ids.eq(audio_token_index)
            if attention_mask is not None:
                audio_token_mask = audio_token_mask & attention_mask.bool()

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        position_ids = None
        if attention_mask is not None:
            delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
            position_ids, rope_deltas = thinker.get_rope_index(
                input_ids,
                None,
                None,
                attention_mask,
                None,
                audio_feature_lengths,
                None,
            )
            thinker.rope_deltas = rope_deltas - delta0

        outputs = thinker.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            return_dict=True,
        )
        return outputs.last_hidden_state, audio_token_mask

    def _forward_last_hidden(self, inputs: dict[str, Any]) -> torch.Tensor:
        hidden, _ = self._forward_hidden_with_audio_mask(inputs)
        return hidden

    def _pool_context_features(
        self,
        hidden: torch.Tensor,
        inputs: dict[str, Any],
        audio_token_mask: torch.Tensor | None,
        *,
        context_pooling: str,
    ) -> np.ndarray:
        if context_pooling not in CONTEXT_POOLING_MODES:
            raise ValueError(f"Unknown context pooling: {context_pooling}")
        prompt_last = self._pool_last_token(hidden, inputs)
        if context_pooling == "prompt_last":
            return prompt_last
        if audio_token_mask is None:
            raise ValueError(f"{context_pooling} requires audio tokens")
        audio_last = _pool_masked_hidden(hidden, audio_token_mask, mode="last")
        if context_pooling == "paper_audio_last":
            return audio_last
        audio_tail = _pool_masked_hidden(
            hidden, audio_token_mask, mode="tail_mean", tail_tokens=8
        )
        audio_mean = _pool_masked_hidden(hidden, audio_token_mask, mode="mean")
        return np.concatenate(
            [prompt_last, audio_last, audio_tail, audio_mean], axis=1
        ).astype(np.float32)

    @torch.no_grad()
    def encode_text(self, prompt: str) -> np.ndarray:
        conversation = [{"role": "user", "content": prompt}]
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = _move_tensor_inputs(inputs, self.device)
        hidden = self._forward_last_hidden(inputs)
        return self._pool_last_token(hidden, inputs)[0]

    @torch.no_grad()
    def encode_text_batch(self, prompts: list[str]) -> np.ndarray:
        conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
        texts = [
            self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            for conversation in conversations
        ]
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = _move_tensor_inputs(inputs, self.device)
        hidden = self._forward_last_hidden(inputs)
        return self._pool_last_token(hidden, inputs)

    @torch.no_grad()
    def encode_audio_text(
        self,
        prompt: str,
        audio: np.ndarray,
        *,
        sample_rate: int = 16_000,
        context_pooling: str = "prompt_last",
    ) -> np.ndarray:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "causal_audio.wav"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(
            text=[text],
            audio=[np.asarray(audio, dtype=np.float32)],
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = _move_tensor_inputs(inputs, self.device)
        hidden, audio_token_mask = self._forward_hidden_with_audio_mask(inputs)
        return self._pool_context_features(
            hidden,
            inputs,
            audio_token_mask,
            context_pooling=context_pooling,
        )[0]

    @torch.no_grad()
    def encode_audio_text_batch(
        self,
        prompts: list[str],
        audios: list[np.ndarray],
        *,
        sample_rate: int = 16_000,
        context_pooling: str = "prompt_last",
    ) -> np.ndarray:
        if len(prompts) != len(audios) or not prompts:
            raise ValueError("prompts and audios must be non-empty and have equal length")
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": "causal_audio.wav"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            for prompt in prompts
        ]
        texts = [
            self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            for conversation in conversations
        ]
        inputs = self.processor(
            text=texts,
            audio=[np.asarray(audio, dtype=np.float32) for audio in audios],
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = _move_tensor_inputs(inputs, self.device)
        hidden, audio_token_mask = self._forward_hidden_with_audio_mask(inputs)
        return self._pool_context_features(
            hidden,
            inputs,
            audio_token_mask,
            context_pooling=context_pooling,
        )


def build_qwen_hidden_cache(
    run_dir: str | Path,
    cache_path: str | Path,
    *,
    model_name: str = DEFAULT_QWEN_MODEL,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    local_files_only: bool = False,
    model_cache_dir: str | Path | None = None,
    offload_folder: str | Path | None = None,
    limit: int | None = None,
    sample_rate: int = 16_000,
    encoder: QwenThinkerHiddenEncoder | None = None,
    context_mode: str = "audio_transcript",
    encoder_batch_size: int = 1,
    checkpoint_every: int = 100,
    resume: bool = True,
    context_pooling: str = "prompt_last",
) -> dict[str, Any]:
    """Encode one split into cached Qwen context/profile hidden vectors."""

    if context_mode not in {"audio_transcript", "audio_only"}:
        raise ValueError("context_mode must be 'audio_transcript' or 'audio_only'")
    if context_pooling not in CONTEXT_POOLING_MODES:
        raise ValueError(f"context_pooling must be one of {CONTEXT_POOLING_MODES}")

    records, paired_audit = load_and_audit_paired_requests(run_dir)
    if limit is not None:
        records = records[: max(0, int(limit))]
    if not records:
        raise ValueError("No records selected for Qwen hidden cache")

    if encoder is None:
        encoder = QwenThinkerHiddenEncoder(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            local_files_only=local_files_only,
            cache_dir=model_cache_dir,
            offload_folder=offload_folder,
        )
    encoder_batch_size = max(1, int(encoder_batch_size))
    checkpoint_every = max(1, int(checkpoint_every))
    destination = Path(cache_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(".partial.npz")
    context_vectors: list[np.ndarray] = []
    context_prompt_hashes: list[str] = []
    timings: list[float] = []
    start_index = 0
    if resume and partial_path.is_file():
        partial = np.load(partial_path, allow_pickle=False)
        partial_pooling = (
            str(partial["context_pooling"].reshape(-1)[0])
            if "context_pooling" in partial.files
            else "prompt_last"
        )
        if partial_pooling != context_pooling:
            raise ValueError(
                f"Partial cache pooling mismatch: {partial_pooling} != {context_pooling}"
            )
        saved_ids = partial["sample_ids"].astype(str).tolist()
        expected_ids = [record["sample_id"] for record in records[: len(saved_ids)]]
        if saved_ids != expected_ids:
            raise ValueError(f"Partial cache sample prefix mismatch: {partial_path}")
        context_vectors = [row for row in partial["qwen_context"].astype(np.float32)]
        context_prompt_hashes = partial["context_prompt_sha256"].astype(str).tolist()
        start_index = len(saved_ids)
        print(f"[qwen-hidden-cache] resumed {start_index}/{len(records)} from {partial_path}", flush=True)

    for batch_start in range(start_index, len(records), encoder_batch_size):
        batch_records = records[batch_start : batch_start + encoder_batch_size]
        prompts: list[str] = []
        audios: list[np.ndarray] = []
        for record in batch_records:
            prompt = build_qwen_context_prompt(record, context_mode=context_mode)
            if record.get("source_audio_path"):
                audio = read_wav_window_robust_mix(
                    record["source_audio_path"],
                    float(record["audio_window_start_s"]),
                    float(record["audio_window_end_s"]),
                    target_rate=sample_rate,
                )
                expected_window_sha = str(record["audio_window_sha256"])
                if expected_window_sha and _sha256_audio_array(audio) != expected_window_sha:
                    raise ValueError(
                        f"Audio window SHA mismatch for {record['sample_id']}: "
                        f"{_sha256_audio_array(audio)} != {expected_window_sha}"
                    )
            else:
                audio = read_wav_window(
                    record["audio_path"],
                    0.0,
                    float(record["audio_duration_s"]),
                    target_rate=sample_rate,
                )
            prompts.append(prompt)
            audios.append(audio)
            context_prompt_hashes.append(_sha256_text(prompt))
        started = time.perf_counter()
        vectors = encoder.encode_audio_text_batch(
            prompts,
            audios,
            sample_rate=sample_rate,
            context_pooling=context_pooling,
        )
        batch_seconds = time.perf_counter() - started
        context_vectors.extend([row.astype(np.float32) for row in vectors])
        timings.extend([batch_seconds / len(batch_records)] * len(batch_records))
        completed = batch_start + len(batch_records)
        print(
            f"[qwen-hidden-cache] context {completed}/{len(records)} "
            f"batch={len(batch_records)} seconds={batch_seconds:.2f} "
            f"per_sample={batch_seconds / len(batch_records):.2f}",
            flush=True,
        )
        if completed % checkpoint_every < len(batch_records) or completed == len(records):
            np.savez_compressed(
                partial_path,
                sample_ids=np.asarray([record["sample_id"] for record in records[:completed]]),
                qwen_context=np.stack(context_vectors).astype(np.float32),
                context_prompt_sha256=np.asarray(context_prompt_hashes),
                context_pooling=np.asarray([context_pooling]),
            )

    unique_profiles = sorted(
        {
            record[f"profile_text_{mode}"]
            for record in records
            for mode in ("given", "shuffled")
        }
    )
    profile_by_text: dict[str, np.ndarray] = {}
    for batch_start in range(0, len(unique_profiles), encoder_batch_size):
        texts = unique_profiles[batch_start : batch_start + encoder_batch_size]
        vectors = encoder.encode_text_batch([build_qwen_profile_prompt(text) for text in texts])
        for text, vector in zip(texts, vectors):
            profile_by_text[text] = vector.astype(np.float32)
        print(
            f"[qwen-hidden-cache] profile {batch_start + len(texts)}/{len(unique_profiles)}",
            flush=True,
        )

    context_array = np.stack(context_vectors).astype(np.float32)
    profile_given = np.stack(
        [profile_by_text[record["profile_text_given"]] for record in records]
    ).astype(np.float32)
    profile_shuffled = np.stack(
        [profile_by_text[record["profile_text_shuffled"]] for record in records]
    ).astype(np.float32)
    labels = np.asarray([LABEL_TO_ID[record["label"]] for record in records], dtype=np.int64)
    paper_targets: np.ndarray | None = None
    if any(record.get("paper_binary_targets") for record in records):
        task_order = ("turn_change", "backchannel", "interruption", "floor_taking")
        paper_targets = np.full((len(records), len(task_order)), -100, dtype=np.int64)
        for row_index, record in enumerate(records):
            targets = record.get("paper_binary_targets", {})
            for task_index, task in enumerate(task_order):
                value = targets.get(task) if isinstance(targets, dict) else None
                if value is not None:
                    paper_targets[row_index, task_index] = int(value)
    save_payload: dict[str, Any] = {
        "sample_ids": np.asarray([record["sample_id"] for record in records]),
        "conversation_ids": np.asarray([record["conversation_id"] for record in records]),
        "labels": labels,
        "qwen_context": context_array,
        "profile_given": profile_given,
        "profile_shuffled": profile_shuffled,
        "context_prompt_sha256": np.asarray(context_prompt_hashes),
    }
    if paper_targets is not None:
        save_payload["paper_targets"] = paper_targets
    np.savez_compressed(
        destination,
        **save_payload,
    )
    if partial_path.is_file():
        partial_path.unlink()
    profile_catalog = [
        {
            "profile_sha256": _sha256_text(text),
            "profile_text": text,
            "embedding_norm": float(np.linalg.norm(vector)),
        }
        for text, vector in profile_by_text.items()
    ]
    write_jsonl(destination.with_suffix(".profiles.jsonl"), profile_catalog)
    meta = {
        "schema_version": QWEN_CACHE_SCHEMA,
        "source_run_dir": str(Path(run_dir).resolve()),
        "cache_path": str(destination.resolve()),
        "cache_sha256": _sha256_file(destination),
        "model_name": model_name,
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "samples": len(records),
        "class_counts": _label_counts(labels),
        "context_mode": context_mode,
        "context_pooling": context_pooling,
        "context_pooling_definition": {
            "prompt_last": "last non-padding Thinker token after causal audio+text prompt",
            "paper_audio_last": "final Thinker hidden state at the last causal audio token",
            "rich": "concatenate prompt_last, audio_last, last-8-audio-token mean, and all-audio-token mean",
        }[context_pooling],
        "encoder_batch_size": encoder_batch_size,
        "checkpoint_every": checkpoint_every,
        "qwen_context_dimension": int(context_array.shape[1]),
        "profile_dimension": int(profile_given.shape[1]),
        "paper_binary_targets_saved": paper_targets is not None,
        "paper_binary_task_order": (
            ["turn_change", "backchannel", "interruption", "floor_taking"]
            if paper_targets is not None
            else None
        ),
        "unique_nonhidden_profile_texts": len(unique_profiles),
        "hidden_profile_representation": "all-zero vector",
        "qwen_context_prompt_contains_profile": False,
        "profile_text_encoded_by_qwen": True,
        "paired_input_audit": paired_audit,
        "mean_context_encode_seconds": float(np.mean(timings)) if timings else None,
        "total_context_encode_seconds": float(np.sum(timings)),
    }
    write_json(destination.with_suffix(".meta.json"), meta)
    return meta


def load_qwen_hidden_cache(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


@dataclass
class QwenHiddenTrainConfig:
    hidden_dimension: int = 256
    dropout: float = 0.2
    profile_dropout: float = 0.5
    epochs: int = 40
    patience: int = 8
    batch_size: int = 64
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    seeds: tuple[int, ...] = (13, 37, 71)
    device: str = "cpu"
    fusion: str = "gate"


class QwenHiddenProfileClassifier(nn.Module):
    """Small adapter over frozen Qwen context/profile vectors."""

    def __init__(
        self,
        context_dimension: int,
        profile_dimension: int,
        hidden_dimension: int,
        dropout: float,
        *,
        fusion: str = "gate",
    ) -> None:
        super().__init__()
        if fusion not in {"gate", "concat"}:
            raise ValueError("fusion must be 'gate' or 'concat'")
        self.fusion = fusion
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dimension),
        )
        self.profile_encoder = nn.Sequential(
            nn.Linear(profile_dimension, hidden_dimension, bias=False),
            nn.GELU(),
            nn.LayerNorm(hidden_dimension, elementwise_affine=False),
        )
        self.profile_gate = nn.Linear(hidden_dimension * 2, hidden_dimension)
        self.concat_adapter = nn.Sequential(
            nn.Linear(hidden_dimension * 2, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dimension),
        )
        self.output_norm = nn.LayerNorm(hidden_dimension)
        self.classifier = nn.Linear(hidden_dimension, len(LABELS))

    def forward(self, context: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        shared = self.context_encoder(context)
        profile_state = self.profile_encoder(profile)
        present = (profile.abs().sum(dim=-1, keepdim=True) > 0).to(profile_state.dtype)
        profile_state = profile_state * present
        if self.fusion == "concat":
            hidden = self.concat_adapter(torch.cat([shared, profile_state], dim=-1))
        else:
            gate = torch.sigmoid(self.profile_gate(torch.cat([shared, profile_state], dim=-1)))
            hidden = self.output_norm(shared + gate * profile_state)
        return self.classifier(hidden)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _standardize(
    train: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(train.std(axis=0, keepdims=True).astype(np.float32), 1e-5)
    return tuple(((array.astype(np.float32) - mean) / std) for array in (train, *others))


def _profile_array(data: dict[str, np.ndarray], mode: str) -> np.ndarray:
    if mode == "hidden":
        return np.zeros_like(data["profile_given"], dtype=np.float32)
    if mode in {"given", "shuffled"}:
        return data[f"profile_{mode}"].astype(np.float32)
    raise ValueError(f"Unknown profile mode: {mode}")


def _make_dataset(
    data: dict[str, np.ndarray],
    context: np.ndarray,
    profile_given: np.ndarray,
) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(context.astype(np.float32)),
        torch.from_numpy(profile_given.astype(np.float32)),
        torch.from_numpy(data["labels"].astype(np.int64)),
    )


def _ece(probabilities: np.ndarray, targets: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    result = 0.0
    for lower in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        upper = lower + 1.0 / bins
        mask = (confidence > lower) & (confidence <= upper)
        if bool(mask.any()):
            accuracy = float(np.mean(predictions[mask] == targets[mask]))
            result += float(mask.mean()) * abs(accuracy - float(confidence[mask].mean()))
    return result


@torch.no_grad()
def _predict(
    model: QwenHiddenProfileClassifier,
    data: dict[str, np.ndarray],
    *,
    context: np.ndarray,
    profile: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(context.astype(np.float32)), torch.from_numpy(profile.astype(np.float32))),
        batch_size=256,
        shuffle=False,
    )
    probabilities: list[np.ndarray] = []
    model.eval()
    for batch_context, batch_profile in loader:
        logits = model(batch_context.to(device), batch_profile.to(device))
        probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probs = np.concatenate(probabilities)
    predictions = probs.argmax(axis=1).astype(np.int64)
    targets = data["labels"].astype(np.int64)
    report = classification_metrics(targets.tolist(), predictions.tolist())
    one_hot = np.eye(len(LABELS), dtype=np.float32)[targets]
    report["brier_score"] = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    report["log_loss"] = float(
        -np.mean(np.log(np.maximum(probs[np.arange(len(targets)), targets], 1e-12)))
    )
    report["ece"] = _ece(probs, targets)
    report["prediction_distribution"] = _prediction_distribution(predictions.tolist())
    dominant = max(Counter(predictions.tolist()).values(), default=0) / max(1, len(predictions))
    report["noncollapsed"] = len(set(predictions.tolist())) >= 3 and dominant <= 0.8
    report["dominant_fraction"] = float(dominant)
    return report, predictions, probs


def _train_one_seed(
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    *,
    train_context: np.ndarray,
    val_context: np.ndarray,
    train_profile_given: np.ndarray,
    val_profiles: dict[str, np.ndarray],
    config: QwenHiddenTrainConfig,
    seed: int,
    output_dir: Path,
) -> tuple[QwenHiddenProfileClassifier, list[dict[str, Any]]]:
    _set_seed(seed)
    device = torch.device(config.device)
    dataset = _make_dataset(train, train_context, train_profile_given)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
    model = QwenHiddenProfileClassifier(
        context_dimension=int(train_context.shape[1]),
        profile_dimension=int(train_profile_given.shape[1]),
        hidden_dimension=config.hidden_dimension,
        dropout=config.dropout,
        fusion=config.fusion,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    counts = Counter(train["labels"].tolist())
    weights = torch.tensor(
        [len(train["labels"]) / max(1, len(LABELS) * counts.get(index, 0)) for index in range(len(LABELS))],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights / weights.mean())
    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for context, profile, labels in loader:
            context = context.to(device)
            profile = profile.to(device)
            labels = labels.to(device)
            if config.profile_dropout > 0:
                drop = torch.rand(profile.shape[0], device=device) < config.profile_dropout
                profile = profile.clone()
                profile[drop] = 0.0
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(context, profile), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            seen += len(labels)
        val_reports: dict[str, dict[str, Any]] = {}
        for mode in PROFILE_MODES:
            val_reports[mode], _, _ = _predict(
                model,
                val,
                context=val_context,
                profile=val_profiles[mode],
                device=device,
            )
        score = 0.5 * (
            val_reports["hidden"]["macro_f1"] + val_reports["given"]["macro_f1"]
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(1, seen),
                "val_selection_score": score,
                "val_hidden_macro_f1": val_reports["hidden"]["macro_f1"],
                "val_given_macro_f1": val_reports["given"]["macro_f1"],
                "val_shuffled_macro_f1": val_reports["shuffled"]["macro_f1"],
            }
        )
        if score > best_score + 1e-6:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "seed": seed,
            "config": asdict(config),
            "state_dict": best_state,
            "history": history,
            "context_dimension": int(train_context.shape[1]),
            "profile_dimension": int(train_profile_given.shape[1]),
        },
        output_dir / f"seed-{seed}.pt",
    )
    write_json(output_dir / f"seed-{seed}.train.json", {"seed": seed, "history": history})
    return model, history


def _write_profile_comparison_csv(path: Path, aggregate: dict[str, Any]) -> None:
    rows = []
    for mode in PROFILE_MODES:
        rows.append(
            {
                "profile_mode": mode,
                "macro_f1_mean": aggregate[mode]["macro_f1_mean"],
                "macro_f1_std": aggregate[mode]["macro_f1_std"],
                "balanced_accuracy_mean": aggregate[mode]["balanced_accuracy_mean"],
                "log_loss_mean": aggregate[mode]["log_loss_mean"],
                "brier_score_mean": aggregate[mode]["brier_score_mean"],
                "ece_mean": aggregate[mode]["ece_mean"],
            }
        )
    rows.append(
        {
            "profile_mode": "given_minus_hidden",
            "macro_f1_mean": aggregate["given_minus_hidden_macro_f1"]["mean"],
            "macro_f1_std": aggregate["given_minus_hidden_macro_f1"]["std"],
            "balanced_accuracy_mean": "",
            "log_loss_mean": "",
            "brier_score_mean": "",
            "ece_mean": "",
        }
    )
    rows.append(
        {
            "profile_mode": "given_minus_shuffled",
            "macro_f1_mean": aggregate["given_minus_shuffled_macro_f1"]["mean"],
            "macro_f1_std": aggregate["given_minus_shuffled_macro_f1"]["std"],
            "balanced_accuracy_mean": "",
            "log_loss_mean": "",
            "brier_score_mean": "",
            "ece_mean": "",
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_qwen_hidden_profile_experiment(
    train_cache: str | Path,
    val_cache: str | Path,
    test_cache: str | Path,
    output_dir: str | Path,
    *,
    config: QwenHiddenTrainConfig | None = None,
) -> dict[str, Any]:
    """Train/evaluate the small adapter over cached Qwen hidden vectors."""

    cfg = config or QwenHiddenTrainConfig()
    train = load_qwen_hidden_cache(train_cache)
    val = load_qwen_hidden_cache(val_cache)
    test = load_qwen_hidden_cache(test_cache)
    train_conversations = set(train["conversation_ids"].tolist())
    val_conversations = set(val["conversation_ids"].tolist())
    test_conversations = set(test["conversation_ids"].tolist())
    if train_conversations & val_conversations or train_conversations & test_conversations or val_conversations & test_conversations:
        raise ValueError("Conversation leakage across train/val/test")

    train_context, val_context, test_context = _standardize(
        train["qwen_context"], val["qwen_context"], test["qwen_context"]
    )
    given_train, given_val, given_test, shuffled_val, shuffled_test = _standardize(
        train["profile_given"],
        val["profile_given"],
        test["profile_given"],
        val["profile_shuffled"],
        test["profile_shuffled"],
    )
    zero_train = np.zeros_like(given_train, dtype=np.float32)
    val_profiles = {
        "hidden": np.zeros_like(given_val, dtype=np.float32),
        "given": given_val,
        "shuffled": shuffled_val,
    }
    test_profiles = {
        "hidden": np.zeros_like(given_test, dtype=np.float32),
        "given": given_test,
        "shuffled": shuffled_test,
    }

    destination = Path(output_dir)
    seed_reports: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        model, history = _train_one_seed(
            train,
            val,
            train_context=train_context,
            val_context=val_context,
            train_profile_given=given_train,
            val_profiles=val_profiles,
            config=cfg,
            seed=seed,
            output_dir=destination,
        )
        device = torch.device(cfg.device)
        validation_reports: dict[str, dict[str, Any]] = {}
        reports: dict[str, dict[str, Any]] = {}
        predictions_by_mode: dict[str, np.ndarray] = {}
        probabilities_by_mode: dict[str, np.ndarray] = {}
        for mode in PROFILE_MODES:
            validation_reports[mode], _, _ = _predict(
                model,
                val,
                context=val_context,
                profile=val_profiles[mode],
                device=device,
            )
            reports[mode], predictions_by_mode[mode], probabilities_by_mode[mode] = _predict(
                model,
                test,
                context=test_context,
                profile=test_profiles[mode],
                device=device,
            )
        reports["paired_effects"] = {
            "given_minus_hidden_macro_f1": reports["given"]["macro_f1"] - reports["hidden"]["macro_f1"],
            "given_minus_shuffled_macro_f1": reports["given"]["macro_f1"] - reports["shuffled"]["macro_f1"],
            "hidden_minus_given_log_loss": reports["hidden"]["log_loss"] - reports["given"]["log_loss"],
            "shuffled_minus_given_log_loss": reports["shuffled"]["log_loss"] - reports["given"]["log_loss"],
            "given_vs_hidden_changed_fraction": float(
                np.mean(predictions_by_mode["given"] != predictions_by_mode["hidden"])
            ),
            "given_vs_shuffled_changed_fraction": float(
                np.mean(predictions_by_mode["given"] != predictions_by_mode["shuffled"])
            ),
        }
        validation_reports["paired_effects"] = {
            "given_minus_hidden_macro_f1": validation_reports["given"]["macro_f1"] - validation_reports["hidden"]["macro_f1"],
            "given_minus_shuffled_macro_f1": validation_reports["given"]["macro_f1"] - validation_reports["shuffled"]["macro_f1"],
            "hidden_minus_given_log_loss": validation_reports["hidden"]["log_loss"] - validation_reports["given"]["log_loss"],
            "shuffled_minus_given_log_loss": validation_reports["shuffled"]["log_loss"] - validation_reports["given"]["log_loss"],
        }
        seed_payload = {
            "seed": seed,
            "epochs_run": len(history),
            "validation_reports": validation_reports,
            "reports": reports,
        }
        seed_reports.append(seed_payload)
        write_json(destination / f"seed-{seed}.metrics.json", seed_payload)
        for index, sample_id in enumerate(test["sample_ids"].tolist()):
            row = {
                "seed": seed,
                "sample_id": str(sample_id),
                "conversation_id": str(test["conversation_ids"][index]),
                "target": LABELS[int(test["labels"][index])],
            }
            for mode in PROFILE_MODES:
                row[f"prediction_{mode}"] = LABELS[int(predictions_by_mode[mode][index])]
                row[f"probabilities_{mode}"] = {
                    label: float(probabilities_by_mode[mode][index, label_index])
                    for label_index, label in enumerate(LABELS)
                }
            prediction_rows.append(row)
    write_jsonl(destination / "predictions.jsonl", prediction_rows)

    aggregate: dict[str, Any] = {}
    validation_aggregate: dict[str, Any] = {}
    for mode in PROFILE_MODES:
        aggregate[mode] = {
            "macro_f1_mean": float(np.mean([r["reports"][mode]["macro_f1"] for r in seed_reports])),
            "macro_f1_std": float(np.std([r["reports"][mode]["macro_f1"] for r in seed_reports])),
            "balanced_accuracy_mean": float(np.mean([r["reports"][mode]["balanced_accuracy"] for r in seed_reports])),
            "balanced_accuracy_std": float(np.std([r["reports"][mode]["balanced_accuracy"] for r in seed_reports])),
            "log_loss_mean": float(np.mean([r["reports"][mode]["log_loss"] for r in seed_reports])),
            "brier_score_mean": float(np.mean([r["reports"][mode]["brier_score"] for r in seed_reports])),
            "ece_mean": float(np.mean([r["reports"][mode]["ece"] for r in seed_reports])),
            "per_class_f1_mean": {
                label: float(np.mean([r["reports"][mode]["per_class"][label]["f1"] for r in seed_reports]))
                for label in LABELS
            },
            "all_seeds_noncollapsed": all(bool(r["reports"][mode]["noncollapsed"]) for r in seed_reports),
        }
        validation_aggregate[mode] = {
            "macro_f1_mean": float(np.mean([r["validation_reports"][mode]["macro_f1"] for r in seed_reports])),
            "macro_f1_std": float(np.std([r["validation_reports"][mode]["macro_f1"] for r in seed_reports])),
            "log_loss_mean": float(np.mean([r["validation_reports"][mode]["log_loss"] for r in seed_reports])),
            "all_seeds_noncollapsed": all(
                bool(r["validation_reports"][mode]["noncollapsed"]) for r in seed_reports
            ),
        }
    for key in (
        "given_minus_hidden_macro_f1",
        "given_minus_shuffled_macro_f1",
        "hidden_minus_given_log_loss",
        "shuffled_minus_given_log_loss",
    ):
        values = np.asarray([r["reports"]["paired_effects"][key] for r in seed_reports], dtype=float)
        val_values = np.asarray([r["validation_reports"]["paired_effects"][key] for r in seed_reports], dtype=float)
        aggregate[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "values": values.tolist(),
        }
        validation_aggregate[key] = {
            "mean": float(val_values.mean()),
            "std": float(val_values.std()),
            "values": val_values.tolist(),
        }

    summary = {
        "experiment": "qwen-omni-thinker-hidden-profile-adapter-v1",
        "qwen_finetuned": False,
        "training_performed": True,
        "trained_part": "small MLP/gate classifier over frozen Qwen hidden vectors",
        "input_contract": [
            "Qwen context vector: causal audio ending at t + matching causal transcript before t",
            "profile vector: fixed-template profile encoded separately by Qwen text path",
            "hidden/given/shuffled change only the profile vector",
        ],
        "config": asdict(cfg),
        "train_cache": str(Path(train_cache).resolve()),
        "val_cache": str(Path(val_cache).resolve()),
        "test_cache": str(Path(test_cache).resolve()),
        "split_conversations": {
            "train": sorted(train_conversations),
            "val": sorted(val_conversations),
            "test": sorted(test_conversations),
        },
        "samples": {
            "train": int(len(train["labels"])),
            "val": int(len(val["labels"])),
            "test": int(len(test["labels"])),
        },
        "class_counts": {
            "train": _label_counts(train["labels"]),
            "val": _label_counts(val["labels"]),
            "test": _label_counts(test["labels"]),
        },
        "validation_aggregate": validation_aggregate,
        "aggregate": aggregate,
        "seed_reports": seed_reports,
        "interpretation_gate": {
            "all_hidden_noncollapsed": all(bool(r["reports"]["hidden"]["noncollapsed"]) for r in seed_reports),
            "all_modes_noncollapsed": all(
                bool(r["reports"][mode]["noncollapsed"]) for r in seed_reports for mode in PROFILE_MODES
            ),
            "profile_effect_claim_allowed": (
                validation_aggregate["given_minus_hidden_macro_f1"]["mean"] > 0
                and validation_aggregate["given_minus_shuffled_macro_f1"]["mean"] > 0
                and aggregate["given_minus_hidden_macro_f1"]["mean"] > 0
                and aggregate["given_minus_shuffled_macro_f1"]["mean"] > 0
            ),
        },
    }
    write_json(destination / "summary.json", summary)
    _write_profile_comparison_csv(destination / "profile_comparison.csv", aggregate)
    return summary


__all__ = [
    "CONTEXT_POOLING_MODES",
    "DEFAULT_QWEN_MODEL",
    "QwenHiddenTrainConfig",
    "QwenThinkerHiddenEncoder",
    "build_qwen_context_prompt",
    "build_qwen_hidden_cache",
    "build_qwen_profile_prompt",
    "load_qwen_hidden_cache",
    "run_qwen_hidden_profile_experiment",
]
