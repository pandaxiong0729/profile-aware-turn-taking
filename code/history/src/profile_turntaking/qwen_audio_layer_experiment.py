"""Talking-Turns-style Qwen audio-layer boundary cache.

The frozen Qwen2.5-Omni audio tower returns the convolutional input state plus
all 32 encoder-layer states.  For every 30 s causal audio window, this module
stores the final valid frame from each layer.  Adapter training can then learn
a softmax-weighted sum over layers, matching the supervised topline design in
Talking Turns while retaining the existing causal transcript representation.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .audio import read_wav_window_robust_mix
from .constants import LABELS
from .qwen_hidden_profile_experiment import _sha256_audio_array
from .semantic_profile_experiment import load_and_audit_paired_requests
from .utils import write_json


AUDIO_LAYER_CACHE_SCHEMA = "qwen-omni-audio-layer-boundary-cache-v1"
AUDIO_TOWER_PREFIX = "thinker.audio_tower."


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _label_counts(labels: np.ndarray) -> dict[str, int]:
    counts = Counter(int(value) for value in labels.tolist())
    return {label: int(counts.get(index, 0)) for index, label in enumerate(LABELS)}


def _savez_compressed_atomic(path: Path, **payload: np.ndarray) -> None:
    """Write an NPZ checkpoint completely before replacing the prior file."""

    temporary = path.with_name(f"{path.stem}.tmp.npz")
    if temporary.exists():
        temporary.unlink()
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def _resolve_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"float16", "fp16", "half", "auto"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unknown torch dtype: {name}")


class QwenAudioLayerBoundaryEncoder:
    """Load only Qwen's audio tower and return one boundary vector per layer."""

    def __init__(
        self,
        model_name: str | Path,
        *,
        torch_dtype: str = "float16",
        device: str = "cuda",
        local_files_only: bool = True,
    ) -> None:
        try:
            from safetensors import safe_open  # type: ignore
            from transformers import (  # type: ignore
                Qwen2_5OmniConfig,
                Qwen2_5OmniProcessor,
            )
            from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (  # type: ignore
                Qwen2_5OmniAudioEncoder,
            )
        except Exception as exc:  # pragma: no cover - dependency path
            raise RuntimeError(
                "Qwen audio-layer extraction requires Transformers with "
                "Qwen2.5-Omni and safetensors support."
            ) from exc

        model_dir = Path(model_name).expanduser().resolve()
        if not model_dir.is_dir():
            raise ValueError(
                "Selective audio-tower loading currently requires a local Qwen "
                f"checkpoint directory, got: {model_name}"
            )
        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(index_path)

        dtype = _resolve_dtype(torch_dtype)
        self.device = torch.device(device)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            model_dir,
            local_files_only=local_files_only,
        )
        full_config = Qwen2_5OmniConfig.from_pretrained(
            model_dir,
            local_files_only=local_files_only,
        )
        audio_config = full_config.thinker_config.audio_config
        model = Qwen2_5OmniAudioEncoder(audio_config).to(dtype=dtype)

        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        audio_keys = sorted(key for key in weight_map if key.startswith(AUDIO_TOWER_PREFIX))
        if not audio_keys:
            raise ValueError(f"No {AUDIO_TOWER_PREFIX} weights found in {index_path}")
        keys_by_shard: dict[str, list[str]] = defaultdict(list)
        for key in audio_keys:
            keys_by_shard[str(weight_map[key])].append(key)
        state: dict[str, torch.Tensor] = {}
        for shard, keys in sorted(keys_by_shard.items()):
            with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
                for key in keys:
                    state[key.removeprefix(AUDIO_TOWER_PREFIX)] = handle.get_tensor(key)
        model.load_state_dict(state, strict=True)
        del state
        self.model = model.to(self.device).eval()
        self.model_name = str(model_dir)
        self.torch_dtype = str(dtype).removeprefix("torch.")
        self.layer_count = int(audio_config.encoder_layers) + 1
        self.layer_dimension = int(audio_config.d_model)
        self.audio_weight_keys = len(audio_keys)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "causal_audio.wav"},
                    {"type": "text", "text": "Causal audio boundary representation."},
                ],
            }
        ]
        self._audio_placeholder_text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

    def _processor_audio_inputs(
        self,
        audios: Sequence[np.ndarray],
        *,
        sample_rate: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.processor(
            text=[self._audio_placeholder_text] * len(audios),
            audio=[np.asarray(audio, dtype=np.float32) for audio in audios],
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs["input_features"]
        feature_mask = inputs["feature_attention_mask"]
        if input_features.ndim != 3:
            raise ValueError(f"Expected [batch, mel, time], got {input_features.shape}")
        if feature_mask.shape != (input_features.shape[0], input_features.shape[2]):
            raise ValueError(
                "Feature mask must use Mel-frame units: "
                f"{feature_mask.shape} vs {input_features.shape}"
            )
        return input_features, feature_mask

    @torch.no_grad()
    def encode_batch(
        self,
        audios: Sequence[np.ndarray],
        *,
        sample_rate: int = 16_000,
    ) -> np.ndarray:
        if not audios:
            return np.empty((0, self.layer_count, self.layer_dimension), dtype=np.float32)
        input_features, feature_mask = self._processor_audio_inputs(
            audios,
            sample_rate=sample_rate,
        )
        input_features = input_features.to(self.device, dtype=self.model.dtype)
        feature_mask = feature_mask.to(self.device)
        feature_lens = feature_mask.sum(dim=1)
        packed_features = input_features.permute(0, 2, 1)[feature_mask.bool()].permute(1, 0)
        aftercnn_lens, _ = self.model._get_feat_extract_output_lengths(feature_lens)
        outputs = self.model(
            packed_features,
            feature_lens=feature_lens,
            aftercnn_lens=aftercnn_lens,
            return_dict=True,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) != self.layer_count:
            raise ValueError(
                f"Expected {self.layer_count} audio hidden states, got "
                f"{None if hidden_states is None else len(hidden_states)}"
            )
        split_lengths = [int(value) for value in aftercnn_lens.tolist()]
        layer_boundaries: list[torch.Tensor] = []
        for hidden in hidden_states:
            if hidden.ndim != 2 or hidden.shape[1] != self.layer_dimension:
                raise ValueError(f"Unexpected audio layer hidden shape: {hidden.shape}")
            pieces = hidden.split(split_lengths, dim=0)
            if len(pieces) != len(audios) or any(len(piece) == 0 for piece in pieces):
                raise ValueError("Could not split packed audio hidden states by sample")
            layer_boundaries.append(torch.stack([piece[-1] for piece in pieces], dim=0))
        stacked = torch.stack(layer_boundaries, dim=1)
        if not bool(torch.isfinite(stacked).all()):
            raise ValueError("Non-finite Qwen audio-layer boundary vector")
        return stacked.detach().float().cpu().numpy().astype(np.float32)


def build_qwen_audio_layer_cache(
    run_dir: str | Path,
    base_cache_path: str | Path,
    cache_path: str | Path,
    *,
    model_name: str | Path,
    torch_dtype: str = "float16",
    device: str = "cuda",
    sample_rate: int = 16_000,
    encoder_batch_size: int = 8,
    checkpoint_every: int = 100,
    resume: bool = True,
    limit: int | None = None,
    encoder: QwenAudioLayerBoundaryEncoder | None = None,
) -> dict[str, Any]:
    """Build one split with transcript context + all audio-layer boundaries."""

    records, paired_audit = load_and_audit_paired_requests(run_dir)
    if limit is not None:
        records = records[: max(0, int(limit))]
    if not records:
        raise ValueError("No records selected for Qwen audio-layer cache")

    base_path = Path(base_cache_path).resolve()
    with np.load(base_path, allow_pickle=False) as payload:
        base = {key: payload[key] for key in payload.files}
    expected_ids = np.asarray([record["sample_id"] for record in records]).astype(str)
    if base["sample_ids"].astype(str).tolist()[: len(records)] != expected_ids.tolist():
        raise ValueError("Base Qwen cache sample IDs do not match run records")
    rich_context = base["qwen_context"][: len(records)]
    if rich_context.ndim != 2 or rich_context.shape[1] < 2048:
        raise ValueError(f"Base cache does not contain Qwen prompt-last context: {rich_context.shape}")
    prompt_last = rich_context[:, :2048].astype(np.float32)

    if encoder is None:
        encoder = QwenAudioLayerBoundaryEncoder(
            model_name,
            torch_dtype=torch_dtype,
            device=device,
        )
    encoder_batch_size = max(1, int(encoder_batch_size))
    checkpoint_every = max(1, int(checkpoint_every))
    destination = Path(cache_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_suffix(".partial.npz")
    layer_vectors: list[np.ndarray] = []
    timings: list[float] = []
    start_index = 0
    if resume and partial_path.is_file():
        partial = np.load(partial_path, allow_pickle=False)
        saved_ids = partial["sample_ids"].astype(str).tolist()
        if saved_ids != expected_ids[: len(saved_ids)].tolist():
            raise ValueError(f"Partial cache sample prefix mismatch: {partial_path}")
        saved_layers = partial["qwen_audio_layers"]
        if saved_layers.shape[1:] != (encoder.layer_count, encoder.layer_dimension):
            raise ValueError(f"Partial audio-layer shape mismatch: {saved_layers.shape}")
        layer_vectors = [row.astype(np.float32) for row in saved_layers]
        start_index = len(saved_ids)
        print(
            f"[qwen-audio-layers] resumed {start_index}/{len(records)} from {partial_path}",
            flush=True,
        )

    for batch_start in range(start_index, len(records), encoder_batch_size):
        batch_records = records[batch_start : batch_start + encoder_batch_size]
        audios: list[np.ndarray] = []
        for record in batch_records:
            audio = read_wav_window_robust_mix(
                record["source_audio_path"],
                float(record["audio_window_start_s"]),
                float(record["audio_window_end_s"]),
                target_rate=sample_rate,
            )
            expected_window_sha = str(record.get("audio_window_sha256", ""))
            if expected_window_sha and _sha256_audio_array(audio) != expected_window_sha:
                raise ValueError(f"Audio window SHA mismatch for {record['sample_id']}")
            audios.append(audio)
        started = time.perf_counter()
        vectors = encoder.encode_batch(audios, sample_rate=sample_rate)
        seconds = time.perf_counter() - started
        layer_vectors.extend([row.astype(np.float32) for row in vectors])
        timings.extend([seconds / len(batch_records)] * len(batch_records))
        completed = batch_start + len(batch_records)
        print(
            f"[qwen-audio-layers] {completed}/{len(records)} batch={len(batch_records)} "
            f"seconds={seconds:.2f} per_sample={seconds / len(batch_records):.2f}",
            flush=True,
        )
        if completed % checkpoint_every < len(batch_records) or completed == len(records):
            _savez_compressed_atomic(
                partial_path,
                sample_ids=expected_ids[:completed],
                qwen_audio_layers=np.stack(layer_vectors).astype(np.float16),
                layer_count=np.asarray([encoder.layer_count], dtype=np.int64),
                layer_dimension=np.asarray([encoder.layer_dimension], dtype=np.int64),
            )

    audio_layers = np.stack(layer_vectors).astype(np.float16)
    if audio_layers.shape != (len(records), encoder.layer_count, encoder.layer_dimension):
        raise ValueError(f"Final Qwen audio-layer cache shape mismatch: {audio_layers.shape}")
    payload: dict[str, np.ndarray] = {
        "sample_ids": expected_ids,
        "conversation_ids": base["conversation_ids"][: len(records)],
        "labels": base["labels"][: len(records)],
        "qwen_context": prompt_last,
        "qwen_audio_layers": audio_layers,
        "profile_given": base["profile_given"][: len(records)],
        "profile_shuffled": base["profile_shuffled"][: len(records)],
    }
    for optional in ("paper_targets", "context_prompt_sha256"):
        if optional in base:
            payload[optional] = base[optional][: len(records)]
    _savez_compressed_atomic(destination, **payload)
    if partial_path.exists():
        partial_path.unlink()

    metadata = {
        "schema_version": AUDIO_LAYER_CACHE_SCHEMA,
        "source_run_dir": str(Path(run_dir).resolve()),
        "base_cache_path": str(base_path),
        "base_cache_sha256": _sha256_file(base_path),
        "cache_path": str(destination),
        "cache_sha256": _sha256_file(destination),
        "model_name": encoder.model_name,
        "torch_dtype": encoder.torch_dtype,
        "device": str(encoder.device),
        "samples": len(records),
        "class_counts": _label_counts(payload["labels"]),
        "context_contract": "Qwen prompt-last causal audio+transcript vector plus Qwen audio-tower boundary vectors",
        "prompt_last_dimension": int(prompt_last.shape[1]),
        "audio_layer_count": encoder.layer_count,
        "audio_layer_dimension": encoder.layer_dimension,
        "audio_weight_keys": encoder.audio_weight_keys,
        "boundary_pooling": "last valid pre-pooling frame from convolutional input plus all 32 encoder layers",
        "mean_encode_seconds": float(np.mean(timings)) if timings else None,
        "paired_input_audit": paired_audit,
    }
    write_json(destination.with_suffix(".meta.json"), metadata)
    return metadata


__all__ = [
    "AUDIO_LAYER_CACHE_SCHEMA",
    "QwenAudioLayerBoundaryEncoder",
    "build_qwen_audio_layer_cache",
]
