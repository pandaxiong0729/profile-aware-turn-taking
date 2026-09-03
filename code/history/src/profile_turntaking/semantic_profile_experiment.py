"""Frozen semantic-profile embeddings for paired turn-taking experiments.

The module deliberately reuses the profile text already rendered for the Qwen
prompt experiment.  Audio, causal transcript, boundary state, sample identity,
and reference label are loaded once per event.  Hidden/given/shuffled evaluation
changes only the profile vector supplied to the same trained checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from .audio import read_wav_window, statistical_audio_features
from .constants import LABELS, LABEL_TO_ID
from .metrics import classification_metrics
from .utils import read_jsonl, write_json, write_jsonl


PROFILE_MODES = ("hidden", "given", "shuffled")
SHARED_REQUEST_FIELDS = (
    "sample_id",
    "conversation_id",
    "audio_path",
    "audio_sha256",
    "source_audio_path",
    "source_audio_sha256",
    "audio_window_start_s",
    "audio_window_end_s",
    "audio_window_sha256",
    "audio_logical_sha256",
    "audio_duration_s",
    "audio_sample_rate",
    "decision_time_in_conversation_s",
    "forecast_offset_ms",
    "evaluation_window_ms",
    "horizon_ms",
    "transcript_prefix",
    "transcript_units",
    "transcript_sha256",
    "boundary_state",
    "boundary_state_text",
    "boundary_state_sha256",
    "causal_asr_transcript",
    "causal_asr_sha256",
    "prompt_template_sha256",
)
DEFAULT_SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _load_reference_map(run_dir: Path) -> dict[str, dict[str, Any]]:
    references = {
        str(row["sample_id"]): row
        for row in read_jsonl(run_dir / "reference_labels.jsonl")
    }
    if not references:
        raise ValueError(f"No reference labels in {run_dir}")
    return references


def load_and_audit_paired_requests(run_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load one base record per event and fail if paired inputs differ."""

    root = Path(run_dir).resolve()
    references = _load_reference_map(root)
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(root / "requests.jsonl"):
        sample_id = str(row.get("sample_id", ""))
        mode = str(row.get("profile_mode", ""))
        if mode not in PROFILE_MODES:
            raise ValueError(f"Unexpected profile mode {mode!r} for {sample_id}")
        if mode in grouped[sample_id]:
            raise ValueError(f"Duplicate {mode} request for {sample_id}")
        grouped[sample_id][mode] = row
    if set(grouped) != set(references):
        raise ValueError("Request and reference sample IDs differ")

    errors: list[str] = []
    records: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    audio_sha_cache: dict[Path, str] = {}
    for sample_id in sorted(grouped):
        modes = grouped[sample_id]
        if set(modes) != set(PROFILE_MODES):
            errors.append(f"{sample_id}: modes={sorted(modes)}")
            continue
        hidden, given, shuffled = (modes[name] for name in PROFILE_MODES)
        for field in SHARED_REQUEST_FIELDS:
            values = [row.get(field) for row in (hidden, given, shuffled)]
            serialized = [json.dumps(value, sort_keys=True) for value in values]
            if len(set(serialized)) != 1:
                errors.append(f"{sample_id}: shared field differs: {field}")
        if str(given.get("profile_text", "")) == str(shuffled.get("profile_text", "")):
            errors.append(f"{sample_id}: given and shuffled profile text are identical")
        for mode, row in modes.items():
            text = str(row.get("profile_text", ""))
            if not text:
                errors.append(f"{sample_id}: empty profile text for {mode}")
            if _sha256_text(text) != row.get("profile_sha256"):
                errors.append(f"{sample_id}: profile SHA mismatch for {mode}")
            transcript = str(row.get("transcript_prefix", ""))
            if _sha256_text(transcript) != row.get("transcript_sha256"):
                errors.append(f"{sample_id}: transcript SHA mismatch for {mode}")
            if any(key in row for key in ("reference_label", "candidate_label", "human_label", "label_evidence")):
                errors.append(f"{sample_id}: target or annotation field leaked into {mode} request")
        label = str(references[sample_id].get("reference_label", ""))
        if label not in LABEL_TO_ID:
            errors.append(f"{sample_id}: invalid label {label!r}")
            continue
        windowed_audio = bool(given.get("source_audio_path")) and given.get("audio_window_start_s") is not None
        if windowed_audio:
            audio_path = Path(str(given["source_audio_path"]))
            if not audio_path.is_absolute():
                audio_path = root / audio_path
            if not audio_path.is_file():
                errors.append(f"{sample_id}: missing source audio {audio_path}")
                continue
            source_sha = str(given.get("source_audio_sha256", ""))
            actual_sha = audio_sha_cache.get(audio_path)
            if actual_sha is None:
                actual_sha = _sha256_file(audio_path)
                audio_sha_cache[audio_path] = actual_sha
            if source_sha and actual_sha != source_sha:
                errors.append(f"{sample_id}: source audio SHA mismatch")
                continue
        else:
            audio_path = Path(str(given["audio_path"]))
            if not audio_path.is_absolute():
                audio_path = root / audio_path
            if not audio_path.is_file():
                errors.append(f"{sample_id}: missing audio {audio_path}")
                continue
            actual_sha = audio_sha_cache.get(audio_path)
            if actual_sha is None:
                actual_sha = _sha256_file(audio_path)
                audio_sha_cache[audio_path] = actual_sha
            if actual_sha != str(given["audio_sha256"]):
                errors.append(f"{sample_id}: audio SHA mismatch")
                continue
        class_counts[label] += 1
        records.append(
            {
                "sample_id": sample_id,
                "conversation_id": str(given["conversation_id"]),
                "label": label,
                "audio_path": str(audio_path),
                "source_audio_path": str(audio_path) if windowed_audio else "",
                "source_audio_sha256": str(given.get("source_audio_sha256", "")),
                "audio_window_start_s": (
                    float(given["audio_window_start_s"]) if windowed_audio else 0.0
                ),
                "audio_window_end_s": (
                    float(given["audio_window_end_s"])
                    if windowed_audio
                    else float(given["audio_duration_s"])
                ),
                "audio_window_sha256": str(given.get("audio_window_sha256", "")),
                "audio_duration_s": float(given["audio_duration_s"]),
                "audio_sha256": str(given["audio_sha256"]),
                "transcript_prefix": str(given.get("transcript_prefix", "")),
                "transcript_units": list(given.get("transcript_units", [])),
                "transcript_sha256": str(given["transcript_sha256"]),
                "boundary_state": dict(given.get("boundary_state", {})),
                "boundary_state_text": str(given.get("boundary_state_text", "")),
                "boundary_state_sha256": str(given["boundary_state_sha256"]),
                "causal_asr_transcript": str(given.get("causal_asr_transcript", "")),
                "causal_asr_sha256": str(given["causal_asr_sha256"]),
                "forecast_offset_ms": int(given["forecast_offset_ms"]),
                "evaluation_window_ms": int(given["evaluation_window_ms"]),
                "prompt_template_sha256": str(given["prompt_template_sha256"]),
                "profile_text_hidden": str(hidden["profile_text"]),
                "profile_text_given": str(given["profile_text"]),
                "profile_text_shuffled": str(shuffled["profile_text"]),
                "profile_sha256_hidden": str(hidden["profile_sha256"]),
                "profile_sha256_given": str(given["profile_sha256"]),
                "profile_sha256_shuffled": str(shuffled["profile_sha256"]),
                "paper_binary_targets": dict(
                    references[sample_id].get("paper_binary_targets", {})
                ),
            }
        )
    if errors:
        raise ValueError("Paired request audit failed:\n" + "\n".join(errors[:50]))

    audit = {
        "passed": True,
        "run_dir": str(root),
        "samples": len(records),
        "requests": len(records) * len(PROFILE_MODES),
        "class_counts": {label: class_counts.get(label, 0) for label in LABELS},
        "shared_fields_checked": list(SHARED_REQUEST_FIELDS),
        "profile_only_change_verified": True,
        "audio_file_sha256_verified": True,
        "windowed_source_audio_supported": True,
        "reference_labels_outside_requests": True,
        "profile_text_source": "exact profile_text stored by the Qwen natural-language prompt pipeline",
    }
    return records, audit


def context_text(record: dict[str, Any]) -> str:
    """Serialize only the same causal text fields used by the Qwen prompt."""

    transcript = record["transcript_prefix"] or "No completed transcript unit is available."
    boundary = record["boundary_state_text"] or "No speaker activity summary is available."
    causal_asr = record["causal_asr_transcript"] or "No separate causal ASR is available."
    return "\n".join(
        [
            "Causal transcript available before the prediction boundary:",
            transcript,
            "Speaker activity at the prediction boundary:",
            boundary,
            "Causal ASR of the same audio:",
            causal_asr,
        ]
    )


def multiscale_audio_features(samples: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
    """Summarize the full causal audio and progressively shorter boundary windows."""

    audio = np.asarray(samples, dtype=np.float32)
    features: list[np.ndarray] = []
    for seconds in (None, 2.0, 1.0, 0.5, 0.25):
        if seconds is None:
            window = audio
        else:
            wanted = max(1, int(round(seconds * sample_rate)))
            window = audio[-wanted:]
        features.append(statistical_audio_features(window))
    return np.concatenate(features).astype(np.float32)


def causal_structure_features(record: dict[str, Any]) -> np.ndarray:
    """Numeric form of causal speaker timing already present in the prompt."""

    duration = max(1e-6, float(record["audio_duration_s"]))
    active = list(record.get("boundary_state", {}).get("active_speakers_at_t", []))
    active_names = {str(item.get("speaker", "")) for item in active}
    active_durations = {"speaker_00": 0.0, "speaker_01": 0.0}
    for item in active:
        speaker = str(item.get("speaker", ""))
        if speaker in active_durations:
            active_durations[speaker] = max(
                0.0,
                duration - float(item.get("observed_start_s", duration)),
            )
    units = list(record.get("transcript_units", []))
    last_end = max((float(unit.get("end_s", 0.0)) for unit in units), default=0.0)
    last_speaker = str(units[-1].get("speaker", "")) if units else ""
    recent = units[-4:]
    changes = sum(
        str(left.get("speaker")) != str(right.get("speaker"))
        for left, right in zip(recent, recent[1:])
    )
    mean_unit = float(
        np.mean(
            [
                max(0.0, float(unit.get("end_s", 0.0)) - float(unit.get("start_s", 0.0)))
                for unit in recent
            ]
        )
    ) if recent else 0.0
    values = np.asarray(
        [
            min(2.0, len(active) / 2.0),
            float("speaker_00" in active_names),
            float("speaker_01" in active_names),
            min(1.0, active_durations["speaker_00"] / duration),
            min(1.0, active_durations["speaker_01"] / duration),
            math.log1p(len(units)) / 5.0,
            min(1.0, max(0.0, duration - last_end) / duration),
            float(last_speaker == "speaker_00"),
            float(last_speaker == "speaker_01"),
            changes / 3.0,
            min(1.0, mean_unit / 5.0),
            float(bool(record.get("causal_asr_transcript"))),
        ],
        dtype=np.float32,
    )
    return values


def _load_sentence_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "Install the semantic-profile dependencies: pip install -e .[semantic-profile]"
        ) from exc
    return SentenceTransformer(model_name, device="cpu")


def build_feature_cache(
    run_dir: str | Path,
    cache_path: str | Path,
    *,
    sentence_model: str = DEFAULT_SENTENCE_MODEL,
    batch_size: int = 64,
    profile_encoding: str = "whole",
) -> dict[str, Any]:
    """Encode causal context and the exact prompt profile text once."""

    if profile_encoding not in {"whole", "linewise"}:
        raise ValueError("profile_encoding must be 'whole' or 'linewise'")

    records, paired_audit = load_and_audit_paired_requests(run_dir)
    encoder = _load_sentence_model(sentence_model)
    contexts = [context_text(record) for record in records]
    context_vectors = encoder.encode(
        contexts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    unique_profile_texts = sorted(
        {
            record[f"profile_text_{mode}"]
            for record in records
            for mode in ("given", "shuffled")
        }
    )
    if profile_encoding == "whole":
        encoded_profiles = encoder.encode(
            unique_profile_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        vector_by_text = dict(zip(unique_profile_texts, encoded_profiles))
    else:
        lines_by_text = {
            text: [line.strip() for line in text.splitlines() if line.strip()]
            for text in unique_profile_texts
        }
        line_counts = {len(lines) for lines in lines_by_text.values()}
        if line_counts != {4}:
            raise ValueError(
                "Linewise encoding expects the unchanged four-line Qwen profile template; "
                f"found line counts {sorted(line_counts)}"
            )
        unique_lines = sorted({line for lines in lines_by_text.values() for line in lines})
        line_vectors = encoder.encode(
            unique_lines,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        vector_by_line = dict(zip(unique_lines, line_vectors))
        vector_by_text = {
            text: np.concatenate([vector_by_line[line] for line in lines])
            / math.sqrt(len(lines))
            for text, lines in lines_by_text.items()
        }
        encoded_profiles = np.stack([vector_by_text[text] for text in unique_profile_texts])

    audio_vectors: list[np.ndarray] = []
    for record in records:
        audio = read_wav_window(
            record["audio_path"],
            0.0,
            float(record["audio_duration_s"]),
        )
        audio_vectors.append(
            np.concatenate(
                [multiscale_audio_features(audio), causal_structure_features(record)]
            ).astype(np.float32)
        )

    destination = Path(cache_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        sample_ids=np.asarray([record["sample_id"] for record in records]),
        conversation_ids=np.asarray([record["conversation_id"] for record in records]),
        labels=np.asarray([LABEL_TO_ID[record["label"]] for record in records], dtype=np.int64),
        audio=np.stack(audio_vectors),
        context=context_vectors,
        profile_given=np.stack([vector_by_text[record["profile_text_given"]] for record in records]),
        profile_shuffled=np.stack(
            [vector_by_text[record["profile_text_shuffled"]] for record in records]
        ),
    )
    catalog = [
        {
            "profile_sha256": _sha256_text(text),
            "profile_text": text,
            "embedding_norm": float(np.linalg.norm(vector_by_text[text])),
        }
        for text in unique_profile_texts
    ]
    write_jsonl(destination.with_suffix(".profiles.jsonl"), catalog)
    meta = {
        "schema_version": "semantic-profile-cache-v1",
        "source_run_dir": str(Path(run_dir).resolve()),
        "cache_path": str(destination.resolve()),
        "cache_sha256": _sha256_file(destination),
        "sentence_model": sentence_model,
        "sentence_model_frozen": True,
        "profile_encoding": profile_encoding,
        "profile_encoding_description": (
            "one normalized vector for the complete unchanged profile text"
            if profile_encoding == "whole"
            else "four normalized sentence vectors from the unchanged four-line prompt profile, concatenated in original order"
        ),
        "samples": len(records),
        "audio_dimension": int(len(audio_vectors[0])),
        "context_dimension": int(context_vectors.shape[1]),
        "profile_dimension": int(encoded_profiles.shape[1]),
        "unique_nonhidden_profile_texts": len(unique_profile_texts),
        "hidden_representation": "all-zero vector",
        "paired_input_audit": paired_audit,
        "profile_text_exactly_reused_from_prompt": True,
        "profile_text_cleaning_or_field_remapping_applied": False,
    }
    write_json(destination.with_suffix(".meta.json"), meta)
    return meta


def load_feature_cache(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


@dataclass
class SemanticTrainConfig:
    hidden_dimension: int = 128
    dropout: float = 0.2
    profile_dropout: float = 0.5
    epochs: int = 30
    patience: int = 6
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seeds: tuple[int, ...] = (13, 37, 71)
    device: str = "cpu"
    profile_fusion: str = "additive"
    profile_preprocessing: str = "raw"


class SemanticProfileClassifier(nn.Module):
    """Small classifier whose hidden condition is exactly context-only."""

    def __init__(
        self,
        audio_dimension: int,
        context_dimension: int,
        profile_dimension: int,
        hidden_dimension: int,
        dropout: float,
        profile_fusion: str = "additive",
    ) -> None:
        super().__init__()
        if profile_fusion not in {"additive", "interaction"}:
            raise ValueError("profile_fusion must be 'additive' or 'interaction'")
        self.profile_fusion = profile_fusion
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dimension),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dimension),
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dimension * 2, hidden_dimension),
            nn.GELU(),
            nn.LayerNorm(hidden_dimension),
        )
        self.profile_projection = nn.Sequential(
            nn.Linear(profile_dimension, hidden_dimension, bias=False),
            nn.GELU(),
            nn.LayerNorm(hidden_dimension, elementwise_affine=False),
        )
        self.profile_adapter = (
            nn.Sequential(
                nn.Linear(hidden_dimension * 3, hidden_dimension),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dimension, hidden_dimension),
            )
            if profile_fusion == "interaction"
            else nn.Identity()
        )
        self.profile_gate = nn.Linear(hidden_dimension * 2, hidden_dimension)
        self.output_norm = nn.LayerNorm(hidden_dimension)
        self.classifier = nn.Linear(hidden_dimension, len(LABELS))

    def forward(
        self,
        audio: torch.Tensor,
        context: torch.Tensor,
        profile: torch.Tensor,
    ) -> torch.Tensor:
        audio_state = self.audio_encoder(audio)
        text_state = self.context_encoder(context)
        shared = self.context_fusion(torch.cat([audio_state, text_state], dim=-1))
        profile_state = self.profile_projection(profile)
        present = (profile.abs().sum(dim=-1, keepdim=True) > 0).to(profile_state.dtype)
        profile_state = profile_state * present
        gate = torch.sigmoid(self.profile_gate(torch.cat([shared, profile_state], dim=-1)))
        if self.profile_fusion == "interaction":
            delta = self.profile_adapter(
                torch.cat([shared, profile_state, shared * profile_state], dim=-1)
            ) * present
        else:
            delta = profile_state
        hidden = self.output_norm(shared + gate * delta)
        return self.classifier(hidden)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _prepare_profiles(
    values: np.ndarray,
    *,
    profile_mean: np.ndarray,
    preprocessing: str,
) -> np.ndarray:
    profiles = values.astype(np.float32)
    if preprocessing == "raw":
        return profiles
    if preprocessing != "center_l2":
        raise ValueError("profile preprocessing must be 'raw' or 'center_l2'")
    centered = profiles - profile_mean
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return centered / np.maximum(norms, 1e-8)


def _tensor_dataset(
    data: dict[str, np.ndarray],
    audio_mean: np.ndarray,
    audio_std: np.ndarray,
    profile_mean: np.ndarray,
    profile_preprocessing: str,
) -> TensorDataset:
    audio = (data["audio"].astype(np.float32) - audio_mean) / audio_std
    return TensorDataset(
        torch.from_numpy(audio),
        torch.from_numpy(data["context"].astype(np.float32)),
        torch.from_numpy(
            _prepare_profiles(
                data["profile_given"],
                profile_mean=profile_mean,
                preprocessing=profile_preprocessing,
            )
        ),
        torch.from_numpy(data["labels"].astype(np.int64)),
    )


def _ece(probabilities: np.ndarray, targets: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    result = 0.0
    for lower in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        upper = lower + 1.0 / bins
        mask = (confidence > lower) & (confidence <= upper)
        if not bool(mask.any()):
            continue
        accuracy = float(np.mean(predictions[mask] == targets[mask]))
        result += float(mask.mean()) * abs(accuracy - float(confidence[mask].mean()))
    return result


@torch.no_grad()
def _predict(
    model: SemanticProfileClassifier,
    data: dict[str, np.ndarray],
    *,
    mode: str,
    audio_mean: np.ndarray,
    audio_std: np.ndarray,
    profile_mean: np.ndarray,
    profile_preprocessing: str,
    device: torch.device,
    zero_audio: bool = False,
    zero_context: bool = False,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if mode not in PROFILE_MODES:
        raise ValueError(f"Unknown profile mode: {mode}")
    audio = (data["audio"].astype(np.float32) - audio_mean) / audio_std
    context = data["context"].astype(np.float32)
    if zero_audio:
        audio = np.zeros_like(audio)
    if zero_context:
        context = np.zeros_like(context)
    if mode == "hidden":
        profile = np.zeros_like(data["profile_given"], dtype=np.float32)
    else:
        profile = _prepare_profiles(
            data[f"profile_{mode}"],
            profile_mean=profile_mean,
            preprocessing=profile_preprocessing,
        )
    dataset = TensorDataset(
        torch.from_numpy(audio),
        torch.from_numpy(context),
        torch.from_numpy(profile),
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    probabilities: list[np.ndarray] = []
    model.eval()
    for batch_audio, batch_context, batch_profile in loader:
        logits = model(
            batch_audio.to(device),
            batch_context.to(device),
            batch_profile.to(device),
        )
        probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probs = np.concatenate(probabilities)
    predictions = probs.argmax(axis=1)
    targets = data["labels"].astype(np.int64)
    report = classification_metrics(targets.tolist(), predictions.tolist())
    one_hot = np.eye(len(LABELS), dtype=np.float32)[targets]
    report["brier_score"] = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    report["log_loss"] = float(
        -np.mean(np.log(np.maximum(probs[np.arange(len(targets)), targets], 1e-12)))
    )
    report["ece"] = _ece(probs, targets)
    distribution = Counter(LABELS[index] for index in predictions.tolist())
    report["prediction_distribution"] = {
        label: distribution.get(label, 0) for label in LABELS
    }
    dominant = max(distribution.values(), default=0) / max(1, len(predictions))
    report["noncollapsed"] = len(distribution) >= 3 and dominant <= 0.8
    report["dominant_fraction"] = dominant
    return report, predictions, probs


def _train_one_seed(
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    *,
    config: SemanticTrainConfig,
    seed: int,
    output_dir: Path,
) -> tuple[
    SemanticProfileClassifier,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
]:
    _set_seed(seed)
    device = torch.device(config.device)
    audio_mean = train["audio"].astype(np.float32).mean(axis=0, keepdims=True)
    audio_std = train["audio"].astype(np.float32).std(axis=0, keepdims=True)
    audio_std = np.maximum(audio_std, 1e-5)
    profile_mean = train["profile_given"].astype(np.float32).mean(axis=0, keepdims=True)
    train_dataset = _tensor_dataset(
        train,
        audio_mean,
        audio_std,
        profile_mean,
        config.profile_preprocessing,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    model = SemanticProfileClassifier(
        audio_dimension=int(train["audio"].shape[1]),
        context_dimension=int(train["context"].shape[1]),
        profile_dimension=int(train["profile_given"].shape[1]),
        hidden_dimension=config.hidden_dimension,
        dropout=config.dropout,
        profile_fusion=config.profile_fusion,
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
        for audio, context, profile, labels in loader:
            audio = audio.to(device)
            context = context.to(device)
            profile = profile.to(device)
            labels = labels.to(device)
            if config.profile_dropout > 0:
                drop = torch.rand(profile.shape[0], device=device) < config.profile_dropout
                profile = profile.clone()
                profile[drop] = 0.0
            optimizer.zero_grad(set_to_none=True)
            logits = model(audio, context, profile)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            seen += len(labels)
        given_report, _, _ = _predict(
            model,
            val,
            mode="given",
            audio_mean=audio_mean,
            audio_std=audio_std,
            profile_mean=profile_mean,
            profile_preprocessing=config.profile_preprocessing,
            device=device,
        )
        hidden_report, _, _ = _predict(
            model,
            val,
            mode="hidden",
            audio_mean=audio_mean,
            audio_std=audio_std,
            profile_mean=profile_mean,
            profile_preprocessing=config.profile_preprocessing,
            device=device,
        )
        score = 0.5 * (given_report["macro_f1"] + hidden_report["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(1, seen),
                "val_selection_score": score,
                "val_given_macro_f1": given_report["macro_f1"],
                "val_hidden_macro_f1": hidden_report["macro_f1"],
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
    checkpoint = {
        "format_version": 1,
        "seed": seed,
        "config": asdict(config),
        "model_dimensions": {
            "audio": int(train["audio"].shape[1]),
            "context": int(train["context"].shape[1]),
            "profile": int(train["profile_given"].shape[1]),
        },
        "audio_mean": audio_mean,
        "audio_std": audio_std,
        "profile_mean": profile_mean,
        "state_dict": best_state,
        "history": history,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_dir / f"seed-{seed}.pt")
    write_json(output_dir / f"seed-{seed}.train.json", {"seed": seed, "history": history})
    return model, audio_mean, audio_std, profile_mean, history


def run_semantic_profile_experiment(
    train_cache: str | Path,
    val_cache: str | Path,
    test_cache: str | Path,
    output_dir: str | Path,
    *,
    config: SemanticTrainConfig | None = None,
) -> dict[str, Any]:
    """Train multiple seeds and evaluate paired profile conditions."""

    cfg = config or SemanticTrainConfig()
    train = load_feature_cache(train_cache)
    val = load_feature_cache(val_cache)
    test = load_feature_cache(test_cache)
    for name, data in (("train", train), ("val", val), ("test", test)):
        if len(data["labels"]) == 0:
            raise ValueError(f"Empty {name} cache")
    train_conversations = set(train["conversation_ids"].tolist())
    val_conversations = set(val["conversation_ids"].tolist())
    test_conversations = set(test["conversation_ids"].tolist())
    if train_conversations & val_conversations or train_conversations & test_conversations or val_conversations & test_conversations:
        raise ValueError("Conversation leakage across train/val/test")

    destination = Path(output_dir)
    seed_reports: list[dict[str, Any]] = []
    all_prediction_rows: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        model, audio_mean, audio_std, profile_mean, history = _train_one_seed(
            train,
            val,
            config=cfg,
            seed=seed,
            output_dir=destination,
        )
        device = torch.device(cfg.device)
        validation_reports: dict[str, dict[str, Any]] = {}
        for mode in PROFILE_MODES:
            validation_report, _, _ = _predict(
                model,
                val,
                mode=mode,
                audio_mean=audio_mean,
                audio_std=audio_std,
                profile_mean=profile_mean,
                profile_preprocessing=cfg.profile_preprocessing,
                device=device,
            )
            validation_reports[mode] = validation_report
        validation_reports["paired_effects"] = {
            "given_minus_hidden_macro_f1": (
                validation_reports["given"]["macro_f1"]
                - validation_reports["hidden"]["macro_f1"]
            ),
            "given_minus_shuffled_macro_f1": (
                validation_reports["given"]["macro_f1"]
                - validation_reports["shuffled"]["macro_f1"]
            ),
            "hidden_minus_given_log_loss": (
                validation_reports["hidden"]["log_loss"]
                - validation_reports["given"]["log_loss"]
            ),
            "shuffled_minus_given_log_loss": (
                validation_reports["shuffled"]["log_loss"]
                - validation_reports["given"]["log_loss"]
            ),
        }
        reports: dict[str, dict[str, Any]] = {}
        predictions_by_mode: dict[str, np.ndarray] = {}
        probabilities_by_mode: dict[str, np.ndarray] = {}
        for mode in PROFILE_MODES:
            report, predictions, probabilities = _predict(
                model,
                test,
                mode=mode,
                audio_mean=audio_mean,
                audio_std=audio_std,
                profile_mean=profile_mean,
                profile_preprocessing=cfg.profile_preprocessing,
                device=device,
            )
            reports[mode] = report
            predictions_by_mode[mode] = predictions
            probabilities_by_mode[mode] = probabilities
        no_audio_report, no_audio_predictions, _ = _predict(
            model,
            test,
            mode="hidden",
            audio_mean=audio_mean,
            audio_std=audio_std,
            profile_mean=profile_mean,
            profile_preprocessing=cfg.profile_preprocessing,
            device=device,
            zero_audio=True,
        )
        no_text_report, no_text_predictions, _ = _predict(
            model,
            test,
            mode="hidden",
            audio_mean=audio_mean,
            audio_std=audio_std,
            profile_mean=profile_mean,
            profile_preprocessing=cfg.profile_preprocessing,
            device=device,
            zero_context=True,
        )
        reports["controls"] = {
            "zero_audio_hidden": no_audio_report,
            "zero_context_hidden": no_text_report,
            "audio_changed_fraction": float(
                np.mean(no_audio_predictions != predictions_by_mode["hidden"])
            ),
            "context_changed_fraction": float(
                np.mean(no_text_predictions != predictions_by_mode["hidden"])
            ),
        }
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
                "sample_id": sample_id,
                "conversation_id": test["conversation_ids"][index].item(),
                "target": LABELS[int(test["labels"][index])],
            }
            for mode in PROFILE_MODES:
                row[f"prediction_{mode}"] = LABELS[int(predictions_by_mode[mode][index])]
                row[f"probabilities_{mode}"] = {
                    label: float(probabilities_by_mode[mode][index, label_index])
                    for label_index, label in enumerate(LABELS)
                }
            all_prediction_rows.append(row)

    write_jsonl(destination / "predictions.jsonl", all_prediction_rows)
    aggregate: dict[str, Any] = {}
    validation_aggregate: dict[str, Any] = {}
    for mode in PROFILE_MODES:
        macro = np.asarray(
            [seed["reports"][mode]["macro_f1"] for seed in seed_reports], dtype=float
        )
        balanced = np.asarray(
            [seed["reports"][mode]["balanced_accuracy"] for seed in seed_reports], dtype=float
        )
        aggregate[mode] = {
            "macro_f1_mean": float(macro.mean()),
            "macro_f1_std": float(macro.std()),
            "balanced_accuracy_mean": float(balanced.mean()),
            "balanced_accuracy_std": float(balanced.std()),
            "log_loss_mean": float(
                np.mean([seed["reports"][mode]["log_loss"] for seed in seed_reports])
            ),
            "brier_score_mean": float(
                np.mean([seed["reports"][mode]["brier_score"] for seed in seed_reports])
            ),
            "per_class_f1_mean": {
                label: float(
                    np.mean(
                        [
                            seed["reports"][mode]["per_class"][label]["f1"]
                            for seed in seed_reports
                        ]
                    )
                )
                for label in LABELS
            },
            "all_seeds_noncollapsed": all(
                bool(seed["reports"][mode]["noncollapsed"]) for seed in seed_reports
            ),
        }
        validation_macro = np.asarray(
            [seed["validation_reports"][mode]["macro_f1"] for seed in seed_reports],
            dtype=float,
        )
        validation_aggregate[mode] = {
            "macro_f1_mean": float(validation_macro.mean()),
            "macro_f1_std": float(validation_macro.std()),
            "log_loss_mean": float(
                np.mean(
                    [
                        seed["validation_reports"][mode]["log_loss"]
                        for seed in seed_reports
                    ]
                )
            ),
            "all_seeds_noncollapsed": all(
                bool(seed["validation_reports"][mode]["noncollapsed"])
                for seed in seed_reports
            ),
        }
    for key in (
        "given_minus_hidden_macro_f1",
        "given_minus_shuffled_macro_f1",
        "hidden_minus_given_log_loss",
        "shuffled_minus_given_log_loss",
    ):
        values = np.asarray(
            [seed["reports"]["paired_effects"][key] for seed in seed_reports], dtype=float
        )
        aggregate[key] = {"mean": float(values.mean()), "std": float(values.std()), "values": values.tolist()}
        validation_values = np.asarray(
            [seed["validation_reports"]["paired_effects"][key] for seed in seed_reports],
            dtype=float,
        )
        validation_aggregate[key] = {
            "mean": float(validation_values.mean()),
            "std": float(validation_values.std()),
            "values": validation_values.tolist(),
        }
    summary = {
        "experiment": "frozen-natural-language-profile-semantic-embedding-v2",
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
            "train": len(train["labels"]),
            "val": len(val["labels"]),
            "test": len(test["labels"]),
        },
        "validation_aggregate": validation_aggregate,
        "aggregate": aggregate,
        "seed_reports": seed_reports,
        "interpretation_gate": {
            "all_hidden_noncollapsed": all(
                bool(seed["reports"]["hidden"]["noncollapsed"]) for seed in seed_reports
            ),
            "all_modes_noncollapsed": all(
                bool(seed["reports"][mode]["noncollapsed"])
                for seed in seed_reports
                for mode in PROFILE_MODES
            ),
            "profile_effect_claim_allowed": (
                validation_aggregate["given_minus_hidden_macro_f1"]["mean"] > 0
                and validation_aggregate["given_minus_shuffled_macro_f1"]["mean"] > 0
                and aggregate["given_minus_hidden_macro_f1"]["mean"] > 0
                and aggregate["given_minus_shuffled_macro_f1"]["mean"] > 0
                and all(
                    value > 0
                    for value in aggregate["given_minus_hidden_macro_f1"]["values"]
                )
                and all(
                    value > 0
                    for value in aggregate["given_minus_shuffled_macro_f1"]["values"]
                )
            ),
            "probability_effect_claim_allowed": (
                validation_aggregate["hidden_minus_given_log_loss"]["mean"] > 0
                and validation_aggregate["shuffled_minus_given_log_loss"]["mean"] > 0
                and aggregate["hidden_minus_given_log_loss"]["mean"] > 0
                and aggregate["shuffled_minus_given_log_loss"]["mean"] > 0
            ),
        },
    }
    write_json(destination / "summary.json", summary)
    return summary


def cache_class_counts(path: str | Path) -> dict[str, int]:
    data = load_feature_cache(path)
    counts = Counter(int(value) for value in data["labels"].tolist())
    return {label: counts.get(index, 0) for index, label in enumerate(LABELS)}
