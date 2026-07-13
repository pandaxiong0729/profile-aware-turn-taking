"""Small profile-conditioned adapter with interchangeable audio backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .constants import LABELS, PROFILE_FIELDS


@dataclass
class ModelConfig:
    audio_backend: str = "statistical"
    whisper_model: str = "openai/whisper-tiny"
    freeze_audio_encoder: bool = True
    statistical_dimension: int = 24
    text_dimension: int = 128
    profile_buckets: int = 512
    profile_embedding_dimension: int = 16
    hidden_dimension: int = 64
    dropout: float = 0.15

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in payload.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StatisticalAudioEncoder(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dimension),
        )

    def forward(self, audio_input: torch.Tensor) -> torch.Tensor:
        return self.network(audio_input)


class WhisperAudioEncoder(nn.Module):
    """Optional frozen Hugging Face Whisper encoder for cloud runs."""

    def __init__(self, model_name: str, hidden_dimension: int, freeze: bool) -> None:
        super().__init__()
        try:
            from transformers import WhisperFeatureExtractor, WhisperModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install profile-turntaking[whisper] to use Whisper") from exc
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        backbone = WhisperModel.from_pretrained(model_name)
        self.encoder = backbone.encoder
        self.freeze = freeze
        if freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        self.projection = nn.Sequential(
            nn.Linear(backbone.config.d_model, hidden_dimension),
            nn.GELU(),
            nn.LayerNorm(hidden_dimension),
        )

    def forward(self, audio_input: torch.Tensor) -> torch.Tensor:  # pragma: no cover - cloud path
        arrays = audio_input.detach().cpu().numpy()
        features = self.feature_extractor(
            list(arrays), sampling_rate=16_000, return_tensors="pt"
        ).input_features.to(audio_input.device)
        with torch.set_grad_enabled(not self.freeze):
            encoded = self.encoder(features).last_hidden_state
        return self.projection(encoded.mean(dim=1))


class StructuredProfileEncoder(nn.Module):
    def __init__(self, buckets: int, embedding_dimension: int, hidden_dimension: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(buckets, embedding_dimension, padding_idx=0) for _ in PROFILE_FIELDS]
        )
        self.projection = nn.Sequential(
            nn.Linear(len(PROFILE_FIELDS) * embedding_dimension, hidden_dimension),
            nn.GELU(),
            nn.LayerNorm(hidden_dimension),
        )

    def forward(self, profile_ids: torch.Tensor) -> torch.Tensor:
        encoded = [embedding(profile_ids[:, index]) for index, embedding in enumerate(self.embeddings)]
        return self.projection(torch.cat(encoded, dim=-1))


class ProfileTurnModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dimension
        if config.audio_backend == "statistical":
            self.audio_encoder: nn.Module = StatisticalAudioEncoder(
                config.statistical_dimension, hidden, config.dropout
            )
        elif config.audio_backend == "whisper":
            self.audio_encoder = WhisperAudioEncoder(
                config.whisper_model, hidden, config.freeze_audio_encoder
            )
        else:
            raise ValueError(f"Unsupported audio backend: {config.audio_backend}")
        self.text_encoder = nn.Sequential(
            nn.Linear(config.text_dimension, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(hidden),
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.profile_encoder = StructuredProfileEncoder(
            config.profile_buckets, config.profile_embedding_dimension, hidden
        )
        self.adapter = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden),
        )
        self.profile_gate = nn.Parameter(torch.tensor(-2.0))
        self.classifier = nn.Linear(hidden, len(LABELS))

    def forward(
        self, audio_input: torch.Tensor, text_input: torch.Tensor, profile_ids: torch.Tensor
    ) -> torch.Tensor:
        audio_state = self.audio_encoder(audio_input)
        text_state = self.text_encoder(text_input)
        context = self.context_fusion(torch.cat([audio_state, text_state], dim=-1))
        profile_state = self.profile_encoder(profile_ids)
        delta = self.adapter(torch.cat([context, profile_state], dim=-1))
        hidden = context + torch.sigmoid(self.profile_gate) * delta
        return self.classifier(hidden)

    def checkpoint_state_dict(self) -> dict[str, torch.Tensor]:
        state = self.state_dict()
        if self.config.audio_backend == "whisper" and self.config.freeze_audio_encoder:
            state = {
                key: value
                for key, value in state.items()
                if not key.startswith("audio_encoder.encoder.")
            }
        return state
