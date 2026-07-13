"""PyTorch dataset backed by the portable JSONL sample manifest."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio import read_wav_window, statistical_audio_features
from .constants import LABEL_TO_ID, UNKNOWN_PROFILE
from .features import hash_text_vector, profile_bucket_ids
from .utils import read_jsonl


class ManifestDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: str,
        *,
        split: str,
        audio_backend: str = "statistical",
        text_dimension: int = 128,
        profile_buckets: int = 512,
        profile_mode: str = "given",
    ) -> None:
        self.rows = [row for row in read_jsonl(manifest_path) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows for split={split!r} in {manifest_path}")
        if audio_backend not in {"statistical", "whisper"}:
            raise ValueError(f"Unsupported audio backend: {audio_backend}")
        if profile_mode not in {"given", "hidden", "shuffled"}:
            raise ValueError(f"Unsupported profile mode: {profile_mode}")
        self.audio_backend = audio_backend
        self.text_dimension = text_dimension
        self.profile_buckets = profile_buckets
        self.profile_mode = profile_mode

    def __len__(self) -> int:
        return len(self.rows)

    def _profile_for_index(self, index: int) -> dict[str, Any]:
        if self.profile_mode == "hidden":
            return UNKNOWN_PROFILE
        if self.profile_mode == "shuffled":
            return self.rows[(index + max(1, len(self.rows) // 2)) % len(self.rows)]["profile"]
        return self.rows[index]["profile"]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        waveform = read_wav_window(
            row["audio_path"],
            float(row["window_start_s"]),
            float(row["window_end_s"]),
        )
        if self.audio_backend == "statistical":
            audio_input = statistical_audio_features(waveform)
        else:
            audio_input = waveform
        return {
            "audio_input": torch.from_numpy(np.asarray(audio_input, dtype=np.float32)),
            "text_input": torch.from_numpy(hash_text_vector(row["transcript_prefix"], self.text_dimension)),
            "profile_ids": torch.from_numpy(
                profile_bucket_ids(self._profile_for_index(index), self.profile_buckets)
            ),
            "label": torch.tensor(LABEL_TO_ID[row["label"]], dtype=torch.long),
            "sample_id": row["sample_id"],
        }
