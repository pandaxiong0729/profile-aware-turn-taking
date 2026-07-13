"""Serializable records used by data preparation and training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Utterance:
    start_s: float
    end_s: float
    speaker: str
    text: str

    def overlaps(self, start_s: float, end_s: float) -> bool:
        return self.start_s < end_s and self.end_s > start_s


@dataclass(frozen=True)
class Sample:
    sample_id: str
    conversation_id: str
    split_group: str
    split: str
    prediction_time_s: float
    horizon_ms: int
    window_start_s: float
    window_end_s: float
    audio_path: str
    transcript_prefix: str
    profile: dict[str, Any]
    label: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
