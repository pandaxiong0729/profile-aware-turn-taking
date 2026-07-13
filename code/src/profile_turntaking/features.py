"""Text and structured-profile feature encoding."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from .constants import PROFILE_FIELDS
from .utils import stable_bucket

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def hash_text_vector(text: str, dimension: int = 128) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    for token in _TOKEN_RE.findall(text.lower()):
        bucket = stable_bucket(token, dimension, reserve_zero=False)
        vector[bucket] += 1.0
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def _profile_value(profile: dict[str, Any], field: str) -> str:
    if "." not in field:
        return str(profile.get(field, "unknown"))
    speaker, name = field.split(".", 1)
    return str(profile.get(speaker, {}).get(name, "unknown"))


def profile_bucket_ids(profile: dict[str, Any], buckets: int = 512) -> np.ndarray:
    return np.asarray(
        [stable_bucket(_profile_value(profile, field), buckets) for field in PROFILE_FIELDS],
        dtype=np.int64,
    )
