"""Deterministic helpers for hashing, JSONL, and random seeds."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch


def stable_bucket(value: str, buckets: int, *, reserve_zero: bool = True) -> int:
    """Map text to a stable bucket; bucket zero is reserved for unknown."""
    normalized = str(value or "unknown").strip().lower()
    if reserve_zero and normalized in {"", "unknown", "none", "?"}:
        return 0
    digest = hashlib.blake2b(normalized.encode("utf-8"), digest_size=8).digest()
    start = 1 if reserve_zero else 0
    width = buckets - start
    return start + (int.from_bytes(digest, "little") % width)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
