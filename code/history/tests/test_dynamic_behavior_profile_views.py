from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_dynamic_behavior_profile_views.py"
SPEC = importlib.util.spec_from_file_location("dynamic_behavior_profile_views", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_profile_cutoff_excludes_an_utterance_that_finishes_in_reserved_final_seconds() -> None:
    units = [
        {"speaker": "speaker_00", "start_s": 20.0, "end_s": 24.0, "text": "complete"},
        {"speaker": "speaker_01", "start_s": 24.5, "end_s": 26.0, "text": "not complete at cutoff"},
    ]
    clipped = MODULE.clip_units(units)
    assert [unit["text"] for unit in clipped] == ["complete"]


def test_behavior_features_are_fixed_length_and_finite() -> None:
    units = [
        {"speaker": "speaker_00", "start_s": 1.0, "end_s": 2.0, "text": "hello there"},
        {"speaker": "speaker_01", "start_s": 1.8, "end_s": 2.2, "text": "mhm"},
        {"speaker": "speaker_00", "start_s": 3.0, "end_s": 5.0, "text": "another turn"},
    ]
    values, names = MODULE.behavior_features(units)
    assert values.shape == (38,)
    assert len(names) == 38
    assert bool(np.isfinite(values).all())
    assert len(set(names)) == len(names)


def test_dynamic_shuffle_always_uses_a_different_conversation() -> None:
    sample_ids = np.asarray(["a", "b", "c", "d"])
    conversation_ids = np.asarray(["x", "x", "y", "z"])
    indices = MODULE.deterministic_shuffled_indices(sample_ids, conversation_ids)
    assert np.all(conversation_ids[indices] != conversation_ids)
    assert np.array_equal(
        indices, MODULE.deterministic_shuffled_indices(sample_ids, conversation_ids)
    )


def test_role_normalization_puts_latest_causal_speaker_first() -> None:
    values = np.arange(38, dtype=np.float32)
    names = [
        *[f"speaker_00.f{i}" for i in range(16)],
        *[f"speaker_01.f{i}" for i in range(16)],
        *[f"global.f{i}" for i in range(6)],
    ]
    units = [
        {"speaker": "speaker_00", "start_s": 1.0, "end_s": 2.0, "text": "first"},
        {"speaker": "speaker_01", "start_s": 27.0, "end_s": 28.0, "text": "latest"},
    ]

    ordered, ordered_names, inferred = MODULE.role_normalize_behavior(values, names, units)

    assert inferred == "speaker_01"
    assert ordered[:16].tolist() == values[16:32].tolist()
    assert ordered[16:32].tolist() == values[:16].tolist()
    assert ordered[32:].tolist() == values[32:].tolist()
    assert ordered_names[0] == "current_speaker.f0"
    assert ordered_names[16] == "other_participant.f0"
