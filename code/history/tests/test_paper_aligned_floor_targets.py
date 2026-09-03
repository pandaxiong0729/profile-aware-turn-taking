from __future__ import annotations

import numpy as np

from profile_turntaking.paper_aligned_floor_targets import outcome_boundary_targets


def test_floor_targets_move_from_interruption_onset_to_outcome_boundary() -> None:
    sample_ids = np.asarray(["onset", "failed", "success", "other"])
    original = np.full((4, 4), -100, dtype=np.int64)
    original[0, 3] = 1
    original[:, 0] = np.asarray([0, 1, -100, -100])
    references = [
        {"sample_id": "onset", "source_kind": "interruption_candidate"},
        {"sample_id": "failed", "source_kind": "hold_after_unsuccessful_interruption"},
        {"sample_id": "success", "source_kind": "shift_after_successful_interruption"},
        {"sample_id": "other", "source_kind": "natural_turn_shift"},
    ]

    rewritten, counts = outcome_boundary_targets(sample_ids, original, references)

    assert rewritten[:, 3].tolist() == [-100, 0, 1, -100]
    assert rewritten[:, 0].tolist() == original[:, 0].tolist()
    assert counts == {"A": 1, "B": 1, "ignored": 2}
