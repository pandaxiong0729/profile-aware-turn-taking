from __future__ import annotations

from pathlib import Path

import pytest

from profile_turntaking.experiment_preflight import audit_prompt_pilot_data


def test_expected_samples_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected_samples must be positive"):
        audit_prompt_pilot_data(
            catalog_dir=tmp_path,
            event_manifest=tmp_path / "events.jsonl",
            run_dir=tmp_path / "run",
            output_path=tmp_path / "audit.json",
            expected_samples=0,
        )
