from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_final_report_uses_authoritative_json_values(tmp_path: Path) -> None:
    result_dir = tmp_path / "final" / "rich" / "gate_base"
    result_dir.mkdir(parents=True)
    modes = {
        "hidden": {"paper_balanced_accuracy_mean": 0.70},
        "given": {"paper_balanced_accuracy_mean": 0.74},
        "shuffled": {"paper_balanced_accuracy_mean": 0.71},
    }
    aggregate = {task: modes for task in ("turn_change", "backchannel", "interruption", "floor_taking")}
    (result_dir / "summary.json").write_text(
        json.dumps({"aggregate": aggregate}), encoding="utf-8"
    )
    final_summary = {
        "validation_candidates": [
            {
                "context_view": "rich",
                "name": "gate_base",
                "hidden_accuracy": 0.70,
                "given_accuracy": 0.74,
                "shuffled_accuracy": 0.71,
                "given_minus_hidden": 0.04,
                "given_minus_shuffled": 0.03,
            }
        ],
        "selected_from_validation": {"context_view": "rich", "name": "gate_base"},
        "final_test": {
            "hidden_accuracy": 0.70,
            "given_accuracy": 0.74,
            "shuffled_accuracy": 0.71,
            "given_minus_hidden": 0.04,
            "given_minus_shuffled": 0.03,
        },
        "final_result_dir": str(result_dir),
    }
    (tmp_path / "FINAL_SUMMARY.json").write_text(
        json.dumps(final_summary), encoding="utf-8"
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "write_qwen_shared_ab_final_report.py"
    subprocess.run(
        [sys.executable, str(script), "--search-root", str(tmp_path)],
        check=True,
    )
    report = (tmp_path / "FINAL_REPORT_ZH.md").read_text(encoding="utf-8")
    assert "74.00%" in report
    assert "given > hidden`：通过" in report
    assert "given > shuffled`：通过" in report
