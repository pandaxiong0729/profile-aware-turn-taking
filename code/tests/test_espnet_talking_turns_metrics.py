from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_espnet_talking_turns_baseline.py"
SPEC = importlib.util.spec_from_file_location("espnet_talking_turns_baseline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_named_ovr_auc_uses_probability_names_not_dictionary_or_alphabetic_order() -> None:
    labels = ("C", "BC", "T", "I", "NA")
    rows = []
    for expected in labels:
        # Deliberately store keys in the released checkpoint's native order.
        probabilities = {
            "C": 0.01,
            "NA": 0.01,
            "I": 0.01,
            "BC": 0.01,
            "T": 0.01,
        }
        probabilities[expected] = 0.96
        rows.append({"reference_label": expected, "probabilities": probabilities})

    result = MODULE.named_ovr_roc_auc(rows, labels)

    assert result == pytest.approx({label: 1.0 for label in labels})
