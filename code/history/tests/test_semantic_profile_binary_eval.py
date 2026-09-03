from __future__ import annotations

import numpy as np

from profile_turntaking.constants import LABELS
from profile_turntaking.semantic_profile_binary_eval import (
    BINARY_STAGES,
    _output_gate_summary,
    five_probabilities_to_binary,
)


def test_one_hot_five_classes_recover_expected_binary_hierarchy() -> None:
    probabilities = np.eye(len(LABELS), dtype=np.float32)
    p_a, answers, predictions = five_probabilities_to_binary(probabilities)
    assert [LABELS[int(index)] for index in predictions] == list(LABELS)
    by_label = {label: answers[index] for index, label in enumerate(LABELS)}
    assert by_label["NA"]["silence"] == "A"
    assert by_label["C"]["listener_onset"] == "A"
    assert by_label["BC"]["brief_response"] == "A"
    assert by_label["T"]["yield"] == "A"
    assert by_label["I"]["yield"] == "B"
    assert p_a.shape == (5, len(BINARY_STAGES))


def test_binary_projection_is_conditional_not_new_training() -> None:
    probabilities = np.asarray([[0.10, 0.20, 0.30, 0.15, 0.25]], dtype=np.float32)
    p_a, answers, _ = five_probabilities_to_binary(probabilities)
    # LABEL order is C, BC, T, I, NA.
    assert np.isclose(p_a[0, 0], 0.25)
    assert np.isclose(p_a[0, 1], 0.10 / 0.75)
    assert np.isclose(p_a[0, 2], 0.20 / 0.65)
    assert np.isclose(p_a[0, 3], 0.30 / 0.45)
    assert answers[0] == {
        "silence": "B",
        "listener_onset": "B",
        "brief_response": "B",
        "yield": "A",
    }


def test_custom_binary_thresholds_change_only_decision_rule() -> None:
    probabilities = np.asarray([[0.20, 0.20, 0.20, 0.20, 0.20]], dtype=np.float32)
    _, raw_answers, raw_predictions = five_probabilities_to_binary(probabilities)
    _, calibrated_answers, calibrated_predictions = five_probabilities_to_binary(
        probabilities, thresholds=[0.19, 0.5, 0.5, 0.5]
    )
    assert raw_answers[0]["silence"] == "B"
    assert calibrated_answers[0]["silence"] == "A"
    assert LABELS[int(raw_predictions[0])] != "NA"
    assert LABELS[int(calibrated_predictions[0])] == "NA"


def test_output_gate_rejects_single_sided_binary_stage() -> None:
    reports = {
        "set": {
            "hidden": {
                "prediction_distribution": {
                    "C": 5,
                    "BC": 5,
                    "T": 5,
                    "I": 5,
                    "NA": 5,
                },
                "dominant_fraction": 0.2,
                "binary_stages": {
                    stage: {"two_sided_predictions": stage != "silence"}
                    for stage in BINARY_STAGES
                },
            },
            "given": {
                "prediction_distribution": {
                    "C": 5,
                    "BC": 5,
                    "T": 5,
                    "I": 5,
                    "NA": 5,
                },
                "dominant_fraction": 0.2,
                "binary_stages": {
                    stage: {"two_sided_predictions": True}
                    for stage in BINARY_STAGES
                },
            },
            "shuffled": {
                "prediction_distribution": {
                    "C": 5,
                    "BC": 5,
                    "T": 5,
                    "I": 5,
                    "NA": 5,
                },
                "dominant_fraction": 0.2,
                "binary_stages": {
                    stage: {"two_sided_predictions": True}
                    for stage in BINARY_STAGES
                },
            },
        }
    }
    summary = _output_gate_summary(
        reports, min_class_count=1, max_dominant_fraction=0.65
    )
    assert not summary["passed"]
    assert not summary["by_evaluation_set_and_profile_mode"]["set"]["hidden"]
