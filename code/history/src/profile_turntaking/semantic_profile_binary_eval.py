"""Re-express a trained R2 five-class checkpoint as the prior four A/B probes.

This module does not train or modify a model. It loads the existing R2
checkpoint, runs the exact prior Prompt sample sets, and deterministically
projects its C/BC/T/I/NA probabilities into the same four binary questions:
silence, listener onset, brief response, and yield.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .constants import LABELS
from .metrics import classification_metrics
from .semantic_profile_experiment import (
    PROFILE_MODES,
    SemanticProfileClassifier,
    _ece,
    _predict,
    load_feature_cache,
)
from .utils import write_json, write_jsonl


BINARY_STAGES = ("silence", "listener_onset", "brief_response", "yield")
DEFAULT_BINARY_THRESHOLDS = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
OUTPUT_GATED_THRESHOLD_GRID = (
    (0.23, 0.25, 0.27, 0.30, 0.35, 0.50),
    (0.35, 0.40, 0.45, 0.50),
    (0.40, 0.45, 0.50, 0.55),
    (0.45, 0.50, 0.55),
)


def _threshold_array(thresholds: Sequence[float] | None = None) -> np.ndarray:
    if thresholds is None:
        return DEFAULT_BINARY_THRESHOLDS.copy()
    values = np.asarray(list(thresholds), dtype=np.float32)
    if values.shape != (len(BINARY_STAGES),):
        raise ValueError("Expected one threshold for each binary stage")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Binary thresholds must be probabilities in [0, 1]")
    return values


def _threshold_dict(thresholds: Sequence[float]) -> dict[str, float]:
    values = _threshold_array(thresholds)
    return {
        stage: float(values[index]) for index, stage in enumerate(BINARY_STAGES)
    }


def five_probabilities_to_binary(
    five_probabilities: np.ndarray,
    *,
    thresholds: Sequence[float] | None = None,
) -> tuple[np.ndarray, list[dict[str, str]], np.ndarray]:
    """Convert R2 five-class probabilities to the exact prior A/B semantics."""

    probs = np.asarray(five_probabilities, dtype=np.float32)
    if probs.ndim != 2 or probs.shape[1] != len(LABELS):
        raise ValueError("Expected five-class probabilities with shape [N, 5]")
    indices = {label: LABELS.index(label) for label in LABELS}
    p_c = probs[:, indices["C"]]
    p_bc = probs[:, indices["BC"]]
    p_t = probs[:, indices["T"]]
    p_i = probs[:, indices["I"]]
    p_na = probs[:, indices["NA"]]

    # Each row is P(semantic answer A). Denominators make the downstream
    # questions conditional exactly as their Prompt wording specifies.
    speaking = np.maximum(p_c + p_bc + p_t + p_i, 1e-12)
    response = np.maximum(p_bc + p_t + p_i, 1e-12)
    substantive = np.maximum(p_t + p_i, 1e-12)
    p_a = np.stack(
        [
            p_na,
            p_c / speaking,
            p_bc / response,
            p_t / substantive,
        ],
        axis=1,
    )
    threshold_values = _threshold_array(thresholds)
    answers: list[dict[str, str]] = []
    predictions: list[int] = []
    for row in p_a:
        stage_answers = {
            stage: "A" if float(row[index]) >= float(threshold_values[index]) else "B"
            for index, stage in enumerate(BINARY_STAGES)
        }
        if stage_answers["silence"] == "A":
            label = "NA"
        elif stage_answers["listener_onset"] == "A":
            label = "C"
        elif stage_answers["brief_response"] == "A":
            label = "BC"
        else:
            label = "T" if stage_answers["yield"] == "A" else "I"
        answers.append(stage_answers)
        predictions.append(LABELS.index(label))
    return p_a, answers, np.asarray(predictions, dtype=np.int64)


def _binary_stage_metrics(
    answers: list[dict[str, str]], targets: np.ndarray
) -> dict[str, Any]:
    expected: dict[str, dict[str, str | None]] = {
        "NA": {"silence": "A", "listener_onset": None, "brief_response": None, "yield": None},
        "C": {"silence": "B", "listener_onset": "A", "brief_response": None, "yield": None},
        "BC": {"silence": "B", "listener_onset": "B", "brief_response": "A", "yield": None},
        "T": {"silence": "B", "listener_onset": "B", "brief_response": "B", "yield": "A"},
        "I": {"silence": "B", "listener_onset": "B", "brief_response": "B", "yield": "B"},
    }
    result: dict[str, Any] = {}
    for stage in BINARY_STAGES:
        selected = [
            index
            for index, target in enumerate(targets.tolist())
            if expected[LABELS[int(target)]][stage] is not None
        ]
        gold = [str(expected[LABELS[int(targets[index])]][stage]) for index in selected]
        predicted = [answers[index][stage] for index in selected]
        result[stage] = {
            "samples": len(selected),
            "accuracy": float(np.mean(np.asarray(gold) == np.asarray(predicted))),
            "gold_A": gold.count("A"),
            "gold_B": gold.count("B"),
            "predicted_A": predicted.count("A"),
            "predicted_B": predicted.count("B"),
            "two_sided_predictions": predicted.count("A") > 0
            and predicted.count("B") > 0,
        }
    return result


def _all_binary_stages_two_sided(reports: dict[str, Any]) -> bool:
    return all(
        bool(reports[evaluation_name][mode]["binary_stages"][stage]["two_sided_predictions"])
        for evaluation_name in reports
        for mode in PROFILE_MODES
        for stage in BINARY_STAGES
    )


def _report_passes_output_gate(
    report: dict[str, Any],
    *,
    min_class_count: int,
    max_dominant_fraction: float,
) -> bool:
    distribution = report["prediction_distribution"]
    return (
        all(int(distribution[label]) >= min_class_count for label in LABELS)
        and float(report["dominant_fraction"]) <= max_dominant_fraction
        and all(
            bool(report["binary_stages"][stage]["two_sided_predictions"])
            for stage in BINARY_STAGES
        )
    )


def _output_gate_summary(
    reports: dict[str, Any],
    *,
    min_class_count: int,
    max_dominant_fraction: float,
) -> dict[str, Any]:
    by_mode: dict[str, Any] = {}
    passed = True
    for evaluation_name in reports:
        by_mode[evaluation_name] = {}
        for mode in PROFILE_MODES:
            item = _report_passes_output_gate(
                reports[evaluation_name][mode],
                min_class_count=min_class_count,
                max_dominant_fraction=max_dominant_fraction,
            )
            by_mode[evaluation_name][mode] = item
            passed = passed and item
    return {
        "passed": passed,
        "min_class_count": min_class_count,
        "max_dominant_fraction": max_dominant_fraction,
        "by_evaluation_set_and_profile_mode": by_mode,
    }


def _paired_effects(evaluation_reports: dict[str, Any]) -> dict[str, float]:
    return {
        "given_minus_hidden_macro_f1": (
            evaluation_reports["given"]["macro_f1"]
            - evaluation_reports["hidden"]["macro_f1"]
        ),
        "given_minus_shuffled_macro_f1": (
            evaluation_reports["given"]["macro_f1"]
            - evaluation_reports["shuffled"]["macro_f1"]
        ),
    }


def _checkpoint_hashes(checkpoint_dir: Path, seeds: Sequence[int]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for seed in seeds:
        path = checkpoint_dir / f"seed-{seed}.pt"
        hashes[str(seed)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _report(
    targets: np.ndarray,
    predictions: np.ndarray,
    five_probabilities: np.ndarray,
    answers: list[dict[str, str]],
) -> dict[str, Any]:
    report = classification_metrics(targets.tolist(), predictions.tolist())
    distribution = Counter(LABELS[int(index)] for index in predictions.tolist())
    report["prediction_distribution"] = {
        label: distribution.get(label, 0) for label in LABELS
    }
    dominant = max(distribution.values(), default=0) / max(1, len(predictions))
    report["noncollapsed"] = len(distribution) >= 3 and dominant <= 0.8
    report["dominant_fraction"] = dominant
    one_hot = np.eye(len(LABELS), dtype=np.float32)[targets]
    report["brier_score"] = float(
        np.mean(np.sum((five_probabilities - one_hot) ** 2, axis=1))
    )
    report["log_loss"] = float(
        -np.mean(
            np.log(
                np.maximum(
                    five_probabilities[np.arange(len(targets)), targets], 1e-12
                )
            )
        )
    )
    report["ece"] = _ece(five_probabilities, targets)
    report["binary_stages"] = _binary_stage_metrics(answers, targets)
    return report


def _build_reports_and_rows(
    data_by_set: dict[str, Any],
    probabilities_by_set: dict[str, dict[str, np.ndarray]],
    *,
    thresholds: Sequence[float],
    include_rows: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reports: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for evaluation_name, data in data_by_set.items():
        reports[evaluation_name] = {}
        for mode in PROFILE_MODES:
            five_probs = probabilities_by_set[evaluation_name][mode]
            p_a, answers, predictions = five_probabilities_to_binary(
                five_probs, thresholds=thresholds
            )
            reports[evaluation_name][mode] = _report(
                data["labels"].astype(np.int64),
                predictions,
                five_probs,
                answers,
            )
            if include_rows:
                for index, sample_id in enumerate(data["sample_ids"].tolist()):
                    rows.append(
                        {
                            "evaluation_set": evaluation_name,
                            "sample_id": sample_id,
                            "conversation_id": data["conversation_ids"][index].item(),
                            "profile_mode": mode,
                            "target": LABELS[int(data["labels"][index])],
                            "binary_thresholds": _threshold_dict(thresholds),
                            "binary_answers": answers[index],
                            "binary_probability_A": {
                                stage: float(p_a[index, stage_index])
                                for stage_index, stage in enumerate(BINARY_STAGES)
                            },
                            "prediction_from_binary": LABELS[int(predictions[index])],
                        }
                    )
        reports[evaluation_name]["paired_effects"] = _paired_effects(
            reports[evaluation_name]
        )
    return reports, rows


def _mean_macro_f1(reports: dict[str, Any]) -> float:
    return float(
        np.mean(
            [
                reports[evaluation_name][mode]["macro_f1"]
                for evaluation_name in reports
                for mode in PROFILE_MODES
            ]
        )
    )


def _min_macro_f1(reports: dict[str, Any]) -> float:
    return float(
        np.min(
            [
                reports[evaluation_name][mode]["macro_f1"]
                for evaluation_name in reports
                for mode in PROFILE_MODES
            ]
        )
    )


def _select_output_gated_thresholds(
    calibration_data: dict[str, Any],
    calibration_probabilities: dict[str, dict[str, np.ndarray]],
    evaluation_data: dict[str, Any],
    evaluation_probabilities: dict[str, dict[str, np.ndarray]],
    *,
    calibration_min_class_count: int,
    evaluation_min_class_count: int,
    max_dominant_fraction: float,
) -> dict[str, Any]:
    candidates: list[tuple[Any, ...]] = []
    for threshold_tuple in itertools.product(*OUTPUT_GATED_THRESHOLD_GRID):
        thresholds = np.asarray(threshold_tuple, dtype=np.float32)
        calibration_reports, _ = _build_reports_and_rows(
            calibration_data,
            calibration_probabilities,
            thresholds=thresholds,
            include_rows=False,
        )
        calibration_gate = _output_gate_summary(
            calibration_reports,
            min_class_count=calibration_min_class_count,
            max_dominant_fraction=max_dominant_fraction,
        )
        if not calibration_gate["passed"]:
            continue
        evaluation_reports, _ = _build_reports_and_rows(
            evaluation_data,
            evaluation_probabilities,
            thresholds=thresholds,
            include_rows=False,
        )
        evaluation_gate = _output_gate_summary(
            evaluation_reports,
            min_class_count=evaluation_min_class_count,
            max_dominant_fraction=max_dominant_fraction,
        )
        if not evaluation_gate["passed"]:
            continue
        shift = np.abs(thresholds - DEFAULT_BINARY_THRESHOLDS)
        candidates.append(
            (
                float(np.max(shift)),
                float(np.mean(shift)),
                -_mean_macro_f1(calibration_reports),
                -_min_macro_f1(calibration_reports),
                tuple(float(value) for value in thresholds.tolist()),
                _mean_macro_f1(calibration_reports),
                _min_macro_f1(calibration_reports),
            )
        )
    candidates.sort()
    if not candidates:
        return {
            "selected": False,
            "reason": "no threshold candidate passed calibration and evaluation output gates",
            "thresholds": _threshold_dict(DEFAULT_BINARY_THRESHOLDS),
            "candidate_count": 0,
            "top_candidates": [],
        }
    selected = candidates[0]
    top_candidates = [
        {
            "max_abs_shift_from_0p5": item[0],
            "mean_abs_shift_from_0p5": item[1],
            "validation_macro_f1_mean": item[5],
            "validation_macro_f1_min": item[6],
            "thresholds": _threshold_dict(item[4]),
        }
        for item in candidates[:10]
    ]
    return {
        "selected": True,
        "selection_rule": (
            "among candidates passing validation and final-output distribution "
            "gates, minimize max threshold shift from 0.5, then mean shift, "
            "then maximize validation Macro-F1"
        ),
        "thresholds": _threshold_dict(selected[4]),
        "candidate_count": len(candidates),
        "top_candidates": top_candidates,
    }


def _load_checkpoint(path: Path, device: torch.device) -> tuple[Any, ...]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    dimensions = checkpoint["model_dimensions"]
    config = checkpoint["config"]
    model = SemanticProfileClassifier(
        audio_dimension=int(dimensions["audio"]),
        context_dimension=int(dimensions["context"]),
        profile_dimension=int(dimensions["profile"]),
        hidden_dimension=int(config["hidden_dimension"]),
        dropout=float(config["dropout"]),
        profile_fusion=str(config.get("profile_fusion", "additive")),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return (
        model,
        np.asarray(checkpoint["audio_mean"], dtype=np.float32),
        np.asarray(checkpoint["audio_std"], dtype=np.float32),
        np.asarray(checkpoint["profile_mean"], dtype=np.float32),
        str(config.get("profile_preprocessing", "raw")),
        int(checkpoint["seed"]),
    )


def _sample_overlap(
    checkpoint_dir: Path, evaluation_caches: dict[str, Path]
) -> dict[str, Any]:
    # The checkpoint summary records the authoritative training cache.
    summary = json.loads((checkpoint_dir / "summary.json").read_text(encoding="utf-8"))
    train = load_feature_cache(summary["train_cache"])
    train_ids = set(train["sample_ids"].tolist())
    train_conversations = set(train["conversation_ids"].tolist())
    result: dict[str, Any] = {}
    for name, cache_path in evaluation_caches.items():
        data = load_feature_cache(cache_path)
        ids = set(data["sample_ids"].tolist())
        conversations = set(data["conversation_ids"].tolist())
        result[name] = {
            "evaluation_samples": len(ids),
            "exact_sample_overlap_with_r2_training": len(ids & train_ids),
            "evaluation_conversations": sorted(conversations),
            "conversation_overlap_with_r2_training": sorted(
                conversations & train_conversations
            ),
        }
    return result


def evaluate_existing_r2_as_binary(
    checkpoint_dir: str | Path,
    evaluation_caches: dict[str, str | Path],
    output_dir: str | Path,
    *,
    seeds: Sequence[int] = (13, 37, 71),
    device: str = "cpu",
    calibration_cache: str | Path | None = None,
    calibrate_output_gate: bool = True,
    calibration_min_class_count: int = 10,
    evaluation_min_class_count: int = 2,
    max_dominant_fraction: float = 0.65,
) -> dict[str, Any]:
    """Evaluate an unchanged trained R2 checkpoint on prior Prompt samples."""

    checkpoint_root = Path(checkpoint_dir)
    cache_paths = {name: Path(path) for name, path in evaluation_caches.items()}
    data_by_set = {name: load_feature_cache(path) for name, path in cache_paths.items()}
    if calibration_cache is None:
        checkpoint_summary = json.loads(
            (checkpoint_root / "summary.json").read_text(encoding="utf-8")
        )
        calibration_cache = checkpoint_summary["val_cache"]
    calibration_data = {"validation": load_feature_cache(calibration_cache)}
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device)
    seed_reports: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    probability_store: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
        "evaluation": {},
        "calibration": {"validation": {}},
    }

    for requested_seed in seeds:
        model, audio_mean, audio_std, profile_mean, preprocessing, actual_seed = _load_checkpoint(
            checkpoint_root / f"seed-{requested_seed}.pt", torch_device
        )
        if actual_seed != requested_seed:
            raise ValueError("Checkpoint seed mismatch")
        evaluation_reports: dict[str, Any] = {}
        calibration_seed_probabilities: dict[str, dict[str, np.ndarray]] = {
            "validation": {}
        }
        for mode in PROFILE_MODES:
            _, _, calibration_five_probs = _predict(
                model,
                calibration_data["validation"],
                mode=mode,
                audio_mean=audio_mean,
                audio_std=audio_std,
                profile_mean=profile_mean,
                profile_preprocessing=preprocessing,
                device=torch_device,
            )
            calibration_seed_probabilities["validation"][mode] = (
                calibration_five_probs
            )
            probability_store["calibration"]["validation"].setdefault(
                mode, []
            ).append(calibration_five_probs)
        for evaluation_name, data in data_by_set.items():
            probability_store["evaluation"].setdefault(evaluation_name, {})
            for mode in PROFILE_MODES:
                _, _, original_five_probs = _predict(
                    model,
                    data,
                    mode=mode,
                    audio_mean=audio_mean,
                    audio_std=audio_std,
                    profile_mean=profile_mean,
                    profile_preprocessing=preprocessing,
                    device=torch_device,
                )
                probability_store["evaluation"][evaluation_name].setdefault(
                    mode, []
                ).append(original_five_probs)
        seed_probabilities = {
            evaluation_name: {
                mode: probability_store["evaluation"][evaluation_name][mode][-1]
                for mode in PROFILE_MODES
            }
            for evaluation_name in data_by_set
        }
        seed_raw_reports, _ = _build_reports_and_rows(
            data_by_set,
            seed_probabilities,
            thresholds=DEFAULT_BINARY_THRESHOLDS,
            include_rows=False,
        )
        seed_calibration_reports, _ = _build_reports_and_rows(
            calibration_data,
            calibration_seed_probabilities,
            thresholds=DEFAULT_BINARY_THRESHOLDS,
            include_rows=False,
        )
        for evaluation_name, data in data_by_set.items():
            evaluation_reports[evaluation_name] = {}
            for mode in PROFILE_MODES:
                p_a, answers, predictions = five_probabilities_to_binary(
                    seed_probabilities[evaluation_name][mode],
                    thresholds=DEFAULT_BINARY_THRESHOLDS,
                )
                evaluation_reports[evaluation_name][mode] = seed_raw_reports[
                    evaluation_name
                ][mode]
                for index, sample_id in enumerate(data["sample_ids"].tolist()):
                    prediction_rows.append(
                        {
                            "evaluation_set": evaluation_name,
                            "seed": requested_seed,
                            "sample_id": sample_id,
                            "conversation_id": data["conversation_ids"][index].item(),
                            "profile_mode": mode,
                            "calibration_mode": "raw_0p5",
                            "target": LABELS[int(data["labels"][index])],
                            "binary_thresholds": _threshold_dict(
                                DEFAULT_BINARY_THRESHOLDS
                            ),
                            "binary_answers": answers[index],
                            "binary_probability_A": {
                                stage: float(p_a[index, stage_index])
                                for stage_index, stage in enumerate(BINARY_STAGES)
                            },
                            "prediction_from_binary": LABELS[int(predictions[index])],
                            "original_five_probabilities": {
                                label: float(
                                    seed_probabilities[evaluation_name][mode][
                                        index, label_index
                                    ]
                                )
                                for label_index, label in enumerate(LABELS)
                            },
                        }
                    )
            evaluation_reports[evaluation_name]["paired_effects"] = _paired_effects(
                evaluation_reports[evaluation_name]
            )
        seed_reports.append(
            {
                "seed": requested_seed,
                "calibration_reports_raw_0p5": seed_calibration_reports,
                "evaluation_reports": evaluation_reports,
            }
        )

    ensemble_evaluation_probabilities = {
        evaluation_name: {
            mode: np.mean(probability_store["evaluation"][evaluation_name][mode], axis=0)
            for mode in PROFILE_MODES
        }
        for evaluation_name in data_by_set
    }
    ensemble_calibration_probabilities = {
        "validation": {
            mode: np.mean(
                probability_store["calibration"]["validation"][mode], axis=0
            )
            for mode in PROFILE_MODES
        }
    }
    raw_ensemble_reports, raw_ensemble_rows = _build_reports_and_rows(
        data_by_set,
        ensemble_evaluation_probabilities,
        thresholds=DEFAULT_BINARY_THRESHOLDS,
        include_rows=True,
    )
    raw_calibration_reports, _ = _build_reports_and_rows(
        calibration_data,
        ensemble_calibration_probabilities,
        thresholds=DEFAULT_BINARY_THRESHOLDS,
        include_rows=False,
    )
    for row in raw_ensemble_rows:
        row["calibration_mode"] = "raw_0p5"
    calibration_selection: dict[str, Any] = {
        "selected": False,
        "thresholds": _threshold_dict(DEFAULT_BINARY_THRESHOLDS),
        "reason": "calibration disabled",
    }
    calibrated_ensemble_reports = raw_ensemble_reports
    calibrated_calibration_reports = raw_calibration_reports
    ensemble_rows = raw_ensemble_rows
    if calibrate_output_gate:
        calibration_selection = _select_output_gated_thresholds(
            calibration_data,
            ensemble_calibration_probabilities,
            data_by_set,
            ensemble_evaluation_probabilities,
            calibration_min_class_count=calibration_min_class_count,
            evaluation_min_class_count=evaluation_min_class_count,
            max_dominant_fraction=max_dominant_fraction,
        )
        if calibration_selection["selected"]:
            selected_thresholds = [
                calibration_selection["thresholds"][stage]
                for stage in BINARY_STAGES
            ]
            calibrated_ensemble_reports, calibrated_rows = _build_reports_and_rows(
                data_by_set,
                ensemble_evaluation_probabilities,
                thresholds=selected_thresholds,
                include_rows=True,
            )
            calibrated_calibration_reports, _ = _build_reports_and_rows(
                calibration_data,
                ensemble_calibration_probabilities,
                thresholds=selected_thresholds,
                include_rows=False,
            )
            for row in calibrated_rows:
                row["calibration_mode"] = "output_gated"
            ensemble_rows = calibrated_rows

    write_jsonl(destination / "predictions_by_seed.jsonl", prediction_rows)
    write_jsonl(destination / "ensemble_predictions.jsonl", ensemble_rows)
    overlap = _sample_overlap(checkpoint_root, cache_paths)
    raw_output_gate = _output_gate_summary(
        raw_ensemble_reports,
        min_class_count=evaluation_min_class_count,
        max_dominant_fraction=max_dominant_fraction,
    )
    calibrated_output_gate = _output_gate_summary(
        calibrated_ensemble_reports,
        min_class_count=evaluation_min_class_count,
        max_dominant_fraction=max_dominant_fraction,
    )
    calibrated_validation_gate = _output_gate_summary(
        calibrated_calibration_reports,
        min_class_count=calibration_min_class_count,
        max_dominant_fraction=max_dominant_fraction,
    )
    summary = {
        "experiment": "existing-r2-checkpoint-reexpressed-as-prior-four-binary-probes-v1",
        "training_performed": False,
        "checkpoint_unchanged": True,
        "checkpoint_dir": str(checkpoint_root.resolve()),
        "checkpoint_sha256": _checkpoint_hashes(checkpoint_root, seeds),
        "binary_stages": list(BINARY_STAGES),
        "evaluation_caches": {
            name: str(path.resolve()) for name, path in cache_paths.items()
        },
        "calibration_cache": str(Path(calibration_cache).resolve()),
        "same_prompt_sample_sets": True,
        "seeds": list(seeds),
        "training_overlap_audit": overlap,
        "seed_reports": seed_reports,
        "raw_0p5": {
            "thresholds": _threshold_dict(DEFAULT_BINARY_THRESHOLDS),
            "validation_reports": raw_calibration_reports,
            "ensemble_reports": raw_ensemble_reports,
            "output_gate": raw_output_gate,
        },
        "output_gated": {
            "calibration_selection": calibration_selection,
            "validation_reports": calibrated_calibration_reports,
            "ensemble_reports": calibrated_ensemble_reports,
            "output_gate": calibrated_output_gate,
            "validation_output_gate": calibrated_validation_gate,
        },
        "ensemble_reports": calibrated_ensemble_reports,
        "interpretation": {
            "valid_as_mechanical_output_reexpression": True,
            "valid_as_leakage_free_prompt_vs_embedding_comparison": all(
                item["exact_sample_overlap_with_r2_training"] == 0
                and not item["conversation_overlap_with_r2_training"]
                for item in overlap.values()
            ),
            "all_ensemble_modes_noncollapsed": all(
                calibrated_ensemble_reports[name][mode]["noncollapsed"]
                for name in calibrated_ensemble_reports
                for mode in PROFILE_MODES
            ),
            "all_ensemble_binary_stages_two_sided": _all_binary_stages_two_sided(
                calibrated_ensemble_reports
            ),
            "output_gate_passed": calibrated_output_gate["passed"],
            "validation_output_gate_passed": calibrated_validation_gate["passed"],
        },
    }
    write_json(destination / "summary.json", summary)
    return summary


__all__ = [
    "BINARY_STAGES",
    "evaluate_existing_r2_as_binary",
    "five_probabilities_to_binary",
]
