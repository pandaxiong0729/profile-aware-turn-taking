"""Hybrid prompt-context + profile-vector pilot.

This module tests a deliberately narrow question:

* keep the Qwen prompt side responsible for causal audio + causal transcript;
* do not let Qwen see the profile text for the tested comparison;
* encode only the profile text as a frozen 384-dimensional MiniLM vector;
* train a tiny classifier on Qwen hidden-prompt binary scores plus the profile
  vector, then compare hidden/given/shuffled profile conditions.

It is a pilot adapter, not a Qwen fine-tune and not soft-token injection into
Qwen.  The Qwen feature for every profile mode is copied from the hidden-profile
binary prompt output, so only the profile vector changes across
hidden/given/shuffled.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .constants import ID_TO_LABEL, LABELS, LABEL_TO_ID
from .metrics import classification_metrics
from .semantic_profile_experiment import DEFAULT_SENTENCE_MODEL, _load_sentence_model
from .utils import read_jsonl, write_json, write_jsonl


BINARY_STAGES = ("silence", "listener_onset", "brief_response", "yield")
PROFILE_MODES = ("hidden", "given", "shuffled")


@dataclass(frozen=True)
class HybridRun:
    name: str
    root: Path
    sample_ids: list[str]
    labels: np.ndarray
    qwen_hidden_context: np.ndarray
    profile_texts: dict[str, list[str]]
    profile_vectors: dict[str, np.ndarray]
    baseline_predictions: np.ndarray
    audit: dict[str, Any]


def _prediction_distribution(predictions: Iterable[int]) -> dict[str, int]:
    counts = Counter(ID_TO_LABEL[int(index)] for index in predictions)
    return {label: int(counts.get(label, 0)) for label in LABELS}


def _rows_by_sample_mode(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(path):
        grouped[str(row["sample_id"])][str(row["profile_mode"])] = row
    return grouped


def _encode_profile_texts(
    texts: list[str],
    *,
    sentence_model: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    unique = sorted(set(texts))
    encoder = _load_sentence_model(sentence_model)
    vectors = encoder.encode(
        unique,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    return dict(zip(unique, vectors))


def load_hybrid_run(
    root: str | Path,
    *,
    name: str | None = None,
    sentence_model: str = DEFAULT_SENTENCE_MODEL,
    batch_size: int = 64,
) -> HybridRun:
    """Load one Qwen binary run as prompt-context features plus profile vectors."""

    run_root = Path(root).resolve()
    run_name = name or run_root.name
    references = {
        str(row["sample_id"]): str(row["reference_label"])
        for row in read_jsonl(run_root / "reference_labels.jsonl")
    }
    requests = _rows_by_sample_mode(run_root / "requests.jsonl")
    binary_predictions = _rows_by_sample_mode(run_root / "binary_predictions.jsonl")

    errors: list[str] = []
    sample_ids = sorted(references)
    labels: list[int] = []
    qwen_features: list[list[float]] = []
    baseline_predictions: list[int] = []
    profile_texts: dict[str, list[str]] = {mode: [] for mode in PROFILE_MODES}

    for sample_id in sample_ids:
        label = references[sample_id]
        if label not in LABEL_TO_ID:
            errors.append(f"{sample_id}: invalid label {label!r}")
            continue
        if set(requests.get(sample_id, {})) != set(PROFILE_MODES):
            errors.append(f"{sample_id}: missing profile-mode requests")
            continue
        hidden_prediction = binary_predictions.get(sample_id, {}).get("hidden")
        if hidden_prediction is None:
            errors.append(f"{sample_id}: missing hidden Qwen binary prediction")
            continue
        log_odds = hidden_prediction.get("semantic_A_log_odds", {})
        if set(log_odds) != set(BINARY_STAGES):
            errors.append(f"{sample_id}: incomplete hidden Qwen log-odds")
            continue
        baseline = str(hidden_prediction.get("prediction", ""))
        if baseline not in LABEL_TO_ID:
            errors.append(f"{sample_id}: invalid baseline prediction {baseline!r}")
            continue

        labels.append(LABEL_TO_ID[label])
        qwen_features.append([float(log_odds[stage]) for stage in BINARY_STAGES])
        baseline_predictions.append(LABEL_TO_ID[baseline])
        for mode in PROFILE_MODES:
            profile_texts[mode].append(str(requests[sample_id][mode]["profile_text"]))

    if errors:
        raise ValueError("Hybrid run audit failed:\n" + "\n".join(errors[:50]))

    vector_by_text = _encode_profile_texts(
        [text for mode in ("given", "shuffled") for text in profile_texts[mode]],
        sentence_model=sentence_model,
        batch_size=batch_size,
    )
    profile_vectors = {
        "hidden": np.zeros((len(labels), 384), dtype=np.float32),
        "given": np.stack([vector_by_text[text] for text in profile_texts["given"]]),
        "shuffled": np.stack([vector_by_text[text] for text in profile_texts["shuffled"]]),
    }
    qwen = np.asarray(qwen_features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    baseline = np.asarray(baseline_predictions, dtype=np.int64)

    # Verify only the profile vector changes across profile modes in this hybrid.
    audit = {
        "run_name": run_name,
        "run_root": str(run_root),
        "samples": int(len(y)),
        "class_counts": {
            label: int(np.sum(y == index)) for index, label in enumerate(LABELS)
        },
        "qwen_context_source": "hidden-profile binary prompt output only",
        "qwen_context_dimension": int(qwen.shape[1]),
        "profile_dimension": 384,
        "profile_hidden_representation": "all-zero vector",
        "profile_modes": list(PROFILE_MODES),
        "binary_stages": list(BINARY_STAGES),
        "qwen_features_identical_across_profile_modes": True,
        "qwen_saw_correct_or_shuffled_profile_for_hybrid": False,
        "profile_vector_only_change_in_hybrid": True,
        "sentence_model": sentence_model,
        "sentence_model_frozen": True,
    }
    return HybridRun(
        name=run_name,
        root=run_root,
        sample_ids=sample_ids,
        labels=y,
        qwen_hidden_context=qwen,
        profile_texts=profile_texts,
        profile_vectors=profile_vectors,
        baseline_predictions=baseline,
        audit=audit,
    )


def _features(run: HybridRun, mode: str) -> np.ndarray:
    if mode not in PROFILE_MODES:
        raise ValueError(f"Unknown profile mode: {mode}")
    return np.concatenate([run.qwen_hidden_context, run.profile_vectors[mode]], axis=1)


def _training_features(run: HybridRun) -> tuple[np.ndarray, np.ndarray]:
    """Train on hidden + correct profile only; shuffled is held for counterfactual eval."""

    x = np.concatenate([_features(run, "hidden"), _features(run, "given")], axis=0)
    y = np.concatenate([run.labels, run.labels], axis=0)
    return x, y


def train_profile_vector_prompt_classifier(run: HybridRun) -> Pipeline:
    """Fit a small deterministic multinomial classifier."""

    x_train, y_train = _training_features(run)
    classifier = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.25,
                    class_weight="balanced",
                    max_iter=3000,
                    multi_class="auto",
                    random_state=0,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    classifier.fit(x_train, y_train)
    return classifier


def evaluate_classifier(
    classifier: Pipeline,
    run: HybridRun,
    *,
    train_run_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reports: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for mode in PROFILE_MODES:
        predictions = classifier.predict(_features(run, mode)).astype(np.int64)
        report = classification_metrics(run.labels.tolist(), predictions.tolist())
        report["prediction_distribution"] = _prediction_distribution(predictions)
        reports[mode] = report
        for sample_id, target, prediction in zip(run.sample_ids, run.labels, predictions):
            prediction_rows.append(
                {
                    "train_run": train_run_name,
                    "eval_run": run.name,
                    "sample_id": sample_id,
                    "profile_mode": mode,
                    "reference_label": ID_TO_LABEL[int(target)],
                    "prediction": ID_TO_LABEL[int(prediction)],
                }
            )
    reports["qwen_hidden_prompt_baseline"] = {
        **classification_metrics(run.labels.tolist(), run.baseline_predictions.tolist()),
        "prediction_distribution": _prediction_distribution(run.baseline_predictions),
        "note": (
            "This is Qwen's hidden-profile prompt prediction. It is copied as the "
            "same prompt-context feature for all hybrid profile modes."
        ),
    }
    reports["paired_effects"] = {
        "given_minus_hidden_macro_f1": (
            reports["given"]["macro_f1"] - reports["hidden"]["macro_f1"]
        ),
        "given_minus_shuffled_macro_f1": (
            reports["given"]["macro_f1"] - reports["shuffled"]["macro_f1"]
        ),
    }
    return reports, prediction_rows


def run_cross_prompt_profile_vector_pilot(
    run_a: str | Path,
    run_b: str | Path,
    output_dir: str | Path,
    *,
    name_a: str = "prompt_seed137",
    name_b: str = "prompt_seed237",
    sentence_model: str = DEFAULT_SENTENCE_MODEL,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Run A->B and B->A cross evaluation for the hybrid pilot."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    loaded_a = load_hybrid_run(
        run_a, name=name_a, sentence_model=sentence_model, batch_size=batch_size
    )
    loaded_b = load_hybrid_run(
        run_b, name=name_b, sentence_model=sentence_model, batch_size=batch_size
    )

    all_predictions: list[dict[str, Any]] = []
    evaluations: dict[str, Any] = {}
    for train_run, eval_run in ((loaded_a, loaded_b), (loaded_b, loaded_a)):
        classifier = train_profile_vector_prompt_classifier(train_run)
        reports, rows = evaluate_classifier(
            classifier,
            eval_run,
            train_run_name=train_run.name,
        )
        key = f"train_{train_run.name}__eval_{eval_run.name}"
        evaluations[key] = reports
        all_predictions.extend(rows)

    sample_overlap = sorted(set(loaded_a.sample_ids) & set(loaded_b.sample_ids))
    summary = {
        "experiment": "profile-vector-plus-qwen-hidden-prompt-context-v1",
        "what_changed": (
            "Qwen reads only causal audio + causal transcript through the hidden-profile "
            "prompt. Correct/given/shuffled profile is encoded separately as a frozen "
            "384-dimensional MiniLM vector and consumed by a small classifier."
        ),
        "training_performed": True,
        "qwen_finetuned": False,
        "qwen_saw_profile_text": False,
        "profile_encoder": sentence_model,
        "profile_encoder_frozen": True,
        "classifier": "StandardScaler + multinomial LogisticRegression",
        "train_modes": ["hidden", "given"],
        "eval_modes": list(PROFILE_MODES),
        "shuffled_used_for_training": False,
        "runs": {loaded_a.name: loaded_a.audit, loaded_b.name: loaded_b.audit},
        "sample_overlap": {
            "count": len(sample_overlap),
            "examples": sample_overlap[:10],
        },
        "evaluations": evaluations,
    }
    write_json(output / "summary.json", summary)
    write_jsonl(output / "predictions.jsonl", all_predictions)
    write_json(
        output / "README.json",
        {
            "plain_language": [
                "This is the requested hybrid check.",
                "Transcript/audio stay on the Qwen prompt side.",
                "Profile is not written into Qwen's prompt here; it is encoded as a 384-dimensional MiniLM vector.",
                "A tiny classifier learns from Qwen hidden-prompt scores plus the profile vector.",
                "Hidden/given/shuffled evaluation changes only that profile vector.",
            ]
        },
    )
    return summary


__all__ = [
    "run_cross_prompt_profile_vector_pilot",
    "load_hybrid_run",
    "train_profile_vector_prompt_classifier",
    "evaluate_classifier",
]
