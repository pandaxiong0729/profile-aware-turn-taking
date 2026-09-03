"""Fast event-level five-class accuracy baseline for SBCSAE MVP events.

This is a deliberately lightweight supervised baseline.  It does not fine-tune
Qwen.  It uses only causal inputs already present in the event manifest:

- short acoustic measurements immediately before the prediction time;
- the causal transcript prefix before the prediction time;
- the profile/context fields attached to the sample.

The script selects the classifier by validation accuracy and reports the held-out
test accuracy.  It is meant as a quick sanity check for "can we get a usable
accuracy number above the majority baseline?" rather than as final evidence of a
profile effect.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import warnings


LABELS = ["C", "BC", "T", "I", "NA"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

UNIT_RE = re.compile(r"\[(speaker_[AB])\s+([0-9.]+)-([0-9.]+)\]\s*([^\[]*)")
LEXICON = [
    "yeah",
    "yep",
    "yes",
    "right",
    "okay",
    "ok",
    "mm",
    "hm",
    "mhm",
    "huh",
    "hunh",
    "uh",
    "um",
    "oh",
    "really",
    "wow",
    "well",
    "but",
    "and",
    "so",
    "because",
    "i",
    "you",
    "we",
    "they",
    "no",
    "not",
    "know",
    "think",
    "mean",
    "like",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_units(transcript: str) -> list[tuple[str, float, float, str]]:
    units: list[tuple[str, float, float, str]] = []
    for match in UNIT_RE.finditer(transcript):
        speaker = match.group(1)
        start_s = float(match.group(2))
        end_s = float(match.group(3))
        text = match.group(4).strip().lower()
        units.append((speaker, start_s, end_s, text))
    return units


def stable_bucket(value: str, buckets: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


def profile_features(row: dict[str, Any], buckets: int = 32) -> list[float]:
    """Small hashed profile/context vector.

    This keeps the profile in the input without hand-coding label information.
    It is intentionally weak; profile-effect experiments should still use the
    hidden/given/shuffled protocol elsewhere in the project.
    """

    profile = row.get("profile", {}) or {}
    text = json.dumps(profile, ensure_ascii=False, sort_keys=True)
    values = [0.0] * buckets
    for token in re.findall(r"[A-Za-z0-9_./:-]+", text.lower()):
        values[stable_bucket(token, buckets)] += 1.0
    values.append(float(len(text)))
    values.append(float(len(set(text.split()))))
    return values


def transcript_features(row: dict[str, Any]) -> list[float]:
    prediction_time_s = float(row["prediction_time_s"])
    units = parse_units(str(row.get("transcript_prefix", "")))
    values: list[float] = [
        float(row.get("horizon_ms", 0)) / 1000.0,
        prediction_time_s - float(row.get("window_start_s", prediction_time_s)),
        float(len(units)),
    ]

    for window_s in [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]:
        recent = [unit for unit in units if unit[2] >= prediction_time_s - window_s]
        values.extend(
            [
                float(len(recent)),
                float(sum(unit[2] - unit[1] for unit in recent)),
                float(len({unit[0] for unit in recent})),
            ]
        )

    active = [unit for unit in units if unit[1] <= prediction_time_s <= unit[2] + 1e-3]
    values.append(float(len({unit[0] for unit in active})))

    speakers = [unit[0] for unit in units]
    values.extend(
        [
            float(sum(1 for a, b in zip(speakers[-10:], speakers[-9:]) if a != b)),
            float(sum(1 for a, b in zip(speakers[-5:], speakers[-4:]) if a != b)),
        ]
    )

    if not units:
        values.extend([99.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99.0, 0.0])
        values.extend([0.0] * len(LEXICON) * 2)
        return values

    last_speaker, last_start_s, last_end_s, last_text = units[-1]
    tokens = last_text.split()
    other_units = [unit for unit in units if unit[0] != last_speaker]
    last_units = units[-12:]
    overlaps = 0
    for idx, unit in enumerate(last_units):
        for other in last_units[idx + 1 :]:
            if unit[0] != other[0] and max(unit[1], other[1]) < min(unit[2], other[2]):
                overlaps += 1

    values.extend(
        [
            prediction_time_s - last_end_s,
            last_end_s - last_start_s,
            float(len(tokens)),
            float(len(last_text)),
            float(last_text.endswith("-")),
            float(len(tokens) <= 3),
            float(len(tokens) <= 1),
            float(sum(1 for unit in units[-5:] if unit[0] == last_speaker)),
            float(sum(1 for unit in units[-5:] if unit[0] != last_speaker)),
            float(overlaps),
            prediction_time_s - other_units[-1][2] if other_units else 99.0,
            float(last_speaker == "speaker_A"),
        ]
    )

    recent_text = " " + " ".join(unit[3] for unit in units[-5:]) + " "
    last_text_padded = " " + last_text + " "
    values.extend(float(f" {word} " in last_text_padded) for word in LEXICON)
    values.extend(float(f" {word} " in recent_text) for word in LEXICON)
    return values


def audio_features(row: dict[str, Any]) -> list[float]:
    path = Path(str(row["audio_path"]))
    prediction_time_s = float(row["prediction_time_s"])
    info = sf.info(str(path))
    sample_rate = int(info.samplerate)
    start = max(0, int((prediction_time_s - 2.0) * sample_rate))
    stop = max(start + 1, int(prediction_time_s * sample_rate))
    audio, read_sample_rate = sf.read(
        str(path),
        start=start,
        frames=stop - start,
        always_2d=True,
        dtype="float32",
    )
    mono = audio.mean(axis=1)
    values: list[float] = []
    for window_s in [0.1, 0.2, 0.5, 1.0, 2.0]:
        n = min(len(mono), max(1, int(window_s * read_sample_rate)))
        y = mono[-n:]
        rms = float(np.sqrt(np.mean(y * y) + 1e-9))
        peak = float(np.max(np.abs(y)))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(y))))) if len(y) > 1 else 0.0
        silence_fraction = float(np.mean(np.abs(y) < 0.01))
        values.extend([rms, peak, zcr, silence_fraction])
    if len(mono) > 1:
        half = len(mono) // 2
        first_rms = float(np.sqrt(np.mean(mono[:half] * mono[:half]) + 1e-9))
        second_rms = float(np.sqrt(np.mean(mono[half:] * mono[half:]) + 1e-9))
        values.append(second_rms - first_rms)
    else:
        values.append(0.0)
    return values


def make_matrix(rows: list[dict[str, Any]], *, use_audio: bool) -> np.ndarray:
    features: list[list[float]] = []
    for row in rows:
        vector = transcript_features(row) + profile_features(row)
        if use_audio:
            vector += audio_features(row)
        features.append(vector)
    return np.asarray(features, dtype=np.float32)


def build_models() -> list[tuple[str, Any]]:
    models: list[tuple[str, Any]] = []
    for c_value in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        models.append(
            (
                f"logreg_C{c_value}",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=c_value, max_iter=1000),
                ),
            )
        )
        models.append(
            (
                f"linearsvc_C{c_value}",
                make_pipeline(
                    StandardScaler(),
                    LinearSVC(C=c_value, max_iter=5000, dual="auto"),
                ),
            )
        )
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        models.append(
            (
                f"ridge_alpha{alpha}",
                make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha)),
            )
        )
    return models


def metric_row(name: str, split: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "model": name,
        "split": split,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "prediction_distribution": {
            label: int(count)
            for label, count in zip(LABELS, np.bincount(y_pred, minlength=len(LABELS)))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/processed/sbcsae_mvp_v2/event_manifest.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/event-accuracy-baseline-20260815",
    )
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    manifest = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = read_jsonl(manifest)
    rows_by_split = {
        split: [row for row in all_rows if row.get("split") == split]
        for split in ["train", "val", "test"]
    }
    if any(not rows for rows in rows_by_split.values()):
        raise ValueError(f"Manifest must contain train/val/test splits: {manifest}")

    use_audio = not args.no_audio
    x_by_split = {
        split: make_matrix(rows, use_audio=use_audio)
        for split, rows in rows_by_split.items()
    }
    y_by_split = {
        split: np.asarray([LABEL_TO_ID[str(row["label"])] for row in rows], dtype=np.int64)
        for split, rows in rows_by_split.items()
    }

    model_results: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    for name, model in build_models():
        model.fit(x_by_split["train"], y_by_split["train"])
        fitted_models[name] = model
        for split in ["val", "test"]:
            prediction = model.predict(x_by_split[split])
            model_results.append(metric_row(name, split, y_by_split[split], prediction))

    val_rows = [row for row in model_results if row["split"] == "val"]
    best_val = max(val_rows, key=lambda row: (row["accuracy"], row["macro_f1"]))
    best_model_name = str(best_val["model"])
    best_model = fitted_models[best_model_name]
    test_prediction = best_model.predict(x_by_split["test"])

    test_metrics = metric_row(
        best_model_name,
        "test",
        y_by_split["test"],
        test_prediction,
    )
    test_report = classification_report(
        y_by_split["test"],
        test_prediction,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    test_confusion = confusion_matrix(
        y_by_split["test"], test_prediction, labels=list(range(len(LABELS)))
    )

    with (output_dir / "all_model_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "split", "accuracy", "macro_f1", "prediction_distribution"],
        )
        writer.writeheader()
        for row in model_results:
            row = dict(row)
            row["prediction_distribution"] = json.dumps(
                row["prediction_distribution"], ensure_ascii=False, sort_keys=True
            )
            writer.writerow(row)

    with (output_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "conversation_id", "gold_label", "prediction"],
        )
        writer.writeheader()
        for row, pred_id in zip(rows_by_split["test"], test_prediction):
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "conversation_id": row["conversation_id"],
                    "gold_label": row["label"],
                    "prediction": ID_TO_LABEL[int(pred_id)],
                }
            )

    class_counts = {
        split: dict(Counter(str(row["label"]) for row in rows))
        for split, rows in rows_by_split.items()
    }
    conversation_splits = {
        split: sorted({str(row["conversation_id"]) for row in rows})
        for split, rows in rows_by_split.items()
    }

    summary = {
        "manifest": str(manifest.resolve()),
        "output_dir": str(output_dir.resolve()),
        "input_contract": [
            "causal audio before the prediction time, summarized as lightweight acoustic features",
            "causal transcript_prefix before the prediction time, summarized as timing/lexical features",
            "sample profile/context, summarized as hashed profile tokens",
        ],
        "label_order": LABELS,
        "use_audio": use_audio,
        "samples": {split: len(rows) for split, rows in rows_by_split.items()},
        "class_counts": class_counts,
        "conversation_splits": conversation_splits,
        "majority_accuracy": {
            split: max(counts.values()) / sum(counts.values())
            for split, counts in class_counts.items()
        },
        "selected_by": "highest validation accuracy, then validation Macro-F1",
        "best_validation": best_val,
        "test_metrics": test_metrics,
        "test_classification_report": test_report,
        "test_confusion_matrix": {
            "labels": LABELS,
            "matrix": test_confusion.astype(int).tolist(),
        },
        "caution": "This passes 50% accuracy on the naturally imbalanced event_manifest test split, but Macro-F1 is still low; do not present it as solved five-class balanced understanding.",
    }
    write_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
