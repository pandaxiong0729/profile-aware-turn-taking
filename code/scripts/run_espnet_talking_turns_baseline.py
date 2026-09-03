#!/usr/bin/env python
"""Run the released Talking Turns ESPnet checkpoint on the SBCSAE test set.

The released model produces five probabilities in its native order
``C, NA, I, BC, T``.  This runner preserves the project's 30 s causal audio
windows, evaluates the five-way output, and also converts the probabilities to
the four paper-style binary tasks used by our Qwen experiments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "code" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from profile_turntaking.audio import read_wav_window_robust_mix


FIVE_LABELS = ("C", "BC", "T", "I", "NA")
ESP_NATIVE_LABELS = ("C", "NA", "I", "BC", "T")
TASKS = ("turn_change", "backchannel", "interruption", "floor_taking")
TASK_PAIRS = {
    "turn_change": ("C", "T"),
    "backchannel": ("C", "BC"),
    "interruption": ("C", "I"),
    "floor_taking": ("C", "T"),
}
FLOOR_SOURCE_TARGET = {
    "hold_after_unsuccessful_interruption": 0,
    "shift_after_successful_interruption": 1,
}


def parse_args() -> argparse.Namespace:
    model_root = REPO_ROOT / "models" / "talking_turns" / "checkpoint"
    model_exp = model_root / "exp" / "asr_train_asr_whisper_turn_taking_raw_en_word"
    dataset = REPO_ROOT / "data" / "processed" / "sbcsae_qwen_shared_ab_30s_causal_v1"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--espnet-source",
        type=Path,
        default=REPO_ROOT / "models" / "talking_turns" / "espnet_source",
    )
    parser.add_argument("--config", type=Path, default=model_exp / "config.yaml")
    parser.add_argument("--checkpoint", type=Path, default=model_exp / "valid.loss.ave.pth")
    parser.add_argument("--selected-inputs", type=Path, default=dataset / "test" / "selected_inputs.jsonl")
    parser.add_argument("--references", type=Path, default=dataset / "test" / "reference_labels.jsonl")
    parser.add_argument("--dataset-audit", type=Path, default=dataset / "test" / "input_audit.json")
    parser.add_argument(
        "--qwen-predictions",
        type=Path,
        default=REPO_ROOT
        / "artifacts"
        / "main_experiment"
        / "audio_only_baseline"
        / "test_predictions.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "talking_turns" / "sbcsae_test",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Recompute metrics from an existing test_predictions.jsonl without loading the model.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_audio(samples: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(samples, dtype="<f4").tobytes()).hexdigest()


def resolve_audio_path(row: dict[str, Any]) -> Path:
    """Resolve old recorded paths and the portable repository-relative fallback."""
    recorded = Path(str(row["source_audio_path"]))
    if recorded.is_file():
        return recorded
    portable = REPO_ROOT / "data/sbcsae/openslr/WAV" / f"{row['conversation_id']}.wav"
    if portable.is_file():
        return portable
    raise FileNotFoundError(
        f"Audio for {row['sample_id']} was not found at the recorded path or {portable}"
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_floor_targets(references: list[dict[str, Any]]) -> dict[str, int]:
    return {
        str(row["sample_id"]): FLOOR_SOURCE_TARGET[str(row.get("source_kind", ""))]
        for row in references
        if str(row.get("source_kind", "")) in FLOOR_SOURCE_TARGET
    }


def load_balanced_ids(path: Path) -> dict[str, set[str]]:
    rows = read_jsonl(path)
    # The subset is deterministic and identical across seeds/profile modes.
    chosen = [
        row
        for row in rows
        if int(row["seed"]) == 13
        and str(row["profile_mode"]) == "hidden"
        and bool(row.get("paper_balanced_subset", False))
    ]
    result = {task: set() for task in TASKS}
    for row in chosen:
        result[str(row["task"])].add(str(row["sample_id"]))
    return result


def preflight_audit(
    *,
    records: list[dict[str, Any]],
    references: list[dict[str, Any]],
    dataset_audit: dict[str, Any],
    checkpoint: Path,
    config: Path,
    balanced_ids: dict[str, set[str]],
    output_path: Path,
) -> dict[str, Any]:
    ids = [str(row["sample_id"]) for row in records]
    reference_ids = [str(row["sample_id"]) for row in references]
    if ids != reference_ids:
        raise ValueError("selected_inputs and reference_labels are not identically ordered")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample IDs")
    if not bool(dataset_audit.get("passed")):
        raise ValueError("source dataset audit did not pass")
    if not bool(dataset_audit.get("causal_transcript_timestamps_checked")):
        raise ValueError("causal transcript timestamps were not checked")
    if not bool(dataset_audit.get("audio_window_sha256_checked")):
        raise ValueError("source audio-window SHA audit is missing")
    if not bool(dataset_audit.get("reference_labels_kept_outside_requests")):
        raise ValueError("target-leakage audit is missing")

    bad_boundaries = []
    bad_durations = []
    for row in records:
        boundary = float(row["prediction_boundary_in_conversation_s"])
        start = float(row["audio_window_start_s"])
        end = float(row["audio_window_end_s"])
        if abs(end - boundary) > 1e-6:
            bad_boundaries.append(str(row["sample_id"]))
        if abs((end - start) - 30.0) > 1e-5:
            bad_durations.append(str(row["sample_id"]))
    if bad_boundaries or bad_durations:
        raise ValueError(
            f"invalid causal windows: boundary={len(bad_boundaries)} duration={len(bad_durations)}"
        )

    source_hashes: dict[str, str] = {}
    for row in records:
        path = str(resolve_audio_path(row).resolve())
        expected = str(row["source_audio_sha256"])
        if path not in source_hashes:
            source_hashes[path] = sha256_file(Path(path))
        if source_hashes[path] != expected:
            raise ValueError(f"source audio SHA mismatch: {path}")

    class_counts = Counter(str(row["reference_label"]) for row in references)
    missing_labels = set(FIVE_LABELS) - class_counts.keys()
    if missing_labels:
        raise ValueError(f"test set is missing labels: {sorted(missing_labels)}")
    if any(not str(row.get("transcript_sha256", "")) for row in records):
        raise ValueError("matching causal transcript SHA is missing")
    if any(len(balanced_ids[task]) == 0 for task in TASKS):
        raise ValueError("one or more deterministic A/B subsets are empty")

    audit = {
        "passed": True,
        "samples": len(records),
        "class_counts": dict(class_counts),
        "causal_audio": {
            "all_windows_end_at_prediction_boundary": True,
            "all_windows_seconds": 30.0,
            "source_audio_files": len(source_hashes),
            "source_audio_sha256_equality_checked": True,
            "window_audio_sha256_rechecked_during_inference": True,
        },
        "matching_causal_transcript": {
            "available_for_every_sample": True,
            "sha256_available_for_every_sample": True,
            "timestamps_checked_by_source_audit": True,
            "fed_to_released_espnet_model": False,
            "reason": "The released Talking Turns checkpoint has an audio-only input interface.",
        },
        "profile": {
            "fed_to_released_espnet_model": False,
            "reason": "This is an external audio-only baseline, not a hidden/given/shuffled profile comparison.",
        },
        "target_absence": {
            "model_input_fields": ["causal_audio_float32"],
            "labels_or_annotation_evidence_in_model_input": False,
            "references_loaded_only_for_scoring": True,
        },
        "output_schema": {
            "native_probability_order": list(ESP_NATIVE_LABELS),
            "saved_probability_order": list(FIVE_LABELS),
            "binary_tasks": {task: list(TASK_PAIRS[task]) for task in TASKS},
        },
        "paper_balanced_subset_counts": {task: len(ids_) for task, ids_ in balanced_ids.items()},
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": str(config.resolve()),
        "config_sha256": sha256_file(config),
    }
    write_json(output_path, audit)
    return audit


def load_model(espnet_source: Path, config: Path, checkpoint: Path, device: str):
    source = str(espnet_source.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from espnet2.bin.asr_inference import Speech2Text

    return Speech2Text(
        str(config.resolve()),
        str(checkpoint.resolve()),
        device=device,
        run_chunk=False,
    )


@torch.inference_mode()
def infer_batch(model: Any, audio_batch: np.ndarray, device: str) -> np.ndarray:
    speech = torch.from_numpy(audio_batch).to(device=device, dtype=torch.float32)
    lengths = torch.full(
        (speech.shape[0],), speech.shape[1], device=device, dtype=torch.long
    )
    enc, enc_lens = model.asr_model.encode(speech=speech, speech_lengths=lengths)
    transformed = model.asr_model.transform_mean(model.asr_model.act_fn(enc))
    final_frames = torch.stack(
        [transformed[index, enc_lens[index] - 1] for index in range(len(enc_lens))]
    )
    logits = model.asr_model.transform_linear(final_frames)
    return torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float64)


def run_inference(
    *,
    model: Any,
    records: list[dict[str, Any]],
    references: list[dict[str, Any]],
    output_path: Path,
    batch_size: int,
    device: str,
    sample_rate: int,
    resume: bool,
) -> list[dict[str, Any]]:
    existing: list[dict[str, Any]] = []
    if resume and output_path.is_file():
        existing = read_jsonl(output_path)
        expected_prefix = [str(row["sample_id"]) for row in records[: len(existing)]]
        actual_prefix = [str(row["sample_id"]) for row in existing]
        if actual_prefix != expected_prefix:
            raise ValueError("resume predictions do not match the test-set prefix")
    elif output_path.exists():
        output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference_by_id = {str(row["sample_id"]): row for row in references}
    completed = len(existing)
    started_all = time.perf_counter()
    current_batch_size = max(1, int(batch_size))
    while completed < len(records):
        batch_records = records[completed : completed + current_batch_size]
        audios = []
        audio_hashes = []
        for row in batch_records:
            audio = read_wav_window_robust_mix(
                resolve_audio_path(row),
                float(row["audio_window_start_s"]),
                float(row["audio_window_end_s"]),
                target_rate=sample_rate,
            )
            expected_samples = int(round(30.0 * sample_rate))
            if audio.shape != (expected_samples,):
                raise ValueError(f"unexpected audio shape for {row['sample_id']}: {audio.shape}")
            digest = sha256_audio(audio)
            expected_digest = str(row.get("audio_window_sha256", ""))
            if expected_digest and digest != expected_digest:
                raise ValueError(f"audio-window SHA mismatch: {row['sample_id']}")
            audios.append(audio)
            audio_hashes.append(digest)
        batch_audio = np.stack(audios).astype(np.float32, copy=False)
        started = time.perf_counter()
        try:
            native_probabilities = infer_batch(model, batch_audio, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current_batch_size == 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            print(f"CUDA OOM; retrying with batch_size={current_batch_size}", flush=True)
            continue
        elapsed = time.perf_counter() - started
        native_index = {label: index for index, label in enumerate(ESP_NATIVE_LABELS)}
        with output_path.open("a", encoding="utf-8") as handle:
            for row, digest, native_probs in zip(batch_records, audio_hashes, native_probabilities):
                probs = {label: float(native_probs[native_index[label]]) for label in FIVE_LABELS}
                prediction = max(probs, key=probs.get)
                reference = reference_by_id[str(row["sample_id"])]
                payload = {
                    "sample_id": str(row["sample_id"]),
                    "conversation_id": str(row["conversation_id"]),
                    "prediction_boundary_in_conversation_s": float(
                        row["prediction_boundary_in_conversation_s"]
                    ),
                    "audio_window_start_s": float(row["audio_window_start_s"]),
                    "audio_window_end_s": float(row["audio_window_end_s"]),
                    "audio_window_sha256": digest,
                    "reference_label": str(reference["reference_label"]),
                    "prediction_label": prediction,
                    "probabilities": probs,
                    "source_kind": str(reference.get("source_kind", "")),
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                existing.append(payload)
        completed += len(batch_records)
        print(
            f"[espnet] {completed}/{len(records)} batch={len(batch_records)} "
            f"seconds={elapsed:.2f} per_sample={elapsed / len(batch_records):.3f}",
            flush=True,
        )
    print(f"[espnet] total_seconds={time.perf_counter() - started_all:.2f}", flush=True)
    return existing


def task_targets(references: list[dict[str, Any]], task: str) -> dict[str, int]:
    if task == "floor_taking":
        return build_floor_targets(references)
    result = {}
    for row in references:
        value = row.get("paper_binary_targets", {}).get(task)
        if value is not None:
            result[str(row["sample_id"])] = int(value)
    return result


def binary_metrics(
    predictions: list[dict[str, Any]],
    references: list[dict[str, Any]],
    balanced_ids: dict[str, set[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prediction_by_id = {str(row["sample_id"]): row for row in predictions}
    metrics: dict[str, Any] = {}
    binary_rows: list[dict[str, Any]] = []
    for task in TASKS:
        label_a, label_b = TASK_PAIRS[task]
        targets = task_targets(references, task)
        y_true = []
        y_pred = []
        y_score = []
        subset_mask = []
        for sample_id, target in targets.items():
            row = prediction_by_id[sample_id]
            probability_a = float(row["probabilities"][label_a])
            probability_b = float(row["probabilities"][label_b])
            denominator = probability_a + probability_b
            if denominator <= 0:
                probability_a = probability_b = 0.5
            else:
                probability_a /= denominator
                probability_b /= denominator
            predicted = int(probability_b > probability_a)
            balanced = sample_id in balanced_ids[task]
            y_true.append(target)
            y_pred.append(predicted)
            y_score.append(probability_b)
            subset_mask.append(balanced)
            binary_rows.append(
                {
                    "task": task,
                    "sample_id": sample_id,
                    "reference_answer": "B" if target else "A",
                    "prediction_answer": "B" if predicted else "A",
                    "prob_A": probability_a,
                    "prob_B": probability_b,
                    "paper_balanced_subset": balanced,
                }
            )
        y_true_arr = np.asarray(y_true, dtype=np.int64)
        y_pred_arr = np.asarray(y_pred, dtype=np.int64)
        y_score_arr = np.asarray(y_score, dtype=np.float64)
        mask = np.asarray(subset_mask, dtype=bool)
        metrics[task] = {
            "eligible_samples": int(len(y_true_arr)),
            "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
            "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro")),
            "roc_auc": float(roc_auc_score(y_true_arr, y_score_arr)),
            "paper_balanced_samples": int(mask.sum()),
            "paper_balanced_accuracy": float(accuracy_score(y_true_arr[mask], y_pred_arr[mask])),
        }
    metrics["overall"] = {
        "paper_balanced_accuracy_mean": float(
            np.mean([metrics[task]["paper_balanced_accuracy"] for task in TASKS])
        ),
        "eligible_accuracy_mean": float(np.mean([metrics[task]["accuracy"] for task in TASKS])),
    }
    return metrics, binary_rows


def named_ovr_roc_auc(
    predictions: list[dict[str, Any]], labels: tuple[str, ...] = FIVE_LABELS
) -> dict[str, float]:
    """Compute one-vs-rest AUCs from named probability fields.

    Using the probability names directly avoids silently pairing a non-alphabetic
    probability-column order with scikit-learn's alphabetic class order.
    """
    y_true = np.asarray([str(row["reference_label"]) for row in predictions])
    result: dict[str, float] = {}
    for label in labels:
        binary_target = (y_true == label).astype(np.int64)
        if np.unique(binary_target).size != 2:
            raise ValueError(f"ROC-AUC for {label} requires both positive and negative samples")
        scores = np.asarray(
            [float(row["probabilities"][label]) for row in predictions], dtype=np.float64
        )
        result[label] = float(roc_auc_score(binary_target, scores))
    return result


def evaluate(
    *,
    predictions: list[dict[str, Any]],
    references: list[dict[str, Any]],
    balanced_ids: dict[str, set[str]],
    output_dir: Path,
) -> dict[str, Any]:
    y_true = [str(row["reference_label"]) for row in predictions]
    y_pred = [str(row["prediction_label"]) for row in predictions]
    per_class_auc = named_ovr_roc_auc(predictions)
    five_way = {
        "samples": len(predictions),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(FIVE_LABELS), average="macro")),
        "per_class_ovr_roc_auc": per_class_auc,
        "macro_ovr_roc_auc": float(np.mean(list(per_class_auc.values()))),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=list(FIVE_LABELS),
            output_dict=True,
            zero_division=0,
        ),
        "prediction_counts": dict(Counter(y_pred)),
        "confusion_matrix_label_order": list(FIVE_LABELS),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(FIVE_LABELS)).tolist(),
    }
    binary, binary_rows = binary_metrics(predictions, references, balanced_ids)
    result = {"five_way": five_way, "binary": binary}
    write_json(output_dir / "metrics.json", result)
    with (output_dir / "binary_predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(binary_rows[0]))
        writer.writeheader()
        writer.writerows(binary_rows)
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reference\\prediction", *FIVE_LABELS])
        for label, row in zip(FIVE_LABELS, five_way["confusion_matrix"]):
            writer.writerow([label, *row])
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.selected_inputs)
    references = read_jsonl(args.references)
    dataset_audit = json.loads(args.dataset_audit.read_text(encoding="utf-8"))
    balanced_ids = load_balanced_ids(args.qwen_predictions)
    preflight_audit(
        records=records,
        references=references,
        dataset_audit=dataset_audit,
        checkpoint=args.checkpoint,
        config=args.config,
        balanced_ids=balanced_ids,
        output_path=args.output_dir / "input_audit.json",
    )
    prediction_path = args.output_dir / "test_predictions.jsonl"
    if args.score_only:
        if not prediction_path.is_file():
            raise FileNotFoundError(f"score-only predictions not found: {prediction_path}")
        predictions = read_jsonl(prediction_path)
        if [str(row["sample_id"]) for row in predictions] != [
            str(row["sample_id"]) for row in records
        ]:
            raise ValueError("score-only predictions do not exactly match selected inputs")
    else:
        model = load_model(args.espnet_source, args.config, args.checkpoint, args.device)
        predictions = run_inference(
            model=model,
            records=records,
            references=references,
            output_path=prediction_path,
            batch_size=args.batch_size,
            device=args.device,
            sample_rate=args.sample_rate,
            resume=args.resume,
        )
    result = evaluate(
        predictions=predictions,
        references=references,
        balanced_ids=balanced_ids,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
