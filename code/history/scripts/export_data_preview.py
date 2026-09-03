"""Export human-readable SBCSAE, PAChat, and training-output examples."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np

from profile_turntaking.audio import read_wav_window
from profile_turntaking.constants import ID_TO_LABEL


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _find_jsonl(path: Path, predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if predicate(row):
                    return row
    raise ValueError(f"No matching row in {path}")


def _write_pcm_wav(path: Path, samples: np.ndarray, sample_rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def export_sbcsae(
    manifest_path: Path,
    output_dir: Path,
    *,
    sample_id: str | None,
    split: str,
    label: str,
) -> dict[str, Any]:
    if sample_id:
        row = _find_jsonl(manifest_path, lambda item: item["sample_id"] == sample_id)
    else:
        row = _find_jsonl(
            manifest_path,
            lambda item: item["split"] == split and item["label"] == label,
        )
    waveform = read_wav_window(
        row["audio_path"],
        float(row["window_start_s"]),
        float(row["window_end_s"]),
    )
    destination = output_dir / "sbcsae"
    audio_path = destination / "audio_input_30s.wav"
    _write_pcm_wav(audio_path, waveform)
    preview = {
        "what_this_row_means": (
            "At prediction_time_s, the model receives this 30-second mono audio window, "
            "the causal transcript prefix, and the selected profile condition. The target is "
            "the weak five-class event label for the next horizon_ms."
        ),
        "input": {
            "sample_id": row["sample_id"],
            "conversation_id": row["conversation_id"],
            "split": row["split"],
            "prediction_time_s": row["prediction_time_s"],
            "window_start_s": row["window_start_s"],
            "window_end_s": row["window_end_s"],
            "exported_audio": str(audio_path.resolve()),
            "source_audio": row["audio_path"],
            "transcript_prefix": row["transcript_prefix"],
            "profile": row["profile"],
            "profile_provenance": row.get("profile_provenance"),
        },
        "target_output": {
            "label": row["label"],
            "horizon_ms": row["horizon_ms"],
            "label_source": row.get("label_source"),
            "gold_label": row.get("gold_label", False),
        },
    }
    _write_json(destination / "sample.json", preview)
    return preview


def export_pachat(
    pachat_dir: Path,
    output_dir: Path,
    *,
    turn_id: str | None,
) -> dict[str, Any]:
    turn = _find_jsonl(
        pachat_dir / "turns.jsonl",
        (lambda item: item["turn_id"] == turn_id) if turn_id else (lambda item: True),
    )
    profile = _find_jsonl(
        pachat_dir / "profiles.jsonl",
        lambda item: item["profile_id"] == turn["profile_id"],
    )
    case = _find_jsonl(
        pachat_dir / "cases.jsonl",
        lambda item: item["case_id"] == turn["case_id"],
    )
    destination = output_dir / "pachat"
    destination.mkdir(parents=True, exist_ok=True)
    audio_path = destination / "turn_audio.wav"
    shutil.copy2(turn["audio_path"], audio_path)
    preview = {
        "case": case,
        "profile": profile,
        "turn": {**turn, "exported_audio": str(audio_path.resolve())},
        "turntaking_output": None,
        "why_no_turntaking_output": (
            "The official demo stores each turn as an isolated WAV. It has no continuous gap "
            "or overlap timeline and is therefore not eligible for 40 ms turn-taking labels."
        ),
    }
    _write_json(destination / "demo.json", preview)
    return preview


def export_training_output(smoke_dir: Path, output_dir: Path) -> dict[str, Any]:
    predictions = _read_json(smoke_dir / "evaluation" / "predictions.json")
    first_mode = next(iter(predictions))
    first_prediction = predictions[first_mode][0]
    sample_id = first_prediction["sample_id"]
    sample = _find_jsonl(
        smoke_dir / "samples.jsonl",
        lambda item: item["sample_id"] == sample_id,
    )
    by_mode: dict[str, Any] = {}
    for mode, rows in predictions.items():
        row = next(item for item in rows if item["sample_id"] == sample_id)
        by_mode[mode] = {
            "target_id": row["target"],
            "target_label": ID_TO_LABEL[row["target"]],
            "prediction_id": row["prediction"],
            "prediction_label": ID_TO_LABEL[row["prediction"]],
        }
    with (smoke_dir / "evaluation" / "profile_comparison.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        comparison = list(csv.DictReader(handle))
    destination = output_dir / "training_output"
    destination.mkdir(parents=True, exist_ok=True)
    for relative_path in (
        Path("model.train.json"),
        Path("evaluation/metrics.json"),
        Path("evaluation/predictions.json"),
        Path("evaluation/profile_comparison.csv"),
    ):
        source = smoke_dir / relative_path
        target = destination / relative_path.name
        shutil.copy2(source, target)
    preview = {
        "checkpoint": {
            "path": str((smoke_dir / "model.pt").resolve()),
            "bytes": (smoke_dir / "model.pt").stat().st_size,
            "note": "Functional smoke checkpoint only; not a full SBCSAE research model.",
        },
        "one_test_sample": sample,
        "same_sample_predictions": by_mode,
        "profile_comparison": comparison,
        "output_files": {
            "training_history": "model.train.json",
            "aggregate_metrics": "metrics.json",
            "per_sample_argmax_predictions": "predictions.json",
            "paired_profile_table": "profile_comparison.csv",
        },
    }
    _write_json(destination / "example.json", preview)
    return preview


def write_readme(output_dir: Path, sbcsae: dict[str, Any], pachat: dict[str, Any]) -> None:
    content = f"""# Local data preview

This directory is generated and Git-ignored.

## SBCSAE

- Sample: `{sbcsae['input']['sample_id']}`
- Target weak label: `{sbcsae['target_output']['label']}`
- Listen: `sbcsae/audio_input_30s.wav`
- Full input/profile/target: `sbcsae/sample.json`

## PAChat official demo

- Turn: `{pachat['turn']['turn_id']}`
- Speaker: `{pachat['turn']['speaker']}`
- Listen: `pachat/turn_audio.wav`
- Case/profile/turn: `pachat/demo.json`
- This isolated WAV has no continuous turn-taking target.

## Training output

- `training_output/example.json`: one test sample and its hidden/given/shuffled predictions
- `training_output/model.train.json`: epoch-level training history
- `training_output/metrics.json`: aggregate metrics and confusion matrices
- `training_output/predictions.json`: per-sample target/prediction IDs
- `training_output/profile_comparison.csv`: paired headline table
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sbcsae-manifest",
        default="data/processed/sbcsae_mvp/manifest.jsonl",
    )
    parser.add_argument("--sample-id")
    parser.add_argument("--split", default="test")
    parser.add_argument("--label", default="I")
    parser.add_argument("--pachat-dir", default="data/processed/pachat_demo")
    parser.add_argument("--pachat-turn-id")
    parser.add_argument("--smoke-dir", default="artifacts/paired-profile-smoke-final")
    parser.add_argument("--output-dir", default="artifacts/data-preview")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    sbcsae = export_sbcsae(
        Path(args.sbcsae_manifest),
        output_dir,
        sample_id=args.sample_id,
        split=args.split,
        label=args.label,
    )
    pachat = export_pachat(
        Path(args.pachat_dir),
        output_dir,
        turn_id=args.pachat_turn_id,
    )
    training = export_training_output(Path(args.smoke_dir), output_dir)
    write_readme(output_dir, sbcsae, pachat)
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir.resolve()),
                "sbcsae_sample_id": sbcsae["input"]["sample_id"],
                "pachat_turn_id": pachat["turn"]["turn_id"],
                "training_sample_id": training["one_test_sample"]["sample_id"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
