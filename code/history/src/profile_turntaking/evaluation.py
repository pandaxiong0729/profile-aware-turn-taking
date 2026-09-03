"""Paired profile hidden/given/shuffled evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from .dataset import ManifestDataset
from .metrics import classification_metrics, write_metrics_csv
from .training import load_checkpoint, predict_loader
from .utils import write_json


def evaluate_checkpoint(
    manifest_path: str,
    checkpoint_path: str,
    output_dir: str,
    *,
    split: str = "test",
    batch_size: int = 64,
    device: str = "auto",
) -> dict[str, dict[str, Any]]:
    model, resolved_device = load_checkpoint(checkpoint_path, device=device)
    config = model.config
    reports: dict[str, dict[str, Any]] = {}
    prediction_rows: dict[str, list[dict[str, Any]]] = {}
    for mode in ("hidden", "given", "shuffled"):
        dataset = ManifestDataset(
            manifest_path,
            split=split,
            audio_backend=config.audio_backend,
            text_dimension=config.text_dimension,
            profile_buckets=config.profile_buckets,
            profile_mode=mode,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        targets, predictions, sample_ids = predict_loader(model, loader, resolved_device)
        reports[mode] = classification_metrics(targets, predictions)
        prediction_rows[mode] = [
            {"sample_id": sample_id, "target": target, "prediction": prediction}
            for sample_id, target, prediction in zip(sample_ids, targets, predictions)
        ]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "metrics.json", reports)
    write_json(destination / "predictions.json", prediction_rows)
    write_metrics_csv(destination / "profile_comparison.csv", reports)
    return reports
