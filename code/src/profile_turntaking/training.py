"""Training and checkpoint loading for the five-class MVP."""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .constants import LABELS
from .dataset import ManifestDataset
from .metrics import classification_metrics
from .model import ModelConfig, ProfileTurnModel
from .utils import set_seed, write_json


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    profile_dropout: float = 0.5
    seed: int = 13
    device: str = "auto"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainConfig":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in payload.items() if key in allowed})


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _class_weights(dataset: ManifestDataset, device: torch.device) -> torch.Tensor:
    from .constants import LABEL_TO_ID

    counts = Counter(LABEL_TO_ID[row["label"]] for row in dataset.rows)
    total = sum(counts.values())
    weights = [total / max(1, len(LABELS) * counts.get(index, 0)) for index in range(len(LABELS))]
    tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return tensor / tensor.mean()


@torch.no_grad()
def predict_loader(
    model: ProfileTurnModel, loader: DataLoader, device: torch.device
) -> tuple[list[int], list[int], list[str]]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    sample_ids: list[str] = []
    for batch in loader:
        logits = model(
            batch["audio_input"].to(device),
            batch["text_input"].to(device),
            batch["profile_ids"].to(device),
        )
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        targets.extend(batch["label"].cpu().tolist())
        sample_ids.extend(list(batch["sample_id"]))
    return targets, predictions, sample_ids


def train_model(
    manifest_path: str,
    checkpoint_path: str,
    *,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> dict[str, Any]:
    set_seed(train_config.seed)
    device = resolve_device(train_config.device)
    dataset_kwargs = {
        "audio_backend": model_config.audio_backend,
        "text_dimension": model_config.text_dimension,
        "profile_buckets": model_config.profile_buckets,
    }
    train_dataset = ManifestDataset(manifest_path, split="train", profile_mode="given", **dataset_kwargs)
    val_dataset = ManifestDataset(manifest_path, split="val", profile_mode="given", **dataset_kwargs)
    generator = torch.Generator().manual_seed(train_config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_config.batch_size, shuffle=False, num_workers=0
    )
    model = ProfileTurnModel(model_config).to(device)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(weight=_class_weights(train_dataset, device))
    history: list[dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, train_config.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            audio_input = batch["audio_input"].to(device)
            text_input = batch["text_input"].to(device)
            profile_ids = batch["profile_ids"].to(device)
            labels = batch["label"].to(device)
            if train_config.profile_dropout > 0:
                drop = torch.rand(profile_ids.shape[0], device=device) < train_config.profile_dropout
                profile_ids = profile_ids.clone()
                profile_ids[drop] = 0
            optimizer.zero_grad(set_to_none=True)
            logits = model(audio_input, text_input, profile_ids)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.item()) * labels.shape[0]
            seen += labels.shape[0]
        targets, predictions, _ = predict_loader(model, val_loader, device)
        report = classification_metrics(targets, predictions)
        epoch_row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, seen),
            "val_macro_f1": report["macro_f1"],
            "val_accuracy": report["accuracy"],
        }
        history.append(epoch_row)
        if report["macro_f1"] > best_macro_f1:
            best_macro_f1 = report["macro_f1"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.checkpoint_state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint state")
    checkpoint = {
        "format_version": 1,
        "model_config": model_config.to_dict(),
        "train_config": asdict(train_config),
        "state_dict": best_state,
        "labels": list(LABELS),
        "best_val_macro_f1": best_macro_f1,
        "history": history,
    }
    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    report = {
        "checkpoint": str(destination.resolve()),
        "device": str(device),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "best_val_macro_f1": best_macro_f1,
        "history": history,
    }
    write_json(destination.with_suffix(".train.json"), report)
    return report


def load_checkpoint(path: str, *, device: str = "auto") -> tuple[ProfileTurnModel, torch.device]:
    resolved_device = resolve_device(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = ProfileTurnModel(config)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    allowed_missing = {
        name for name in missing if name.startswith("audio_encoder.encoder.") and config.freeze_audio_encoder
    }
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.to(resolved_device).eval()
    return model, resolved_device
