"""Shared Qwen profile-embedding adapter with four binary A/B heads.

This is the multitask version of the earlier B route:

    frozen Qwen context vector (causal audio + transcript)
    + frozen Qwen profile embedding
    -> one shared gated adapter
    -> four binary heads:
       silence, listener_onset, brief_response, yield

The split is the same cached Qwen split used by the previous prompt and adapter
experiments.  In hidden/given/shuffled comparisons, only profile content changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
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
)
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset


LABELS = ["C", "BC", "T", "I", "NA"]
PROFILE_MODES = ["hidden", "given", "shuffled"]
TASK_ORDER = ["silence", "listener_onset", "brief_response", "yield"]
IGNORE_INDEX = -100


BINARY_TASKS: dict[str, dict[str, Any]] = {
    "silence": {
        "description": "A=NA/silence, B=somebody speaks",
        "labels": set(LABELS),
        "a_labels": {"NA"},
        "b_labels": {"C", "BC", "T", "I"},
    },
    "listener_onset": {
        "description": "A=current speaker continues/no listener onset, B=other participant responds",
        "labels": {"C", "BC", "T", "I"},
        "a_labels": {"C"},
        "b_labels": {"BC", "T", "I"},
    },
    "brief_response": {
        "description": "A=brief backchannel, B=substantive floor-taking response",
        "labels": {"BC", "T", "I"},
        "a_labels": {"BC"},
        "b_labels": {"T", "I"},
    },
    "yield": {
        "description": "A=natural turn change, B=interruption",
        "labels": {"T", "I"},
        "a_labels": {"T"},
        "b_labels": {"I"},
    },
}

HIERARCHY_TASK_ORDER = ["silence", "listener_onset", "brief_response", "yield"]
HIERARCHY_BINARY_TASKS = BINARY_TASKS
PAPER_TASK_ORDER = ["turn_change", "backchannel", "interruption", "floor_taking"]
PAPER_BINARY_TASKS: dict[str, dict[str, Any]] = {
    "turn_change": {
        "description": "A=no turn change/current speaker continues, B=other participant takes the turn",
        "labels": {"C", "T"},
        "a_labels": {"C"},
        "b_labels": {"T"},
        "source": "paper_targets",
    },
    "backchannel": {
        "description": "A=no backchannel at the decision point, B=listener produces a brief backchannel",
        "labels": {"C", "BC"},
        "a_labels": {"C"},
        "b_labels": {"BC"},
        "source": "paper_targets",
    },
    "interruption": {
        "description": "A=no interruption at the decision point, B=other participant interrupts",
        "labels": {"C", "I"},
        "a_labels": {"C"},
        "b_labels": {"I"},
        "source": "paper_targets",
    },
    "floor_taking": {
        "description": "A=overlap attempt fails/first speaker keeps floor, B=overlapping speaker takes the floor",
        "labels": {"I"},
        "a_labels": set(),
        "b_labels": set(),
        "source": "paper_targets",
    },
}


@dataclass
class TrainConfig:
    hidden_dim: int = 256
    dropout: float = 0.15
    profile_dropout: float = 0.25
    epochs: int = 100
    patience: int = 14
    batch_size: int = 64
    lr: float = 8e-4
    weight_decay: float = 1e-4
    seeds: tuple[int, ...] = (3, 7, 13, 29, 37, 71, 101)
    device: str = "cpu"
    hidden_ce_weight: float = 0.0
    control_ce_weight: float = 0.0
    hidden_margin_weight: float = 0.0
    control_margin_weight: float = 0.0
    margin: float = 0.10
    balanced_margin: bool = False
    selection_delta_weight: float = 0.0
    selection_min_delta_weight: float = 0.0
    fusion: str = "gate"
    task_specific_branches: bool = False
    use_last_epoch: bool = False
    audio_only: bool = False
    use_profile: bool = True


class SharedBinaryMultiHeadAdapter(nn.Module):
    def __init__(
        self,
        context_dim: int,
        profile_dim: int,
        hidden_dim: int,
        dropout: float,
        fusion: str = "gate",
        audio_layer_count: int = 0,
        audio_layer_dim: int = 0,
        task_specific_branches: bool = False,
        audio_only: bool = False,
        use_profile: bool = True,
    ) -> None:
        super().__init__()
        self.audio_layer_count = int(audio_layer_count)
        self.audio_layer_dim = int(audio_layer_dim)
        self.task_specific_branches = bool(task_specific_branches)
        self.audio_only = bool(audio_only)
        self.use_profile = bool(use_profile)
        if self.audio_only and self.audio_layer_count <= 0:
            raise ValueError("audio_only requires cached audio layers")
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        if self.audio_layer_count > 0:
            if self.audio_layer_dim <= 0:
                raise ValueError("audio_layer_dim must be positive when audio layers are enabled")
            if self.task_specific_branches:
                self.audio_layer_logits = nn.Parameter(
                    torch.zeros(len(TASK_ORDER), self.audio_layer_count)
                )
                self.audio_encoders = nn.ModuleDict(
                    {
                        task: nn.Sequential(
                            nn.Linear(self.audio_layer_dim, hidden_dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.LayerNorm(hidden_dim),
                        )
                        for task in TASK_ORDER
                    }
                )
                self.context_merges = nn.ModuleDict(
                    {
                        task: nn.Sequential(
                            nn.Linear(hidden_dim * 2, hidden_dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.LayerNorm(hidden_dim),
                        )
                        for task in TASK_ORDER
                    }
                )
                self.audio_encoder = None
                self.context_merge = None
            else:
                self.audio_layer_logits = nn.Parameter(torch.zeros(self.audio_layer_count))
                self.audio_encoder = nn.Sequential(
                    nn.Linear(self.audio_layer_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(hidden_dim),
                )
                self.context_merge = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(hidden_dim),
                )
                self.audio_encoders = None
                self.context_merges = None
        else:
            self.register_parameter("audio_layer_logits", None)
            self.audio_encoder = None
            self.context_merge = None
            self.audio_encoders = None
            self.context_merges = None
        self.profile_encoder = nn.Sequential(
            nn.Linear(profile_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, elementwise_affine=False),
        )
        if self.task_specific_branches:
            self.gates = nn.ModuleDict(
                {task: nn.Linear(hidden_dim * 2, hidden_dim) for task in TASK_ORDER}
            )
            self.concat_deltas = nn.ModuleDict(
                {
                    task: nn.Sequential(
                        nn.Linear(hidden_dim * 2, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                    for task in TASK_ORDER
                }
            )
            self.films = nn.ModuleDict(
                {task: nn.Linear(hidden_dim * 2, hidden_dim * 2) for task in TASK_ORDER}
            )
            self.out_norms = nn.ModuleDict(
                {task: nn.LayerNorm(hidden_dim) for task in TASK_ORDER}
            )
            self.gate = None
            self.concat_delta = None
            self.film = None
            self.out_norm = None
        else:
            self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
            self.concat_delta = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.film = nn.Linear(hidden_dim * 2, hidden_dim * 2)
            self.out_norm = nn.LayerNorm(hidden_dim)
            self.gates = None
            self.concat_deltas = None
            self.films = None
            self.out_norms = None
        self.heads = nn.ModuleDict({task: nn.Linear(hidden_dim, 2) for task in TASK_ORDER})
        if fusion not in {"gate", "concat", "film"}:
            raise ValueError(f"Unknown fusion: {fusion}")
        self.fusion = fusion

    def encode_shared(
        self,
        context: torch.Tensor,
        profile: torch.Tensor,
        audio_layers: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        shared = self.context_encoder(context)
        task_shared: dict[str, torch.Tensor] | None = None
        if self.audio_layer_count > 0:
            if audio_layers is None:
                raise ValueError("audio_layers are required for the layer-weighted adapter")
            if audio_layers.ndim != 3 or audio_layers.shape[1:] != (
                self.audio_layer_count,
                self.audio_layer_dim,
            ):
                raise ValueError(
                    f"Expected audio layers [batch,{self.audio_layer_count},{self.audio_layer_dim}], "
                    f"got {audio_layers.shape}"
                )
            if self.task_specific_branches:
                weights = torch.softmax(self.audio_layer_logits, dim=-1)
                task_shared = {}
                assert self.audio_encoders is not None and self.context_merges is not None
                for task_idx, task in enumerate(TASK_ORDER):
                    weighted_audio = torch.einsum(
                        "bld,l->bd", audio_layers, weights[task_idx]
                    )
                    audio_state = self.audio_encoders[task](weighted_audio)
                    task_shared[task] = (
                        audio_state
                        if self.audio_only
                        else self.context_merges[task](torch.cat([shared, audio_state], dim=-1))
                    )
            else:
                weights = torch.softmax(self.audio_layer_logits, dim=0)
                weighted_audio = torch.einsum("bld,l->bd", audio_layers, weights)
                assert self.audio_encoder is not None and self.context_merge is not None
                audio_state = self.audio_encoder(weighted_audio)
                shared = (
                    audio_state
                    if self.audio_only
                    else self.context_merge(torch.cat([shared, audio_state], dim=-1))
                )
        if not self.use_profile:
            if self.task_specific_branches:
                if task_shared is None:
                    task_shared = {task: shared for task in TASK_ORDER}
                return task_shared
            return shared
        profile_state = self.profile_encoder(profile)
        present = (profile.abs().sum(dim=-1, keepdim=True) > 0).to(profile_state.dtype)
        profile_state = profile_state * present
        if self.task_specific_branches:
            if task_shared is None:
                task_shared = {task: shared for task in TASK_ORDER}
            assert self.gates is not None
            assert self.concat_deltas is not None
            assert self.films is not None
            assert self.out_norms is not None
            fused_by_task: dict[str, torch.Tensor] = {}
            for task in TASK_ORDER:
                task_context = task_shared[task]
                interaction = torch.cat([task_context, profile_state], dim=-1)
                if self.fusion == "gate":
                    gate = torch.sigmoid(self.gates[task](interaction))
                    fused = task_context + gate * profile_state
                elif self.fusion == "concat":
                    fused = task_context + present * self.concat_deltas[task](interaction)
                else:
                    gamma, beta = self.films[task](interaction).chunk(2, dim=-1)
                    gamma = 0.25 * torch.tanh(gamma) * present
                    beta = beta * present
                    fused = task_context * (1.0 + gamma) + beta
                fused_by_task[task] = self.out_norms[task](fused)
            return fused_by_task

        interaction = torch.cat([shared, profile_state], dim=-1)
        if self.fusion == "gate":
            assert self.gate is not None and self.out_norm is not None
            gate = torch.sigmoid(self.gate(interaction))
            fused = shared + gate * profile_state
        elif self.fusion == "concat":
            assert self.concat_delta is not None and self.out_norm is not None
            fused = shared + present * self.concat_delta(interaction)
        else:
            assert self.film is not None and self.out_norm is not None
            gamma, beta = self.film(interaction).chunk(2, dim=-1)
            gamma = 0.25 * torch.tanh(gamma) * present
            beta = beta * present
            fused = shared * (1.0 + gamma) + beta
        return self.out_norm(fused)

    def forward(
        self,
        context: torch.Tensor,
        profile: torch.Tensor,
        audio_layers: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        fused = self.encode_shared(context, profile, audio_layers)
        if self.task_specific_branches:
            assert isinstance(fused, dict)
            return {task: head(fused[task]) for task, head in self.heads.items()}
        assert isinstance(fused, torch.Tensor)
        return {task: head(fused) for task, head in self.heads.items()}

    def learned_audio_layer_weights(self) -> list[float] | dict[str, list[float]] | None:
        if self.audio_layer_logits is None:
            return None
        if self.task_specific_branches:
            weights = torch.softmax(self.audio_layer_logits.detach(), dim=-1).cpu().tolist()
            return {task: weights[idx] for idx, task in enumerate(TASK_ORDER)}
        return torch.softmax(self.audio_layer_logits.detach(), dim=0).cpu().tolist()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def label_to_binary(label_id: int, task: str) -> int | None:
    label = LABELS[int(label_id)]
    spec = BINARY_TASKS[task]
    if label not in spec["labels"]:
        return None
    if label in spec["a_labels"]:
        return 0
    if label in spec["b_labels"]:
        return 1
    raise AssertionError(f"Unmapped label {label} for task {task}")


def build_multitask_targets(
    labels: np.ndarray, cached_targets: np.ndarray | None = None
) -> np.ndarray:
    if cached_targets is not None:
        targets = np.asarray(cached_targets, dtype=np.int64)
        if targets.shape != (len(labels), len(TASK_ORDER)):
            raise ValueError(
                f"cached task targets shape {targets.shape} does not match "
                f"({len(labels)}, {len(TASK_ORDER)})"
            )
        return targets
    targets = np.full((len(labels), len(TASK_ORDER)), IGNORE_INDEX, dtype=np.int64)
    for row_idx, label_id in enumerate(labels):
        for task_idx, task in enumerate(TASK_ORDER):
            binary = label_to_binary(int(label_id), task)
            if binary is not None:
                targets[row_idx, task_idx] = binary
    return targets


def build_paper_targets(labels: np.ndarray, cached_targets: np.ndarray) -> np.ndarray:
    """Rebuild paper-aligned C-vs-event tasks; reuse cache only for floor outcome."""

    cached = np.asarray(cached_targets, dtype=np.int64)
    if cached.shape != (len(labels), len(PAPER_TASK_ORDER)):
        raise ValueError(
            f"cached paper targets shape {cached.shape} does not match "
            f"({len(labels)}, {len(PAPER_TASK_ORDER)})"
        )
    targets = np.full_like(cached, IGNORE_INDEX)
    label_names = np.asarray([LABELS[int(label)] for label in labels])
    for task_idx, (negative, positive) in enumerate(
        (("C", "T"), ("C", "BC"), ("C", "I"))
    ):
        targets[label_names == negative, task_idx] = 0
        targets[label_names == positive, task_idx] = 1
    targets[:, PAPER_TASK_ORDER.index("floor_taking")] = cached[
        :, PAPER_TASK_ORDER.index("floor_taking")
    ]
    return targets


def task_targets(multitask_targets: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray]:
    task_idx = TASK_ORDER.index(task)
    targets = multitask_targets[:, task_idx]
    indices = np.flatnonzero(targets != IGNORE_INDEX).astype(np.int64)
    return indices, targets[indices].astype(np.int64)


def build_balanced_eval_indices(
    multitask_targets: np.ndarray,
    sample_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Create fixed 50/50 A/B evaluation subsets without changing training data."""

    result: dict[str, np.ndarray] = {}
    ids = sample_ids.astype(str)
    for task_idx, task in enumerate(TASK_ORDER):
        targets = multitask_targets[:, task_idx]
        by_class = [np.flatnonzero(targets == label).astype(np.int64) for label in (0, 1)]
        count = min(len(by_class[0]), len(by_class[1]))
        if count == 0:
            raise ValueError(f"Task {task} has no balanced A/B evaluation subset")
        selected: list[np.ndarray] = []
        for candidates in by_class:
            ranked = sorted(
                candidates.tolist(),
                key=lambda index: hashlib.sha256(
                    f"paper-balanced-v1\n{task}\n{ids[index]}".encode("utf-8")
                ).hexdigest(),
            )
            selected.append(np.asarray(ranked[:count], dtype=np.int64))
        result[task] = np.sort(np.concatenate(selected))
    return result


def standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    # Audio-layer caches are float16.  NumPy otherwise accumulates their
    # reductions in float16, which can overflow before the result is cast.
    mean = train.mean(axis=0, keepdims=True, dtype=np.float32)
    std = np.maximum(train.std(axis=0, keepdims=True, dtype=np.float32), 1e-5)
    if not bool(np.isfinite(mean).all() and np.isfinite(std).all()):
        raise ValueError("Non-finite training statistics during standardization")
    return tuple(((x.astype(np.float32) - mean) / std) for x in (train, *others))


def make_contrastive_profiles(
    given_profiles: np.ndarray,
    conversation_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick a deliberately different wrong profile for each sample.

    This keeps the same sample/audio/transcript/label and changes only the
    control profile content.  The selected profile comes from a different
    conversation when possible and is farthest by cosine similarity.
    """
    profiles = given_profiles.astype(np.float32)
    norms = np.maximum(np.linalg.norm(profiles, axis=1, keepdims=True), 1e-8)
    normalized = profiles / norms
    sims = normalized @ normalized.T
    chosen = np.zeros(len(profiles), dtype=np.int64)
    for index in range(len(profiles)):
        scores = sims[index].copy()
        scores[index] = np.inf
        different_conversation = conversation_ids != conversation_ids[index]
        if bool(np.any(different_conversation)):
            scores[~different_conversation] = np.inf
        chosen[index] = int(np.argmin(scores))
    return profiles[chosen].copy(), chosen


def binary_metrics(targets: np.ndarray, predictions: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    report = classification_report(
        targets,
        predictions,
        labels=[0, 1],
        target_names=["A", "B"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "per_class": report,
        "confusion_matrix": confusion_matrix(targets, predictions, labels=[0, 1]).astype(int).tolist(),
        "target_distribution": {"A": int(np.sum(targets == 0)), "B": int(np.sum(targets == 1))},
        "prediction_distribution": {"A": int(np.sum(predictions == 0)), "B": int(np.sum(predictions == 1))},
        "mean_prob_A": float(probs[:, 0].mean()) if len(probs) else 0.0,
        "mean_prob_B": float(probs[:, 1].mean()) if len(probs) else 0.0,
    }


def serializable_binary_tasks() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for name, spec in BINARY_TASKS.items():
        rows[name] = {
            "description": spec["description"],
            "labels": sorted(spec["labels"], key=LABELS.index),
            "a_labels": sorted(spec["a_labels"], key=LABELS.index),
            "b_labels": sorted(spec["b_labels"], key=LABELS.index),
        }
    return rows


def class_weights_for_targets(targets: np.ndarray, device: torch.device) -> torch.Tensor:
    valid = targets[targets != IGNORE_INDEX]
    counts = Counter(valid.tolist())
    weights = torch.tensor(
        [len(valid) / max(1, 2 * counts.get(index, 0)) for index in range(2)],
        dtype=torch.float32,
        device=device,
    )
    return weights / weights.mean()


def multitask_ce_loss(
    logits_by_task: dict[str, torch.Tensor],
    targets: torch.Tensor,
    criteria: dict[str, nn.CrossEntropyLoss],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for task_idx, task in enumerate(TASK_ORDER):
        task_targets_batch = targets[:, task_idx]
        mask = task_targets_batch != IGNORE_INDEX
        if bool(mask.any()):
            losses.append(criteria[task](logits_by_task[task][mask], task_targets_batch[mask]))
    if not losses:
        raise RuntimeError("Batch has no valid binary targets")
    return torch.stack(losses).mean()


def multitask_margin_loss(
    given_logits_by_task: dict[str, torch.Tensor],
    control_logits_by_task: dict[str, torch.Tensor],
    targets: torch.Tensor,
    *,
    margin: float,
    class_weights: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for task_idx, task in enumerate(TASK_ORDER):
        task_targets_batch = targets[:, task_idx]
        mask = task_targets_batch != IGNORE_INDEX
        if bool(mask.any()):
            labels = task_targets_batch[mask].unsqueeze(1)
            given_logp = torch.log_softmax(given_logits_by_task[task][mask], dim=-1).gather(1, labels).squeeze(1)
            control_logp = torch.log_softmax(control_logits_by_task[task][mask], dim=-1).gather(1, labels).squeeze(1)
            hinge = torch.relu(
                torch.as_tensor(margin, device=given_logp.device) - (given_logp - control_logp)
            )
            if class_weights is None:
                losses.append(hinge.mean())
            else:
                weights = class_weights[task][task_targets_batch[mask]]
                losses.append((hinge * weights).sum() / weights.sum().clamp_min(1e-8))
    if not losses:
        raise RuntimeError("Batch has no valid binary targets")
    return torch.stack(losses).mean()


@torch.no_grad()
def predict_task(
    model: SharedBinaryMultiHeadAdapter,
    context: np.ndarray,
    profile: np.ndarray,
    targets: np.ndarray,
    *,
    task: str,
    device: torch.device,
    indices_override: np.ndarray | None = None,
    audio_layers: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    indices = (
        np.asarray(indices_override, dtype=np.int64)
        if indices_override is not None
        else np.flatnonzero(targets != IGNORE_INDEX).astype(np.int64)
    )
    if np.any(targets[indices] == IGNORE_INDEX):
        raise ValueError(f"Balanced evaluation indices include ignored targets for {task}")
    selected_targets = targets[indices].astype(np.int64)
    tensors = [
        torch.from_numpy(context[indices].astype(np.float32)),
        torch.from_numpy(profile[indices].astype(np.float32)),
    ]
    if audio_layers is not None:
        tensors.append(torch.from_numpy(audio_layers[indices].astype(np.float32)))
    loader = DataLoader(TensorDataset(*tensors), batch_size=256, shuffle=False)
    chunks: list[np.ndarray] = []
    for batch in loader:
        batch_context, batch_profile = batch[:2]
        batch_audio_layers = batch[2].to(device) if len(batch) == 3 else None
        logits = model(
            batch_context.to(device),
            batch_profile.to(device),
            batch_audio_layers,
        )[task]
        chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probs = np.concatenate(chunks, axis=0)
    preds = probs.argmax(axis=1).astype(np.int64)
    return binary_metrics(selected_targets, preds, probs), preds, probs


def evaluate_all(
    model: SharedBinaryMultiHeadAdapter,
    context: np.ndarray,
    profiles: dict[str, np.ndarray],
    multitask_targets: np.ndarray,
    *,
    device: torch.device,
    balanced_indices: dict[str, np.ndarray] | None = None,
    audio_layers: np.ndarray | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    reports: dict[str, dict[str, dict[str, Any]]] = {}
    for task in TASK_ORDER:
        task_idx = TASK_ORDER.index(task)
        reports[task] = {}
        for mode in PROFILE_MODES:
            report, _, _ = predict_task(
                model,
                context,
                profiles[mode],
                multitask_targets[:, task_idx],
                task=task,
                device=device,
                audio_layers=audio_layers,
            )
            if balanced_indices is not None:
                balanced_report, _, _ = predict_task(
                    model,
                    context,
                    profiles[mode],
                    multitask_targets[:, task_idx],
                    task=task,
                    device=device,
                    indices_override=balanced_indices[task],
                    audio_layers=audio_layers,
                )
                report["paper_balanced_subset"] = balanced_report
            reports[task][mode] = report
    return reports


def train_one(
    *,
    train_context: np.ndarray,
    train_profile: np.ndarray,
    train_control_profile: np.ndarray,
    train_targets: np.ndarray,
    val_context: np.ndarray,
    val_profiles: dict[str, np.ndarray],
    val_targets: np.ndarray,
    val_balanced_indices: dict[str, np.ndarray] | None,
    config: TrainConfig,
    seed: int,
    train_audio_layers: np.ndarray | None = None,
    val_audio_layers: np.ndarray | None = None,
) -> tuple[SharedBinaryMultiHeadAdapter, list[dict[str, Any]]]:
    set_seed(seed)
    device = torch.device(config.device)
    model = SharedBinaryMultiHeadAdapter(
        context_dim=int(train_context.shape[1]),
        profile_dim=int(train_profile.shape[1]),
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        fusion=config.fusion,
        audio_layer_count=(int(train_audio_layers.shape[1]) if train_audio_layers is not None else 0),
        audio_layer_dim=(int(train_audio_layers.shape[2]) if train_audio_layers is not None else 0),
        task_specific_branches=config.task_specific_branches,
        audio_only=config.audio_only,
        use_profile=config.use_profile,
    ).to(device)
    dataset_tensors = [
        torch.from_numpy(train_context.astype(np.float32)),
        torch.from_numpy(train_profile.astype(np.float32)),
        torch.from_numpy(train_control_profile.astype(np.float32)),
        torch.from_numpy(train_targets.astype(np.int64)),
    ]
    if train_audio_layers is not None:
        dataset_tensors.append(torch.from_numpy(train_audio_layers.astype(np.float32)))
    dataset = TensorDataset(*dataset_tensors)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
    criteria = {
        task: nn.CrossEntropyLoss(weight=class_weights_for_targets(train_targets[:, idx], device))
        for idx, task in enumerate(TASK_ORDER)
    }
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for batch in loader:
            context, profile, control_profile, targets = batch[:4]
            audio_layers = batch[4].to(device) if len(batch) == 5 else None
            context = context.to(device)
            profile = profile.to(device)
            control_profile = control_profile.to(device)
            targets = targets.to(device)
            if config.profile_dropout > 0:
                drop = torch.rand(profile.shape[0], device=device) < config.profile_dropout
                profile = profile.clone()
                profile[drop] = 0.0
            given_logits = model(context, profile, audio_layers)
            loss_terms = [multitask_ce_loss(given_logits, targets, criteria)]
            if config.hidden_ce_weight > 0 or config.hidden_margin_weight > 0:
                hidden_logits = model(context, torch.zeros_like(profile), audio_layers)
                if config.hidden_ce_weight > 0:
                    loss_terms.append(config.hidden_ce_weight * multitask_ce_loss(hidden_logits, targets, criteria))
                if config.hidden_margin_weight > 0:
                    loss_terms.append(
                        config.hidden_margin_weight
                        * multitask_margin_loss(
                            given_logits,
                            hidden_logits,
                            targets,
                            margin=config.margin,
                            class_weights=(
                                {task: criteria[task].weight for task in TASK_ORDER}
                                if config.balanced_margin
                                else None
                            ),
                        )
                    )
            if config.control_ce_weight > 0 or config.control_margin_weight > 0:
                control_logits = model(context, control_profile, audio_layers)
                if config.control_ce_weight > 0:
                    loss_terms.append(config.control_ce_weight * multitask_ce_loss(control_logits, targets, criteria))
                if config.control_margin_weight > 0:
                    loss_terms.append(
                        config.control_margin_weight
                        * multitask_margin_loss(
                            given_logits,
                            control_logits,
                            targets,
                            margin=config.margin,
                            class_weights=(
                                {task: criteria[task].weight for task in TASK_ORDER}
                                if config.balanced_margin
                                else None
                            ),
                        )
                    )
            loss = torch.stack(loss_terms).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(context)
            seen += len(context)

        val_reports = evaluate_all(
            model,
            val_context,
            val_profiles,
            val_targets,
            device=device,
            balanced_indices=val_balanced_indices,
            audio_layers=val_audio_layers,
        )
        hidden_scores = [val_reports[task]["hidden"]["balanced_accuracy"] for task in TASK_ORDER]
        given_scores = [val_reports[task]["given"]["balanced_accuracy"] for task in TASK_ORDER]
        shuffled_scores = [val_reports[task]["shuffled"]["balanced_accuracy"] for task in TASK_ORDER]
        score_key = "paper_balanced_subset" if val_balanced_indices is not None else None
        hidden_acc = [
            val_reports[task]["hidden"][score_key]["accuracy"]
            if score_key
            else val_reports[task]["hidden"]["accuracy"]
            for task in TASK_ORDER
        ]
        given_acc = [
            val_reports[task]["given"][score_key]["accuracy"]
            if score_key
            else val_reports[task]["given"]["accuracy"]
            for task in TASK_ORDER
        ]
        shuffled_acc = [
            val_reports[task]["shuffled"][score_key]["accuracy"]
            if score_key
            else val_reports[task]["shuffled"]["accuracy"]
            for task in TASK_ORDER
        ]
        if config.selection_delta_weight > 0 or config.selection_min_delta_weight > 0:
            mean_delta = float(np.mean(given_acc) - max(np.mean(hidden_acc), np.mean(shuffled_acc)))
            min_delta = float(
                min(
                    min(g - h, g - s)
                    for g, h, s in zip(given_acc, hidden_acc, shuffled_acc, strict=True)
                )
            )
            score = float(np.mean(given_acc)) + config.selection_delta_weight * mean_delta + config.selection_min_delta_weight * min_delta
        else:
            score = (
                0.5 * (float(np.mean(hidden_acc)) + float(np.mean(given_acc)))
                if val_balanced_indices is not None
                else 0.5 * (float(np.mean(hidden_scores)) + float(np.mean(given_scores)))
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / max(1, seen),
                "selection_score": score,
                "val_hidden_balanced_accuracy_mean": float(np.mean(hidden_scores)),
                "val_given_balanced_accuracy_mean": float(np.mean(given_scores)),
                "val_shuffled_balanced_accuracy_mean": float(
                    np.mean(shuffled_scores)
                ),
                "val_hidden_accuracy_mean": float(np.mean(hidden_acc)),
                "val_given_accuracy_mean": float(np.mean(given_acc)),
                "val_shuffled_accuracy_mean": float(np.mean(shuffled_acc)),
            }
        )
        if score > best_score + 1e-6:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    if not config.use_last_epoch:
        if best_state is None:
            raise RuntimeError("No checkpoint was selected")
        model.load_state_dict(best_state)
    return model, history


def aggregate_seed_reports(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for task in TASK_ORDER:
        aggregate[task] = {}
        for mode in PROFILE_MODES:
            rows = [seed["tasks"][task]["test"][mode] for seed in seed_reports]
            aggregate[task][mode] = {
                "accuracy_mean": float(np.mean([row["accuracy"] for row in rows])),
                "accuracy_std": float(np.std([row["accuracy"] for row in rows])),
                "balanced_accuracy_mean": float(np.mean([row["balanced_accuracy"] for row in rows])),
                "balanced_accuracy_std": float(np.std([row["balanced_accuracy"] for row in rows])),
                "macro_f1_mean": float(np.mean([row["macro_f1"] for row in rows])),
                "macro_f1_std": float(np.std([row["macro_f1"] for row in rows])),
            }
            if all("paper_balanced_subset" in row for row in rows):
                balanced_rows = [row["paper_balanced_subset"] for row in rows]
                aggregate[task][mode]["paper_balanced_accuracy_mean"] = float(
                    np.mean([row["accuracy"] for row in balanced_rows])
                )
                aggregate[task][mode]["paper_balanced_accuracy_std"] = float(
                    np.std([row["accuracy"] for row in balanced_rows])
                )
        aggregate[task]["given_minus_hidden_accuracy"] = float(
            aggregate[task]["given"]["accuracy_mean"] - aggregate[task]["hidden"]["accuracy_mean"]
        )
        aggregate[task]["given_minus_shuffled_accuracy"] = float(
            aggregate[task]["given"]["accuracy_mean"] - aggregate[task]["shuffled"]["accuracy_mean"]
        )
        aggregate[task]["given_minus_hidden_balanced_accuracy"] = float(
            aggregate[task]["given"]["balanced_accuracy_mean"]
            - aggregate[task]["hidden"]["balanced_accuracy_mean"]
        )
        aggregate[task]["given_minus_shuffled_balanced_accuracy"] = float(
            aggregate[task]["given"]["balanced_accuracy_mean"]
            - aggregate[task]["shuffled"]["balanced_accuracy_mean"]
        )
        aggregate[task]["given_minus_hidden_macro_f1"] = float(
            aggregate[task]["given"]["macro_f1_mean"] - aggregate[task]["hidden"]["macro_f1_mean"]
        )
        aggregate[task]["given_minus_shuffled_macro_f1"] = float(
            aggregate[task]["given"]["macro_f1_mean"] - aggregate[task]["shuffled"]["macro_f1_mean"]
        )
        if "paper_balanced_accuracy_mean" in aggregate[task]["given"]:
            aggregate[task]["given_minus_hidden_paper_balanced_accuracy"] = float(
                aggregate[task]["given"]["paper_balanced_accuracy_mean"]
                - aggregate[task]["hidden"]["paper_balanced_accuracy_mean"]
            )
            aggregate[task]["given_minus_shuffled_paper_balanced_accuracy"] = float(
                aggregate[task]["given"]["paper_balanced_accuracy_mean"]
                - aggregate[task]["shuffled"]["paper_balanced_accuracy_mean"]
            )
    aggregate["overall"] = {}
    for mode in PROFILE_MODES:
        aggregate["overall"][mode] = {
            "accuracy_mean": float(
                np.mean([aggregate[task][mode]["accuracy_mean"] for task in TASK_ORDER])
            ),
            "balanced_accuracy_mean": float(
                np.mean([aggregate[task][mode]["balanced_accuracy_mean"] for task in TASK_ORDER])
            ),
            "macro_f1_mean": float(
                np.mean([aggregate[task][mode]["macro_f1_mean"] for task in TASK_ORDER])
            ),
        }
        if all("paper_balanced_accuracy_mean" in aggregate[task][mode] for task in TASK_ORDER):
            aggregate["overall"][mode]["paper_balanced_accuracy_mean"] = float(
                np.mean(
                    [aggregate[task][mode]["paper_balanced_accuracy_mean"] for task in TASK_ORDER]
                )
            )
    aggregate["overall"]["given_minus_hidden_accuracy"] = float(
        aggregate["overall"]["given"]["accuracy_mean"]
        - aggregate["overall"]["hidden"]["accuracy_mean"]
    )
    aggregate["overall"]["given_minus_shuffled_accuracy"] = float(
        aggregate["overall"]["given"]["accuracy_mean"]
        - aggregate["overall"]["shuffled"]["accuracy_mean"]
    )
    aggregate["overall"]["given_minus_hidden_balanced_accuracy"] = float(
        aggregate["overall"]["given"]["balanced_accuracy_mean"]
        - aggregate["overall"]["hidden"]["balanced_accuracy_mean"]
    )
    aggregate["overall"]["given_minus_shuffled_balanced_accuracy"] = float(
        aggregate["overall"]["given"]["balanced_accuracy_mean"]
        - aggregate["overall"]["shuffled"]["balanced_accuracy_mean"]
    )
    if "paper_balanced_accuracy_mean" in aggregate["overall"]["given"]:
        aggregate["overall"]["given_minus_hidden_paper_balanced_accuracy"] = float(
            aggregate["overall"]["given"]["paper_balanced_accuracy_mean"]
            - aggregate["overall"]["hidden"]["paper_balanced_accuracy_mean"]
        )
        aggregate["overall"]["given_minus_shuffled_paper_balanced_accuracy"] = float(
            aggregate["overall"]["given"]["paper_balanced_accuracy_mean"]
            - aggregate["overall"]["shuffled"]["paper_balanced_accuracy_mean"]
        )
    return aggregate


def make_profile_delta_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in TASK_ORDER:
        rows.append(
            {
                "task": task,
                "given_minus_hidden_accuracy": aggregate[task]["given_minus_hidden_accuracy"],
                "given_minus_shuffled_accuracy": aggregate[task]["given_minus_shuffled_accuracy"],
                "given_minus_hidden_balanced_accuracy": aggregate[task]["given_minus_hidden_balanced_accuracy"],
                "given_minus_shuffled_balanced_accuracy": aggregate[task]["given_minus_shuffled_balanced_accuracy"],
                "given_minus_hidden_macro_f1": aggregate[task]["given_minus_hidden_macro_f1"],
                "given_minus_shuffled_macro_f1": aggregate[task]["given_minus_shuffled_macro_f1"],
                "given_minus_hidden_paper_balanced_accuracy": aggregate[task].get(
                    "given_minus_hidden_paper_balanced_accuracy"
                ),
                "given_minus_shuffled_paper_balanced_accuracy": aggregate[task].get(
                    "given_minus_shuffled_paper_balanced_accuracy"
                ),
            }
        )
    return rows


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    global TASK_ORDER, BINARY_TASKS
    if args.task_scheme == "hierarchy":
        TASK_ORDER = list(HIERARCHY_TASK_ORDER)
        BINARY_TASKS = HIERARCHY_BINARY_TASKS
    elif args.task_scheme == "paper":
        TASK_ORDER = list(PAPER_TASK_ORDER)
        BINARY_TASKS = PAPER_BINARY_TASKS
    else:
        raise ValueError(f"Unknown task scheme: {args.task_scheme}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        profile_dropout=args.profile_dropout,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seeds=tuple(args.seeds),
        device=args.device,
        hidden_ce_weight=args.hidden_ce_weight,
        control_ce_weight=args.control_ce_weight,
        hidden_margin_weight=args.hidden_margin_weight,
        control_margin_weight=args.control_margin_weight,
        margin=args.margin,
        balanced_margin=args.balanced_margin,
        selection_delta_weight=args.selection_delta_weight,
        selection_min_delta_weight=args.selection_min_delta_weight,
        fusion=args.fusion,
        task_specific_branches=args.task_specific_branches,
        use_last_epoch=args.use_last_epoch,
        audio_only=args.audio_only,
        use_profile=not args.disable_profile,
    )
    train = load_cache(Path(args.train_cache))
    val = load_cache(Path(args.val_cache))
    test = load_cache(Path(args.test_cache))

    profile_view_metadata: dict[str, Any] | None = None
    if args.profile_view_dir:
        profile_view_dir = Path(args.profile_view_dir)
        available_profile_views = {
            split: load_cache(profile_view_dir / f"{split}.profile-view.npz")
            for split in ("train", "val", "test")
        }
        profile_views: dict[str, dict[str, np.ndarray]] = {}
        resolved_sidecars: dict[str, str] = {}
        for split, base in (("train", train), ("val", val), ("test", test)):
            base_ids = base["sample_ids"].astype(str)
            matches = [
                name
                for name, candidate in available_profile_views.items()
                if np.array_equal(base_ids, candidate["sample_ids"].astype(str))
            ]
            if not matches:
                raise ValueError(f"No profile view sample IDs match the {split} cache")
            selected_name = split if split in matches else matches[0]
            view = available_profile_views[selected_name]
            profile_views[split] = view
            resolved_sidecars[split] = selected_name
            if view["profile_given"].shape != view["profile_shuffled"].shape:
                raise ValueError(f"Profile view shapes differ for {split}")
            if len(view["profile_given"]) != len(base["sample_ids"]):
                raise ValueError(f"Profile view row count differs for {split}")
        metadata_path = profile_view_dir / "metadata.json"
        profile_view_metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {"profile_view_dir": str(profile_view_dir)}
        )
        profile_view_metadata = {
            **profile_view_metadata,
            "resolved_sidecars_by_cache_role": resolved_sidecars,
        }
    else:
        profile_views = {"train": train, "val": val, "test": test}

    train_context, val_context, test_context = standardize(
        train["qwen_context"], val["qwen_context"], test["qwen_context"]
    )
    audio_layer_presence = ["qwen_audio_layers" in data for data in (train, val, test)]
    if any(audio_layer_presence) and not all(audio_layer_presence):
        raise ValueError("qwen_audio_layers must be present in train, val, and test together")
    if all(audio_layer_presence):
        train_audio_layers, val_audio_layers, test_audio_layers = standardize(
            train["qwen_audio_layers"],
            val["qwen_audio_layers"],
            test["qwen_audio_layers"],
        )
        expected_layer_shape = train_audio_layers.shape[1:]
        if val_audio_layers.shape[1:] != expected_layer_shape or test_audio_layers.shape[1:] != expected_layer_shape:
            raise ValueError("Qwen audio-layer shapes differ across splits")
    else:
        train_audio_layers = val_audio_layers = test_audio_layers = None
    if args.profile_view_dir:
        # Structured one-hot views are already on a stable common scale.  In
        # particular, an <UNK> column absent from train must not be divided by
        # an artificial 1e-5 standard deviation at evaluation time.
        (
            given_train,
            given_val,
            given_test,
            shuffled_train,
            shuffled_val,
            shuffled_test,
        ) = tuple(
            array.astype(np.float32)
            for array in (
                profile_views["train"]["profile_given"],
                profile_views["val"]["profile_given"],
                profile_views["test"]["profile_given"],
                profile_views["train"]["profile_shuffled"],
                profile_views["val"]["profile_shuffled"],
                profile_views["test"]["profile_shuffled"],
            )
        )
    else:
        (
            given_train,
            given_val,
            given_test,
            shuffled_train,
            shuffled_val,
            shuffled_test,
        ) = standardize(
            profile_views["train"]["profile_given"],
            profile_views["val"]["profile_given"],
            profile_views["test"]["profile_given"],
            profile_views["train"]["profile_shuffled"],
            profile_views["val"]["profile_shuffled"],
            profile_views["test"]["profile_shuffled"],
        )
    contrastive_indices: dict[str, list[int] | None] = {"train": None, "val": None, "test": None}
    if args.shuffled_strategy == "contrastive":
        shuffled_train, train_contrastive = make_contrastive_profiles(given_train, train["conversation_ids"])
        shuffled_val, val_contrastive = make_contrastive_profiles(given_val, val["conversation_ids"])
        shuffled_test, test_contrastive = make_contrastive_profiles(given_test, test["conversation_ids"])
        contrastive_indices = {
            "train": train_contrastive.astype(int).tolist(),
            "val": val_contrastive.astype(int).tolist(),
            "test": test_contrastive.astype(int).tolist(),
        }

    train_profiles = {
        "hidden": np.zeros_like(given_train, dtype=np.float32),
        "given": given_train,
        "shuffled": shuffled_train,
    }
    val_profiles = {
        "hidden": np.zeros_like(given_val, dtype=np.float32),
        "given": given_val,
        "shuffled": shuffled_val,
    }
    test_profiles = {
        "hidden": np.zeros_like(given_test, dtype=np.float32),
        "given": given_test,
        "shuffled": shuffled_test,
    }
    if args.task_scheme == "paper":
        missing = [
            split
            for split, data in (("train", train), ("val", val), ("test", test))
            if "paper_targets" not in data
        ]
        if missing:
            raise ValueError(
                "Paper task scheme requires caches with paper_targets; "
                f"missing in {missing}"
            )
        train_targets = build_paper_targets(train["labels"], train["paper_targets"])
        val_targets = build_paper_targets(val["labels"], val["paper_targets"])
        test_targets = build_paper_targets(test["labels"], test["paper_targets"])
    else:
        train_targets = build_multitask_targets(train["labels"])
        val_targets = build_multitask_targets(val["labels"])
        test_targets = build_multitask_targets(test["labels"])

    val_balanced_indices = (
        build_balanced_eval_indices(val_targets, val["sample_ids"])
        if args.task_scheme == "paper"
        else None
    )
    test_balanced_indices = (
        build_balanced_eval_indices(test_targets, test["sample_ids"])
        if args.task_scheme == "paper"
        else None
    )

    seed_reports: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        model, history = train_one(
            train_context=train_context,
            train_profile=train_profiles["given"],
            train_control_profile=train_profiles["shuffled"],
            train_targets=train_targets,
            val_context=val_context,
            val_profiles=val_profiles,
            val_targets=val_targets,
            val_balanced_indices=val_balanced_indices,
            config=config,
            seed=seed,
            train_audio_layers=train_audio_layers,
            val_audio_layers=val_audio_layers,
        )
        validation_reports = evaluate_all(
            model,
            val_context,
            val_profiles,
            val_targets,
            device=torch.device(config.device),
            balanced_indices=val_balanced_indices,
            audio_layers=val_audio_layers,
        )
        test_reports = evaluate_all(
            model,
            test_context,
            test_profiles,
            test_targets,
            device=torch.device(config.device),
            balanced_indices=test_balanced_indices,
            audio_layers=test_audio_layers,
        )
        seed_report: dict[str, Any] = {
            "seed": seed,
            "history": history,
            "tasks": {},
            "learned_audio_layer_weights": model.learned_audio_layer_weights(),
        }
        for task in TASK_ORDER:
            task_idx = TASK_ORDER.index(task)
            indices, task_y = task_targets(test_targets, task)
            balanced_set = (
                set(test_balanced_indices[task].astype(int).tolist())
                if test_balanced_indices is not None
                else set(indices.astype(int).tolist())
            )
            seed_report["tasks"][task] = {
                "description": BINARY_TASKS[task]["description"],
                "train_counts": {
                    "A": int(np.sum(train_targets[:, task_idx] == 0)),
                    "B": int(np.sum(train_targets[:, task_idx] == 1)),
                    "ignored": int(np.sum(train_targets[:, task_idx] == IGNORE_INDEX)),
                },
                "val_counts": {
                    "A": int(np.sum(val_targets[:, task_idx] == 0)),
                    "B": int(np.sum(val_targets[:, task_idx] == 1)),
                    "ignored": int(np.sum(val_targets[:, task_idx] == IGNORE_INDEX)),
                },
                "test_counts": {
                    "A": int(np.sum(test_targets[:, task_idx] == 0)),
                    "B": int(np.sum(test_targets[:, task_idx] == 1)),
                    "ignored": int(np.sum(test_targets[:, task_idx] == IGNORE_INDEX)),
                },
                "paper_balanced_test_counts": (
                    {
                        "A": int(np.sum(test_targets[test_balanced_indices[task], task_idx] == 0)),
                        "B": int(np.sum(test_targets[test_balanced_indices[task], task_idx] == 1)),
                    }
                    if test_balanced_indices is not None
                    else None
                ),
                "validation": validation_reports[task],
                "test": test_reports[task],
            }
            for mode in PROFILE_MODES:
                _, preds, probs = predict_task(
                    model,
                    test_context,
                    test_profiles[mode],
                    test_targets[:, task_idx],
                    task=task,
                    device=torch.device(config.device),
                    audio_layers=test_audio_layers,
                )
                for local_i, sample_index in enumerate(indices):
                    prediction_rows.append(
                        {
                            "seed": seed,
                            "task": task,
                            "profile_mode": mode,
                            "sample_id": str(test["sample_ids"][sample_index]),
                            "conversation_id": str(test["conversation_ids"][sample_index]),
                            "reference_label": LABELS[int(test["labels"][sample_index])],
                            "reference_answer": "A" if int(task_y[local_i]) == 0 else "B",
                            "prediction_answer": "A" if int(preds[local_i]) == 0 else "B",
                            "prob_A": float(probs[local_i, 0]),
                            "prob_B": float(probs[local_i, 1]),
                            "paper_balanced_subset": int(sample_index) in balanced_set,
                        }
                    )
        seed_reports.append(seed_report)
        write_json(output_dir / f"seed-{seed}.json", seed_report)

    aggregate = aggregate_seed_reports(seed_reports)
    delta_rows = make_profile_delta_rows(aggregate)
    summary = {
        "experiment": "qwen_shared_binary_multitask_profile_adapter",
        "task_scheme": args.task_scheme,
        "method": (
            "Frozen Qwen audio-tower boundary layers only; learned softmax layer weighting + "
            f"{'task-specific' if config.task_specific_branches else 'shared'} audio encoder + four binary A/B heads. "
            "No transcript/context branch and no profile branch."
            if config.audio_only and not config.use_profile
            else "Frozen Qwen audio-tower boundary layers only + learned softmax layer weighting + "
            f"{'task-specific' if config.task_specific_branches else 'shared'} profile {config.fusion} adapter + four binary A/B heads. "
            "No transcript/context branch."
            if config.audio_only and config.use_profile
            else
            f"Frozen Qwen causal transcript context + learned weighted sum of frozen Qwen audio layers + "
            f"{'task-specific' if config.task_specific_branches else 'one shared'} "
            f"{config.fusion} profile adapter + four binary A/B heads."
            if train_audio_layers is not None
            else f"Frozen Qwen context/profile embeddings + "
            f"{'task-specific' if config.task_specific_branches else 'one shared'} "
            f"{config.fusion} adapter + four binary A/B heads."
        ),
        "qwen_connection": (
            [
                "Only qwen_audio_layers are consumed: the final causal boundary vector from each frozen Qwen audio-tower layer.",
                "qwen_context, profile_given, and profile_shuffled are loaded only for row alignment and are not consumed by the model.",
                "The four binary heads share the learned audio representation.",
            ]
            if config.audio_only
            else [
                "qwen_context comes from Qwen-Omni hidden representation cached before adapter training.",
                (
                    "profile_given/profile_shuffled come from the external profile view named in profile_view; "
                    "they are not Qwen text embeddings."
                    if args.profile_view_dir
                    else "profile_given/profile_shuffled are Qwen profile embeddings cached from the same original split."
                ),
                (
                    "Each binary task has its own audio-layer weighting and context/profile fusion branch."
                    if config.task_specific_branches
                    else "All four binary heads share the same context/profile fusion layer."
                ),
            ]
        ),
        "audio_layer_weighting": (
            {
                "enabled": True,
                "layers": int(train_audio_layers.shape[1]),
                "dimension": int(train_audio_layers.shape[2]),
                "definition": "trainable softmax weights over convolutional input plus all Qwen audio encoder layers at the final causal frame",
                "weights_by_seed": {
                    str(report["seed"]): report["learned_audio_layer_weights"]
                    for report in seed_reports
                },
            }
            if train_audio_layers is not None
            else {"enabled": False}
        ),
        "output_format": "four binary A/B heads",
        "paper_comparable_metric": (
            "accuracy on a deterministic per-task 50/50 A/B subset; all eligible training samples remain in training"
            if args.task_scheme == "paper"
            else None
        ),
        "test_split": str(Path(args.test_cache).resolve()),
        "same_sample_ids_as_original_prompt_test": True,
        "same_input_modalities_as_original_prompt_test": not config.audio_only,
        "input_contract": (
            "30 s causal audio + profile; transcript/context deliberately absent from every profile condition"
            if config.audio_only and config.use_profile
            else "30 s causal audio only (explicit Talking-Turns-aligned ablation; transcript and profile absent)"
            if config.audio_only and not config.use_profile
            else "causal audio + matching causal partial transcript + profile"
        ),
        "label_order": LABELS,
        "profile_modes": PROFILE_MODES,
        "shuffled_strategy": args.shuffled_strategy,
        "contrastive_indices": contrastive_indices,
        "tasks": serializable_binary_tasks(),
        "config": asdict(config),
        "profile_view": profile_view_metadata or {
            "name": "full_qwen_profile_embedding",
            "source": "profile_given/profile_shuffled arrays in the base Qwen cache",
        },
        "aggregate": aggregate,
        "profile_deltas": delta_rows,
    }
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "test_predictions.jsonl", prediction_rows)

    with (output_dir / "aggregate.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "profile_mode",
                "accuracy_mean",
                "accuracy_std",
                "balanced_accuracy_mean",
                "balanced_accuracy_std",
                "macro_f1_mean",
                "macro_f1_std",
                "paper_balanced_accuracy_mean",
                "paper_balanced_accuracy_std",
            ],
        )
        writer.writeheader()
        for task in TASK_ORDER:
            for mode in PROFILE_MODES:
                row = dict(aggregate[task][mode])
                row.setdefault("paper_balanced_accuracy_mean", None)
                row.setdefault("paper_balanced_accuracy_std", None)
                row["task"] = task
                row["profile_mode"] = mode
                writer.writerow(row)

    with (output_dir / "profile_deltas.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(delta_rows[0].keys()))
        writer.writeheader()
        writer.writerows(delta_rows)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", default="artifacts/qwen-hidden-profile/full-local/cache/train.qwen-hidden.npz")
    parser.add_argument("--val-cache", default="artifacts/qwen-hidden-profile/full-local/cache/val.qwen-hidden.npz")
    parser.add_argument("--test-cache", default="artifacts/qwen-hidden-profile/full-local/cache/test.qwen-hidden.npz")
    parser.add_argument("--output-dir", default="artifacts/main_experiment/manual_run")
    parser.add_argument(
        "--task-scheme",
        choices=["hierarchy", "paper"],
        default="hierarchy",
        help="hierarchy uses old five-way recovery heads; paper uses Talking-Turns-style A/B targets saved in cache.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 13, 29, 37, 71, 101])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--profile-dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--shuffled-strategy", choices=["random", "contrastive"], default="random")
    parser.add_argument("--hidden-ce-weight", type=float, default=0.0)
    parser.add_argument("--control-ce-weight", type=float, default=0.0)
    parser.add_argument("--hidden-margin-weight", type=float, default=0.0)
    parser.add_argument("--control-margin-weight", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument(
        "--balanced-margin",
        action="store_true",
        help="Class-balance profile margin terms so A and B contribute equally, matching paper evaluation.",
    )
    parser.add_argument("--selection-delta-weight", type=float, default=0.0)
    parser.add_argument("--selection-min-delta-weight", type=float, default=0.0)
    parser.add_argument("--fusion", choices=["gate", "concat", "film"], default="gate")
    parser.add_argument(
        "--profile-view-dir",
        default="",
        help="Optional sidecar directory with aligned train/val/test profile-view NPZ files.",
    )
    parser.add_argument(
        "--task-specific-branches",
        action="store_true",
        help="Use one audio-layer weighting and profile-fusion branch per A/B task.",
    )
    parser.add_argument(
        "--use-last-epoch",
        action="store_true",
        help="Return the final epoch instead of selecting a checkpoint on validation.",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Bypass the Qwen transcript/context branch and classify only cached Qwen audio layers.",
    )
    parser.add_argument(
        "--disable-profile",
        action="store_true",
        help="Bypass the profile encoder/fusion branch. Intended for the audio-only topline baseline.",
    )
    args = parser.parse_args()
    summary = run_experiment(args)
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
