"""Qwen LM-head binary A/B profile-embedding adapter.

This script is the practical "A-like" route:

    cached Qwen hidden vector for causal audio + transcript
    + cached Qwen profile embedding
    -> small residual adapter in Qwen hidden space
    -> frozen Qwen lm_head scores token "A" vs token "B"

So the final output is still an A/B answer through Qwen's vocabulary head, not a
new five-class classifier.  The train/val/test split and sample IDs are the same
as the earlier Qwen prompt experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
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


@dataclass
class TrainConfig:
    hidden_dim: int = 256
    dropout: float = 0.10
    profile_dropout: float = 0.25
    epochs: int = 80
    patience: int = 12
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 1e-4
    seeds: tuple[int, ...] = (3, 7, 13, 29, 37, 71, 101)
    device: str = "cpu"
    context_residual_scale: float = 0.50
    profile_residual_scale: float = 0.50
    answer_direction_scale: float = 0.0
    hidden_ce_weight: float = 0.0
    control_ce_weight: float = 0.0
    hidden_margin_weight: float = 0.0
    control_margin_weight: float = 0.0
    margin: float = 0.10
    selection_delta_weight: float = 0.0
    selection_min_delta_weight: float = 0.0


class QwenLmHeadBinaryAdapter(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,
        dropout: float,
        ab_weight: torch.Tensor,
        context_residual_scale: float,
        profile_residual_scale: float,
        answer_direction_scale: float,
    ) -> None:
        super().__init__()
        if ab_weight.shape != (2, dim):
            raise ValueError(f"Expected ab_weight shape (2, {dim}), got {tuple(ab_weight.shape)}")
        self.register_buffer("ab_weight", ab_weight.float())
        ab_direction = (ab_weight[1] - ab_weight[0]).float()
        ab_direction = ab_direction / torch.clamp(torch.linalg.norm(ab_direction), min=1e-8)
        self.register_buffer("ab_direction", ab_direction)
        self.context_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.profile_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.context_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.profile_net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=False),
        )
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid(),
        )
        self.answer_direction_net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.answer_bias = nn.Parameter(torch.zeros(2))
        self.context_residual_scale = context_residual_scale
        self.profile_residual_scale = profile_residual_scale
        self.answer_direction_scale = answer_direction_scale
        self._zero_init_last_layers()

    def _zero_init_last_layers(self) -> None:
        for module in (self.context_net[-1], self.profile_net[-1], self.answer_direction_net[-1]):
            nn.init.zeros_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, context: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        context_norm = self.context_norm(context)
        profile_norm = self.profile_norm(profile)
        present = (profile.abs().sum(dim=-1, keepdim=True) > 0).to(context.dtype)

        context_delta = self.context_net(context_norm)
        profile_delta = self.profile_net(profile_norm) * present
        gate = self.gate(torch.cat([context_norm, profile_norm * present], dim=-1))

        hidden = (
            context
            + self.context_residual_scale * context_delta
            + self.profile_residual_scale * gate * profile_delta
        )
        if self.answer_direction_scale:
            answer_shift = self.answer_direction_net(
                torch.cat([context_norm, profile_norm * present], dim=-1)
            )
            hidden = hidden + self.answer_direction_scale * answer_shift * present * self.ab_direction
        return torch.matmul(hidden.float(), self.ab_weight.t()) + self.answer_bias


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


def task_indices_and_targets(data: dict[str, np.ndarray], task: str) -> tuple[np.ndarray, np.ndarray]:
    indices: list[int] = []
    targets: list[int] = []
    for index, label_id in enumerate(data["labels"]):
        target = label_to_binary(int(label_id), task)
        if target is not None:
            indices.append(index)
            targets.append(target)
    return np.asarray(indices, dtype=np.int64), np.asarray(targets, dtype=np.int64)


def make_contrastive_profiles(
    given_profiles: np.ndarray,
    conversation_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick a deliberately different wrong profile for each sample.

    This changes only the control profile.  The sample ID, label, Qwen context
    vector, and task target stay unchanged.  When possible the replacement
    profile is selected from a different conversation and is farthest by cosine
    similarity.
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


def gold_log_prob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs.gather(1, labels.view(-1, 1)).squeeze(1)


def margin_loss(
    given_logits: torch.Tensor,
    control_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    given_gold = gold_log_prob(given_logits, labels)
    control_gold = gold_log_prob(control_logits, labels)
    return torch.relu(margin - (given_gold - control_gold)).mean()


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


def load_qwen_ab_weight_from_safetensors(model_dir: Path, token_a: int, token_b: int) -> torch.Tensor:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_file_name = index["weight_map"].get("thinker.lm_head.weight")
    if weight_file_name is None:
        raise KeyError("thinker.lm_head.weight not found in model index")
    weight_path = model_dir / weight_file_name
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Missing {weight_path}. Download the Qwen HF shard containing thinker.lm_head.weight first."
        )
    weights = load_file(str(weight_path), device="cpu")
    lm_head = weights["thinker.lm_head.weight"].float()
    return torch.stack([lm_head[token_a], lm_head[token_b]], dim=0).contiguous()


def load_qwen_ab_weight_from_gguf(gguf_path: Path, token_a: int, token_b: int) -> torch.Tensor:
    if not gguf_path.exists():
        raise FileNotFoundError(f"Missing {gguf_path}")
    import gguf  # type: ignore

    reader = gguf.GGUFReader(str(gguf_path), "r")
    output = next((tensor for tensor in reader.tensors if tensor.name == "output.weight"), None)
    if output is None:
        raise KeyError("output.weight not found in GGUF")
    rows = []
    for token_id in (token_a, token_b):
        row = gguf.dequantize(output.data[token_id : token_id + 1], output.tensor_type)[0]
        rows.append(torch.from_numpy(np.asarray(row, dtype=np.float32)))
    return torch.stack(rows, dim=0).contiguous()


def load_qwen_ab_weight(args: argparse.Namespace) -> tuple[torch.Tensor, str]:
    if args.gguf_model:
        gguf_path = Path(args.gguf_model)
        if gguf_path.exists():
            return load_qwen_ab_weight_from_gguf(gguf_path, args.token_a, args.token_b), str(gguf_path.resolve())
    model_dir = Path(args.model_dir)
    return load_qwen_ab_weight_from_safetensors(model_dir, args.token_a, args.token_b), str(model_dir.resolve())


@torch.no_grad()
def predict(
    model: QwenLmHeadBinaryAdapter,
    context: np.ndarray,
    profile: np.ndarray,
    targets: np.ndarray,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(context.astype(np.float32)),
            torch.from_numpy(profile.astype(np.float32)),
        ),
        batch_size=256,
        shuffle=False,
    )
    chunks: list[np.ndarray] = []
    for batch_context, batch_profile in loader:
        logits = model(batch_context.to(device), batch_profile.to(device))
        chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probs = np.concatenate(chunks, axis=0)
    preds = probs.argmax(axis=1).astype(np.int64)
    return binary_metrics(targets, preds, probs), preds, probs


def train_one(
    *,
    train_context: np.ndarray,
    train_profile: np.ndarray,
    train_control_profile: np.ndarray,
    train_y: np.ndarray,
    val_context: np.ndarray,
    val_profiles: dict[str, np.ndarray],
    val_y: np.ndarray,
    config: TrainConfig,
    ab_weight: torch.Tensor,
    seed: int,
) -> tuple[QwenLmHeadBinaryAdapter, list[dict[str, Any]]]:
    set_seed(seed)
    device = torch.device(config.device)
    model = QwenLmHeadBinaryAdapter(
        dim=int(train_context.shape[1]),
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        ab_weight=ab_weight.to(device),
        context_residual_scale=config.context_residual_scale,
        profile_residual_scale=config.profile_residual_scale,
        answer_direction_scale=config.answer_direction_scale,
    ).to(device)
    dataset = TensorDataset(
        torch.from_numpy(train_context.astype(np.float32)),
        torch.from_numpy(train_profile.astype(np.float32)),
        torch.from_numpy(train_control_profile.astype(np.float32)),
        torch.from_numpy(train_y.astype(np.int64)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
    counts = Counter(train_y.tolist())
    weights = torch.tensor(
        [len(train_y) / max(1, 2 * counts.get(index, 0)) for index in range(2)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights / weights.mean())
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for context, profile, control_profile, labels in loader:
            context = context.to(device)
            profile = profile.to(device)
            control_profile = control_profile.to(device)
            labels = labels.to(device)
            if config.profile_dropout > 0:
                drop = torch.rand(profile.shape[0], device=device) < config.profile_dropout
                profile = profile.clone()
                profile[drop] = 0.0
            optimizer.zero_grad(set_to_none=True)
            given_logits = model(context, profile)
            loss = criterion(given_logits, labels)
            if config.hidden_ce_weight > 0:
                hidden_profile = torch.zeros_like(profile)
                hidden_logits = model(context, hidden_profile)
                loss = loss + config.hidden_ce_weight * criterion(hidden_logits, labels)
            else:
                hidden_logits = None
            if config.control_ce_weight > 0:
                control_logits = model(context, control_profile)
                loss = loss + config.control_ce_weight * criterion(control_logits, labels)
            else:
                control_logits = None
            if config.hidden_margin_weight > 0:
                if hidden_logits is None:
                    hidden_logits = model(context, torch.zeros_like(profile))
                loss = loss + config.hidden_margin_weight * margin_loss(
                    given_logits,
                    hidden_logits,
                    labels,
                    margin=config.margin,
                )
            if config.control_margin_weight > 0:
                if control_logits is None:
                    control_logits = model(context, control_profile)
                loss = loss + config.control_margin_weight * margin_loss(
                    given_logits,
                    control_logits,
                    labels,
                    margin=config.margin,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(labels)
            seen += len(labels)

        val_reports = {}
        for mode in PROFILE_MODES:
            report, _, _ = predict(
                model,
                val_context,
                val_profiles[mode],
                val_y,
                device=device,
            )
            val_reports[mode] = report
        hidden_acc = val_reports["hidden"]["accuracy"]
        given_acc = val_reports["given"]["accuracy"]
        shuffled_acc = val_reports["shuffled"]["accuracy"]
        delta_hidden = given_acc - hidden_acc
        delta_shuffled = given_acc - shuffled_acc
        if config.selection_delta_weight or config.selection_min_delta_weight:
            score = (
                given_acc
                + config.selection_delta_weight * 0.5 * (delta_hidden + delta_shuffled)
                + config.selection_min_delta_weight * min(delta_hidden, delta_shuffled)
            )
        else:
            score = 0.5 * (
                val_reports["hidden"]["balanced_accuracy"]
                + val_reports["given"]["balanced_accuracy"]
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / max(1, seen),
                "selection_score": score,
                "val_hidden_balanced_accuracy": val_reports["hidden"]["balanced_accuracy"],
                "val_given_balanced_accuracy": val_reports["given"]["balanced_accuracy"],
                "val_shuffled_balanced_accuracy": val_reports["shuffled"]["balanced_accuracy"],
                "val_hidden_accuracy": hidden_acc,
                "val_given_accuracy": given_acc,
                "val_shuffled_accuracy": shuffled_acc,
                "val_given_minus_hidden_accuracy": delta_hidden,
                "val_given_minus_shuffled_accuracy": delta_shuffled,
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
    if best_state is None:
        raise RuntimeError("No checkpoint was selected")
    model.load_state_dict(best_state)
    return model, history


def aggregate_seed_reports(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for task in BINARY_TASKS:
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
        aggregate[task]["given_minus_hidden_balanced_accuracy"] = float(
            aggregate[task]["given"]["balanced_accuracy_mean"]
            - aggregate[task]["hidden"]["balanced_accuracy_mean"]
        )
        aggregate[task]["given_minus_shuffled_balanced_accuracy"] = float(
            aggregate[task]["given"]["balanced_accuracy_mean"]
            - aggregate[task]["shuffled"]["balanced_accuracy_mean"]
        )
        aggregate[task]["given_minus_hidden_accuracy"] = float(
            aggregate[task]["given"]["accuracy_mean"]
            - aggregate[task]["hidden"]["accuracy_mean"]
        )
        aggregate[task]["given_minus_shuffled_accuracy"] = float(
            aggregate[task]["given"]["accuracy_mean"]
            - aggregate[task]["shuffled"]["accuracy_mean"]
        )
    return aggregate


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
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
        context_residual_scale=args.context_residual_scale,
        profile_residual_scale=args.profile_residual_scale,
        answer_direction_scale=args.answer_direction_scale,
        hidden_ce_weight=args.hidden_ce_weight,
        control_ce_weight=args.control_ce_weight,
        hidden_margin_weight=args.hidden_margin_weight,
        control_margin_weight=args.control_margin_weight,
        margin=args.margin,
        selection_delta_weight=args.selection_delta_weight,
        selection_min_delta_weight=args.selection_min_delta_weight,
    )
    train = load_cache(Path(args.train_cache))
    val = load_cache(Path(args.val_cache))
    test = load_cache(Path(args.test_cache))
    ab_weight, qwen_weight_source = load_qwen_ab_weight(args)

    if train["qwen_context"].shape[1] != ab_weight.shape[1]:
        raise ValueError(
            f"Qwen context dim {train['qwen_context'].shape[1]} does not match lm_head dim {ab_weight.shape[1]}"
        )

    train_profiles = {
        "hidden": np.zeros_like(train["profile_given"], dtype=np.float32),
        "given": train["profile_given"].astype(np.float32),
        "shuffled": train["profile_shuffled"].astype(np.float32),
    }
    val_profiles = {
        "hidden": np.zeros_like(val["profile_given"], dtype=np.float32),
        "given": val["profile_given"].astype(np.float32),
        "shuffled": val["profile_shuffled"].astype(np.float32),
    }
    test_profiles = {
        "hidden": np.zeros_like(test["profile_given"], dtype=np.float32),
        "given": test["profile_given"].astype(np.float32),
        "shuffled": test["profile_shuffled"].astype(np.float32),
    }
    contrastive_indices: dict[str, list[int] | None] = {"train": None, "val": None, "test": None}
    if args.shuffled_strategy == "contrastive":
        train_profiles["shuffled"], train_contrastive = make_contrastive_profiles(
            train_profiles["given"], train["conversation_ids"]
        )
        val_profiles["shuffled"], val_contrastive = make_contrastive_profiles(
            val_profiles["given"], val["conversation_ids"]
        )
        test_profiles["shuffled"], test_contrastive = make_contrastive_profiles(
            test_profiles["given"], test["conversation_ids"]
        )
        contrastive_indices = {
            "train": train_contrastive.astype(int).tolist(),
            "val": val_contrastive.astype(int).tolist(),
            "test": test_contrastive.astype(int).tolist(),
        }

    seed_reports: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        seed_report: dict[str, Any] = {"seed": seed, "tasks": {}}
        for task in BINARY_TASKS:
            train_idx, train_y = task_indices_and_targets(train, task)
            val_idx, val_y = task_indices_and_targets(val, task)
            test_idx, test_y = task_indices_and_targets(test, task)
            model, history = train_one(
                train_context=train["qwen_context"][train_idx],
                train_profile=train_profiles["given"][train_idx],
                train_control_profile=train_profiles["shuffled"][train_idx],
                train_y=train_y,
                val_context=val["qwen_context"][val_idx],
                val_profiles={mode: arr[val_idx] for mode, arr in val_profiles.items()},
                val_y=val_y,
                config=config,
                ab_weight=ab_weight,
                seed=seed,
            )
            task_report: dict[str, Any] = {
                "description": BINARY_TASKS[task]["description"],
                "train_counts": {"A": int(np.sum(train_y == 0)), "B": int(np.sum(train_y == 1))},
                "val_counts": {"A": int(np.sum(val_y == 0)), "B": int(np.sum(val_y == 1))},
                "test_counts": {"A": int(np.sum(test_y == 0)), "B": int(np.sum(test_y == 1))},
                "history": history,
                "validation": {},
                "test": {},
            }
            for split_name, data, context_key, profiles, indices, targets in [
                ("validation", val, "qwen_context", val_profiles, val_idx, val_y),
                ("test", test, "qwen_context", test_profiles, test_idx, test_y),
            ]:
                for mode in PROFILE_MODES:
                    report, preds, probs = predict(
                        model,
                        data[context_key][indices],
                        profiles[mode][indices],
                        targets,
                        device=torch.device(config.device),
                    )
                    task_report[split_name][mode] = report
                    if split_name == "test":
                        for local_i, sample_index in enumerate(indices):
                            prediction_rows.append(
                                {
                                    "seed": seed,
                                    "task": task,
                                    "profile_mode": mode,
                                    "sample_id": str(data["sample_ids"][sample_index]),
                                    "conversation_id": str(data["conversation_ids"][sample_index]),
                                    "reference_label": LABELS[int(data["labels"][sample_index])],
                                    "reference_answer": "A" if int(targets[local_i]) == 0 else "B",
                                    "prediction_answer": "A" if int(preds[local_i]) == 0 else "B",
                                    "prob_A": float(probs[local_i, 0]),
                                    "prob_B": float(probs[local_i, 1]),
                                }
                            )
            seed_report["tasks"][task] = task_report
        seed_reports.append(seed_report)
        write_json(output_dir / f"seed-{seed}.json", seed_report)

    aggregate = aggregate_seed_reports(seed_reports)
    summary = {
        "experiment": "qwen_lm_head_binary_profile_embedding_adapter_A_like",
        "method": (
            "Frozen Qwen context/profile embeddings; train residual adapter in Qwen hidden space; "
            "score token A/B with frozen Qwen thinker.lm_head."
        ),
        "qwen_connection": [
            "qwen_context is the Qwen-Omni hidden vector for causal audio plus matching causal transcript.",
            "profile_given/profile_shuffled are Qwen profile embeddings from the same cached split.",
            "The final A/B probabilities are computed using Qwen's own lm_head rows for token A and token B.",
        ],
        "output_format": "binary A/B token probability for each task",
        "test_split": str(Path(args.test_cache).resolve()),
        "same_as_original_prompt_test": True,
        "label_order": LABELS,
        "profile_modes": PROFILE_MODES,
        "shuffled_strategy": args.shuffled_strategy,
        "contrastive_indices": contrastive_indices,
        "tasks": serializable_binary_tasks(),
        "token_a": args.token_a,
        "token_b": args.token_b,
        "qwen_weight_source": qwen_weight_source,
        "config": asdict(config),
        "aggregate": aggregate,
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
            ],
        )
        writer.writeheader()
        for task in BINARY_TASKS:
            for mode in PROFILE_MODES:
                row = dict(aggregate[task][mode])
                row["task"] = task
                row["profile_mode"] = mode
                writer.writerow(row)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", default="artifacts/qwen-hidden-profile/full-local/cache/train.qwen-hidden.npz")
    parser.add_argument("--val-cache", default="artifacts/qwen-hidden-profile/full-local/cache/val.qwen-hidden.npz")
    parser.add_argument("--test-cache", default="artifacts/qwen-hidden-profile/full-local/cache/test.qwen-hidden.npz")
    parser.add_argument(
        "--model-dir",
        default=(
            "models/huggingface/models--Qwen--Qwen2.5-Omni-3B/"
            "snapshots/f75b40e3da2003cdd6e1829b1f420ca70797c34e"
        ),
    )
    parser.add_argument(
        "--gguf-model",
        default="models/huggingface/Qwen2.5-Omni-3B-GGUF/Qwen2.5-Omni-3B-Q4_K_M.gguf",
    )
    parser.add_argument("--output-dir", default="artifacts/qwen-lm-head-binary-profile-adapter-A-like-20260815")
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 13, 29, 37, 71, 101])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--profile-dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--context-residual-scale", type=float, default=0.50)
    parser.add_argument("--profile-residual-scale", type=float, default=0.50)
    parser.add_argument("--answer-direction-scale", type=float, default=0.0)
    parser.add_argument("--shuffled-strategy", choices=["random", "contrastive"], default="random")
    parser.add_argument("--hidden-ce-weight", type=float, default=0.0)
    parser.add_argument("--control-ce-weight", type=float, default=0.0)
    parser.add_argument("--hidden-margin-weight", type=float, default=0.0)
    parser.add_argument("--control-margin-weight", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--selection-delta-weight", type=float, default=0.0)
    parser.add_argument("--selection-min-delta-weight", type=float, default=0.0)
    parser.add_argument("--token-a", type=int, default=32)
    parser.add_argument("--token-b", type=int, default=33)
    args = parser.parse_args()
    summary = run_experiment(args)
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
