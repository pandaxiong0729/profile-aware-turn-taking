"""Qwen-connected binary A/B profile-embedding adapter.

This script implements the "B" route:

    frozen Qwen context vector (causal audio + transcript)
    + frozen Qwen profile embedding
    -> small trainable adapter
    -> A/B binary answer

The test split is the same `sbcsae_semantic_profile_v1/test` split used by the
earlier Qwen prompt experiment.  The script also converts the old five-class
prompt predictions into the same binary tasks for a directly comparable prompt
baseline table.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
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
    dropout: float = 0.15
    profile_dropout: float = 0.25
    epochs: int = 80
    patience: int = 12
    batch_size: int = 64
    lr: float = 8e-4
    weight_decay: float = 1e-4
    seeds: tuple[int, ...] = (3, 7, 13, 29, 37, 71, 101)
    fusion: str = "gate"
    device: str = "cpu"


class BinaryProfileAdapter(nn.Module):
    def __init__(
        self,
        context_dim: int,
        profile_dim: int,
        hidden_dim: int,
        dropout: float,
        *,
        fusion: str,
    ) -> None:
        super().__init__()
        if fusion not in {"gate", "concat"}:
            raise ValueError("fusion must be gate or concat")
        self.fusion = fusion
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.profile_encoder = nn.Sequential(
            nn.Linear(profile_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, elementwise_affine=False),
        )
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.concat = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, context: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        shared = self.context_encoder(context)
        profile_state = self.profile_encoder(profile)
        present = (profile.abs().sum(dim=-1, keepdim=True) > 0).to(profile_state.dtype)
        profile_state = profile_state * present
        if self.fusion == "concat":
            fused = self.concat(torch.cat([shared, profile_state], dim=-1))
        else:
            gate = torch.sigmoid(self.gate(torch.cat([shared, profile_state], dim=-1)))
            fused = self.out_norm(shared + gate * profile_state)
        return self.classifier(fused)


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


def standardize(
    train: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(train.std(axis=0, keepdims=True).astype(np.float32), 1e-5)
    return tuple(((x.astype(np.float32) - mean) / std) for x in (train, *others))


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


@torch.no_grad()
def predict(
    model: BinaryProfileAdapter,
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
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    *,
    train_idx: np.ndarray,
    train_y: np.ndarray,
    val_idx: np.ndarray,
    val_y: np.ndarray,
    train_context: np.ndarray,
    val_context: np.ndarray,
    train_profile: np.ndarray,
    val_profiles: dict[str, np.ndarray],
    config: TrainConfig,
    seed: int,
) -> tuple[BinaryProfileAdapter, list[dict[str, Any]]]:
    set_seed(seed)
    device = torch.device(config.device)
    model = BinaryProfileAdapter(
        context_dim=int(train_context.shape[1]),
        profile_dim=int(train_profile.shape[1]),
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        fusion=config.fusion,
    ).to(device)
    dataset = TensorDataset(
        torch.from_numpy(train_context.astype(np.float32)),
        torch.from_numpy(train_profile.astype(np.float32)),
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
        for context, profile, labels in loader:
            context = context.to(device)
            profile = profile.to(device)
            labels = labels.to(device)
            if config.profile_dropout > 0:
                drop = torch.rand(profile.shape[0], device=device) < config.profile_dropout
                profile = profile.clone()
                profile[drop] = 0.0
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(context, profile), labels)
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
    return aggregate


def convert_prompt_fiveclass_to_binary(
    prompt_predictions_csv: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if not prompt_predictions_csv.exists():
        return {"available": False, "reason": f"missing {prompt_predictions_csv}"}
    rows = list(csv.DictReader(prompt_predictions_csv.open("r", encoding="utf-8")))
    result: dict[str, Any] = {"available": True, "source": str(prompt_predictions_csv.resolve()), "tasks": {}}
    converted_rows: list[dict[str, Any]] = []
    for task in BINARY_TASKS:
        result["tasks"][task] = {}
        for mode in PROFILE_MODES:
            targets: list[int] = []
            preds: list[int] = []
            for row in rows:
                target = label_to_binary(LABELS.index(row["reference_label"]), task)
                pred_label = row[f"{mode}_prediction"]
                pred = label_to_binary(LABELS.index(pred_label), task)
                if target is None or pred is None:
                    continue
                targets.append(target)
                preds.append(pred)
                converted_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "profile_mode": mode,
                        "task": task,
                        "reference_answer": "A" if target == 0 else "B",
                        "prediction_answer": "A" if pred == 0 else "B",
                        "reference_label": row["reference_label"],
                        "prediction_label": pred_label,
                    }
                )
            targets_arr = np.asarray(targets, dtype=np.int64)
            preds_arr = np.asarray(preds, dtype=np.int64)
            probs = np.eye(2, dtype=np.float32)[preds_arr] if len(preds_arr) else np.zeros((0, 2), dtype=np.float32)
            result["tasks"][task][mode] = binary_metrics(targets_arr, preds_arr, probs)
    write_jsonl(output_dir / "prompt_fiveclass_converted_to_binary_predictions.jsonl", converted_rows)
    write_json(output_dir / "prompt_fiveclass_converted_to_binary_summary.json", result)
    return result


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
        fusion=args.fusion,
        device=args.device,
    )
    train = load_cache(Path(args.train_cache))
    val = load_cache(Path(args.val_cache))
    test = load_cache(Path(args.test_cache))

    # Standardize Qwen vectors using train statistics only.
    train_context, val_context, test_context = standardize(
        train["qwen_context"], val["qwen_context"], test["qwen_context"]
    )
    # Use the same profile scale for given and shuffled conditions.  Otherwise
    # the control would change both profile content and preprocessing.
    (
        given_train,
        given_val,
        given_test,
        shuffled_train,
        shuffled_val,
        shuffled_test,
    ) = standardize(
        train["profile_given"],
        val["profile_given"],
        test["profile_given"],
        train["profile_shuffled"],
        val["profile_shuffled"],
        test["profile_shuffled"],
    )
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

    seed_reports: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        seed_report: dict[str, Any] = {"seed": seed, "tasks": {}}
        for task in BINARY_TASKS:
            train_idx, train_y = task_indices_and_targets(train, task)
            val_idx, val_y = task_indices_and_targets(val, task)
            test_idx, test_y = task_indices_and_targets(test, task)
            model, history = train_one(
                train,
                val,
                train_idx=train_idx,
                train_y=train_y,
                val_idx=val_idx,
                val_y=val_y,
                train_context=train_context[train_idx],
                val_context=val_context[val_idx],
                train_profile=train_profiles["given"][train_idx],
                val_profiles={mode: arr[val_idx] for mode, arr in val_profiles.items()},
                config=config,
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
            for split_name, data, context, profiles, indices, targets in [
                ("validation", val, val_context, val_profiles, val_idx, val_y),
                ("test", test, test_context, test_profiles, test_idx, test_y),
            ]:
                for mode in PROFILE_MODES:
                    report, preds, probs = predict(
                        model,
                        context[indices],
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
    prompt_binary = convert_prompt_fiveclass_to_binary(
        Path(args.prompt_predictions_csv),
        output_dir,
    )
    summary = {
        "experiment": "qwen_binary_profile_embedding_adapter_B",
        "method": "Frozen Qwen context embeddings + frozen Qwen profile embeddings + trainable binary A/B adapter.",
        "qwen_connection": [
            "qwen_context comes from Qwen-Omni hidden representation of causal audio plus matching causal transcript.",
            "profile_given/profile_shuffled are Qwen profile embeddings cached from the same original split.",
            "The trainable module only maps Qwen vectors to A/B; it does not use hand-crafted LinearSVC features.",
        ],
        "output_format": "binary A/B for each task",
        "test_split": str(Path(args.test_cache).resolve()),
        "same_as_original_prompt_test": True,
        "label_order": LABELS,
        "profile_modes": PROFILE_MODES,
        "tasks": serializable_binary_tasks(),
        "config": asdict(config),
        "aggregate": aggregate,
        "prompt_fiveclass_converted_to_binary": prompt_binary,
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
    parser.add_argument("--prompt-predictions-csv", default="artifacts/qwen-hidden-profile/prompt-on-semantic-test-250/predictions.csv")
    parser.add_argument("--output-dir", default="artifacts/qwen-binary-embedding-adapter-B-20260815")
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 13, 29, 37, 71, 101])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fusion", choices=["gate", "concat"], default="gate")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--profile-dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    summary = run_experiment(args)
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
