from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_qwen_shared_binary_multitask_adapter.py"
SPEC = importlib.util.spec_from_file_location("qwen_shared_binary_adapter", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_paper_targets_use_c_vs_event_pairs() -> None:
    MODULE.TASK_ORDER = list(MODULE.PAPER_TASK_ORDER)
    labels = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)  # C, BC, T, I, NA
    cached = np.full((5, 4), MODULE.IGNORE_INDEX, dtype=np.int64)
    cached[3, 3] = 1
    targets = MODULE.build_paper_targets(labels, cached)

    assert targets[:, 0].tolist() == [0, -100, 1, -100, -100]
    assert targets[:, 1].tolist() == [0, 1, -100, -100, -100]
    assert targets[:, 2].tolist() == [0, -100, -100, 1, -100]
    assert targets[:, 3].tolist() == [-100, -100, -100, 1, -100]


def test_balanced_eval_indices_are_fixed_and_balanced() -> None:
    MODULE.TASK_ORDER = list(MODULE.PAPER_TASK_ORDER)
    targets = np.asarray(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, MODULE.IGNORE_INDEX],
        ],
        dtype=np.int64,
    )
    sample_ids = np.asarray(["a", "b", "c", "d", "e"])
    first = MODULE.build_balanced_eval_indices(targets, sample_ids)
    second = MODULE.build_balanced_eval_indices(targets, sample_ids)

    for task_index, task in enumerate(MODULE.TASK_ORDER):
        assert np.array_equal(first[task], second[task])
        selected = targets[first[task], task_index]
        assert int(np.sum(selected == 0)) == int(np.sum(selected == 1))


def test_profile_margin_can_weight_a_and_b_equally() -> None:
    previous_order = MODULE.TASK_ORDER
    MODULE.TASK_ORDER = ["turn_change"]
    try:
        # Three A rows already satisfy the margin; the single B row does not.
        # Inverse-frequency weights make that B row half of the loss instead
        # of only one quarter.
        targets = torch.tensor([[0], [0], [0], [1]], dtype=torch.long)
        given = {
            "turn_change": torch.tensor(
                [[4.0, 0.0], [4.0, 0.0], [4.0, 0.0], [0.0, 0.0]], dtype=torch.float32
            )
        }
        control = {
            "turn_change": torch.tensor(
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float32
            )
        }
        unbalanced = MODULE.multitask_margin_loss(given, control, targets, margin=0.1)
        balanced = MODULE.multitask_margin_loss(
            given,
            control,
            targets,
            margin=0.1,
            class_weights={"turn_change": torch.tensor([2.0 / 3.0, 2.0])},
        )
        assert torch.isclose(unbalanced, torch.tensor(0.025), atol=1e-6)
        assert torch.isclose(balanced, torch.tensor(0.05), atol=1e-6)
    finally:
        MODULE.TASK_ORDER = previous_order


def test_standardize_accumulates_float16_inputs_in_float32() -> None:
    train = np.full((7000, 2), 100.0, dtype=np.float16)
    train[:, 1] += (np.arange(len(train)) % 2).astype(np.float16)
    (standardized,) = MODULE.standardize(train)
    assert standardized.dtype == np.float32
    assert bool(np.isfinite(standardized).all())
    assert np.allclose(standardized.mean(axis=0), 0.0, atol=1e-4)


def test_all_fusions_have_exact_repeatable_hidden_path() -> None:
    MODULE.TASK_ORDER = list(MODULE.PAPER_TASK_ORDER)
    torch.manual_seed(7)
    context = torch.randn(5, 16)
    profile = torch.randn(5, 8)
    for fusion in ("gate", "concat", "film"):
        model = MODULE.SharedBinaryMultiHeadAdapter(
            context_dim=16,
            profile_dim=8,
            hidden_dim=32,
            dropout=0.0,
            fusion=fusion,
        ).eval()
        with torch.no_grad():
            outputs = model(context, profile)
            hidden_first = model(context, torch.zeros_like(profile))
            hidden_second = model(context, torch.zeros_like(profile))
        assert all(value.shape == (5, 2) for value in outputs.values())
        assert all(torch.equal(hidden_first[task], hidden_second[task]) for task in MODULE.TASK_ORDER)


def test_layer_weighted_adapter_uses_one_shared_softmax_over_audio_layers() -> None:
    MODULE.TASK_ORDER = list(MODULE.PAPER_TASK_ORDER)
    torch.manual_seed(11)
    model = MODULE.SharedBinaryMultiHeadAdapter(
        context_dim=16,
        profile_dim=8,
        hidden_dim=32,
        dropout=0.0,
        fusion="film",
        audio_layer_count=4,
        audio_layer_dim=12,
    ).eval()
    context = torch.randn(5, 16)
    profile = torch.randn(5, 8)
    audio_layers = torch.randn(5, 4, 12)
    with torch.no_grad():
        outputs = model(context, profile, audio_layers)
    weights = model.learned_audio_layer_weights()
    assert weights is not None
    assert len(weights) == 4
    assert abs(sum(weights) - 1.0) < 1e-6
    assert all(value.shape == (5, 2) for value in outputs.values())


def test_layer_weight_profile_and_binary_head_receive_gradients_together() -> None:
    MODULE.TASK_ORDER = list(MODULE.PAPER_TASK_ORDER)
    torch.manual_seed(19)
    model = MODULE.SharedBinaryMultiHeadAdapter(
        context_dim=10,
        profile_dim=6,
        hidden_dim=16,
        dropout=0.0,
        fusion="film",
        audio_layer_count=3,
        audio_layer_dim=7,
    )
    outputs = model(
        torch.randn(8, 10),
        torch.randn(8, 6),
        torch.randn(8, 3, 7),
    )
    targets = torch.arange(8) % 2
    loss = sum(torch.nn.functional.cross_entropy(logits, targets) for logits in outputs.values())
    loss.backward()
    assert model.audio_layer_logits.grad is not None
    assert bool(torch.any(model.audio_layer_logits.grad != 0))
    assert model.context_encoder[0].weight.grad is not None
    assert model.profile_encoder[0].weight.grad is not None
    assert all(model.heads[task].weight.grad is not None for task in MODULE.TASK_ORDER)


def test_audio_only_baseline_ignores_context_and_profile_exactly() -> None:
    MODULE.TASK_ORDER = list(MODULE.PAPER_TASK_ORDER)
    torch.manual_seed(31)
    model = MODULE.SharedBinaryMultiHeadAdapter(
        context_dim=10,
        profile_dim=6,
        hidden_dim=16,
        dropout=0.0,
        fusion="gate",
        audio_layer_count=3,
        audio_layer_dim=7,
        audio_only=True,
        use_profile=False,
    ).eval()
    audio_layers = torch.randn(8, 3, 7)
    with torch.no_grad():
        first = model(torch.randn(8, 10), torch.randn(8, 6), audio_layers)
        second = model(torch.randn(8, 10), torch.randn(8, 6), audio_layers)
    assert all(torch.equal(first[task], second[task]) for task in MODULE.TASK_ORDER)


def test_audio_only_baseline_requires_audio_layers() -> None:
    MODULE.TASK_ORDER = list(MODULE.PAPER_TASK_ORDER)
    with pytest.raises(ValueError):
        MODULE.SharedBinaryMultiHeadAdapter(
            context_dim=10,
            profile_dim=6,
            hidden_dim=16,
            dropout=0.0,
            audio_only=True,
        )


def test_task_specific_branches_use_one_softmax_and_fusion_path_per_task() -> None:
    MODULE.TASK_ORDER = list(MODULE.PAPER_TASK_ORDER)
    torch.manual_seed(23)
    model = MODULE.SharedBinaryMultiHeadAdapter(
        context_dim=10,
        profile_dim=6,
        hidden_dim=16,
        dropout=0.0,
        fusion="gate",
        audio_layer_count=5,
        audio_layer_dim=7,
        task_specific_branches=True,
    )
    outputs = model(
        torch.randn(8, 10),
        torch.randn(8, 6),
        torch.randn(8, 5, 7),
    )
    targets = torch.arange(8) % 2
    loss = sum(torch.nn.functional.cross_entropy(logits, targets) for logits in outputs.values())
    loss.backward()

    weights = model.learned_audio_layer_weights()
    assert isinstance(weights, dict)
    assert set(weights) == set(MODULE.TASK_ORDER)
    assert all(len(weights[task]) == 5 for task in MODULE.TASK_ORDER)
    assert all(abs(sum(weights[task]) - 1.0) < 1e-6 for task in MODULE.TASK_ORDER)
    assert model.audio_layer_logits.grad is not None
    assert model.audio_layer_logits.grad.shape == (len(MODULE.TASK_ORDER), 5)
    assert all(model.gates[task].weight.grad is not None for task in MODULE.TASK_ORDER)


def test_task_specific_hidden_path_is_repeatable_for_all_fusions() -> None:
    MODULE.TASK_ORDER = list(MODULE.PAPER_TASK_ORDER)
    torch.manual_seed(29)
    context = torch.randn(4, 10)
    profile = torch.zeros(4, 6)
    audio_layers = torch.randn(4, 3, 7)
    for fusion in ("gate", "concat", "film"):
        model = MODULE.SharedBinaryMultiHeadAdapter(
            context_dim=10,
            profile_dim=6,
            hidden_dim=16,
            dropout=0.0,
            fusion=fusion,
            audio_layer_count=3,
            audio_layer_dim=7,
            task_specific_branches=True,
        ).eval()
        with torch.no_grad():
            first = model(context, profile, audio_layers)
            second = model(context, profile, audio_layers)
        assert all(torch.equal(first[task], second[task]) for task in MODULE.TASK_ORDER)
