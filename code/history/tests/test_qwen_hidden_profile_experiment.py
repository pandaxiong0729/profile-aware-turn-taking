from __future__ import annotations

import numpy as np
import torch

from profile_turntaking.constants import LABELS
from profile_turntaking.qwen_hidden_profile_experiment import (
    QwenHiddenTrainConfig,
    _pool_masked_hidden,
    build_qwen_context_prompt,
    build_qwen_profile_prompt,
    run_qwen_hidden_profile_experiment,
)


def test_masked_audio_pooling_uses_each_samples_own_boundary() -> None:
    hidden = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [9.0, 9.0]],
            [[4.0, 4.0], [5.0, 5.0], [8.0, 8.0], [9.0, 9.0]],
        ]
    )
    mask = torch.tensor(
        [
            [False, True, True, False],
            [True, True, False, False],
        ]
    )

    np.testing.assert_allclose(
        _pool_masked_hidden(hidden, mask, mode="last"),
        np.asarray([[3.0, 3.0], [5.0, 5.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        _pool_masked_hidden(hidden, mask, mode="mean"),
        np.asarray([[2.5, 2.5], [4.5, 4.5]], dtype=np.float32),
    )


def test_qwen_context_prompt_excludes_profile_text() -> None:
    record = {
        "transcript_prefix": "[speaker_00 0.00-1.00] hello",
        "boundary_state_text": "speaker_00 is active",
        "causal_asr_transcript": "hello",
        "forecast_offset_ms": 100,
        "evaluation_window_ms": 500,
    }
    profile = "speaker_00 has age group 25-34 and speaker_01 has age group 35-44."

    context_prompt = build_qwen_context_prompt(record)
    profile_prompt = build_qwen_profile_prompt(profile)

    assert "hello" in context_prompt
    assert "speaker_00 is active" in context_prompt
    assert profile not in context_prompt
    assert profile in profile_prompt


def _write_fake_cache(path, *, prefix: str, conversations: list[str], per_class: int) -> None:
    rng = np.random.default_rng(abs(hash(prefix)) % (2**32))
    labels = []
    sample_ids = []
    conversation_ids = []
    qwen = []
    given = []
    shuffled = []
    for label_index, label in enumerate(LABELS):
        for item in range(per_class):
            labels.append(label_index)
            sample_ids.append(f"{prefix}-{label}-{item}")
            conversation_ids.append(conversations[(label_index + item) % len(conversations)])
            base = np.zeros(16, dtype=np.float32)
            base[label_index] = 4.0
            qwen.append(base + rng.normal(0.0, 0.05, size=16).astype(np.float32))
            prof = np.zeros(8, dtype=np.float32)
            prof[label_index % 4] = 1.0
            given.append(prof + rng.normal(0.0, 0.01, size=8).astype(np.float32))
            shuffled.append(np.roll(prof, 1) + rng.normal(0.0, 0.01, size=8).astype(np.float32))
    np.savez_compressed(
        path,
        sample_ids=np.asarray(sample_ids),
        conversation_ids=np.asarray(conversation_ids),
        labels=np.asarray(labels, dtype=np.int64),
        qwen_context=np.stack(qwen).astype(np.float32),
        profile_given=np.stack(given).astype(np.float32),
        profile_shuffled=np.stack(shuffled).astype(np.float32),
    )


def test_qwen_hidden_profile_experiment_trains_on_cached_vectors(tmp_path) -> None:
    train_cache = tmp_path / "train.npz"
    val_cache = tmp_path / "val.npz"
    test_cache = tmp_path / "test.npz"
    _write_fake_cache(train_cache, prefix="train", conversations=["A", "B"], per_class=8)
    _write_fake_cache(val_cache, prefix="val", conversations=["C"], per_class=3)
    _write_fake_cache(test_cache, prefix="test", conversations=["D"], per_class=3)

    summary = run_qwen_hidden_profile_experiment(
        train_cache,
        val_cache,
        test_cache,
        tmp_path / "out",
        config=QwenHiddenTrainConfig(
            hidden_dimension=32,
            dropout=0.0,
            profile_dropout=0.3,
            epochs=8,
            patience=3,
            batch_size=16,
            seeds=(13,),
            device="cpu",
            fusion="gate",
        ),
    )

    assert summary["samples"] == {"train": 40, "val": 15, "test": 15}
    assert summary["aggregate"]["hidden"]["all_seeds_noncollapsed"]
    assert (tmp_path / "out" / "summary.json").is_file()
    assert (tmp_path / "out" / "predictions.jsonl").is_file()
    assert (tmp_path / "out" / "profile_comparison.csv").is_file()
