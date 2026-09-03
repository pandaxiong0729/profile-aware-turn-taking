from __future__ import annotations

import numpy as np
import torch

from profile_turntaking.semantic_profile_experiment import (
    SemanticProfileClassifier,
    _prepare_profiles,
    causal_structure_features,
    context_text,
    multiscale_audio_features,
)


def test_centered_profiles_remove_shared_direction() -> None:
    values = np.asarray([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    prepared = _prepare_profiles(values, profile_mean=mean, preprocessing="center_l2")
    assert np.allclose(prepared.mean(axis=0), 0.0)
    assert np.allclose(np.linalg.norm(prepared, axis=1), 1.0)


def test_multiscale_audio_features_have_fixed_shape() -> None:
    short = multiscale_audio_features(np.zeros(1600, dtype=np.float32))
    long = multiscale_audio_features(np.zeros(16000 * 6, dtype=np.float32))
    assert short.shape == long.shape == (120,)


def test_context_text_uses_only_causal_fields() -> None:
    record = {
        "transcript_prefix": "hello",
        "boundary_state_text": "speaker_00 active",
        "causal_asr_transcript": "hello wor",
        "label": "I",
        "profile_text_given": "private profile",
    }
    rendered = context_text(record)
    assert "hello wor" in rendered
    assert "speaker_00 active" in rendered
    assert "private profile" not in rendered
    assert "\nI\n" not in rendered


def test_structure_features_are_fixed_and_causal() -> None:
    features = causal_structure_features(
        {
            "audio_duration_s": 5.9,
            "boundary_state": {
                "active_speakers_at_t": [
                    {"speaker": "speaker_00", "observed_start_s": 5.4}
                ]
            },
            "transcript_units": [
                {"speaker": "speaker_01", "start_s": 1.0, "end_s": 2.0},
                {"speaker": "speaker_00", "start_s": 2.2, "end_s": 5.0},
            ],
            "causal_asr_transcript": "unfinished",
        }
    )
    assert features.shape == (12,)
    assert np.isfinite(features).all()


def test_hidden_profile_is_exact_context_only_path() -> None:
    torch.manual_seed(1)
    model = SemanticProfileClassifier(132, 384, 384, 32, 0.0).eval()
    audio = torch.randn(3, 132)
    context = torch.randn(3, 384)
    zero_profile = torch.zeros(3, 384)
    with torch.no_grad():
        first = model(audio, context, zero_profile)
        second = model(audio, context, zero_profile)
    assert torch.equal(first, second)
    assert first.shape == (3, 5)
