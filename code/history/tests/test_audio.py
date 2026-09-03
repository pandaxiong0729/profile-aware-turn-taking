from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from profile_turntaking.audio import (
    read_wav_window,
    read_wav_window_robust_mix,
    statistical_audio_features,
    write_synthetic_conversation,
)
from profile_turntaking.schemas import Utterance


def test_synthetic_audio_round_trip(tmp_path: Path) -> None:
    destination = tmp_path / "sample.wav"
    utterances = [
        Utterance(0.0, 0.5, "speaker_A", "hello"),
        Utterance(0.4, 1.0, "speaker_B", "overlap"),
    ]
    write_synthetic_conversation(destination, utterances, duration_s=1.2)
    with wave.open(str(destination), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 16_000
    window = read_wav_window(destination, 0.0, 1.2)
    assert window.shape == (19_200,)
    assert float(np.max(np.abs(window))) > 0.01
    features = statistical_audio_features(window)
    assert features.shape == (24,)
    assert np.isfinite(features).all()


def test_robust_mix_preserves_opposite_polarity_stereo(tmp_path: Path) -> None:
    destination = tmp_path / "opposite-polarity.wav"
    sample_rate = 16_000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    tone = 0.2 * np.sin(2.0 * np.pi * 220.0 * t)
    stereo = np.stack([tone, -tone], axis=1)
    with wave.open(str(destination), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(np.round(stereo * 32767.0).astype("<i2").tobytes())

    naive = read_wav_window(destination, 0.0, 1.0)
    robust = read_wav_window_robust_mix(destination, 0.0, 1.0)
    assert float(np.sqrt(np.mean(naive * naive))) < 1e-4
    assert float(np.sqrt(np.mean(robust * robust))) > 0.1
