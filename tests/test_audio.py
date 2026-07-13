from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from profile_turntaking.audio import (
    read_wav_window,
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
