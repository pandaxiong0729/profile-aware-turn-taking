"""PCM WAV input, resampling, statistical features, and smoke audio generation."""

from __future__ import annotations

import hashlib
import math
import wave
from pathlib import Path
from typing import Sequence

import numpy as np

from .schemas import Utterance


def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")


def _resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    target_length = max(1, int(round(samples.size * target_rate / source_rate)))
    source_positions = np.linspace(0.0, 1.0, samples.size, endpoint=False)
    target_positions = np.linspace(0.0, 1.0, target_length, endpoint=False)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def read_wav_window(
    path: str | Path,
    start_s: float,
    end_s: float,
    *,
    target_rate: int = 16_000,
) -> np.ndarray:
    """Read a mono window from PCM WAV, averaging channels and padding as needed."""
    if end_s <= start_s:
        raise ValueError("end_s must be greater than start_s")
    with wave.open(str(path), "rb") as wav:
        source_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        start_frame = max(0, int(math.floor(start_s * source_rate)))
        end_frame = max(start_frame, int(math.ceil(end_s * source_rate)))
        available = wav.getnframes()
        wav.setpos(min(start_frame, available))
        frames = wav.readframes(max(0, min(end_frame, available) - start_frame))
    decoded = _decode_pcm(frames, sample_width)
    if decoded.size and channels > 1:
        decoded = decoded.reshape(-1, channels).mean(axis=1)
    decoded = _resample_linear(decoded, source_rate, target_rate)
    wanted = int(round((end_s - start_s) * target_rate))
    if decoded.size < wanted:
        decoded = np.pad(decoded, (0, wanted - decoded.size))
    return decoded[:wanted].astype(np.float32, copy=False)


def statistical_audio_features(samples: np.ndarray, *, bands: int = 16) -> np.ndarray:
    """Extract a small deterministic feature vector for CPU smoke training."""
    x = np.asarray(samples, dtype=np.float32)
    if x.size == 0:
        return np.zeros(8 + bands, dtype=np.float32)
    rms = float(np.sqrt(np.mean(x * x) + 1e-10))
    mean_abs = float(np.mean(np.abs(x)))
    peak = float(np.max(np.abs(x)))
    zcr = float(np.mean(np.signbit(x[1:]) != np.signbit(x[:-1]))) if x.size > 1 else 0.0
    quarter = max(1, x.size // 4)
    chunk_rms = [
        float(np.sqrt(np.mean(chunk * chunk) + 1e-10))
        for chunk in np.array_split(x, 4)
    ]
    fft_size = min(4096, max(256, 1 << int(math.log2(max(256, min(x.size, 4096))))))
    if x.size < fft_size:
        fft_input = np.pad(x, (0, fft_size - x.size))
    else:
        center = max(0, x.size - fft_size)
        fft_input = x[center : center + fft_size]
    spectrum = np.abs(np.fft.rfft(fft_input * np.hanning(fft_size)))
    spectrum = np.log1p(spectrum)
    spectral_bands = np.array(
        [float(np.mean(part)) for part in np.array_split(spectrum, bands)],
        dtype=np.float32,
    )
    scalar = np.array([rms, mean_abs, peak, zcr, *chunk_rms], dtype=np.float32)
    return np.concatenate([scalar, spectral_bands]).astype(np.float32)


def write_synthetic_conversation(
    path: str | Path,
    utterances: Sequence[Utterance],
    *,
    duration_s: float | None = None,
    sample_rate: int = 16_000,
    seed: int = 13,
) -> Path:
    """Generate deterministic mono speech-like tones aligned to real transcript intervals."""
    if not utterances:
        raise ValueError("At least one utterance is required")
    duration = duration_s or (max(item.end_s for item in utterances) + 0.5)
    total = int(math.ceil(duration * sample_rate))
    audio = np.zeros(total, dtype=np.float32)
    speakers = {speaker: index for index, speaker in enumerate(sorted({u.speaker for u in utterances}))}
    for utterance in utterances:
        start = max(0, int(utterance.start_s * sample_rate))
        end = min(total, int(utterance.end_s * sample_rate))
        if end <= start:
            continue
        length = end - start
        speaker_index = speakers[utterance.speaker]
        digest = hashlib.blake2b(utterance.text.encode("utf-8"), digest_size=2).digest()
        text_offset = int.from_bytes(digest, "little") % 37
        frequency = 125.0 + speaker_index * 55.0 + text_offset
        t = np.arange(length, dtype=np.float32) / sample_rate
        envelope = np.minimum(1.0, np.minimum(t * 25.0, (length / sample_rate - t) * 25.0))
        carrier = np.sin(2.0 * np.pi * frequency * t)
        modulation = 0.55 + 0.45 * np.sin(2.0 * np.pi * 4.2 * t + speaker_index)
        audio[start:end] += 0.11 * envelope * modulation * carrier
    rng = np.random.default_rng(seed)
    audio += rng.normal(0.0, 0.0015, size=audio.shape).astype(np.float32)
    audio = np.clip(audio, -0.95, 0.95)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pcm = (audio * 32767.0).astype("<i2")
    with wave.open(str(destination), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return destination
